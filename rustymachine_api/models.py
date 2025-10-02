import numpy as np
import cupy as cp
# This imports the compiled Rust extension module
import rusty_machine

class LinearRegression:
    """
    Linear Regression model powered by a Rust/CUDA backend with zero-copy data transfer.

    This class provides a Scikit-learn style interface. It accepts either NumPy or
    CuPy arrays as input. Data is managed on the GPU via CuPy, and its device
    pointers are passed directly to the high-performance Rust solver, eliminating
    unnecessary host-device data copies during computation.
    """
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit the linear model using the GPU-accelerated backend.

        This method moves data to the GPU (if not already there), adds the
        intercept term, and then calls the Rust backend with device pointers
        to solve the Normal Equation entirely on the GPU.

        Args:
            X (np.ndarray or cp.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray or cp.ndarray): Target values of shape (n_samples, 1).
        """
        if isinstance(X, np.ndarray):
            X_gpu = cp.asarray(X, dtype=cp.float32)
        elif isinstance(X, cp.ndarray):
            X_gpu = X.astype(cp.float32)
        else:
            raise TypeError("Input X must be a NumPy or CuPy array.")
            
        if isinstance(y, np.ndarray):
            y_gpu = cp.asarray(y, dtype=cp.float32)
        elif isinstance(y, cp.ndarray):
            y_gpu = y.astype(cp.float32)
        else:
            raise TypeError("Input y must be a NumPy or CuPy array.")
        
        # Add intercept term to X directly on the GPU
        ones = cp.ones((X_gpu.shape[0], 1), dtype=cp.float32)
        X_b_gpu = cp.hstack([ones, X_gpu])
        samples, features = X_b_gpu.shape

        print("🚀 Fitting model on the GPU (zero-copy)...")
        # Allocate an output buffer for theta on the GPU
        theta_gpu = cp.empty((features, 1), dtype=cp.float32)
        
        # Call the device-level solver in Rust, passing only the memory pointers.
        # The `.data.ptr` attribute provides the raw device pointer address.
        rusty_machine.solve_normal_equation_device(
            X_b_gpu.data.ptr, 
            y_gpu.data.ptr,
            theta_gpu.data.ptr,
            samples, 
            features
        )
        
        # The computation is complete. Copy the final result back to the host CPU.
        theta_host = theta_gpu.get()
        self.intercept_ = theta_host[0, 0]
        self.coef_ = theta_host[1:].flatten()

        print("✅ Model fitting complete.")
        return self

    def predict(self, X):
        """
        Predict using the linear model with GPU acceleration via CuPy.

        This method leverages CuPy's highly optimized matrix multiplication
        to perform predictions entirely on the GPU for maximum performance.

        Args:
            X (np.ndarray or cp.ndarray): Samples to predict of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values, returned as a NumPy array on the host.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("You must call fit() before predicting.")
            
        if isinstance(X, np.ndarray):
            X_gpu = cp.asarray(X, dtype=cp.float32)
        elif isinstance(X, cp.ndarray):
            X_gpu = X.astype(cp.float32)
        else:
            raise TypeError("Input X must be a NumPy or CuPy array.")

        # Reconstruct the full theta coefficient vector on the GPU
        intercept_gpu = cp.array([self.intercept_], dtype=cp.float32)
        coef_gpu = cp.asarray(self.coef_.reshape(-1, 1), dtype=cp.float32)
        theta_gpu = cp.vstack([intercept_gpu, coef_gpu])

        # Add the intercept term to the input matrix X on the GPU
        ones = cp.ones((X_gpu.shape[0], 1), dtype=cp.float32)
        X_b_gpu = cp.hstack([ones, X_gpu])
        
        # Perform prediction directly on the GPU using CuPy's matmul operator.
        predictions_gpu = X_b_gpu @ theta_gpu
        
        # Return the final predictions back to the host as a NumPy array.
        return predictions_gpu.get()