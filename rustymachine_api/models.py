import numpy as np
import cupy as cp
import rusty_machine

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(y, dtype=cp.float32).reshape(-1, 1)

        # Add intercept term to X
        ones = cp.ones((X_gpu.shape[0], 1), dtype=cp.float32)
        X_b_gpu = cp.hstack([ones, X_gpu])
        samples, features = X_b_gpu.shape

        theta_gpu = cp.empty((features, 1), dtype=cp.float32)

        rusty_machine.solve_normal_equation_device(
            X_b_gpu.data.ptr,
            y_gpu.data.ptr,
            theta_gpu.data.ptr,
            samples,
            features
        )

        theta_host = theta_gpu.get()
        self.intercept_ = theta_host[0, 0]
        self.coef_ = theta_host[1:].flatten()

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("The model has not been fitted yet. Call fit() before predict().")

        # Ensure X is a numpy array for the dot product, not cupy
        X_np = cp.asnumpy(X) if isinstance(X, cp.ndarray) else X
        
        logits = X_np @ self.coef_ + self.intercept_
        return logits.reshape(-1, 1)


class LogisticRegression:
    def __init__(self, penalty='l2', C=1.0, epochs=1000, tol=1e-4, lr=0.01, solver='gd'):
        """
        GPU-Accelerated Logistic Regression.

        Args:
            penalty (str): The type of regularization. Can be 'l1', 'l2', or None.
            C (float): Inverse of regularization strength.
            epochs (int): Number of passes over the training data.
            tol (float): Tolerance for stopping criteria (only for 'gd' solver).
            lr (float): Learning rate for the optimization algorithm.
            solver (str): The optimization algorithm to use. Can be 'gd' (Gradient Descent)
                          or 'cd' (Fused Coordinate Descent).
        """
        self.coef_ = None
        self.intercept_ = None
        self.penalty = penalty
        self.C = C
        self.epochs = epochs
        self.tol = tol
        self.lr = lr
        self.solver = solver

    def fit(self, X, y):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(y, dtype=cp.float32).reshape(-1, 1)

        # Add intercept term to X
        ones = cp.ones((X_gpu.shape[0], 1), dtype=cp.float32)
        X_b_gpu = cp.hstack([ones, X_gpu])
        samples, features = X_b_gpu.shape

        theta_gpu = cp.zeros((features, 1), dtype=cp.float32)
        
        # --- NEW: Solver selection logic ---
        if self.solver == 'cd':
            if self.penalty not in ['l1', None]:
                raise ValueError("Coordinate Descent ('cd') solver currently only supports 'l1' penalty.")
            
            # The 'cd' solver requires a column-major (Fortran-style) layout for X
            # This is a critical step for performance with the new CUDA kernel
            X_b_gpu_col_major = cp.asfortranarray(X_b_gpu)
            
            l1_penalty = 1.0 / self.C if self.penalty == 'l1' else 0.0

            rusty_machine.train_logistic_sgd_l1_gpu(
                X_b_gpu_col_major.data.ptr,
                y_gpu.data.ptr,
                theta_gpu.data.ptr,
                samples,
                features,
                self.epochs, # Note: max_iter is used as epochs here
                self.lr,
                l1_penalty
            )
        
        elif self.solver == 'gd':
            l1 = 1.0 / self.C if self.penalty == 'l1' else 0.0
            l2 = 1.0 / self.C if self.penalty == 'l2' else 0.0

            rusty_machine.train_logistic_gpu(
                X_b_gpu.data.ptr,
                y_gpu.data.ptr,
                theta_gpu.data.ptr,
                samples,
                features,
                self.epochs, # Note: max_iter is used as epochs here
                self.lr,
                self.tol,
                l1,
                l2
            )
        else:
            raise ValueError(f"Unknown solver '{self.solver}'. Choose 'gd' or 'cd'.")

        theta_host = theta_gpu.get()
        self.intercept_ = theta_host[0, 0]
        self.coef_ = theta_host[1:].flatten()

        return self

    def predict_proba(self, X):
        if self.coef_ is None:
            raise RuntimeError("The model has not been fitted yet. Call fit() before predict_proba().")

        X_np = cp.asnumpy(X) if isinstance(X, cp.ndarray) else X

        # Calculate logits using numpy
        logits = X_np @ self.coef_ + self.intercept_
        
        # Improved numerical stability for the sigmoid function
        logits_clipped = np.clip(logits, -20, 20)
        probs = 1 / (1 + np.exp(-logits_clipped))
        
        # Return probabilities for both classes
        return np.hstack([1 - probs.reshape(-1, 1), probs.reshape(-1, 1)])

    def predict(self, X):
        # Predicts the class with the highest probability
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)