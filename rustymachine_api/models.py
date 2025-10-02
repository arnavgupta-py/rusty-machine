import numpy as np
# This imports the compiled Rust extension module we've been building
import rusty_machine

class LinearRegression:
    """
    Linear Regression model powered by a Rust/CUDA backend.

    This class provides a Scikit-learn style interface for a Linear Regression
    model that computes the solution to the Normal Equation entirely on the GPU.
    """
    def __init__(self):
        # These will store the model coefficients (theta) after fitting
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit the linear model using the GPU-accelerated backend.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples, 1).
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Input must be NumPy arrays.")

        # Add the intercept column (a column of ones) to the feature matrix X
        X_b = np.c_[np.ones((X.shape[0], 1)), X].astype(np.float32)
        samples, features = X_b.shape

        # Flatten the NumPy arrays into 1D lists for the Rust function
        x_flat = X_b.flatten().tolist()
        y_flat = y.flatten().tolist()

        print("🚀 Fitting model on the GPU...")
        # This is the call to our complete Rust solver
        theta = rusty_machine.solve_normal_equation(x_flat, y_flat, samples, features)

        # Unpack the results into the standard scikit-learn attributes
        self.intercept_ = theta[0]
        self.coef_ = np.array(theta[1:])

        print("✅ Model fitting complete.")
        return self

    def predict(self, X):
        """
        Predict using the linear model with GPU acceleration.

        Args:
            X (np.ndarray): Samples to predict of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("You must call fit() before predicting.")

        X_b = np.c_[np.ones((X.shape[0], 1)), X].astype(np.float32)
        full_theta = np.vstack([self.intercept_, self.coef_.reshape(-1, 1)]).astype(np.float32)

        m, n = X_b.shape
        n_theta, k = full_theta.shape
        if n != n_theta:
            raise ValueError("Matrix dimensions are not compatible for prediction.")

        x_flat = X_b.flatten().tolist()
        theta_flat = full_theta.flatten().tolist()

        predictions_flat = rusty_machine.gpu_matrix_multiply(x_flat, theta_flat, m, n, k)

        return np.array(predictions_flat).reshape(m, k)