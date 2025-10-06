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
    def __init__(self, penalty=None, C=1.0, max_iter=1000, tol=1e-4, lr=0.01):
        self.coef_ = None
        self.intercept_ = None
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr

    def fit(self, X, y):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(y, dtype=np.float32).reshape(-1, 1)

        # Add intercept term to X
        ones = cp.ones((X_gpu.shape[0], 1), dtype=cp.float32)
        X_b_gpu = cp.hstack([ones, X_gpu])
        samples, features = X_b_gpu.shape

        theta_gpu = cp.zeros((features, 1), dtype=cp.float32)

        l1 = 1.0 / self.C if self.penalty == 'l1' else 0.0
        l2 = 1.0 / self.C if self.penalty == 'l2' else 0.0

        rusty_machine.train_logistic_gpu(
            X_b_gpu.data.ptr,
            y_gpu.data.ptr,
            theta_gpu.data.ptr,
            samples,
            features,
            self.max_iter,
            self.lr,
            self.tol,
            l1,
            l2
        )

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
        # Clip logits to prevent overflow in np.exp
        logits_clipped = np.clip(logits, -20, 20)
        probs = 1 / (1 + np.exp(-logits_clipped))
        
        # Return probabilities for both classes
        return np.hstack([1 - probs.reshape(-1, 1), probs.reshape(-1, 1)])

    def predict(self, X):
        # Predicts the class with the highest probability
        # The second column (index 1) is the probability of the positive class
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)