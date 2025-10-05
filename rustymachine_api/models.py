import numpy as np
import cupy as cp
import rusty_machine

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32) if isinstance(X, np.ndarray) else X.astype(cp.float32)
        y = np.asarray(y, dtype=np.float32) if isinstance(y, np.ndarray) else y.astype(cp.float32)

        if X.ndim != 2 or y.ndim not in (1, 2):
            raise ValueError("X must be 2D and y must be 1D or 2D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match")

        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y).reshape(-1, 1)

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
            raise RuntimeError("Call fit() before predict()")

        X_gpu = cp.asarray(X, dtype=cp.float32)
        predictions_gpu = X_gpu @ cp.asarray(self.coef_.reshape(-1, 1)) + self.intercept_
        return predictions_gpu.get()


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
            raise RuntimeError("Call fit() before predict_proba()")

        logits = X @ self.coef_ + self.intercept_
        probs = 1 / (1 + np.exp(-logits))
        return probs.reshape(-1, 1)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)