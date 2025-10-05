import numpy as np
import cupy as cp
import rusty_machine

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
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
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("You must call fit() before predicting.")
            
        if isinstance(X, np.ndarray):
            X_gpu = cp.asarray(X, dtype=cp.float32)
        elif isinstance(X, cp.ndarray):
            X_gpu = X.astype(cp.float32)
        else:
            raise TypeError("Input X must be a NumPy or CuPy array.")

        intercept_gpu = cp.array([self.intercept_], dtype=cp.float32)
        coef_gpu = cp.asarray(self.coef_.reshape(-1, 1), dtype=cp.float32)
        theta_gpu = cp.vstack([intercept_gpu, coef_gpu])

        ones = cp.ones((X_gpu.shape[0], 1), dtype=cp.float32)
        X_b_gpu = cp.hstack([ones, X_gpu])
        
        predictions_gpu = X_b_gpu @ theta_gpu
        
        return predictions_gpu.get()

class LogisticRegression:
    def __init__(self, max_iter=1000, tol=1e-4):
        self.coef_ = None
        self.intercept_ = None
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
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

        ones = cp.ones((X_gpu.shape[0], 1), dtype=cp.float32)
        X_b_gpu = cp.hstack([ones, X_gpu])
        samples, features = X_b_gpu.shape
        
        theta_gpu = cp.zeros((features, 1), dtype=cp.float32)
        
        rusty_machine.train_logistic_lbfgs(
            X_b_gpu.data.ptr,
            y_gpu.data.ptr,
            theta_gpu.data.ptr,
            samples,
            features,
            self.max_iter,
            self.tol
        )
        
        theta_host = theta_gpu.get()
        self.intercept_ = theta_host[0, 0]
        self.coef_ = theta_host[1:].flatten()

        return self

    def predict_proba(self, X):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("You must call fit() before predicting.")
        
        if isinstance(X, np.ndarray):
            X_gpu = cp.asarray(X, dtype=cp.float32)
        elif isinstance(X, cp.ndarray):
            X_gpu = X.astype(cp.float32)
        else:
            raise TypeError("Input X must be a NumPy or CuPy array.")

        intercept_gpu = cp.array([self.intercept_], dtype=cp.float32)
        coef_gpu = cp.asarray(self.coef_.reshape(-1, 1), dtype=cp.float32)
        theta_gpu = cp.vstack([intercept_gpu, coef_gpu])

        ones = cp.ones((X_gpu.shape[0], 1), dtype=cp.float32)
        X_b_gpu = cp.hstack([ones, X_gpu])

        logits = X_b_gpu @ theta_gpu
        probabilities = 1 / (1 + cp.exp(-logits))
        
        return probabilities.get()

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)