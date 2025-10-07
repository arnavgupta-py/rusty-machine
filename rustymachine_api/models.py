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
            y_gpu = cp.asarray(y, dtype=cp.float32).reshape(-1, 1)
        else:
            X_gpu = X
            y_gpu = y.reshape(-1, 1)

        ones = cp.ones((X_gpu.shape[0], 1), dtype=cp.float32)
        X_b_gpu = cp.hstack([X_gpu, ones])

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
        self.intercept_ = theta_host[-1, 0]
        self.coef_ = theta_host[:-1].flatten()

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before predict().")
        
        X_np = cp.asnumpy(X) if isinstance(X, cp.ndarray) else np.asarray(X)
        return X_np @ self.coef_ + self.intercept_


class LogisticRegression:

    def __init__(self, epochs=1000, lr=0.01, batch_size=256, random_state=None):
        self.coef_ = None
        self.intercept_ = None
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if isinstance(X, np.ndarray):
            X_np = np.asarray(X, dtype=np.float32)
            y_np = np.asarray(y, dtype=np.float32).ravel()
        else: 
            X_np = X.get()
            y_np = y.get().ravel()

        ones = np.ones((X_np.shape[0], 1), dtype=np.float32)
        X_b_np = np.hstack([X_np, ones])
        
        samples, features = X_b_np.shape
        theta_np = np.zeros(features, dtype=np.float32)

        permutation = np.random.permutation(samples)
        X_shuffled = np.ascontiguousarray(X_b_np[permutation])
        y_shuffled = np.ascontiguousarray(y_np[permutation])

        X_gpu = cp.asarray(X_shuffled)
        y_gpu = cp.asarray(y_shuffled)
        theta_gpu = cp.asarray(theta_np)

        rusty_machine.train_logistic_minibatch_gpu(
            X_gpu.data.ptr,
            y_gpu.data.ptr,
            theta_gpu.data.ptr,
            samples,
            features,
            self.epochs,
            self.lr,
            self.batch_size
        )
            
        theta_host = theta_gpu.get()

        self.intercept_ = theta_host[-1]
        self.coef_ = theta_host[:-1]
        return self

    def predict_proba(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet.")
        
        X_np = cp.asnumpy(X) if isinstance(X, cp.ndarray) else np.asarray(X)
        logits = X_np @ self.coef_ + self.intercept_
        probs = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
        return np.hstack([1 - probs.reshape(-1, 1), probs.reshape(-1, 1)])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)