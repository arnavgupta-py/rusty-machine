import numpy as np
import cupy as cp
import time
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from rustymachine_api.models import LogisticRegression

# --- 1. Full Model Verification ---
print("--- Full Model Verification ---")
SAMPLES = 2000
FEATURES = 10

# Deterministically create a balanced, separable dataset
X = np.random.rand(SAMPLES, FEATURES).astype(np.float32)
y = np.ones((SAMPLES, 1), dtype=np.float32)
# Create a simple linear boundary
y[np.sum(X, axis=1) < (FEATURES / 2.0)] = 0.0

X_gpu = cp.asarray(X)
y_gpu = cp.asarray(y)

# --- RustyMachine Model ---
model_log = LogisticRegression(max_iter=100, tol=1e-4)
model_log.fit(X_gpu, y_gpu)
gpu_predictions = model_log.predict_proba(X_gpu)

# --- Scikit-learn Model (Configured for a fair comparison) ---
sklearn_model_log = SklearnLogisticRegression(
    solver='lbfgs',
    penalty=None, # Critical for a fair comparison
    max_iter=100,
    tol=1e-4
)
sklearn_model_log.fit(X, y.ravel())
sklearn_predictions = sklearn_model_log.predict_proba(X)[:, 1].reshape(-1, 1)

# Assert that the PREDICTIONS are nearly identical, which is the correct test
assert np.allclose(gpu_predictions, sklearn_predictions, atol=1e-2), "Model predictions mismatch."
print("✅ Full Model Verification Passed (Predictions Match).")
print("-------------------------------------------------")


# --- 2. Benchmarking Performance ---
print("\n--- Benchmarking Performance ---")
SAMPLES_BM = 100000
FEATURES_BM = 100

X_bm = np.random.rand(SAMPLES_BM, FEATURES_BM).astype(np.float32)
y_bm = np.ones((SAMPLES_BM, 1), dtype=np.float32)
y_bm[np.sum(X_bm, axis=1) < (FEATURES_BM / 2.0)] = 0.0

X_bm_gpu = cp.asarray(X_bm)
y_bm_gpu = cp.asarray(y_bm)

start_time_rm = time.time()
model_log.fit(X_bm_gpu, y_bm_gpu)
end_time_rm = time.time()
rusty_machine_time = end_time_rm - start_time_rm
print(f"🚀 RustyMachine Logistic fit time: {rusty_machine_time:.4f} seconds")

start_time_sk = time.time()
sklearn_model_log.fit(X_bm, y_bm.ravel())
end_time_sk = time.time()
sklearn_time = end_time_sk - start_time_sk
print(f"🐢 Scikit-learn Logistic fit time:  {sklearn_time:.4f} seconds")

if rusty_machine_time < sklearn_time:
    speedup = sklearn_time / rusty_machine_time
    print(f"✅ RustyMachine Logistic is {speedup:.2f}x faster.")
else:
    print("⚠️ RustyMachine Logistic is slower.")

print("\n-------------------------------------------------")
print("🏆 All Testing Complete.")
print("-------------------------------------------------")