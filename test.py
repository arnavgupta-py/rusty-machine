import numpy as np
import cupy as cp
import time
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from rustymachine_api.models import LinearRegression

# --- 1. Verification Section ---
print("--- Verification Phase ---")
SAMPLES = 2000
FEATURES = 10

print(f"Generating synthetic data for verification...")
X = np.random.rand(SAMPLES, FEATURES).astype(np.float32)
true_theta = np.random.rand(FEATURES + 1, 1).astype(np.float32)
X_b = np.c_[np.ones((SAMPLES, 1)), X].astype(np.float32)
y = X_b.dot(true_theta) + (np.random.randn(SAMPLES, 1) * 0.5).astype(np.float32)

# --- Fit and Predict with RustyMachine ---
print("\n--- Testing RustyMachine (Zero-Copy) ---")
X_gpu = cp.asarray(X)
y_gpu = cp.asarray(y)

model = LinearRegression()
model.fit(X_gpu, y_gpu)
predictions = model.predict(X_gpu)
gpu_theta = np.vstack([model.intercept_, model.coef_.reshape(-1, 1)]).astype(np.float32)

# --- Fit and Predict with Scikit-learn ---
print("\n--- Testing Scikit-learn (Baseline) ---")
sklearn_model = SklearnLinearRegression()
sklearn_model.fit(X, y)
sklearn_predictions = sklearn_model.predict(X)
sklearn_theta = np.vstack([sklearn_model.intercept_, sklearn_model.coef_.T]).astype(np.float32)

# --- Compare Coefficients ---
print("\n🔎 Comparing final model coefficients...")
if np.allclose(gpu_theta, sklearn_theta, atol=1e-3):
    print("✅ Success! The model coefficients match Scikit-learn.")
else:
    print("❌ Failure! The coefficients do not match.")

# --- Compare Predictions ---
print("\n🔎 Comparing model predictions...")
if np.allclose(predictions, sklearn_predictions, atol=1e-3):
    print("✅ Success! The GPU-accelerated predictions match Scikit-learn.")
else:
    print("❌ Failure! The predictions do not match.")

print("\n-------------------------------------------------")
print("🏆 Verification Complete.")
print("-------------------------------------------------")


# --- 2. Benchmarking Section ---
print("\n--- Benchmarking Performance ---")
SAMPLES_BM = 100000
FEATURES_BM = 100

print(f"Generating synthetic data for benchmarking ({SAMPLES_BM} samples, {FEATURES_BM} features)...")
X_bm = np.random.rand(SAMPLES_BM, FEATURES_BM).astype(np.float32)
true_theta_bm = np.random.rand(FEATURES_BM + 1, 1).astype(np.float32)
X_b_bm = np.c_[np.ones((SAMPLES_BM, 1)), X_bm].astype(np.float32)
y_bm = X_b_bm.dot(true_theta_bm) + np.random.randn(SAMPLES_BM, 1).astype(np.float32)

X_bm_gpu = cp.asarray(X_bm)
y_bm_gpu = cp.asarray(y_bm)

# Benchmark RustyMachine
print("\nBenchmarking RustyMachine fit()...")
start_time_rm = time.time()
model.fit(X_bm_gpu, y_bm_gpu)
end_time_rm = time.time()
rusty_machine_time = end_time_rm - start_time_rm
print(f"🚀 RustyMachine fit time: {rusty_machine_time:.4f} seconds")

# Benchmark Scikit-learn
print("\nBenchmarking Scikit-learn fit()...")
start_time_sk = time.time()
sklearn_model.fit(X_bm, y_bm)
end_time_sk = time.time()
sklearn_time = end_time_sk - start_time_sk
print(f"🐢 Scikit-learn fit time:  {sklearn_time:.4f} seconds")

# --- Performance Comparison ---
if rusty_machine_time < sklearn_time:
    speedup = sklearn_time / rusty_machine_time
    print(f"\n✅ RustyMachine is {speedup:.2f}x faster than Scikit-learn.")
else:
    print("\n⚠️ RustyMachine is slower than Scikit-learn.")

print("\n-------------------------------------------------")
print("🏆 Benchmarking Complete.")
print("-------------------------------------------------")