import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from rustymachine_api.models import LinearRegression # Import our new class!

# --- 1. Generate Synthetic Data ---
SAMPLES = 2000
FEATURES = 10

print(f"Generating synthetic data...")
X = np.random.rand(SAMPLES, FEATURES).astype(np.float32)
true_theta = np.random.rand(FEATURES + 1, 1).astype(np.float32)
X_b = np.c_[np.ones((SAMPLES, 1)), X].astype(np.float32)
y = X_b.dot(true_theta) + (np.random.randn(SAMPLES, 1) * 0.5).astype(np.float32)


# --- 2. Fit and Predict with RustyMachine ---
print("\n--- Testing RustyMachine ---")
model = LinearRegression()
model.fit(X, y) # This one call runs our entire Rust/CUDA backend
predictions = model.predict(X) # This call now also runs on the GPU
gpu_theta = np.vstack([model.intercept_, model.coef_.reshape(-1, 1)])


# --- 3. Fit and Predict with Scikit-learn ---
print("\n--- Testing Scikit-learn ---")
sklearn_model = SklearnLinearRegression()
sklearn_model.fit(X, y)
sklearn_predictions = sklearn_model.predict(X)
sklearn_theta = np.vstack([sklearn_model.intercept_, sklearn_model.coef_.T])


# --- 4. Compare Coefficients ---
print("\n🔎 Comparing final model coefficients...")
if np.allclose(gpu_theta, sklearn_theta, atol=1e-4):
    print("✅ Success! The model coefficients match Scikit-learn.")
else:
    print("❌ Failure! The coefficients do not match.")


# --- 5. Compare Predictions ---
print("\n🔎 Comparing model predictions...")
if np.allclose(predictions, sklearn_predictions, atol=1e-4):
    print("✅ Success! The GPU-accelerated predictions match Scikit-learn.")
else:
    print("❌ Failure! The predictions do not match.")

print("\n-------------------------------------------------")
print("🏆 The Linear Regression model is complete and verified.")
print("-------------------------------------------------")