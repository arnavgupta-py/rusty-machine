import numpy as np
import cupy as cp
import time
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from rustymachine_api.models import LogisticRegression

np.set_printoptions(precision=6, suppress=True)

# --- 1. Full Model Verification ---
print("--- Full Model Verification ---")
SAMPLES = 2000
FEATURES = 10

np.random.seed(42)
X = np.random.rand(SAMPLES, FEATURES).astype(np.float32)
y = np.zeros((SAMPLES, 1), dtype=np.float32)
y[np.sum(X, axis=1) > (FEATURES / 2.0)] = 1.0

print(f"Dataset: {SAMPLES} samples, {FEATURES} features")
print(f"Class balance: {np.sum(y==0)} zeros, {np.sum(y==1)} ones")

X_gpu = cp.asarray(X)
y_gpu = cp.asarray(y)

# Try different learning rates
learning_rates = [0.5, 1.0, 5.0, 10.0, 20.0]  

print("\nTrying different learning rates:")
for lr in learning_rates:
    model_log = LogisticRegression(
        max_iter=5000,  # Increased iterations
        tol=1e-7,
        lr=lr,
        penalty=None
    )
    model_log.fit(X_gpu, y_gpu)
    gpu_predictions_labels = model_log.predict(X)
    gpu_accuracy = accuracy_score(y, gpu_predictions_labels)
    print(f"  lr={lr:4.2f}: Accuracy = {gpu_accuracy:.4f}")

# Use best learning rate
print("\n--- Final Test with lr=0.5 ---")
model_log = LogisticRegression(
    max_iter=10000,
    tol=1e-7,
    lr=10,
    penalty=None
)
model_log.fit(X_gpu, y_gpu)
gpu_predictions_labels = model_log.predict(X)
gpu_accuracy = accuracy_score(y, gpu_predictions_labels)

# --- Scikit-learn Model ---
sklearn_model_log = SklearnLogisticRegression(
    solver='lbfgs',
    penalty=None,
    max_iter=1000,
    tol=1e-7
)
sklearn_model_log.fit(X, y.ravel())
sklearn_predictions_labels = sklearn_model_log.predict(X)
sklearn_accuracy = accuracy_score(y, sklearn_predictions_labels)

# --- Comparing Model Performance ---
print("\n--- Comparing Model Performance ---")
print(f"RustyMachine Accuracy: {gpu_accuracy:.4f}")
print(f"Scikit-learn Accuracy: {sklearn_accuracy:.4f}")

# Compare coefficients
print("\nCoefficient comparison (first 5):")
print(f"RustyMachine: {model_log.coef_[:5]}")
print(f"Scikit-learn: {sklearn_model_log.coef_[0][:5]}")
print(f"Intercept - RustyMachine: {model_log.intercept_:.4f}")
print(f"Intercept - Scikit-learn: {sklearn_model_log.intercept_[0]:.4f}")

# Show some predictions
print("\nFirst 10 predictions vs actual:")
print(f"RustyMachine: {gpu_predictions_labels[:10].flatten()}")
print(f"Scikit-learn: {sklearn_predictions_labels[:10]}")
print(f"Actual:       {y[:10].flatten().astype(int)}")

if gpu_accuracy > 0.95:
    print("\n✅ Model achieves good accuracy!")
else:
    print(f"\n⚠️ Model accuracy {gpu_accuracy:.4f} is too low")
    print("Possible issues:")
    print("  - Learning rate may need tuning")
    print("  - May need more iterations")
    print("  - Check kernel implementations")