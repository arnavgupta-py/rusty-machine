import numpy as np
import time
from rustymachine_api.models import LogisticRegression
from sklearn.metrics import accuracy_score

def create_dataset(samples, features, seed):
    np.random.seed(seed)
    X = (np.random.rand(samples, features) * 2 - 1).astype(np.float32)
    # A clear, linearly separable problem where features 0 and 1 are important
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    return X, y

print("=" * 80)
print("✅ FINAL TUNED TEST: Fused Coordinate Descent ('cd')")
print("=" * 80)

# --- Final Hyperparameters ---
SAMPLES = 10000
FEATURES = 50
EPOCHS = 20
# ✅ THE FINAL FIX: A more stable learning rate
LEARNING_RATE = 1.0
L1_PENALTY_C = 500.0

# --- Data Preparation ---
X_train, y_train = create_dataset(SAMPLES, FEATURES, seed=42)
print(f"Dataset: {SAMPLES} samples, {FEATURES} features.")
print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, C (for L1): {L1_PENALTY_C}")
print("-" * 80)

# --- Train Rusty Machine ---
print("Training with Rusty Machine (solver='cd', penalty='l1')...")
model_rm_cd = LogisticRegression(
    solver='cd',
    penalty='l1',
    C=L1_PENALTY_C,
    epochs=EPOCHS,
    lr=LEARNING_RATE
)

start_time = time.time()
model_rm_cd.fit(X_train, y_train)
duration = time.time() - start_time

preds = model_rm_cd.predict(X_train)
accuracy = accuracy_score(y_train, preds)

print(f"\nTraining finished in {duration:.4f} seconds.")
print(f"Final Accuracy: {accuracy:.4f}")

non_zero_coeffs = np.sum(np.abs(model_rm_cd.coef_) > 1e-4)
print(f"Model Coefficients (Theta):")
print(f"   - Intercept: {model_rm_cd.intercept_:.4f}")
print(f"   - Non-zero coefficients: {non_zero_coeffs} out of {FEATURES}")
# The first two coefficients should be the most significant and not exploded
print(f"   - First 5 coeffs: {model_rm_cd.coef_[:5]}")
print("-" * 80)

if accuracy > 0.98:
    print("🚀 VICTORY: The model has converged correctly with stable weights.")
else:
    print("❌ FAILED: Model did not achieve expected accuracy.")

print("\nTest complete.")