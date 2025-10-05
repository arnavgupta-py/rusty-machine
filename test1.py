import numpy as np
import cupy as cp
import time
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from rustymachine_api.models import LogisticRegression

np.set_printoptions(precision=6, suppress=True, linewidth=120)

def create_dataset(samples, features, seed):
    """Creates a linearly separable dataset: y=1 if sum(features) > features/2"""
    np.random.seed(seed)
    X = np.random.rand(samples, features).astype(np.float32)
    y = np.zeros((samples, 1), dtype=np.float32)
    y[np.sum(X, axis=1) > (features / 2.0)] = 1.0
    return X, y

print("=" * 80)
print("COMPREHENSIVE LOGISTIC REGRESSION TEST")
print("=" * 80)

# Test 1: Small dataset convergence test
print("\n[TEST 1] Small Dataset (100 samples, 5 features)")
print("-" * 80)
X_small, y_small = create_dataset(100, 5, seed=42)
X_small_gpu = cp.asarray(X_small)
y_small_gpu = cp.asarray(y_small)

model_small = LogisticRegression(max_iter=2000, tol=0.0, lr=10.0, penalty=None)
model_small.fit(X_small_gpu, y_small_gpu)
preds_small = model_small.predict(X_small)
acc_small = accuracy_score(y_small, preds_small)

print(f"Accuracy: {acc_small:.4f}")
print(f"Theta (first 3): {model_small.coef_[:3]}")
print(f"Intercept: {model_small.intercept_:.4f}")

if acc_small > 0.95:
    print("✓ PASSED: Achieved >95% accuracy on small dataset")
else:
    print(f"✗ FAILED: Only achieved {acc_small:.4f} accuracy (expected >0.95)")
    exit(1)

# Test 2: Medium dataset vs sklearn
print("\n[TEST 2] Medium Dataset vs Sklearn (2000 samples, 10 features)")
print("-" * 80)
X_med, y_med = create_dataset(2000, 10, seed=42)
X_med_gpu = cp.asarray(X_med)
y_med_gpu = cp.asarray(y_med)

print(f"Class balance: {np.sum(y_med==0)} zeros, {np.sum(y_med==1)} ones")

# Train RustyMachine
print("\nTraining RustyMachine...")
model_rm = LogisticRegression(max_iter=50000, tol=0.0, lr=10.0, penalty=None)
start_rm = time.time()
model_rm.fit(X_med_gpu, y_med_gpu)
time_rm = time.time() - start_rm
preds_rm = model_rm.predict(X_med)
acc_rm = accuracy_score(y_med, preds_rm)

# Train sklearn
print("Training Sklearn...")
model_sk = SklearnLogisticRegression(solver='lbfgs', penalty=None, max_iter=1000, tol=1e-7)
start_sk = time.time()
model_sk.fit(X_med, y_med.ravel())
time_sk = time.time() - start_sk
preds_sk = model_sk.predict(X_med)
acc_sk = accuracy_score(y_med, preds_sk)

print(f"\nRustyMachine: {acc_rm:.4f} accuracy in {time_rm:.3f}s")
print(f"Sklearn:      {acc_sk:.4f} accuracy in {time_sk:.3f}s")
print(f"\nRustyMachine Coefficients (first 5): {model_rm.coef_[:5]}")
print(f"Sklearn Coefficients (first 5):      {model_sk.coef_[0][:5]}")

if acc_rm > 0.99:
    print("✓ PASSED: Achieved >99% accuracy on medium dataset")
else:
    print(f"✗ FAILED: Only achieved {acc_rm:.4f} accuracy (expected >0.99)")
    exit(1)

# Test 3: Learning rate sensitivity
print("\n[TEST 3] Learning Rate Sensitivity")
print("-" * 80)
learning_rates = [1.0, 5.0, 10.0, 20.0]
X_lr, y_lr = create_dataset(1000, 8, seed=123)
X_lr_gpu = cp.asarray(X_lr)
y_lr_gpu = cp.asarray(y_lr)

print("Testing different learning rates...")
best_acc = 0
for lr in learning_rates:
    model_lr = LogisticRegression(max_iter=10000, tol=0.0, lr=lr, penalty=None)
    model_lr.fit(X_lr_gpu, y_lr_gpu)
    preds_lr = model_lr.predict(X_lr)
    acc_lr = accuracy_score(y_lr, preds_lr)
    print(f"   lr={lr:5.1f}: Accuracy = {acc_lr:.4f}")
    best_acc = max(best_acc, acc_lr)

if best_acc > 0.95:
    print(f"✓ PASSED: Best accuracy {best_acc:.4f} >0.95")
else:
    print(f"✗ FAILED: Best accuracy only {best_acc:.4f} (expected >0.95)")
    exit(1)

# Test 4: Large dataset performance
print("\n[TEST 4] Large Dataset Performance (10000 samples, 50 features)")
print("-" * 80)
X_large, y_large = create_dataset(10000, 50, seed=456)
X_large_gpu = cp.asarray(X_large)
y_large_gpu = cp.asarray(y_large)

print("Training on large dataset...")
model_large = LogisticRegression(max_iter=20000, tol=0.0, lr=10.0, penalty=None)
start_large = time.time()
model_large.fit(X_large_gpu, y_large_gpu)
time_large = time.time() - start_large
preds_large = model_large.predict(X_large)
acc_large = accuracy_score(y_large, preds_large)

print(f"Accuracy: {acc_large:.4f}")
print(f"Training time: {time_large:.3f}s")

if acc_large > 0.95:
    print(f"✓ PASSED: Large dataset accuracy {acc_large:.4f} >0.95")
else:
    print(f"✗ FAILED: Large dataset accuracy only {acc_large:.4f} (expected >0.95)")
    exit(1)

# Final summary
print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nSummary:")
print(f"   Small dataset (100x5):     {acc_small:.4f} accuracy")
print(f"   Medium dataset (2000x10):  {acc_rm:.4f} accuracy in {time_rm:.3f}s")
print(f"   Learning rate test (1000x8): {best_acc:.4f} best accuracy")
print(f"   Large dataset (10000x50):  {acc_large:.4f} accuracy in {time_large:.3f}s")
print(f"\nYour GPU-accelerated logistic regression is working correctly!")