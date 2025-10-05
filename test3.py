import numpy as np
import cupy as cp
import rusty_machine

# Test each kernel individually
print("=== Testing Individual Kernels ===\n")

# 1. Test Matrix Multiplication
print("1. Testing Matrix Multiplication")
A = np.array([[1, 2], [3, 4]], dtype=np.float32)
B = np.array([[5], [6]], dtype=np.float32)
A_gpu = cp.asarray(A)
B_gpu = cp.asarray(B)
C_gpu = cp.zeros((2, 1), dtype=np.float32)

rusty_machine.gpu_matrix_multiply(
    A_gpu.data.ptr, B_gpu.data.ptr, C_gpu.data.ptr,
    2, 2, 1
)
C_result = C_gpu.get()
C_expected = A @ B
print(f"Result: {C_result.flatten()}")
print(f"Expected: {C_expected.flatten()}")
print(f"Match: {np.allclose(C_result, C_expected)}\n")

# 2. Test Transpose
print("2. Testing Transpose")
X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
X_gpu = cp.asarray(X)
X_t_gpu = cp.zeros((3, 2), dtype=np.float32)

rusty_machine.gpu_transpose(
    X_gpu.data.ptr, X_t_gpu.data.ptr,
    2, 3
)
X_t_result = X_t_gpu.get()
X_t_expected = X.T
print(f"Result:\n{X_t_result}")
print(f"Expected:\n{X_t_expected}")
print(f"Match: {np.allclose(X_t_result, X_t_expected)}\n")

# 3. Test Full Forward Pass (Manual)
print("3. Testing Full Forward Pass Manually")
np.random.seed(42)
X_test = np.random.rand(10, 3).astype(np.float32)
# Add bias column
X_b_test = np.hstack([np.ones((10, 1), dtype=np.float32), X_test])
y_test = np.array([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]], dtype=np.float32)
theta_test = np.zeros((4, 1), dtype=np.float32)

# Move to GPU
X_b_gpu = cp.asarray(X_b_test)
y_gpu = cp.asarray(y_test)
theta_gpu = cp.asarray(theta_test)

# Manual forward pass
print("Initial theta:", theta_gpu.get().flatten())

# h = X * theta
h_gpu = cp.zeros((10, 1), dtype=np.float32)
rusty_machine.gpu_matrix_multiply(
    X_b_gpu.data.ptr, theta_gpu.data.ptr, h_gpu.data.ptr,
    10, 4, 1
)
print(f"h after matmul (should be all zeros): {h_gpu.get().flatten()[:5]}...")

# Test with non-zero theta
theta_test_nonzero = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
theta_gpu = cp.asarray(theta_test_nonzero)
rusty_machine.gpu_matrix_multiply(
    X_b_gpu.data.ptr, theta_gpu.data.ptr, h_gpu.data.ptr,
    10, 4, 1
)
h_result = h_gpu.get()
h_expected = X_b_test @ theta_test_nonzero
print(f"\nWith non-zero theta:")
print(f"GPU result: {h_result.flatten()[:5]}...")
print(f"Expected:   {h_expected.flatten()[:5]}...")
print(f"Match: {np.allclose(h_result, h_expected)}\n")

# 4. Test one full gradient descent iteration manually
print("4. Testing One Gradient Descent Iteration")
# Simple dataset: y = 1 if x[0] + x[1] > 1 else 0
X_simple = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=np.float32)
y_simple = np.array([[0], [0], [0], [1]], dtype=np.float32)

X_b_simple = np.hstack([np.ones((4, 1), dtype=np.float32), X_simple])
theta_simple = np.zeros((3, 1), dtype=np.float32)

print("Data:")
print("X_b:", X_b_simple)
print("y:", y_simple.flatten())

# NumPy reference implementation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

h_np = sigmoid(X_b_simple @ theta_simple)
error_np = h_np - y_simple
grad_np = X_b_simple.T @ error_np
theta_new_np = theta_simple - 0.1 * grad_np / 4

print("\nNumPy computation:")
print(f"h (predictions): {h_np.flatten()}")
print(f"error: {error_np.flatten()}")
print(f"gradient: {grad_np.flatten()}")
print(f"new theta: {theta_new_np.flatten()}")

# GPU computation using train_logistic_gpu
X_b_simple_gpu = cp.asarray(X_b_simple)
y_simple_gpu = cp.asarray(y_simple)
theta_simple_gpu = cp.zeros((3, 1), dtype=np.float32)

print("\nRunning GPU training for 1 iteration...")
rusty_machine.train_logistic_gpu(
    X_b_simple_gpu.data.ptr,
    y_simple_gpu.data.ptr,
    theta_simple_gpu.data.ptr,
    4, 3, 1, 0.1, 1e-10, 0.0, 0.0
)

theta_result = theta_simple_gpu.get()
print(f"GPU theta after 1 iter: {theta_result.flatten()}")
print(f"Expected (NumPy):       {theta_new_np.flatten()}")
print(f"Match: {np.allclose(theta_result, theta_new_np, atol=1e-5)}\n")

# 5. Test with your actual failing case
print("5. Testing Small Version of Failing Case")
np.random.seed(42)
SAMPLES = 100
FEATURES = 5

X = np.random.rand(SAMPLES, FEATURES).astype(np.float32)
y = np.zeros((SAMPLES, 1), dtype=np.float32)
y[np.sum(X, axis=1) > (FEATURES / 2.0)] = 1.0

print(f"Class distribution: {np.sum(y==0)} zeros, {np.sum(y==1)} ones")

X_gpu = cp.asarray(X)
y_gpu = cp.asarray(y)

# Add intercept
ones = cp.ones((SAMPLES, 1), dtype=cp.float32)
X_b_gpu = cp.hstack([ones, X_gpu])
theta_gpu = cp.zeros((FEATURES + 1, 1), dtype=cp.float32)

print("Training for 1000 iterations...")
rusty_machine.train_logistic_gpu(
    X_b_gpu.data.ptr,
    y_gpu.data.ptr,
    theta_gpu.data.ptr,
    SAMPLES, FEATURES + 1, 1000, 10.0, 1e-7, 0.0, 0.0
)

# Make predictions
theta_final = theta_gpu.get()
print(f"\nFinal theta: {theta_final.flatten()}")

# Compute predictions using NumPy
logits = X @ theta_final[1:] + theta_final[0]
probs = sigmoid(logits)
preds = (probs > 0.5).astype(int)
accuracy = np.mean(preds == y)

print(f"Accuracy: {accuracy:.4f}")
print(f"First 10 predictions: {preds.flatten()[:10]}")
print(f"First 10 actual:      {y.flatten()[:10]}")

if accuracy < 0.8:
    print("\n⚠️ PROBLEM FOUND: Accuracy is too low!")
    print("Checking predictions distribution:")
    print(f"  Predicted 0s: {np.sum(preds==0)}")
    print(f"  Predicted 1s: {np.sum(preds==1)}")
else:
    print("\n✅ Small test passed!")