import numpy as np
import rusty_machine

N = 10 # A square N x N matrix

print(f"Creating a {N}x{N} invertible matrix...")
A = np.random.rand(N, N).astype(np.float32) + np.eye(N, dtype=np.float32)

print("🚀 Launching GPU matrix inversion via Rust and cuSOLVER...")
a_flat = A.flatten().tolist()
result_flat = rusty_machine.gpu_inverse(a_flat, N)
gpu_inverse_result = np.array(result_flat).reshape(N, N)

print("⚙️  Verifying the result...")
identity_check = np.dot(A, gpu_inverse_result)
identity_matrix = np.eye(N, dtype=np.float32)

print("🔎 Comparing (Original @ GPU_Inverse) with the Identity Matrix...")
if np.allclose(identity_check, identity_matrix, atol=1e-5):
    print("\n✅ Success! The GPU inverse is correct.")
else:
    print("\n❌ Failure! The matrix inverse is incorrect.")
    print("Original @ GPU_Inverse (should be Identity):\n", identity_check)