#include <math.h>

// --- HIGH-PERFORMANCE TILED MATRIX MULTIPLICATION ---
#define TILE_WIDTH 16
extern "C" __global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < M && (t * TILE_WIDTH + tx) < N) {
            As[ty][tx] = A[row * N + (t * TILE_WIDTH + tx)];
        } else { As[ty][tx] = 0.0f; }
        if ((t * TILE_WIDTH + ty) < N && col < K) {
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * K + col];
        } else { Bs[ty][tx] = 0.0f; }
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; ++i) { sum += As[ty][i] * Bs[i][tx]; }
        __syncthreads();
    }
    if (row < M && col < K) { C[row * K + col] = sum; }
}

extern "C" __global__ void transpose(const float* In, float* Out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) { Out[col * M + row] = In[row * N + col]; }
}

extern "C" __global__ void elementwise_sigmoid(float* vec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { vec[idx] = 1.0f / (1.0f + expf(-vec[idx])); }
}

// --- FUSED GRADIENT KERNEL (STABLE VERSION) ---
// Calculates gradient using logits (z) instead of probabilities (h)
extern "C" __global__ void fused_gradient_from_logits(
    const float* X_col, const float* y, const float* z, float* grad_j, int n_samples
) {
    extern __shared__ float cache[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float partial_dot = 0.0f;
    while (i < n_samples) {
        float h = 1.0f / (1.0f + expf(-z[i])); // Sigmoid calculated on-the-fly
        float error = h - y[i];
        partial_dot += error * X_col[i];
        i += gridDim.x * blockDim.x;
    }
    cache[tid] = partial_dot;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { cache[tid] += cache[tid + s]; }
        __syncthreads();
    }

    // ✅ **THE FIX**: Accumulate from each block atomically to get the correct total gradient.
    if (tid == 0) {
        atomicAdd(grad_j, cache[0]);
    }
}

// --- KERNEL for Theta and Z Update (STABLE VERSION) ---
extern "C" __global__ void update_theta_and_z(
    float* theta,           // Full theta vector
    float* z,               // Full logit vector
    const float* X_col,     // A single column of X
    const float* grad_j,    // The pre-computed gradient for feature j
    int j,                  // Index of the feature to update
    int n_samples,
    float lr,
    float l1_penalty
) {
    __shared__ float delta_theta;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float old_theta_j = theta[j];
        float gradient = *grad_j / (float)n_samples;
        float updated_theta_j = old_theta_j - lr * gradient;

        float l1_threshold = lr * l1_penalty;
        float new_theta_j;
        if (updated_theta_j > l1_threshold) {
            new_theta_j = updated_theta_j - l1_threshold;
        } else if (updated_theta_j < -l1_threshold) {
            new_theta_j = updated_theta_j + l1_threshold;
        } else {
            new_theta_j = 0.0f;
        }
        
        theta[j] = new_theta_j;
        delta_theta = new_theta_j - old_theta_j;
    }
    __syncthreads();

    // Update 'z' in parallel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (delta_theta != 0.0f) {
        while (idx < n_samples) {
            z[idx] += X_col[idx] * delta_theta;
            idx += gridDim.x * blockDim.x;
        }
    }
}