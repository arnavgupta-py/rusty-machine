#include <math.h>

// --- HIGH-PERFORMANCE TILED MATRIX MULTIPLICATION ---
// This kernel uses shared memory to dramatically reduce global memory access
#define TILE_WIDTH 16

extern "C" __global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    // Loop over the tiles of A and B required to compute the C element
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Collaboratively load a tile of A and B into shared memory
        if (row < M && (t * TILE_WIDTH + tx) < N) {
            As[ty][tx] = A[row * N + (t * TILE_WIDTH + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((t * TILE_WIDTH + ty) < N && col < K) {
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * K + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded before starting computation
        __syncthreads();

        // Multiply the two tiles from shared memory
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        // Synchronize to make sure all threads are done with the current tile
        __syncthreads();
    }

    // Write the final result to global memory
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}


extern "C" __global__ void transpose(const float* In, float* Out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        Out[col * M + row] = In[row * N + col];
    }
}

extern "C" __global__ void elementwise_sigmoid(float* vec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vec[idx] = 1.0f / (1.0f + expf(-vec[idx]));
    }
}

extern "C" __global__ void elementwise_sub(const float* a, const float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] - b[idx];
    }
}

extern "C" __global__ void axpy(float alpha, const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

extern "C" __global__ void sum_of_squares_reduction(const float* vec, float* partial_sums, int n) {
    extern __shared__ float cache[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float temp = 0;
    while (i < n) {
        float val = vec[i];
        temp += val * val;
        i += gridDim.x * blockDim.x;
    }
    cache[tid] = temp;
    
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = cache[0];
    }
}

extern "C" __global__ void proximal_update_l1(float* theta, float threshold, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = theta[idx];
        if (val > threshold) {
            theta[idx] = val - threshold;
        } else if (val < -threshold) {
            theta[idx] = val + threshold;
        } else {
            theta[idx] = 0.0f;
        }
    }
}

// Unused kernels from the L-BFGS implementation can be kept or removed
extern "C" __global__ void dot_product(const float* a, const float* b, float* partial_sums, int n) {
    extern __shared__ float cache[];
    int tid = threadIdx.x; int i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0; while (i < n) { temp += a[i] * b[i]; i += gridDim.x * blockDim.x; }
    cache[tid] = temp; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) { cache[tid] += cache[tid + s]; } __syncthreads(); }
    if (tid == 0) { partial_sums[blockIdx.x] = cache[0]; }
}
extern "C" __global__ void elementwise_log(float* vec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { const float epsilon = 1e-9f; vec[idx] = logf(vec[idx] + epsilon); }
}
extern "C" __global__ void cost_kernel(const float* y, const float* h_log, const float* h_one_minus_log, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { float y_i = y[idx]; result[idx] = - (y_i * h_log[idx] + (1.0f - y_i) * h_one_minus_log[idx]); }
}
extern "C" __global__ void sum_reduction(const float* vec, float* partial_sums, int n) {
    extern __shared__ float cache[];
    int tid = threadIdx.x; int i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0; while (i < n) { temp += vec[i]; i += gridDim.x * blockDim.x; }
    cache[tid] = temp; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) { cache[tid] += cache[tid + s]; } __syncthreads(); }
    if (tid == 0) { partial_sums[blockIdx.x] = cache[0]; }
}