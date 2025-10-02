#include <math.h> // For expf

// Your existing, working kernels
extern "C" __global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
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

// --- NEW KERNELS FOR LOGISTIC REGRESSION ---

/**
 * @brief Applies the sigmoid function element-wise to a vector, in-place.
 * @param vec The vector to modify.
 * @param n The number of elements in the vector.
 */
extern "C" __global__ void elementwise_sigmoid(float* vec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vec[idx] = 1.0f / (1.0f + expf(-vec[idx]));
    }
}

/**
 * @brief Subtracts one vector from another element-wise (a - b).
 * @param a The first vector.
 * @param b The second vector.
 * @param result The output vector where results are stored.
 * @param n The number of elements in the vectors.
 */
extern "C" __global__ void elementwise_sub(const float* a, const float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] - b[idx];
    }
}

/**
 * @brief Computes the AXPY operation (y = alpha * x + y).
 * @param alpha A scalar float.
 * @param x The first vector.
 * @param y The second vector, which is modified in-place.
 * @param n The number of elements in the vectors.
 */
extern "C" __global__ void axpy(float alpha, const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

/**
 * @brief Computes the dot product of two vectors using parallel reduction.
 * Each block computes a partial sum, which must be summed on the CPU.
 * @param a The first vector.
 * @param b The second vector.
 * @param partial_sums An output vector of size equal to the number of blocks.
 * @param n The number of elements in vectors a and b.
 */
extern "C" __global__ void dot_product(const float* a, const float* b, float* partial_sums, int n) {
    // Shared memory for this block's partial sums. Fast on-chip memory.
    extern __shared__ float cache[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread calculates a partial sum and stores it in shared memory.
    // A grid-stride loop allows one thread to handle multiple elements if n > threads.
    float temp = 0;
    while (i < n) {
        temp += a[i] * b[i];
        i += gridDim.x * blockDim.x;
    }
    cache[tid] = temp;
    
    // Synchronize to make sure all threads in the block are done.
    __syncthreads();
    
    // Perform reduction in shared memory.
    // Each step halves the number of active threads.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    
    // The first thread in each block writes its block's total sum to global memory.
    if (tid == 0) {
        partial_sums[blockIdx.x] = cache[0];
    }
}