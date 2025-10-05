#include <math.h>

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

extern "C" __global__ void dot_product(const float* a, const float* b, float* partial_sums, int n) {
    extern __shared__ float cache[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float temp = 0;
    while (i < n) {
        temp += a[i] * b[i];
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

extern "C" __global__ void elementwise_log(float* vec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float epsilon = 1e-9f;
        vec[idx] = logf(vec[idx] + epsilon);
    }
}

extern "C" __global__ void cost_kernel(const float* y, const float* h_log, const float* h_one_minus_log, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float y_i = y[idx];
        result[idx] = - (y_i * h_log[idx] + (1.0f - y_i) * h_one_minus_log[idx]);
    }
}

extern "C" __global__ void sum_reduction(const float* vec, float* partial_sums, int n) {
    extern __shared__ float cache[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float temp = 0;
    while (i < n) {
        temp += vec[i];
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