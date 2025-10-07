#include <math.h>

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
    if (row < M && col < N) {
        Out[col * M + row] = In[row * N + col];
    }
}

extern "C" __global__ void fused_sigmoid_sub(
    const float* z,
    const float* y,
    float* error,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float h = 1.0f / (1.0f + expf(-z[idx]));
        error[idx] = h - y[idx];
    }
}

extern "C" __global__ void axpy(float alpha, const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}