#pragma once

#include <cuda_runtime.h>

constexpr size_t threads_per_block = 256;
#include "vectors.cuh"

#include <cuda_bf16.h>
#include <cstdlib>
#include <cstdio>

namespace device
{
template <typename dtype = float>
__global__
void vector_dot_step(const dtype* A, const dtype* B, dtype* partial, size_t N)
{
    extern __shared__ dtype cache[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    cache[tid] = (i < N) ? A[i] * B[i] : dtype(); // operation step
    __syncthreads();
    
    //printf("hi from idx");
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {   
        partial[blockIdx.x] = cache[0];
    }
}

extern "C" float host_vector_dot_float(const float* h_A, const float* h_B, size_t N)
{
    /*  DEPRECATED
        Copies host vector to cudaMalloc'd vector, and performs dot product,
        accumulating partial sums and returning their sum.
    */
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    const size_t size = N * sizeof(float);

    float *A, *B, *partial;
    cudaMalloc(&A, size);
    cudaMalloc(&B, size);
    cudaMalloc(&partial, blocks_per_grid * sizeof(float));
    cudaMemcpy(A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, size, cudaMemcpyHostToDevice);

    vector_dot_step<float> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>
        (A, B, partial, N);
    
    float* h_partial = new float[blocks_per_grid];
    cudaMemcpy(h_partial, partial, blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost);
    float prod = 0;
    for (size_t i = 0; i < blocks_per_grid; ++i)
    {
        prod += h_partial[i];
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(partial);
    delete[] h_partial;

    return prod;
}

template <class dtype>
__global__
void vector_add_vector_step(dtype* dest, const dtype* V, const dtype* U, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] = (i < N) ? V[i] + U[i] : dtype(); // operation step
}

template <class dtype>
__global__
void vector_addassign_vector_step(dtype* V, const dtype* U, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    V[i] += (i < N) ? U[i] : dtype();
}

template <class dtype>
__global__
void vector_sub_scalar_step(dtype* dest, const dtype* V, dtype A, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] = (i < N) ? V[i] - A : dtype();
}

template <class dtype>
__global__
void vector_sub_vector_step(dtype* dest, const dtype* V, const dtype* U, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] = (i < N) ? V[i] - U[i] : dtype(); // operation step
}

template <class dtype>
__global__
void vector_subassign_vector_step(dtype* V, const dtype* U, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    V[i] -= (i < N) ? U[i] : dtype();
}

template <class dtype>
__global__
void vector_mul_scalar_step(dtype* dest, const dtype* V, dtype A, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] = (i < N) ? V[i] * A : dtype();
}

template <class dtype>
__global__
void vector_div_scalar_step(dtype* dest, const dtype* V, dtype A, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] = (i < N) ? V[i] / A : dtype();
}

template <typename dtype>
__global__
void vector_dot_matrix_step(dtype* dest, const dtype* T, const dtype* X, size_t M, size_t N, dtype bias)
{
    unsigned int row = blockIdx.x;
    if (row >= N)
        return;
    
    extern __shared__ dtype cache[];

    unsigned int tid = threadIdx.x;
    const dtype* row_ptr = X + row * M;
    dtype partial = 0;
    for (int j = tid; j < M; j += blockDim.x)
    {
        partial += T[j] * row_ptr[j];
    }
    cache[tid] = partial;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        dest[row] = cache[0] + bias;
    }
}

template <typename dtype, class UnaryOp>
__global__
dtype* vector_transform_step(dtype* dest, const dtype* X, size_t N,
                             dtype bias, UnaryOp thunk)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        dest[i] = thunk(X[i] + bias);
}

template <typename dtype, class UnaryOp>
__global__
void vector_dot_matrix_transform_step(dtype* dest, const dtype* T, const dtype* X, 
                                      size_t M, size_t N, dtype bias, UnaryOp thunk)
{
    unsigned int row = blockIdx.x;
    if (row >= N)
        return;
    
    extern __shared__ dtype cache[];

    unsigned int tid = threadIdx.x;
    const dtype* row_ptr = X + row * M;
    dtype partial = 0;
    for (int j = tid; j < M; j += blockDim.x)
    {
        partial += T[j] * row_ptr[j];
    }
    cache[tid] = partial;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        dest[row] = thunk(cache[0] + bias); // Applies the transform operation
    }
}

template <typename dtype, typename itype, class UnaryOp>
__global__
void vector_reduce_step(itype* block_sums, const dtype* V, size_t N, UnaryOp thunk)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    
    extern __shared__ itype cache[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;

    itype acc = 0;
    if (idx < N)
    {
        acc = thunk(V[idx]);
    }
    if (idx + blockDim.x < N)
    {
        acc += thunk(V[idx + blockDim.x]);
    }
    cache[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    
    if (tid < 32)
    {
        volatile itype* vsmem = cache;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0)
    {
        block_sums[blockIdx.x] = cache[0];
    }
}

};

namespace device
{
// TODO replace __host__ with __global__ and compare performance
template <typename dtype>
__host__
dtype vector_dot(const dtype* d_A, const dtype* d_B, size_t N)
{
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    dtype* partial;
    cudaMalloc(&partial, blocks_per_grid * sizeof(dtype));

    vector_dot_step<dtype> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (d_A, d_B, partial, N);
    
    dtype* h_partial = new dtype[blocks_per_grid];
    cudaMemcpy(h_partial, partial, blocks_per_grid * sizeof(dtype), cudaMemcpyDeviceToHost);

    dtype prod = 0;
    for (size_t i = 0; i < blocks_per_grid; ++i)
    {
        prod += h_partial[i];
    }

    cudaFree(partial);
    delete[] h_partial;
    return prod;
}

template <class dtype>
__host__
dtype* vector_add_vector(const dtype* V, const dtype* U, size_t N)
{// Transfers ownership of a live pointer!
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    dtype* dest;
    cudaMalloc(&dest, N * sizeof(dtype));

    vector_add_vector_step<dtype> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (dest, V, U, N);
    
    return dest;
}

template <class dtype>
__host__
void vector_addassign_vector(dtype* V, const dtype* U, size_t N)
{
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    vector_addassign_vector_step<dtype> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (V, U, N);
}

template <class dtype>
__host__
dtype* vector_sub_scalar(const dtype* V, dtype A, size_t N)
{
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    dtype* dest;
    cudaMalloc(&dest, N * sizeof(dtype));

    vector_sub_scalar_step<dtype> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (dest, V, A, N);
    return dest;
}

template <class dtype>
__host__
dtype* vector_sub_vector(const dtype* V, const dtype* U, size_t N)
{// Transfers ownership of a live pointer!
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    dtype* dest;
    cudaMalloc(&dest, N * sizeof(dtype));

    vector_sub_vector_step<dtype> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (dest, V, U, N);
    
    return dest;
}

template <class dtype>
__host__
void vector_subassign_vector(dtype* V, const dtype* U, size_t N)
{
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    vector_subassign_vector_step<dtype> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (V, U, N);
}

template <class dtype>
__host__
dtype* vector_mul_scalar(const dtype* V, dtype A, size_t N)
{// Transfers ownership of a live pointer!
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    dtype* dest;
    cudaMalloc(&dest, N * sizeof(dtype));

    vector_mul_scalar_step<dtype> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (dest, V, A, N);
    
    return dest;
}

template <class dtype>
__host__
dtype* vector_div_scalar(const dtype* V, dtype A, size_t N)
{// Transfers ownership of a live pointer!
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    dtype* dest;
    cudaMalloc(&dest, N * sizeof(dtype));

    vector_div_scalar_step<dtype> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (dest, V, A, N);
    
    return dest;
}

template <typename dtype>
__host__
dtype* vector_dot_matrix(const dtype* T, const dtype* X, size_t M, size_t N, dtype bias)
{
    dtype* dest;
    cudaMalloc(&dest, M * N * sizeof(dtype));
    vector_dot_matrix_step<dtype> <<<N, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (dest, T, X, M, N, bias);
    return dest;
}

template <typename dtype, class UnaryOp>
__host__
dtype* vector_transform(const dtype* X, size_t N, dtype bias, UnaryOp thunk)
{
    dtype* dest;
    cudaMalloc(&dest, N * sizeof(dtype));
    vector_transform_step<dtype> <<<N, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (dest, X, N, bias, thunk);
    return dest;
}

template <typename dtype, class UnaryOp>
__host__
dtype* vector_dot_matrix_transform(const dtype* T, const dtype* X, size_t M, size_t N,
                                   dtype bias, UnaryOp thunk)
{
    dtype* dest;
    cudaMalloc(&dest, M * N * sizeof(dtype));
    vector_dot_matrix_transform_step<dtype> <<<N, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (dest, T, X, M, N, bias, thunk);
    return dest;
}

template <typename dtype, typename itype, class UnaryOp>
__host__
itype vector_reduce(const dtype* V, size_t N, itype acc, UnaryOp thunk)
{
    unsigned int n_blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    itype* block_sums;
    cudaMalloc(&block_sums, n_blocks * sizeof(itype));
    vector_reduce_step<dtype, itype, UnaryOp> <<<n_blocks, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (block_sums, V, N, thunk);
    
    unsigned int n_blocks_2 = (n_blocks + threads_per_block * 2 - 1) / (threads_per_block * 2);
    vector_reduce_step<itype, itype> <<<n_blocks_2, threads_per_block, threads_per_block * sizeof(itype)>>>
        (block_sums, block_sums, n_blocks_2, thunk);
    itype sum;
    cudaMemcpy(&sum, &block_sums[0], sizeof(itype), cudaMemcpyDeviceToHost);
    return acc + sum;
}

template <typename dtype, typename itype, class BinaryOp>
__host__
itype vector_double_reduce(const dtype* V, const dtype* U, size_t N,
                            itype acc, BinaryOp thunk)
{   /*
    vector_double_reduce_step<dtype, itype, BinaryOp> <<<N, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (&acc, V, N, bias, thunk);
    */
    return acc;
}

struct Sigmoid
{
    template <typename dtype>
    __host__ __device__
    dtype operator () (dtype z) const
    {
        return 1 / (1 + std::exp(-z));
    }
} sigmoid;

struct LogLoss
{
    template <typename dtype>
    __host__ __device__
    double operator () (dtype h, dtype y) const
    {
        return -((double) y * log(h + 1e-15) + (double) (1 - y) * log(1 - h + 1e-15));
    }
} log_loss;

template <typename dtype>
struct LogLoser
{
    dtype h;
    LogLoser() = default;
    
    __host__ __device__
    double operator () (dtype y) const
    {
        return -(y * log(h + 1e-15) + (1 - y) * log(1 - h + 1e-15));
    }

    __host__ __device__
    double operator () (dtype h, dtype y) const
    {
        return -(y * log(h + 1e-15) + (1 - y) * log(1 - h + 1e-15));
    }
};
};