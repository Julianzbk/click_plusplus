#pragma once

#include <cuda_runtime.h>

constexpr size_t threads_per_block = 256;
#include "vectors.cuh"

#include <cuda_bf16.h>
#include <cstdlib>
#include <cstdio>

namespace device
{
#pragma region functions
struct NoOp
{
    template <typename dtype>
    __host__ __device__
    dtype operator () (dtype x) const
    {
        return x;
    }
} no_op;

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
#pragma endregion functions

extern __shared__ unsigned char shared_cache[];

template <typename dtype = float>
__global__
void vector_dot_step(const dtype* A, const dtype* B, dtype* partial, size_t N)
{
    dtype* cache = reinterpret_cast<dtype*>(shared_cache);
    
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
    if (i < N)
        dest[i] = V[i] + U[i];
}

template <class dtype>
__global__
void vector_addassign_vector_step(dtype* V, const dtype* U, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        V[i] += U[i];
}

template <class dtype>
__global__
void vector_sub_scalar_step(dtype* dest, const dtype* V, dtype A, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        dest[i] = V[i] - A;
}

template <class dtype>
__global__
void vector_sub_vector_step(dtype* dest, const dtype* V, const dtype* U, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        dest[i] = V[i] - U[i];
}

template <class dtype>
__global__
void vector_subassign_vector_step(dtype* V, const dtype* U, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        V[i] -= U[i];
}

template <class dtype>
__global__
void vector_mul_scalar_step(dtype* dest, const dtype* V, dtype A, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        dest[i] = V[i] * A;
}

template <class dtype>
__global__
void vector_div_scalar_step(dtype* dest, const dtype* V, dtype A, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        dest[i] = V[i] / A;
}

template <typename dtype>
__global__
void vector_dot_matrix_step(dtype* dest, const dtype* T, const dtype* X, size_t M, size_t N, dtype bias)
{
    unsigned int row = blockIdx.x;
    if (row >= N)
        return;

    dtype* cache = reinterpret_cast<dtype*>(shared_cache);
    
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
void vector_transform_step(dtype* dest, const dtype* X, size_t N,
                             dtype bias, UnaryOp thunk)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        dest[i] = thunk(X[i] + bias);
}

template <typename dtype, class UnaryOp>
__global__
void matrix_inplace_transform_step(dtype* X, size_t M, size_t N, UnaryOp thunk)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        thunk(X + i * M);
}

template <typename dtype, class UnaryOp>
__global__
void vector_dot_matrix_transform_step(dtype* dest, const dtype* T, const dtype* X, 
                                      size_t M, size_t N, dtype bias, UnaryOp thunk)
{
    unsigned int row = blockIdx.x;
    if (row >= N)
        return;
    
    dtype* cache = reinterpret_cast<dtype*>(shared_cache);

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
[[nodiscard]] dtype* vector_add_vector(const dtype* V, const dtype* U, size_t N)
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
[[nodiscard]] dtype* vector_sub_scalar(const dtype* V, dtype A, size_t N)
{// Transfers ownership of a live pointer!
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    dtype* dest;
    cudaMalloc(&dest, N * sizeof(dtype));

    vector_sub_scalar_step<dtype> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (dest, V, A, N);
    return dest;
}

template <class dtype>
__host__
[[nodiscard]] dtype* vector_sub_vector(const dtype* V, const dtype* U, size_t N)
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
[[nodiscard]] dtype* vector_mul_scalar(const dtype* V, dtype A, size_t N)
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
[[nodiscard]] dtype* vector_div_scalar(const dtype* V, dtype A, size_t N)
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
[[nodiscard]] dtype* vector_dot_matrix(const dtype* T, const dtype* X, size_t M, size_t N, dtype bias)
{// Transfers ownership of a live pointer!
    dtype* dest;
    cudaMalloc(&dest, M * N * sizeof(dtype));
    vector_dot_matrix_step<dtype> <<<N, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (dest, T, X, M, N, bias);
    return dest;
}

template <typename dtype, class UnaryOp>
__host__
[[nodiscard]] dtype* vector_transform(const dtype* X, size_t N, dtype bias, UnaryOp thunk)
{// Transfers ownership of a live pointer!
    dtype* dest;
    cudaMalloc(&dest, N * sizeof(dtype));
    vector_transform_step<dtype, UnaryOp> <<<N, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (dest, X, N, bias, thunk); 
    return dest;
}

template <typename dtype, class UnaryOp>
__host__
void vector_inplace_transform(dtype* X, size_t N, dtype bias, UnaryOp thunk)
{
    vector_transform_step<dtype, UnaryOp> <<<N, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (X, X, N, bias, thunk);
}

template <typename dtype, class UnaryOp>
__host__
void matrix_inplace_transform(dtype* A, size_t M, size_t N, UnaryOp thunk)
{
    matrix_inplace_transform_step<dtype, UnaryOp> <<<N, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (A, M, N, thunk);
}


template <typename dtype, class UnaryOp>
__host__
[[nodiscard]] dtype* vector_dot_matrix_transform(const dtype* T, const dtype* X, size_t M, size_t N,
                                   dtype bias, UnaryOp thunk)
{// Transfers ownership of a live pointer!
    dtype* dest;
    cudaMalloc(&dest, M * N * sizeof(dtype));
    vector_dot_matrix_transform_step<dtype> <<<N, threads_per_block, threads_per_block * sizeof(dtype)>>>
        (dest, T, X, M, N, bias, thunk);
    return dest;
}

template <typename dtype, typename itype, class UnaryOp>
__global__
void vector_reduce_step(itype* block_sums, const dtype* V, size_t N, UnaryOp thunk)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    itype* cache = reinterpret_cast<itype*>(shared_cache);
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;

    itype acc = itype();
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
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1)
        {
            cache[tid] += __shfl_down_sync(0xffffffff, cache[tid], s);
        }
    }

    if (tid == 0)
    {
        block_sums[blockIdx.x] = cache[0];
    }
}

template <typename dtype, typename itype, class UnaryOp>
__host__
itype vector_reduce(const dtype* V, size_t N, itype acc, UnaryOp thunk)
{
    unsigned int n_blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    itype* block_sums;
    cudaMalloc(&block_sums, n_blocks * sizeof(itype));
    vector_reduce_step<dtype, itype, UnaryOp> <<<n_blocks, threads_per_block, threads_per_block * sizeof(itype)>>>
        (block_sums, V, N, thunk);
    
    unsigned int n_blocks_2 = (n_blocks + threads_per_block * 2 - 1) / (threads_per_block * 2);
    vector_reduce_step<itype, itype> <<<n_blocks_2, threads_per_block, threads_per_block * sizeof(itype)>>>
        (block_sums, block_sums, n_blocks_2, no_op);
    itype sum;
    cudaMemcpy(&sum, &block_sums[0], sizeof(itype), cudaMemcpyDeviceToHost);
    cudaFree(block_sums);
    return acc + sum;
}

template <typename dtype, typename itype, class BinaryOp>
__global__
void vector_double_reduce_step(itype* block_sums, const dtype* V, const dtype* U,
                               size_t N, BinaryOp thunk)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    if (blockIdx.x * blockDim.x >= N)
        return;

    itype* cache = reinterpret_cast<itype*>(shared_cache);

    itype acc = itype();
    for (size_t i = idx; i < N; i += blockDim.x * gridDim.x)
    {
        acc += thunk(V[i], U[i]);
    }

    cache[tid] = acc;
    __syncthreads();

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
        block_sums[blockIdx.x] = cache[0];
    }
}

template <typename dtype, typename itype, class BinaryOp>
__host__
itype vector_double_reduce(const dtype* V, const dtype* U, size_t N,
                            itype acc, BinaryOp thunk)
{
    unsigned int n_blocks = (N + threads_per_block - 1) / threads_per_block;
    itype* d_sums;
    cudaMalloc(&d_sums, n_blocks * sizeof(itype));
    vector_double_reduce_step<dtype, itype, BinaryOp> <<<n_blocks, threads_per_block, threads_per_block * sizeof(itype)>>>
        (d_sums, V, U, N, thunk);
    
    std::vector<itype> h_sums(n_blocks);
    cudaMemcpy(h_sums.data(), d_sums, n_blocks * sizeof(itype), cudaMemcpyDeviceToHost);
    cudaFree(d_sums);
    for (itype sum: h_sums)
        acc += sum;
    return acc;
}

template <typename dtype>
struct WelfordState
{
    dtype mean;
    dtype m_sq;
    size_t count;

    __host__ __device__
    WelfordState()
        :mean(0.0), m_sq(0.0), count(0ULL)
    {
    }
};

template <typename dtype>
__device__
WelfordState<dtype> welford_acc(WelfordState<dtype> state, dtype x)
{
    state.count++;
    dtype delta = x - state.mean;
    state.mean += delta / state.count;
    state.m_sq += delta * (x - state.mean);
    return state;
}

template <typename dtype>
__device__
WelfordState<dtype> welford_merge(WelfordState<dtype> const& lhs, WelfordState<dtype> const& rhs)
{
    if (lhs.count == 0)
        return rhs;
    if (rhs.count == 0)
        return lhs;
    WelfordState<dtype> out;
    out.count = lhs.count + rhs.count;
    dtype delta = rhs.mean - lhs.mean;
    out.mean = lhs.mean + delta * ((double) rhs.count / out.count); // pulls the mean from left to center
    out.m_sq = lhs.m_sq + rhs.m_sq + delta * delta * ((double) lhs.count * rhs.count / out.count);
    return out;
}

template <typename dtype>
__global__
void welford_step(const dtype* A, size_t M, size_t N,
                  dtype* means, dtype* stds)
{
    WelfordState<dtype>* cache = reinterpret_cast<WelfordState<dtype>*>(shared_cache);
    unsigned int col = blockIdx.x;
    if (col >= M)
        return;
    WelfordState<dtype> local_state;
    unsigned int tid = threadIdx.x;
    for (unsigned int i = tid; i < N; i += blockDim.x)
    {
        size_t idx = size_t(i) * M + size_t(col);
        if (idx >= M * N)
        {
            assert(false);
            printf("DEBUG: data=%p rows=%llu cols=%llu maxIdx=%llu, idx=%llu\n",
               A, N, M, M * N, idx);
        }
        dtype x = A[idx];
        local_state = welford_acc(local_state, x);
    }
    cache[tid] = local_state;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            cache[tid] = welford_merge(cache[tid], cache[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        //printf("DEBUG: col=%lu\n", col);
        WelfordState<dtype> tot = cache[0];
        means[col] = tot.mean;
        stds[col] = (tot.count > 1) ? sqrt(tot.m_sq / (tot.count - 1)) : 0;
    }
}

template <typename dtype, size_t M>
__host__
void welford(const dtype* A, size_t N, 
             std::array<dtype, M>& means, std::array<dtype, M>& stds)
{
    size_t row_n_bytes = M * sizeof(dtype);
    dtype* d_means, *d_stds;
    cudaMalloc(&d_means, row_n_bytes);
    cudaMalloc(&d_stds, row_n_bytes);

    welford_step<dtype> <<<M, threads_per_block, threads_per_block * sizeof(WelfordState<dtype>)>>>
        (A, M, N, d_means, d_stds);
    cudaDeviceSynchronize();

    cudaMemcpy(means.data(), d_means, row_n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(stds.data(), d_stds, row_n_bytes, cudaMemcpyDeviceToHost);   
    cudaFree(d_means);
    cudaFree(d_stds);
}
};