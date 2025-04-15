#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <cstdio>

constexpr size_t threads_per_block = 256;

using UnaryOp = float (*)(float);

__device__ float no_op_thunk(float arg)
{
    return arg;
}
constexpr UnaryOp NoOpThunk = &no_op_thunk;


template <typename dtype = float>
__global__
void dot_vector_step(const dtype* A, const dtype* B, dtype* partial, size_t N)
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

extern "C" float host_dot_vector_float(const float* h_A, const float* h_B, size_t N)
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

    dot_vector_step<float> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>
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

template <typename dtype = float>
__global__
dtype dot_vector(const dtype* d_A, const dtype* d_B, size_t N)
{
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    dtype* partial;
    cudaMalloc(&partial, blocks_per_grid * sizeof(dtype));

    dot_vector_step<dtype> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(dtype)>>>
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

extern "C" inline float dot_vector_float(const float* d_A, const float* d_B, size_t N)
{
    /*
        Performs dot product directly on device memory.
    */
   return dot_vector<float>(d_A, d_B, N);
}


template <class dtype>
__global__
void vector_add_vector(const dtype* V, const dtype* U, dtype* dest, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] = (i < N) ? V[i] + U[i] : dtype(); // operation step
}

extern "C" float* vector_add_vector_float(const float* V, const float* U, size_t N)
{// Transfers ownership of a live pointer!
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    float* dest;
    cudaMalloc(&dest, N * sizeof(float));

    vector_add_vector<float> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>
        (V, U, dest, N);
    
    return dest;
}

template <class dtype>
__global__
void vector_addassign_vector(dtype* V, const dtype* U, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    V[i] += (i < N) ? U[i] : dtype();
}

extern "C" void vector_addassign_vector_float(float* V, const float* U, size_t N)
{
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    vector_addassign_vector<float> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>
        (V, U, N);
}

template <class dtype>
__global__
void vector_sub_scalar(const dtype* V, dtype A, dtype* dest, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] += (i < N) ? V[i] - A : dtype();
}

extern "C" float* vector_sub_scalar_float(const float* V, float A, size_t N)
{
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    float* dest;
    cudaMalloc(&dest, N * sizeof(float));

    vector_sub_scalar<float> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>
        (V, A, dest, N);
    return dest;
}

template <class dtype>
__global__
void vector_sub_vector(const dtype* V, const dtype* U, dtype* dest, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] = (i < N) ? V[i] - U[i] : dtype(); // operation step
}

extern "C" float* vector_sub_vector_float(const float* V, const float* U, size_t N)
{// Transfers ownership of a live pointer!
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    float* dest;
    cudaMalloc(&dest, N * sizeof(float));

    vector_sub_vector<float> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>
        (V, U, dest, N);
    
    return dest;
}

template <class dtype>
__global__
void vector_subassign_vector(dtype* V, const dtype* U, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    V[i] -= (i < N) ? U[i] : dtype();
}

extern "C" void vector_subassign_vector_float(float* V, const float* U, size_t N)
{
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    vector_subassign_vector<float> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>
        (V, U, N);
}

template <class dtype>
__global__
void vector_mul_scalar(const dtype* V, dtype A, dtype* dest, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] += (i < N) ? V[i] * A : dtype();
}

extern "C" float* vector_mul_scalar_float(const float* V, float A, size_t N)
{// Transfers ownership of a live pointer!
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    float* dest;
    cudaMalloc(&dest, N * sizeof(float));

    vector_mul_scalar<float> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>
        (V, A, dest, N);
    
    return dest;
}

template <class dtype>
__global__
void vector_div_scalar(const dtype* V, dtype A, dtype* dest, size_t N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] += (i < N) ? V[i] / A : dtype();
}

extern "C" float* vector_div_scalar_float(const float* V, float A, size_t N)
{// Transfers ownership of a live pointer!
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    float* dest;
    cudaMalloc(&dest, N * sizeof(float));

    vector_div_scalar<float> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>
        (V, A, dest, N);
    
    return dest;
}


template <typename dtype>
__global__
void vector_dot_matrix_step(const dtype* T, const dtype* X, dtype* dest, size_t M, size_t N, dtype bias)
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

template <typename dtype>
__global__
dtype* vector_dot_matrix(const dtype* T, const dtype* X, size_t M, size_t N, dtype bias = 0)
{
    dtype* dest;
    cudaMalloc(&dest, M * N * sizeof(dtype));
    vector_dot_matrix<float> <<<N, threads_per_block, threads_per_block * sizeof(float)>>>
        (T, X, dest, M, N, bias);
    return dest;
}

extern "C" float* vector_dot_matrix_float(const float* T, const float* X, size_t M, size_t N, float bias = 0.0f)
{
    return vector_dot_matrix<float>(T, X, M, N, bias);
}

template <typename dtype>
void vector_dot_matrix_transform_step(const dtype* T, const dtype* X, dtype* dest,
                                      dtype bias, size_t M, size_t N, UnaryOp thunk)
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
        dest[row] = (*thunk)(cache[0] + bias); // Applies the transform operation
    }
}

template <typename dtype>
__global__
dtype* vector_dot_matrix_transform(const dtype* T, const dtype* X, size_t M, size_t N,
                                   dtype bias = 0, UnaryOp thunk = NoOpThunk)
{
    dtype* dest;
    cudaMalloc(&dest, M * N * sizeof(dtype));
    vector_dot_matrix_transform_step<float> <<<N, threads_per_block, threads_per_block * sizeof(float)>>>
        (T, X, dest, M, N, bias, thunk);
    return dest;
}

extern "C" float* vector_dot_matrix_transform_float(const float* T, const float* X, size_t M, size_t N,
                                                    float bias = 0.0, UnaryOp thunk = NoOpThunk)
{
    return vector_dot_matrix_transform<float>(T, X, M, N, bias, thunk);
}