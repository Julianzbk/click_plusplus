#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <cstdio>

template <typename dtype = nv_bfloat16>
__global__
void dot_product_vector(const dtype* A, const dtype* B, dtype* partial, size_t N)
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

extern "C" nv_bfloat16 dot_product_vector_bf16(const nv_bfloat16* h_A, const nv_bfloat16* h_B, size_t N)
{
    /*
        Copies host vector to cudaMalloc'd vector, and performs dot product,
        accumulating partial sums and returning their sum.
    */
    constexpr size_t threads_per_block = 256;
    const size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    const size_t size = N * sizeof(nv_bfloat16);

    nv_bfloat16 *A, *B, *partial;
    cudaMalloc(&A, size);
    cudaMalloc(&B, size);
    cudaMalloc(&partial, blocks_per_grid * sizeof(nv_bfloat16));
    cudaMemcpy(A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, size, cudaMemcpyHostToDevice);

    dot_product_vector<nv_bfloat16> <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(nv_bfloat16)>>>
        (A, B, partial, N);
    
    nv_bfloat16* h_partial = new nv_bfloat16[blocks_per_grid];
    cudaMemcpy(h_partial, partial, blocks_per_grid * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
    nv_bfloat16 prod = 0;
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