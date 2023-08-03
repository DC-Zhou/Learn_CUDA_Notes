#include <stdio.h>
#include "reduction.h"

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

// interleaved addressing
// cuda thread synchronization
__global__ void reduction_kernel_0(float* d_out, float* d_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    s_data[threadIdx.x] = (idx_x < size) ? d_in[idx_x] : 0.f;

    __syncthreads();

    // do reduction
    // interleaved addressing
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // thread synchronous reduction
        if ( (idx_x % (stride * 2)) == 0 )
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];

        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_out[blockIdx.x] = s_data[0];
}


//   interleaved addressing
__global__ void reduction_kernel_1(float* d_out, float* d_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    s_data[threadIdx.x] = (idx_x < size) ? d_in[idx_x] : 0.f;

    __syncthreads();

     // do reduction
    // interleaved addressing but strided index an non-divergent branch
    for(unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * threadIdx.x;
        if (index < blockDim.x)
            s_data[index] += s_data[index + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_out[blockIdx.x] = s_data[0];
}


// sequential addressing
__global__ void reduction_kernel_2(float* d_out, float* d_in, unsigned int size)
{
    extern __shared__ float s_data[];

    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    s_data[threadIdx.x] = (idx_x < size) ? d_in[idx_x] : 0.f;

    __syncthreads();

     // do reduction
    // sequential addressing
    for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_out[blockIdx.x] = s_data[0];
}

// with 2 load and first add of the reduction 
// but this is error need to be fixed
__global__ void reduction_kernel_3(float* d_out, float* d_in, unsigned int size)
{
    extern __shared__ float s_data[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid   = threadIdx.x;
    unsigned int idx_x = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    if(idx_x + blockDim.x < size)
        s_data[tid] = d_in[idx_x] + d_in[idx_x + blockDim.x];

    __syncthreads();

     // do reduction
    // sequential addressing
    for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if ((threadIdx.x < stride) && (idx_x + stride < size))            
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_out[blockIdx.x] = s_data[0];
}


void shared_reduction(float *d_out, float *d_in, int n_threads, int size)
{   
    cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);
    while(size > 1)
    {
        int n_blocks = (size + n_threads - 1) / n_threads;
        reduction_kernel_3<<< n_blocks, n_threads, n_threads * sizeof(float), 0 >>>(d_out, d_out, size);
        size = n_blocks;
    } 
}