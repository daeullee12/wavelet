#include <cuda_runtime.h>
#include <cassert>
#include "daubechies4.cuh"
// #include "utils.h"
#include "error.h"

// Ensure no conflicting identifiers with the `numbers` header
#undef numbers

/*  The Daubechies-4 wavelet forward pass
    I adapted this code from http://bearcave.com/misl/misl_tech/wavelets/index.html
    To compute the full wavelet transform of a signal of size N
    We call this kernel log_2(N) times (assuming N is power of 2) */

// Define device constants
__constant__ double g[4];
__constant__ double h[4];
__constant__ double ig[4];
__constant__ double ih[4];

// __host__ __device__
inline double elapsed(clock_t start, clock_t end)
{
    return double(end - start) / CLOCKS_PER_SEC;
}
// __host__ __device__
inline bool check_power_two(int x)
{
    return (x & (x - 1)) == 0;
}


__global__ void gpu_dwt_pass(float *src, float *dest, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n >> 1;

    if (i < half) {
        if (2 * i < (n - 3)) {
            dest[i] = src[2 * i] * h[0] + src[2 * i + 1] * h[1] + src[2 * i + 2] * h[2] + src[2 * i + 3] * h[3];
            dest[i + half] = src[2 * i] * g[0] + src[2 * i + 1] * g[1] + src[2 * i + 2] * g[2] + src[2 * i + 3] * g[3];
        }
        if (2 * i == (n - 2)) {
            dest[i] = src[n - 2] * h[0] + src[n - 1] * h[1] + src[0] * h[2] + src[1] * h[3];
            dest[i + half] = src[n - 2] * g[0] + src[n - 1] * g[1] + src[0] * g[2] + src[1] * g[3];
        }
    }
}

void run_daubechies4_wavelet_gpu(float *channel_img, int width, int height, int levels)
{
    int N = width * height;
    assert(check_power_two(N));

    size_t size = N * sizeof(float);
    float *d_src, *d_dst;
    // int n = N;

    dim3 threadsPerBlock(16,16);
    dim3 blocks_per_grid((width / 2 + 15) / 16, (height + 15) / 16);

    HANDLE_ERROR(cudaMalloc((void**)&d_src, size));
    HANDLE_ERROR(cudaMalloc((void**)&d_dst, size));

    HANDLE_ERROR(cudaMemcpy(d_src, channel_img, size, cudaMemcpyHostToDevice));

    // Apply the Daubechies wavelet transformation to rows

    gpu_dwt_pass<<<blocks_per_grid, threadsPerBlock>>>(d_dst, d_src, width);
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(cuda_status) << std::endl;
        return;
    }
    else
    {
        std::cout << "CUDA kernel launch success" << std::endl;
    }

    cudaDeviceSynchronize();

    //

    // Apply the Daubechies wavelet transformation to columns
    gpu_dwt_pass<<<blocks_per_grid, threadsPerBlock>>>(d_dst, d_src, height);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(cuda_status) << std::endl;
        return;
    }
    else
    {
        std::cout << "CUDA kernel launch success" << std::endl;
    }


    cudaDeviceSynchronize();
    // end = clock();
    
    // copy image to the host
    HANDLE_ERROR(cudaMemcpy(channel_img, d_dst, size, cudaMemcpyDeviceToHost));
    
    // HANDLE_ERROR(cudaMemcpy(channel_img, d_dst, size, cudaMemcpyDeviceToHost));

    // printf("GPU Elapsed: %lfs \n", elapsed(begin, end));

    cudaFree(d_src);
    cudaFree(d_dst);
}