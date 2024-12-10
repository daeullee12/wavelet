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

        float low_sum = 0.0f;
        float high_sum = 0.0f;

        if (2 * i < (n - 3)) {
            low_sum = src[2 * i] * h[0] + src[2 * i + 1] * h[1] + src[2 * i + 2] * h[2] + src[2 * i + 3] * h[3];
            high_sum = src[2 * i] * g[0] + src[2 * i + 1] * g[1] + src[2 * i + 2] * g[2] + src[2 * i + 3] * g[3];
        }
        if (2 * i == (n - 2)) {
            low_sum = src[n - 2] * h[0] + src[n - 1] * h[1] + src[0] * h[2] + src[1] * h[3];
            high_sum = src[n - 2] * g[0] + src[n - 1] * g[1] + src[0] * g[2] + src[1] * g[3];
        }

        dest[i] = low_sum;
        dest[i + half] = high_sum;
    }
}

// Function to perform multi-level wavelet transform
void multi_level_wavelet(float *d_src, float *d_temp, int n, int levels) {
    for (int level = 0; level < levels; ++level) {
        
        int curr_n = n >> level; // Array size reduces by half for each level
        int threads_per_block;
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        threads_per_block = min(prop.maxThreadsPerBlock, curr_n / 2);
        int blocks_per_grid = (curr_n / 2 + threads_per_block - 1) / threads_per_block;

        // Call the kernel for the current level
        gpu_dwt_pass<<<blocks_per_grid, threads_per_block>>>(d_src, d_temp, curr_n);
        cudaDeviceSynchronize();

        // Swap source and destination pointers for next level
        cudaMemcpy(d_src, d_temp, curr_n * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void run_daubechies4_wavelet_gpu(float *channel_img, int width, int height, int levels) {
    int N = width * height;
    assert(check_power_two(N));

    size_t size = N * sizeof(float);
    float *d_src, *d_temp;
    int n = N;

    HANDLE_ERROR(cudaMalloc((void**)&d_src, size));
    HANDLE_ERROR(cudaMalloc((void**)&d_temp, size));

    HANDLE_ERROR(cudaMemcpy(d_src, channel_img, size, cudaMemcpyHostToDevice));

    clock_t begin, end;
    begin = clock();

    // Debugging: Print initial values
    printf("n: %d, levels: %d\n", n, levels);

    // Perform multi-level wavelet transform
    multi_level_wavelet(d_src, d_temp, n, levels);


    end = clock();
    printf("Time taken for wavelet transform: %f\n", elapsed(begin, end));

    // Copy the result back to host
    HANDLE_ERROR(cudaMemcpy(channel_img, d_src, size, cudaMemcpyDeviceToHost));

    // Free device memory
    HANDLE_ERROR(cudaFree(d_src));
    HANDLE_ERROR(cudaFree(d_temp));
}
