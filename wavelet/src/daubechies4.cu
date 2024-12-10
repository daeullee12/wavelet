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

// Define Daubechies 4 coefficients
__constant__ float db4_low[4] = {0.4829629131445341, 0.8365163037378077, 0.2241438680420134, -0.1294095225512603};
__constant__ float db4_high[4] = {-0.1294095225512603, -0.2241438680420134, 0.8365163037378077, -0.4829629131445341};


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



// CUDA Kernel to process rows
__global__ void wavelet_transform_rows(const float* input, float* low_output, float* high_output, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width / 2) {
        float low_sum = 0.0f;
        float high_sum = 0.0f;

        // Apply wavelet transform to the row
        for (int k = 0; k < 4; ++k) {
            int idx = (2 * col + k) % width;
            low_sum += input[row * width + idx] * db4_low[k];
            high_sum += input[row * width + idx] * db4_high[k];
        }

        // Store results
        low_output[row * (width / 2) + col] = low_sum;
        high_output[row * (width / 2) + col] = high_sum;
    }
}

// CUDA Kernel to process columns
__global__ void wavelet_transform_columns(const float* input, float* low_output, float* high_output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height / 2) {
        float low_sum = 0.0f;
        float high_sum = 0.0f;

        // Apply wavelet transform to the column
        for (int k = 0; k < 4; ++k) {
            int idx = (2 * row + k) % height;
            low_sum += input[idx * width + col] * db4_low[k];
            high_sum += input[idx * width + col] * db4_high[k];
        }

        // Store results
        low_output[row * width + col] = low_sum;
        high_output[row * width + col] = high_sum;
    }
}

void run_daubechies4_wavelet_gpu(float *channel_img, int width, int height, int levels) {
    // Allocate device memory
    float *d_image;
    cudaMalloc(&d_image, width * height * sizeof(float));

    // Copy image to device
    cudaMemcpy(d_image, channel_img, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate buffers for the current level
    float *d_temp_low, *d_temp_high, *d_LL, *d_LH, *d_HL, *d_HH;
    cudaMalloc(&d_temp_low, (width / 2) * height * sizeof(float));
    cudaMalloc(&d_temp_high, (width / 2) * height * sizeof(float));
    cudaMalloc(&d_LL, (width / 2) * (height / 2) * sizeof(float));
    cudaMalloc(&d_LH, (width / 2) * (height / 2) * sizeof(float));
    cudaMalloc(&d_HL, (width / 2) * (height / 2) * sizeof(float));
    cudaMalloc(&d_HH, (width / 2) * (height / 2) * sizeof(float));

    // Define CUDA grid and block dimensions
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((width / 2 + 15) / 16, (height + 15) / 16);

    // Transform rows
    wavelet_transform_rows<<<blocks_per_grid, threads_per_block>>>(d_image, d_temp_low, d_temp_high, width, height);
    cudaDeviceSynchronize();

    // Transform columns
    dim3 blocks_per_grid_cols((width / 2 + 15) / 16, (height / 2 + 15) / 16);
    wavelet_transform_columns<<<blocks_per_grid_cols, threads_per_block>>>(
        d_temp_low, d_LL, d_LH, width / 2, height
    );
    cudaDeviceSynchronize();
    wavelet_transform_columns<<<blocks_per_grid_cols, threads_per_block>>>(
        d_temp_high, d_HL, d_HH, width / 2, height
    );
    cudaDeviceSynchronize();

    // Copy LL to the top-left
    cudaMemcpy2D(d_image, width * sizeof(float), d_LL, (width / 2) * sizeof(float), 
                (width / 2) * sizeof(float), height / 2, cudaMemcpyDeviceToDevice);

    // Copy LH to the top-right
    cudaMemcpy2D(d_image + (width / 2), width * sizeof(float), d_LH, (width / 2) * sizeof(float), 
                (width / 2) * sizeof(float), height / 2, cudaMemcpyDeviceToDevice);

    // Copy HL to the bottom-left
    cudaMemcpy2D(d_image + (height / 2) * width, width * sizeof(float), d_HL, (width / 2) * sizeof(float), 
                (width / 2) * sizeof(float), height / 2, cudaMemcpyDeviceToDevice);

    // Copy HH to the bottom-right
    cudaMemcpy2D(d_image + (height / 2) * width + (width / 2), width * sizeof(float), d_HH, 
                (width / 2) * sizeof(float), (width / 2) * sizeof(float), height / 2, cudaMemcpyDeviceToDevice);


    // Copy concatenated result back to host
    cudaMemcpy(channel_img, d_image, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    
    // if levels > 1, recursively apply the wavelet transform
    if (levels > 1) {
        // take the LL subband as the new image d_LL to LL

        float *LL = new float[(width / 2) * (height / 2)];
        cudaMemcpy(LL, d_LL, (width / 2) * (height / 2) * sizeof(float), cudaMemcpyDeviceToHost);

        run_daubechies4_wavelet_gpu(LL, width / 2, height / 2, levels - 1);

        for (int i = 0; i < height / 2; ++i) {
            for (int j = 0; j < width / 2; ++j) {
                channel_img[i * width + j] = LL[i * (width / 2) + j];
            }
        }

    }
    
    // Free device memory
    cudaFree(d_image);
    cudaFree(d_temp_low);
    cudaFree(d_temp_high);
    cudaFree(d_LL);
    cudaFree(d_LH);
    cudaFree(d_HL);
    cudaFree(d_HH);
}
