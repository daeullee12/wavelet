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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

void run_daubechies4_wavelet_gpu(float *channel_img, int width, int height, int levels) {
    // Allocate device memory
    float *d_image;
    CUDA_CHECK(cudaMalloc(&d_image, width * height * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_image, channel_img, width * height * sizeof(float), cudaMemcpyHostToDevice));

    int current_width = width;
    int current_height = height;

    for (int level = 0; level < levels; ++level) {
        float *d_temp_low, *d_temp_high, *d_LL, *d_LH, *d_HL, *d_HH;
        CUDA_CHECK(cudaMalloc(&d_temp_low, (current_width / 2) * current_height * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp_high, (current_width / 2) * current_height * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_LL, (current_width / 2) * (current_height / 2) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_LH, (current_width / 2) * (current_height / 2) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_HL, (current_width / 2) * (current_height / 2) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_HH, (current_width / 2) * (current_height / 2) * sizeof(float)));

        // Define CUDA grid and block dimensions
        dim3 threads_per_block(16, 16);
        dim3 blocks_per_grid((current_width / 2 + 15) / 16, (current_height + 15) / 16);

        // Transform rows
        wavelet_transform_rows<<<blocks_per_grid, threads_per_block>>>(d_image, d_temp_low, d_temp_high, current_width, current_height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Transform columns
        dim3 blocks_per_grid_cols((current_width + 15) / 16, (current_height / 2 + 15) / 16);
        wavelet_transform_columns<<<blocks_per_grid_cols, threads_per_block>>>(
            d_temp_low, d_LL, d_LH, current_width / 2, current_height
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        wavelet_transform_columns<<<blocks_per_grid_cols, threads_per_block>>>(
            d_temp_high, d_HL, d_HH, current_width / 2, current_height
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Calculate offsets for subbands
        int offset_x = (current_width / 2);
        int offset_y = (current_height / 2);

        // Copy LL to the top-left
        CUDA_CHECK(cudaMemcpy2D(d_image, current_width * sizeof(float), d_LL, (current_width / 2) * sizeof(float), (current_width / 2) * sizeof(float), current_height / 2, cudaMemcpyDeviceToDevice));

        // Copy LH to the top-right
        CUDA_CHECK(cudaMemcpy2D(d_image + offset_x, current_width * sizeof(float), d_LH, (current_width / 2) * sizeof(float), (current_width / 2) * sizeof(float), current_height / 2, cudaMemcpyDeviceToDevice));

        // Copy HL to the bottom-left
        CUDA_CHECK(cudaMemcpy2D(d_image + offset_y * current_width, current_width * sizeof(float), d_HL, (current_width / 2) * sizeof(float), (current_width / 2) * sizeof(float), current_height / 2, cudaMemcpyDeviceToDevice));

        // Copy HH to the bottom-right
        CUDA_CHECK(cudaMemcpy2D(d_image + offset_y * current_width + offset_x, current_width * sizeof(float), d_HH, (current_width / 2) * sizeof(float), (current_width / 2) * sizeof(float), current_height / 2, cudaMemcpyDeviceToDevice));

        // Prepare for the next level by copying LH to the input image
        if (level < levels - 1) {
            CUDA_CHECK(cudaMemcpy(d_image, d_LH, (current_width / 2) * (current_height / 2) * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        // Update dimensions for the next level
        current_width /= 2;
        current_height /= 2;

        cudaFree(d_temp_low);
        cudaFree(d_temp_high);
        cudaFree(d_LL);
        cudaFree(d_LH);        
        cudaFree(d_HL);        
        cudaFree(d_HH);    
        }    
        
        // Copy concatenated result back to host    
        CUDA_CHECK(cudaMemcpy(channel_img, d_image, width * height * sizeof(float), cudaMemcpyDeviceToHost));    
        
        // Free device memory    
        cudaFree(d_image);
        
        }

