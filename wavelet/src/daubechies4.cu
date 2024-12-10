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

// // Function to perform multi-level wavelet transform
// void multi_level_wavelet_transform(float* d_image, int width, int height, int levels) {
//     float *d_temp_low, *d_temp_high, *d_LL, *d_LH, *d_HL, *d_HH;

//     for (int level = 0; level < levels; ++level) {
//         int curr_width = width >> level;   // Divide by 2 for each level
//         int curr_height = height >> level;


//         HANDLE_ERROR(cudaMalloc((void**)&d_temp_low, curr_width * curr_height * sizeof(float) / 2));
//         HANDLE_ERROR(cudaMalloc((void**)&d_temp_high, curr_width * curr_height * sizeof(float) / 2));
//         HANDLE_ERROR(cudaMalloc((void**)&d_LL, (curr_width / 2) * (curr_height / 2) * sizeof(float)));
//         HANDLE_ERROR(cudaMalloc((void**)&d_LH, (curr_width / 2) * (curr_height / 2) * sizeof(float)));
//         HANDLE_ERROR(cudaMalloc((void**)&d_HL, (curr_width / 2) * (curr_height / 2) * sizeof(float)));
//         HANDLE_ERROR(cudaMalloc((void**)&d_HH, (curr_width / 2) * (curr_height / 2) * sizeof(float)));

//         // Grid and block sizes
//         dim3 threads_per_block(16, 16);
//         dim3 blocks_per_grid((curr_width / 2 + 15) / 16, (curr_height + 15) / 16);

//         // Apply wavelet transform to rows
//         wavelet_transform_rows<<<blocks_per_grid, threads_per_block>>>(
//             d_image, d_temp_low, d_temp_high, curr_width, curr_height
//         );
//         cudaDeviceSynchronize();

//         // Apply wavelet transform to columns
//         dim3 blocks_per_grid_cols((curr_width / 2 + 15) / 16, (curr_height / 2 + 15) / 16);
//         wavelet_transform_columns<<<blocks_per_grid_cols, threads_per_block>>>(
//             d_temp_low, d_LL, d_LH, curr_width / 2, curr_height
//         );
//         cudaDeviceSynchronize();
//         wavelet_transform_columns<<<blocks_per_grid_cols, threads_per_block>>>(
//             d_temp_high, d_HL, d_HH, curr_width / 2, curr_height
//         );
//         cudaDeviceSynchronize();

//         // Copy LL subband back to `d_image` for the next level


//         cudaMemcpy(d_image, d_LL, (curr_width / 2) * (curr_height / 2) * sizeof(float), cudaMemcpyDeviceToDevice);
//         cudaMemcpy(d_image + (curr_width / 2) * (curr_height / 2), d_LH, (curr_width / 2) * (curr_height / 2) * sizeof(float), cudaMemcpyDeviceToDevice);
//         cudaMemcpy(d_image + (curr_width / 2) * curr_height, d_HL, (curr_width / 2) * (curr_height / 2) * sizeof(float), cudaMemcpyDeviceToDevice);
//         cudaMemcpy(d_image + (curr_width / 2) * curr_height + (curr_width / 2) * (curr_height / 2), d_HH, (curr_width / 2) * (curr_height / 2) * sizeof(float), cudaMemcpyDeviceToDevice);

//         // Free intermediate memory
//         cudaFree(d_temp_low);
//         cudaFree(d_temp_high);
//         cudaFree(d_LL);
//         cudaFree(d_LH);
//         cudaFree(d_HL);
//         cudaFree(d_HH);
//     }
// }


void run_daubechies4_wavelet_gpu(float *channel_img, int width, int height, int levels) {
        // Allocate device memory
    float *d_image, *d_temp_low, *d_temp_high, *d_LL, *d_LH, *d_HL, *d_HH;
    cudaMalloc(&d_image, width * height * sizeof(float));
    cudaMalloc(&d_temp_low, width * height * sizeof(float) / 2);
    cudaMalloc(&d_temp_high, width * height * sizeof(float) / 2);
    cudaMalloc(&d_LL, (width / 2) * (height / 2) * sizeof(float));
    cudaMalloc(&d_LH, (width / 2) * (height / 2) * sizeof(float));
    cudaMalloc(&d_HL, (width / 2) * (height / 2) * sizeof(float));
    cudaMalloc(&d_HH, (width / 2) * (height / 2) * sizeof(float));

    // Copy image to device
    cudaMemcpy(d_image, channel_img, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((width / 2 + 15) / 16, (height + 15) / 16);

    // Transform rows
    wavelet_transform_rows<<<blocks_per_grid, threads_per_block>>>(d_image, d_temp_low, d_temp_high, width, height);

    // Transform columns
    dim3 blocks_per_grid_cols((width + 15) / 16, (height / 2 + 15) / 16);
    wavelet_transform_columns<<<blocks_per_grid_cols, threads_per_block>>>(
        d_temp_low, d_LL, d_LH, width / 2, height
    );
    wavelet_transform_columns<<<blocks_per_grid_cols, threads_per_block>>>(
        d_temp_high, d_HL, d_HH, width / 2, height
    );

    // Retrieve results

    cudaMemcpy(channel_img, d_LL, (width / 2) * (height / 2) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(channel_img + (width / 2) * (height / 2), d_LH, (width / 2) * (height / 2) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(channel_img + (width / 2) * height, d_HL, (width / 2) * (height / 2) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(channel_img + (width / 2) * height + (width / 2) * (height / 2), d_HH, (width / 2) * (height / 2) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_temp_low);
    cudaFree(d_temp_high);
    cudaFree(d_LL);
    cudaFree(d_LH);
    cudaFree(d_HL);
    cudaFree(d_HH);
}
