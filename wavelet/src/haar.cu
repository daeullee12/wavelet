#include <cuda_runtime.h>
// #include "../src/cuda_ptr.h"
// #include "utils.h"
#include "error.h"
#include <iostream>
// #include <studio.h>
#include <cassert>


// 1.0/ sqrt(2)
#define haar 0.5f

// // __host__ __device__
// void disp(double *t, const int n)
// {
//     int i;
//     for(i=0;i<n;i++)
//     {
//         printf("%lf ", t[i]);
//     }
//     printf("\n");
// }
// // __host__ __device__
// void fill_rand(double *t, const int n)
// {
//     int i=0;
//     for(;i<n;i++)
//     {
//         t[i] = ((double)rand())/INT_MAX;
//     }
// }


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

/*  Haar wavelets forward horizontal and vertical passes 
    To get the full decomposition we apply one after the other
    log_2(N) times and its done */

template<typename T>
__global__ void gpu_haar_horizontal(T* in, const int n, T* out, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if( i < n && j < n/2)
	{
		int idx_in 	= i*N + 2*j;    // (i,2*j)
		int idx_out 	= j + i*N;      // (i,j)

		out[idx_out] 		= haar*(in[idx_in] + in[idx_in+1]);
        // out(i,2*j + n/2)
		out[idx_out + n/2] 	= haar*(in[idx_in] - in[idx_in+1]);
	}
}

template<typename T>
__global__ void gpu_haar_vertical(T* in, const int n, T* out, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if( i < n/2 && j < n)
	{
		int in_idx_1 	= 2*i*N + j;
		int in_idx_2 	= (2*i+1)*N + j;
		int out_idx 	= j + i*N;

        out[out_idx]            = haar*(in[in_idx_1] + in[in_idx_2]);
        // out(i+n/2,j)
        out[out_idx + N*n/2]    = haar*(in[in_idx_1] - in[in_idx_2]);
	}
}

template<typename T>
__global__ void gpu_low_pass(T* x, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < n && j < n)
    {
        if(fabs(x[i*n+j]) < 1.5f)
        {
            x[i*n+j] = 0.0f;
        }
    }
}


//define the haar wavelet transform
void run_haar_wavelet_gpu(float *channel_img, int width, int height, int haar_level)
{
    // check if the level is smaller than the maximum possible
    assert(haar_level <= int(log2(width)));
    if (haar_level == 0) return;

    int N = width * height;
    assert(check_power_two(width) && check_power_two(height));

    size_t size = N * sizeof(float);
    float *d_src, *d_dst;

    int threadsPerBlock = 16;
    dim3 threads(threadsPerBlock, threadsPerBlock);
    dim3 blocks((width + threadsPerBlock - 1) / threadsPerBlock, (height + threadsPerBlock - 1) / threadsPerBlock);

    HANDLE_ERROR(cudaMalloc((void**)&d_src, size));
    HANDLE_ERROR(cudaMalloc((void**)&d_dst, size));

    HANDLE_ERROR(cudaMemcpy(d_src, channel_img, size, cudaMemcpyHostToDevice));

    clock_t begin, end;
    begin = clock();

    // Perform a single level Haar wavelet transform
    gpu_haar_horizontal<<<blocks, threads>>>(d_src, width, d_dst, width);
    gpu_low_pass<<<blocks, threads>>>(d_dst, width);
    gpu_haar_vertical<<<blocks, threads>>>(d_dst, height, d_src, width);
    gpu_low_pass<<<blocks, threads>>>(d_src, height);

    cudaDeviceSynchronize();
    end = clock();
    HANDLE_ERROR(cudaMemcpy(channel_img, d_src, size, cudaMemcpyDeviceToHost));


    // if levels > 1, recursively apply the wavelet transform
    if (haar_level > 1) {
        // Recursively apply the wavelet transform
        
        // Allocate buffers for the current level
        float *LL = new float[(width / 2) * (height / 2)];

        // take the left top corner of the image from channel image
        for (int i = 0; i < height / 2; ++i) {
            for (int j = 0; j < width / 2; ++j) {
                LL[i * (width / 2) + j] = channel_img[i * width + j];
            }
        }

        run_haar_wavelet_gpu(LL, width / 2, height / 2, haar_level - 1);

        // Copy the low-frequency coefficients back to channel_img
        for (int i = 0; i < height / 2; ++i) {
            for (int j = 0; j < width / 2; ++j) {
                channel_img[i * width + j] = LL[i * (width / 2) + j];
            }
        }

        delete[] LL;

    }


    printf("GPU Elapsed: %lfs \n", elapsed(begin, end));

    cudaFree(d_src);
    cudaFree(d_dst);
}
