#include <cuda_runtime.h>
#include "../src/cuda_ptr.h"
#include "utils.h"

#include <iostream>

// 1.0/ sqrt(2)
#define haar 0.5f

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

void mat_to_float(unsigned char* in, float* out, int width, int height)
{
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            out[i * width + j] = static_cast<float>(in[i * width + j]);
        }
    }
}

void float_to_mat(float *in, unsigned char* out, int width, int height)
{
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            out[i * width + j] = static_cast<unsigned char>(fabs(in[i * width + j]));
        }
    }
}


