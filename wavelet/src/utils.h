#ifndef _UTILS_H
#define _UTILS_H

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

// #define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
//     printf("Error %s (%d) at %s:%d\n", cudaGetErrorString(x),x, __FILE__,__LINE__); \
//     return EXIT_FAILURE;}} while(0)

#define HAAR 0.70710678118f  // 1.0 / sqrt(2)


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

// __host__ __device__
void disp(double *t, const int n)
{
    int i;
    for(i=0;i<n;i++)
    {
        printf("%lf ", t[i]);
    }
    printf("\n");
}
// __host__ __device__
void fill_rand(double *t, const int n)
{
    int i=0;
    for(;i<n;i++)
    {
        t[i] = ((double)rand())/INT_MAX;
    }
}
// __host__ __device__
void fill_ones(double *t, const int n)
{
    int i=0;
    for(;i<n;i++)
    {
        t[i] = 1.;
    }
}
// __host__ __device__
void fill_rand_2d(double *t, int n)
{
    int i;

    for(i=0;i<n*n;i++)
    {
        t[i] = (double)rand() / INT_MAX;
    }
}
// __host__ __device__
void disp_2d(double *t, int n)
{
    int i,j;

    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
        {
            printf("%lf ", t[i*n+j]);
        }
        printf("\n");
    }
    
}
// __host__ __device__
int test_arrays_equal(double *t1, double *t2, const int n,
    const double tol = 1e-6)
{
    int i=0;
    for(;i<n;i++) 
    {
        if(fabs(t1[i] - t2[i]) > tol) 
        {
            // printf("Arrays not equal!\n");
            return 0;
        }
    }

    // printf("Arrays are equal!\n");
    return 1;
}


#endif