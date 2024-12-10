#ifndef _DAUBECHIES4_CUH
#define _DAUBECHIES4_CUH

/* Definition of coefficients for the Daubechies 4 wavelet */

constexpr double sqrt3 = 1.73205080757;
constexpr double daub = 5.65685424949;

/* Host constants */

const double _h[4] = {
    (1 + sqrt3)/daub, (3 + sqrt3)/daub,
    (3 - sqrt3)/daub, (1 - sqrt3)/daub
};

const double _g[4] = {
    (1 - sqrt3)/daub, -(3 - sqrt3)/daub, (3 + sqrt3)/daub, -(1 + sqrt3)/daub
};

const double _ih[4] = {
    _h[2],_g[2],_h[0],_g[0]
};

const double _ig[4] = {
    _h[3],_g[3],_h[1],_g[1]
};

/* Device constants */

// extern __constant__ double g[4];
// extern __constant__ double h[4];
// extern __constant__ double ig[4];
// extern __constant__ double ih[4];

void run_daubechies4_wavelet_gpu(float *channel_img, int width, int height, int doubechies_level);

// void run_haar_wavelet_gpu(float *channel_img, int width, int height, int haar_level);
#endif