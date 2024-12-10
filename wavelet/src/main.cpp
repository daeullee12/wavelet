#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "daubechies4.cuh"
#include "utils.h"
// #include "haar.cu"
// #include "daubechies4.cu"

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

void cpu_haar_horizontal(float* in, int n, float* out, int N)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n / 2; j++)
        {
            int idx_in = i * N + 2 * j;
            int idx_out = j + i * N;

            out[idx_out] = HAAR * (in[idx_in] + in[idx_in + 1]);
            out[idx_out + n / 2] = HAAR * (in[idx_in] - in[idx_in + 1]);
        }
    }
}

void cpu_haar_vertical(float* in, int n, float* out, int N)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n / 2; j++)
        {
            int idx_in = 2 * j * N + i;
            int idx_out = j * N + i;

            out[idx_out] = HAAR * (in[idx_in] + in[idx_in + N]);
            out[idx_out + n / 2 * N] = HAAR * (in[idx_in] - in[idx_in + N]);
        }
    }
}

void run_haar_wavelet_cpu(float* img, int width, int height, int levels)
{
    const int size = width * height * sizeof(float);
    float *frame_buf = new float[size];
    float *wav_buf = new float[size];

    if (frame_buf == nullptr || wav_buf == nullptr) {
        std::cerr << "Failed to allocate memory for wavelet buffers" << std::endl;
        delete[] frame_buf;
        delete[] wav_buf;
        return;
    }

    for (int level = 0; level < levels; level++) {
        int step = width >> level;
        cpu_haar_horizontal(img, step, wav_buf, width);
        cpu_haar_vertical(wav_buf, step, img, width);
    }

    delete[] frame_buf;
    delete[] wav_buf;
}

void cpu_d4_transform(float *src, float* dest, const int n)
{
    
    if (n >= 4) 
    {
        int i=0,j=0;
        const int half = n>>1;

        for (i = 0; i < half; i++) 
        {
            j = 2*i;
            if (j < n-3) {
                dest[i]      = src[j]*_h[0] + src[j+1]*_h[1] + src[j+2]*_h[2] + src[j+3]*_h[3];
                dest[i+half] = src[j]*_g[0] + src[j+1]*_g[1] + src[j+2]*_g[2] + src[j+3]*_g[3];
            } 
            else { 
                break; 
            }
        }

        dest[i]      = src[n-2]*_h[0] + src[n-1]*_h[1] + src[0]*_h[2] + src[1]*_h[3];
        dest[i+half] = src[n-2]*_g[0] + src[n-1]*_g[1] + src[0]*_g[2] + src[1]*_g[3];
    }
}

float cpu_dwt(float* t, int N)
{
    assert(check_power_two(N));
    int n=N;
    clock_t begin,end;
    float *tmp = (float*)malloc(N*sizeof(float));

    if(!tmp)
    {
        fprintf(stderr,"(host) cannot allocate memory for DWT\n");
        exit(EXIT_FAILURE);
    }

    begin = clock();
    while(n >= 4) 
    {
        cpu_d4_transform(t,tmp,n);
        memcpy(t,tmp,n*sizeof(float));

        n >>= 1;
    }

    end = clock();
    printf("CPU Elapsed: %lfs\n", elapsed(begin,end));
    free(tmp);
    return elapsed(begin,end);
}


void run_daubechies4_wavelet_cpu(float* img, int width, int height, int levels)
{
    if (!check_power_two(width) || !check_power_two(height)) {
        std::cerr << "Error: Width and height must be powers of two." << std::endl;
        return;
    }

    const int size = width * height * sizeof(float);
    float *frame_buf = new float[width * height];
    float *wav_buf = new float[height];

    if (frame_buf == nullptr || wav_buf == nullptr) {
        std::cerr << "Failed to allocate memory for wavelet buffers" << std::endl;
        delete[] frame_buf;
        delete[] wav_buf;
        return;
    }

    for (int level = 0; level < levels; level++) {
        int step = width >> level;

        // Apply the Daubechies wavelet transformation to rows
        for (int i = 0; i < height; i++) {
            cpu_d4_transform(&img[i * width], &frame_buf[i * width], step);
        }
        // Apply the Daubechies wavelet transformation to columns
        for (int j = 0; j < step; j++) {
            for (int i = 0; i < height; i++) {
            wav_buf[i] = frame_buf[i * width + j];
            }
            cpu_d4_transform(wav_buf, frame_buf, step);
            for (int i = 0; i < height; i++) {
            img[i * width + j] = frame_buf[i];
            }
        }
    }

    delete[] frame_buf;
    delete[] wav_buf;
}

void resize_to_power_of_two(unsigned char* src, unsigned char* dst, int src_width, int src_height, int dst_width, int dst_height, int channels)
{
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < dst_height; i++) {
            for (int j = 0; j < dst_width; j++) {
                int src_i = i * src_height / dst_height;
                int src_j = j * src_width / dst_width;
                dst[(i * dst_width + j) * channels + c] = src[(src_i * src_width + src_j) * channels + c];
            }
        }
    }
}

void process_image(unsigned char* img, unsigned char* haar_output_img, unsigned char* daubechies_output_img, int width, int height, int channels, const std::string& mode, int haar_levels, int daubechies_levels)
{
    float* channel_img = new float[width * height];
    float* daubechies_channel_img = new float[width * height];
    if (channel_img == nullptr || daubechies_channel_img == nullptr) {
        std::cerr << "Failed to allocate memory for channel_img" << std::endl;
        delete[] channel_img;
        delete[] daubechies_channel_img;
        return;
    }

    for (int c = 0; c < channels; c++) {
        // Extract the channel data
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                channel_img[i * width + j] = static_cast<float>(img[(i * width + j) * channels + c]);
                daubechies_channel_img[i * width + j] = static_cast<float>(img[(i * width + j) * channels + c]);
            }
        }

        // Apply the wavelet transformations
        if (mode == "gpu") {
            std::cout << "Running GPU wavelet transformations" << std::endl;


            run_daubechies4_wavelet_gpu(daubechies_channel_img, width, height, daubechies_levels);

            cudaError_t cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                std::cerr << "CUDA Daubechies wavelet failed: " << cudaGetErrorString(cuda_status) << std::endl;
            }

            run_haar_wavelet_gpu(channel_img, width, height, haar_levels);

            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                std::cerr << "CUDA HAAR wavelet failed: " << cudaGetErrorString(cuda_status) << std::endl;
            }

        } else if (mode == "cpu") {
            std::cout << "Running CPU wavelet transformations" << std::endl;
            run_haar_wavelet_cpu(channel_img, width, height, haar_levels);
            run_daubechies4_wavelet_cpu(daubechies_channel_img, width, height, daubechies_levels);
        }

        // Normalize the coefficients for visualization
        float max_val_haar = 0.0f;
        float max_val_daubechies = 0.0;
        for (int i = 0; i < width * height; i++) {
            if (fabs(channel_img[i]) > max_val_haar) {
                max_val_haar = fabs(channel_img[i]);
            }
            if (fabs(daubechies_channel_img[i]) > max_val_daubechies) {
                max_val_daubechies = fabs(daubechies_channel_img[i]);
            }
        }
        for (int i = 0; i < width * height; i++) {
            channel_img[i] = (channel_img[i] / max_val_haar) * 255.0f;
            daubechies_channel_img[i] = (daubechies_channel_img[i] / max_val_daubechies) * 255.0;
        }

        // Store the results back into the output images
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                haar_output_img[(i * width + j) * channels + c] = static_cast<unsigned char>(fabs(channel_img[i * width + j]));
                daubechies_output_img[(i * width + j) * channels + c] = static_cast<unsigned char>(fabs(daubechies_channel_img[i * width + j]));
            }
        }
    }

    delete[] channel_img;
    delete[] daubechies_channel_img;
}

int main(int argc, char** argv) {
    // the argument is the image file, mode (cpu or gpu), haar levels, and daubechies levels
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <image file> <mode (cpu|gpu)> <haar levels> <daubechies levels>" << std::endl;
        return 1;
    }

    std::string picture_file = argv[1];
    if (!std::filesystem::exists(picture_file)) {
        std::cerr << "Error: File " << picture_file << " does not exist." << std::endl;
        return 1;
    }

    // Extract the base name of the image file
    std::string base_name = std::filesystem::path(picture_file).stem().string();

    // load the image
    int height, width, channels;
    unsigned char* img = stbi_load(picture_file.c_str(), &width, &height, &channels, 0);
    if (img == NULL) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return -1;
    }


    std::string mode = argv[2];
    int haar_levels = std::stoi(argv[3]);
    int daubechies_levels = std::stoi(argv[4]);

    // Resize image to the nearest power of two dimensions
    int new_width = 1 << static_cast<int>(ceil(log2(width)));
    int new_height = 1 << static_cast<int>(ceil(log2(height)));
    unsigned char* resized_img = new unsigned char[new_width * new_height * channels];
    resize_to_power_of_two(img, resized_img, width, height, new_width, new_height, channels);

    std::cout << "Loaded image: " << std::endl;
    std::cout << "Image Width: " << new_width << std::endl;
    std::cout << "Image Height: " << new_height << std::endl;
    std::cout << "Image Channels: " << channels << std::endl;
    std::cout << "Mode: " << mode << std::endl;
    std::cout << "Haar Levels: " << haar_levels << std::endl;
    std::cout << "Daubechies Levels: " << daubechies_levels << std::endl;

    unsigned char* haar_output_img = new unsigned char[new_width * new_height * channels];
    unsigned char* daubechies_output_img = new unsigned char[new_width * new_height * channels];
    if (haar_output_img == nullptr || daubechies_output_img == nullptr) {
        std::cerr << "Failed to allocate memory for output images" << std::endl;
        stbi_image_free(img);
        delete[] resized_img;
        delete[] haar_output_img;
        delete[] daubechies_output_img;
        return -1;
    }

    process_image(resized_img, haar_output_img, daubechies_output_img, new_width, new_height, channels, mode, haar_levels, daubechies_levels);

    std::string haar_output_filename = base_name + "_haar_output.png";
    std::string daubechies_output_filename = base_name + "_daubechies_output.png";

    stbi_write_png(haar_output_filename.c_str(), new_width, new_height, channels, haar_output_img, new_width * channels);
    stbi_write_png(daubechies_output_filename.c_str(), new_width, new_height, channels, daubechies_output_img, new_width * channels);

    delete[] resized_img;
    delete[] haar_output_img;
    delete[] daubechies_output_img;
    stbi_image_free(img);

    return 0;
}