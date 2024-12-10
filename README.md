# wavelet

This program performs Haar and Daubechies wavelet transformations on an input image using either CPU or GPU.

## Prerequisites

Before running the program, ensure you have the following:

- **CUDA Toolkit**: Required for GPU processing.
- **C++ Compiler**: Compatible with the project.
- **CMake**: Optional, for building the project if needed.
- **stb_image and stb_image_write libraries**: For handling image input/output.

## Building the Project

### Using CMake
1. Create a build directory:
   ```
   mkdir build
   cd build
   ```
2. Run CMake to generate build files:
   ```
   cmake ..
   ```
3. Build the project:
   ```
   make
   ```

### Direct Compilation
If you prefer not to use CMake, you can directly compile the source files with your C++ compiler.

## Running the Program

Run the program using the following format:

```
./wavelet <image file> <kernel size> <mode> <haar levels> <daubechies levels>
```

### Arguments
1. `<image file>`: Path to the input image (e.g., `image.jpg`).
2. `<mode>`: Specify execution mode:
   - `cpu`: Run the program using CPU.
   - `gpu`: Run the program using GPU.
3. `<haar levels>`: Number of levels for Haar wavelet transformation.
4. `<daubechies levels>`: Number of levels for Daubechies wavelet transformation.

### Example Usage
To process `cat.jpg` using GPU mode, and applying 3 levels for both Haar and Daubechies transformations, use:

```
./wavelet ../img/cat.jpg gpu 3 3
```


## Output

The program generates the following output images:
- `<base_name>_haar_output.png`: Image after Haar wavelet transformation.
- `<base_name>_daubechies_output.png`: Image after Daubechies wavelet transformation.

The `<base_name>` corresponds to the original file name without its extension.

## Notes

- Ensure that the input image is in a supported format.
- Ensure that the imput image size to be square.
- Levels for Haar and Daubechies should be less than or equal to log2(width)

