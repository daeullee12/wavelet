#Specify the version
cmake_minimum_required(VERSION 3.29)

#Name your project here
project(wavelet LANGUAGES CXX CUDA)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_TOOLCHAIN_FILE "/home/dlee50/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")
#set(CMAKE_cuda_COMPILER nvcc)
#set(CUDA_TOOLKIT_ROOT_DIR "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6")

# Specify the CUDA toolkit root directory
set(CUDAToolkit_ROOT "/usr/local/cuda-12.3")

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
link_directories(${CUDAToolkit_LIBRARY_DIRS})

set(Boost_DIR "/home/dlee50/vcpkg/installed/x64-linux/share/boost")
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)

#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
else()
	find_package(X11 REQUIRED)
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif ( MSVC )

#set the include directories
include_directories(
			${CMAKE_CURRENT_BINARY_DIR}
			${CMAKE_CURRENT_SOURCE_DIR}
			${TIRA_INCLUDE_DIRS}
			${X11_INCLUDE_DIR}
			${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
			${CUDA_INCLUDE_DIRS}
			${CMAKE_SOURCE_DIR}
			
)



#create an executable
add_executable(wavelet
    src/daubechies4.cu
    src/haar.cu
    src/main.cpp
)


set_target_properties(wavelet PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(wavelet PRIVATE cxx_std_20 cuda_std_20)
set_source_files_properties(daubechies4.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(haar.cu PROPERTIES LANGUAGE CUDA)

target_link_libraries(wavelet
			${X11_LIBRARIES}
			Boost::program_options
			${CUDAToolkit_LIBRARIES}
			CUDA::cudart
			CUDA::cuda_driver
)
