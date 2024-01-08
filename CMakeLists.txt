cmake_minimum_required(VERSION 3.18)

project(histogram_equalization LANGUAGES CUDA CXX)


# Set executable name
add_executable(histogram_equalization src/main.cu)

# Set include directories
target_include_directories(histogram_equalization PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Set compiler options for CUDA
target_compile_options(histogram_equalization PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Set compiler features
target_compile_features(histogram_equalization PUBLIC cxx_std_11)

# Set NVCC flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")