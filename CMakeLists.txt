cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
set(CUDACXX "/opt/cuda/bin/nvcc")

project(smolLLM LANGUAGES CXX CUDA)

# ==========================
# Language standards
# ==========================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ==========================
# Target architecture
# ==========================
set(CMAKE_CUDA_ARCHITECTURES 86) # hardcoded cc 8.6 for rtx 2050

# ==========================
# Build configuration
# ==========================
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")

set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -DDEBUG")

find_package(CUDAToolkit REQUIRED)

add_executable(smolLLM
    src/main.cpp
    src/kernels.cu
)

# ==========================
# Includes and Source directories
# ==========================
target_include_directories(smolLLM PRIVATE src)
target_include_directories(smolLLM PRIVATE include)


target_link_libraries(smolLLM PRIVATE
CUDA::cudart # this is the main lib cuda-run time
CUDA::cublas # provide BLAS functions (basic linear algebra sunprograms)
# using this currently then will switch to Lt
# CUDA::cublasLt # provide BLAS functions (basic linear algebra sunprograms) optimized for LLMs
# we still include the normal one to avoid type def issues(not sure about that might have to try w/o it to check)
)
