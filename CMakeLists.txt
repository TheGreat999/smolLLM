cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
set(CUDACXX "/opt/cuda/bin/nvcc")

project(smolVLLM LANGUAGES CXX CUDA)

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

add_executable(smolVLLM
    src/main.cpp
    src/kernels.cu
)

# ==========================
# Includes and Source directories
# ==========================
target_include_directories(smolVLLM PRIVATE src)
target_include_directories(smolVLLM PRIVATE include)


target_link_libraries(smolVLLM PRIVATE
CUDA::cublas
CUDA::cudart
)
