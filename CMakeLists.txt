cmake_minimum_required(VERSION 3.20)
project(grand-pattern-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70)

find_package(CUDAToolkit REQUIRED)

add_executable(test_cuda_kernels
    src/main.cu
    src/kernels.cu
    src/tests.cu
)

target_include_directories(test_cuda_kernels PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_cuda_kernels CUDA::cudart)

set_target_properties(test_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
