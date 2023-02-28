cmake_minimum_required(VERSION 3.24)
project(test CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(test main.cu)

set_target_properties(test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
