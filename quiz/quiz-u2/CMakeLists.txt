cmake_minimum_required(VERSION 3.24)
project(quiz_u2 CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(quiz_u2 main.cu)

set_target_properties(quiz_u2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)