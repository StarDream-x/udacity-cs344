############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(cs344)

#error fix
set(CMAKE_CXX_FLAGS_DEBUG "/MDd")

set(OpenCV_DIR D:/iniBuild/opencv/bd/opencv/build)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

link_libraries(${OpenCV_LIBS} )

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin/")

if(CUDA_FOUND)
  # compared to class settings, we let NVidia's FindCUDA CMake detect 
  # whether to build x64.  We tell it to support most devices, though, 
  # to make sure more people can easily run class code without knowing 
  # about this compiler argument
#  set(CUDA_NVCC_FLAGS "-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75;")
  set(CUDA_GEN_CODE "-gencode=arch=compute_86,code=sm_86")    # 根据自己的显卡设置compute_**，例如3060为compute_86

  #  set(CMAKE_CUDA_STANDARD 17)

  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

  add_subdirectory (HW1)
  add_subdirectory (HW2)
  add_subdirectory (HW3)
  add_subdirectory (HW4)
#  add_subdirectory (HW5)
#  add_subdirectory (HW6)
else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()
