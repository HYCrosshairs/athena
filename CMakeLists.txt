# Athena Project

cmake_minimum_required(VERSION 3.18)
project(athena LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include cmake helper scripts
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Explicitly set CUDA compiler to NVCC if not already set
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(NVCC_EXECUTABLE nvcc)
    if(NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER "${NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
        message(STATUS "Set CUDA compiler to NVCC: ${CMAKE_CUDA_COMPILER}")
    else()
        message(FATAL_ERROR "NVCC compiler not found. Please ensure CUDA Toolkit is installed and nvcc is in your PATH.")
    endif()
else()
    message(STATUS "Using CUDA compiler: ${CMAKE_CUDA_COMPILER}")
endif()


# Device selection (CPU/GPU)
option(TARGET_DEVICE "Target device for build (CPU/GPU)" "CPU")
if(TARGET_DEVICE STREQUAL "GPU")
    add_compile_definitions(DEVICE_GPU)
    find_package(CUDAToolkit REQUIRED)
else()
    add_compile_definitions(DEVICE_CPU)
endif()

# Toolchain file usage (optional)
# set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/cmake/toolchain.cmake" CACHE FILEPATH "Toolchain file")

option(BUILD_TESTS "Build tests" ON)

add_subdirectory(app)
add_subdirectory(ext)
add_subdirectory(sys)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
