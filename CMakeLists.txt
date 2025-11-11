# Athena Project

cmake_minimum_required(VERSION 3.18)
project(athena LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 23)


set(TARGET_DEVICE "CPU" CACHE STRING "Target device for build (CPU or GPU)")
set_property(CACHE TARGET_DEVICE PROPERTY STRINGS CPU GPU)
message(STATUS "Configured target device: ${TARGET_DEVICE}")

if(TARGET_DEVICE STREQUAL "GPU")
    add_compile_definitions(GPU)
elseif(TARGET_DEVICE STREQUAL "CPU")
    add_compile_definitions(CPU)
else()
    message(FATAL_ERROR "Unknown TARGET_DEVICE: ${TARGET_DEVICE}. Use CPU or GPU (set -DTARGET_DEVICE=CPU or -DTARGET_DEVICE=GPU)")
endif()

option(BUILD_TESTS "Build tests" ON)

add_subdirectory(app)
add_subdirectory(ext)
add_subdirectory(sys)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
