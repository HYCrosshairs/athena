cmake_minimum_required(VERSION 3.10)

project(athena LANGUAGES CUDA CXX)

# Set the output directory for compiled files
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find and include CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find and include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the CUDA and C++ source files
file(GLOB CUDA_SOURCES "src/cuda/*.cu")
file(GLOB CPP_SOURCES "src/cpp/*.cpp")

include_directories(inc/cuda)
include_directories(inc/cpp)

# Add the CUDA and C++ header files
file(GLOB CUDA_HEADERS "inc/cuda/*.cuh")
file(GLOB CPP_HEADERS "inc/cpp/*.hpp")

# Compile the CUDA code into a library
#cuda_add_library(MyCudaLib ${CUDA_SOURCES} ${CUDA_HEADERS})
#target_link_libraries(MyCudaLib ${CUDA_LIBRARIES})

# Compile the C++ code into an executable
add_executable(athena ${CPP_SOURCES} ${CPP_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS})
target_link_libraries(athena ${OpenCV_LIBS})
