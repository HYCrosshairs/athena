cmake_minimum_required(VERSION 3.10)

project(athena LANGUAGES CUDA CXX)

# Set the output directory for compiled files
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find and include CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -rdc=true")
include_directories(${CUDA_INCLUDE_DIRS})

# Find and include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(inc/cuda)
include_directories(inc/cpp)

# Add the CUDA and C++ source files
file(GLOB_RECURSE CUDA_SOURCES "src/cuda/*.cu")
file(GLOB_RECURSE CPP_SOURCES "src/cpp/*.cpp")

# Add the CUDA and C++ header files
file(GLOB_RECURSE CUDA_HEADERS "inc/cuda/*.cuh")
file(GLOB_RECURSE CPP_HEADERS "inc/cpp/*.hpp")

# Compile the C++ code into an executable
add_executable(athena ${CPP_SOURCES} ${CPP_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS})
target_link_libraries(athena ${OpenCV_LIBS} ${CUDA_LIBRARIES} boost_iostreams boost_system)
