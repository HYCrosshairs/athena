cmake_minimum_required(VERSION 3.15)

# set the project name
project(athena CUDA)

# enable C and C++ languages
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)


# find the CUDA package
find_package(CUDA REQUIRED)

# set the CUDA architecture to match your GPU
set(CUDA_ARCH "-arch=sm_61")
set(CMAKE_CUDA_FLAGS "${CUDA_ARCH}")

# set the include directories
include_directories(${CUDA_INCLUDE_DIRS})

# add the executable
cuda_add_executable(athena main.cu)

# link against CUDA libraries and OpenGL libraries
target_link_libraries(athena ${CUDA_LIBRARIES} GL glut)