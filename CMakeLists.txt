cmake_minimum_required(VERSION 3.15)

# set the project name
project(athena)

# enable C and C++ languages
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_C_STANDARD 11)

# enable CUDA language
#enable_language(CUDA)

# find the CUDA package
#find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# set the CUDA architecture to match your GPU
#set(CUDA_ARCH "-arch=sm_61")
#set(CMAKE_CUDA_FLAGS "${CUDA_ARCH}")

# set the include directories
#include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(inc)
set(SOURCES
    src/Socket.cpp
    main.cpp)

# add the executable
add_executable(athena main.cpp)

# link against CUDA libraries and OpenGL libraries
#target_link_libraries(athena ${CUDA_LIBRARIES} GL glut)
target_link_libraries(athena ${OpenCV_LIBS})