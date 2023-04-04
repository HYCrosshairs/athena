cmake_minimum_required(VERSION 3.15)

project(athena VERSION 1.0.0 LANGUAGES CXX)
include(GNUInstallDirs)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_compile_options(-Wall -Wextra -Wconversion -pedantic -Wformat-security)# -Wfatal-errors)

# find the packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# set the CUDA architecture to match your GPU
set(CUDA_ARCH "-arch=sm_61")
set(CMAKE_CUDA_FLAGS "${CUDA_ARCH}")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

include_directories(inc)
set(SOURCES
    src/Socket.cpp
    main.cpp)

# add the executable
add_executable(${PROJECT_NAME} ${SOURCES} ${RESOURCES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_link_libraries(athena ${OpenCV_LIBS} ${CUDA_LIBRARIES} GL)