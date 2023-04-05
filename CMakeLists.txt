cmake_minimum_required(VERSION 3.15)

project(athena VERSION 1.0.0 LANGUAGES CXX)
include(GNUInstallDirs)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_compile_options(-Wall -Wextra -Wconversion -pedantic -Wformat-security)# -Wfatal-errors)

include_directories(inc)
set(SOURCES
    src/Socket.cpp
    main.cu)

# find the packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# set the CUDA architecture to match your GPU
if (CUDA_FOUND)
    set(CUDA_ARCH "-arch=sm_61")
    set(CMAKE_CUDA_FLAGS "${CUDA_ARCH}")
    set(CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS};-O3;-std=c++11)
endif()

# set vectorization library based on processor type
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(VECTORIZE_LIB "intel_vector")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(VECTORIZE_LIB "gomp")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(VECTORIZE_LIB "openmp")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# add the executable
add_executable(${PROJECT_NAME} ${SOURCES} ${RESOURCES})

# set CUDA properties for target executable
if (CUDA_FOUND)
    set_target_properties(athena PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# link against CUDA and vectorization libraries
if (CUDA_FOUND)
    target_link_libraries(athena ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY})
endif()

if (VECTORIZE_LIB)
    target_link_libraries(athena ${VECTORIZE_LIB})  
endif()

if(OpenCV_LIBS)
    target_link_libraries(athena ${OpenCV_LIBS})
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

