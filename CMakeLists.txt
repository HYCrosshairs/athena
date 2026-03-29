cmake_minimum_required(VERSION 3.28)

cmake_policy(SET CMP0102 NEW)
cmake_policy(SET CMP0126 NEW)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.29)
  cmake_policy(SET CMP0159 NEW)
endif()

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

list(
  APPEND
  CMAKE_MODULE_PATH
  "${PROJECT_ROOT_DIR}/cmake"
  "${PROJECT_ROOT_DIR}/cmake/modules"
  "${PROJECT_ROOT_DIR}/cmake/modules/CudaHelpers"
  "${PROJECT_ROOT_DIR}/cmake/modules/CudaOptions")

include(SetDefaultBuildType)

project(
  athena
  VERSION 1.0
  DESCRIPTION
    "Athena: A High-Performance Simulation Framework for Computational Science"
  LANGUAGES C CXX CUDA)

option(BUILD_TESTS "Build test suite" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
option(BUILD_SRC_ABSOLUTE_PATHS
       "Build source files with absolute paths for better debugging" OFF)
option(STRICT_WARNINGS
       "Enable strict compiler warnings and treat them as error" ON)

if(TARGET_DEVICE STREQUAL "GPU")
  option(ENABLE_CUDA_DEBUG "Enable CUDA device debugging symbols" ON)
  option(ENABLE_CUDA_LINEINFO "Enable line information for profiling" ON)
  # option(ENABLE_CUDA_NVTX "Enable NVTX annotations for profiling" ON)
  option(ENABLE_CUDA_PROFILING "Enable CUDA profiling with NVTX" OFF)
  option(BUILD_CUDA_AUTO_DETECT
         "Auto-detect CUDA architecture and build for it" ON)
  option(BUILD_CUDA_MULTI_ARCH "Build for multiple GPU architecture" OFF)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(TARGET_DEVICE STREQUAL "GPU")
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

if(TARGET_DEVICE STREQUAL "GPU")

  if(BUILD_CUDA_AUTO_DETECT)
    include(CudaHelpers/DetectCudaArchitecture)
    detect_cuda_gpu()
    if(DETECTED_CUDA_ARCH)
      message(STATUS "Detected CUDA architecture: ${DETECTED_CUDA_ARCH}")
      set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_CUDA_ARCH})
    else()
      set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
      message(
        WARNING
          "Failed to detect CUDA architecture. Building with default architectures."
      )
    endif()
  elseif(BUILD_CUDA_MULTI_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    message(
      STATUS
        "Building for multiple CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    message(
      STATUS
        "Building for default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  endif()

  if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(NVCC_EXECUTABLE nvcc)
    if(NVCC_EXECUTABLE)
      set(CMAKE_CUDA_COMPILER ${NVCC_EXECUTABLE})
      message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    else()
      message(WARNING "No CUDA compiler found. CUDA support will be disabled.")
    endif()
  else()
    message(STATUS "Using specified CUDA compiler: ${CMAKE_CUDA_COMPILER}")
  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

  if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0")
    message(
      STATUS
        "CUDA debugging enabled (warning: this will significantly reduce performance)"
    )
  else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0")
    message(STATUS "CUDA debugging disabled (optimized for performance)")
  endif()

  if(ENABLE_CUDA_LINEINFO)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    message(STATUS "CUDA line information enabled for profiling")
  else()
    message(STATUS "CUDA line information disabled")
  endif()

  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO
      "${CMAKE_CUDA_FLAGS} -O3 -g -G -lineinfo --use_fast_math")

  if(STRICT_WARNINGS)
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror=unused-variable>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror=unused-but-set-variable>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror=unused-parameter>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror=unused-function>)
    message(STATUS "Strict warnings enabled for CUDA compilation")
  endif()

  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
                      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  set(CMAKE_CUDA_RUNTIME_LIBRARY
      "Shared"
      CACHE STRING "Use shared CUDA runtime library")
  set_property(CACHE CMAKE_CUDA_RUNTIME_LIBRARY PROPERTY STRINGS "Shared"
                                                         "Static" "Auto")

  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  set_property(GLOBAL PROPERTY JOB_POOLS max_cuda_jobs=4
                               max_cuda_linking_jobs=2 max_heavy_linking_jobs=4)

  find_package(CUDAToolkit REQUIRED)
  if(CUDAToolkit_FOUND)
    message(
      STATUS
        "CUDA toolkit found: ${CUDAToolkit_VERSION} found at ${CUDAToolkit_BIN_DIR}"
    )
  endif()

  if(ENABLE_CUDA_PROFILING)
    add_compile_definitions(ENABLE_NVTX USE_NVTX)
    message(STATUS "CUDA profiling enabled with NVTX annotations")
  else()
    message(STATUS "CUDA profiling disabled")
  endif()
  add_compile_definitions(DEVICE_GPU)
endif()

if(TARGET_DEVICE STREQUAL "CPU")
  add_compile_definitions(DEVICE_CPU)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(NOT DEBUG_SRC_ABSOLUTE_PATH AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(TARGET DEVICE STREQUAL "GPU")
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffile-prefix-map=${PROJECT_ROOT_DIR}=.>
      $<$<COMPILE_LANGUAGE:CXX>:-ffile-prefix-map=${PROJECT_ROOT_DIR}=.>)
  else()
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX>:-ffile-prefix-map=${PROJECT_ROOT_DIR}=.>)
  endif()
  message(STATUS "Reproductible builds enabled (normalezed source paths)")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=auto>)
  add_link_options(-fdiagnostics-color=auto)

  if(STRICT_WARNINGS)
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
      $<$<COMPILE_LANGUAGE:CXX>:-Werror=unused-variable>
      $<$<COMPILE_LANGUAGE:CXX>:-Werror=unused-but-set-variable>
      $<$<COMPILE_LANGUAGE:CXX>:-Werror=unused-parameter>
      $<$<COMPILE_LANGUAGE:CXX>:-Werror=unused-function>)
    message(STATUS "Strict warnings enabled for C++ compilation")
  endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(BUILD_TESTS)
  include(CTest)
  include(CTestHelper)
  if(TARGET_DEVICE STREQUAL "GPU")
    include(CudaTestRunner)
  endif()
  enable_testing()
  message(STATUS "Testing enabled")
endif()

add_subdirectory(app)
add_subdirectory(sys)
add_subdirectory(ext)
add_subdirectory(concepts)
add_subdirectory(lib)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Target Device: ${TARGET_DEVICE}")
if(TARGET_DEVICE STREQUAL "GPU")
  message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
  message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "  CUDA Debugging: ${ENABLE_CUDA_DEBUG}")
  message(STATUS "  CUDA Line Info: ${ENABLE_CUDA_LINEINFO}")
  message(STATUS "  CUDA Profiling: ${ENABLE_CUDA_PROFILING}")
endif()
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Strict Warnings: ${STRICT_WARNINGS}")
message(STATUS "  Source Absolute Paths: ${BUILD_SRC_ABSOLUTE_PATHS}")
message(STATUS "")
