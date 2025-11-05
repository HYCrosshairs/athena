# Athena GPU CMake

#find_package(CUDA REQUIRED)
add_library(gpu hello.cu)
target_link_libraries(gpu PUBLIC common)
target_include_directories(gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../common)
set_target_properties(gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

