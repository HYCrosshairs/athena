add_library(common_lib INTERFACE)

if(TARGET_DEVICE STREQUAL "GPU")
  add_library(common_gpu_lib INTERFACE)
  target_include_directories(common_gpu_lib
                             INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/gpu)
endif()

target_link_libraries(common_lib INTERFACE common_gpu_lib)

add_library(HY::COMMON ALIAS common_lib)
