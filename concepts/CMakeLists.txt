add_library(gpu_concepts_lib STATIC LoadVsDirectCompute.cu)

target_include_directories(gpu_concepts_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(TARGET CUDA::cudart)
  target_link_libraries(gpu_concepts_lib PUBLIC CUDA::cudart)
endif()

add_library(HY::CONCEPTS ALIAS gpu_concepts_lib)
