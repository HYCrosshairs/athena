add_library(common utils.cpp
        FastBuffer.hpp)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

