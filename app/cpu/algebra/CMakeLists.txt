add_library(algebra_lib STATIC
    Coordinates.cpp
    Coordinates.hpp
    Point.cpp
    Point.hpp
)

target_include_directories(algebra_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(algebra_lib
    PRIVATE
)

add_library(HYS::ALGEBRA ALIAS algebra_lib)
