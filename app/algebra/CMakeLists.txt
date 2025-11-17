add_library(algebra_lib
    Matrix.cu
    Coordinates.hpp
    Matrix.hpp
    Point.hpp
)

set_target_properties(algebra_lib PROPERTIES LINKER_LANGUAGE CUDA)

target_include_directories(algebra_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(algebra_lib
    PUBLIC
        HY::SystemDefinitions
)

add_library(HYS::ALGEBRA ALIAS algebra_lib)
