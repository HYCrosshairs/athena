# Athena App CMake

add_subdirectory(common)
add_subdirectory(cpu)
add_subdirectory(gpu)

add_executable(main main.cpp
        ../test/math/TestCoordinatesLib.cpp
        cpu/algebra/Matrix.hpp)
target_link_libraries(main PRIVATE common cpu gpu)

