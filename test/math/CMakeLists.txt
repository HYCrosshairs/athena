add_executable(TEST_MathLib
    TestCoordinatesLib.cpp
)

target_link_libraries(TEST_MathLib
    PRIVATE
        CppUTest::CppUTest
        CppUTest::CppUTestExt
)

add_executable(TEST_AlgebraMatrix
    TestAlgebraMatrix.cpp
)

target_link_libraries(TEST_AlgebraMatrix
    PRIVATE
        CppUTest::CppUTest
        CppUTest::CppUTestExt
        HYS::ALGEBRA
)
