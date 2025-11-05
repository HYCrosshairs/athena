#include <CppUTest/CommandLineTestRunner.h>

#include "Matrix.hpp"

TEST_GROUP(TestMatrixLib)
{
    Matrix<int, 2, 2> matrixA;
    Matrix<int, 2, 2> matrixB;

    Matrix<int, 2, 2> matrixC;

    Matrix<int , 2, 2> matrixD;

    void initTestCase()
    {
        matrixA.setValue(0, 0, -1);
        matrixA.setValue(1, 0, 2);
        matrixA.setValue(0, 1, 3);
        matrixA.setValue(1, 1, 0);

        matrixB.setValue(0, 0, 2);
        matrixB.setValue(1, 0, 3);
        matrixB.setValue(0, 1, -1);
        matrixB.setValue(1, 1, 2);
    }

    void runTestCase()
    {
        matrixA.print();
        matrixB.print();

        matrixC = matrixA + matrixB;

        matrixD = matrixA * matrixB;

        matrixC.print();
        matrixD.print();
    }

    void checkTestCase()
    {
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                CHECK_EQUAL(matrixC.getValue(i, j), matrixA.getValue(i, j) + matrixB.getValue(i, j) );
            }
        }
    }
};

TEST(TestMatrixLib, test_example)
{
    initTestCase();
    runTestCase();
    checkTestCase();
}

int main(int argc, char** argv)
{
    return CommandLineTestRunner::RunAllTests(argc, argv);
}