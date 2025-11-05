#include "CppUTest/CommandLineTestRunner.h"

TEST_GROUP(TestCoordinatesLib)
{
};

TEST(TestCoordinatesLib, test_example)
{
    CHECK(true);
}

int main(int argc, char** argv)
{
    return CommandLineTestRunner::RunAllTests(argc, argv);
}