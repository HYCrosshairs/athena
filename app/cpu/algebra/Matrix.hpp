#pragma once

#include <stdint.h>
#include <iostream>
#include <format>
#include <cassert>


template<typename T, uint32_t rows, uint32_t cols>
class Matrix
{
public:
    void print()
    {
        for (uint32_t rowIndex  = 0; rowIndex < rows; rowIndex++)
        {
            for (uint32_t colIndex = 0; colIndex < cols; colIndex++)
            {
                std::cout << std::format("{:<8}", matrix[rowIndex][colIndex]);
            }
            std::cout << std::endl;
        }
    }

    void setValue(uint32_t row, uint32_t col, T value)
    {
        matrix[row][col] = value;
    }

    T getValue(uint32_t row, uint32_t col) const
    {
        return matrix[row][col];
    }

    Matrix operator+(const Matrix& other) const
    {
        Matrix result;
        for (uint32_t rowIndex = 0; rowIndex < rows; rowIndex++)
        {
            for (uint32_t colIndex = 0; colIndex < cols; colIndex++)
            {
                result.setValue(rowIndex, colIndex,
                                this->getValue(rowIndex, colIndex) +
                                other.getValue(rowIndex, colIndex));
            }
        }
        return result;
    }

    Matrix& operator=(const Matrix& other)
    {
        if (this != &other)
        {
            for (uint32_t rowIndex = 0; rowIndex < rows; rowIndex++)
            {
                for (uint32_t colIndex = 0; colIndex < cols; colIndex++)
                {
                    this->setValue(rowIndex, colIndex,
                                   other.getValue(rowIndex, colIndex));
                }
            }
        }
        return *this;
    }

    Matrix operator*(const Matrix& other) const
    {
        static_assert(cols == rows, "Matrix size doesn't match");

        Matrix result;
        for (uint32_t rowIndex = 0; rowIndex < rows; rowIndex++)
        {
            for (uint32_t colIndex = 0; colIndex < cols; colIndex++)
            {
                T sum = 0;
                for (uint32_t elementIndex = 0; elementIndex < cols; elementIndex++)
                {
                    sum += this->getValue(rowIndex, elementIndex) * other.getValue(elementIndex, colIndex);
                }
                result.setValue(rowIndex, colIndex, sum);
            }
        }
        return result;
    }

private:
    T matrix[rows][cols]{};
};
