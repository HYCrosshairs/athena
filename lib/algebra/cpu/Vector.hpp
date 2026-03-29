#pragma once

#include <span>

namespace hyc
{
template <typename T> void addVectors(std::span<T> a, std::span<T> b, std::span<T> c)
{
    for (size_t i = 0; i < a.size(); ++i)
    {
        c[i] = a[i] + b[i];
    }
}
}  // namespace hyc