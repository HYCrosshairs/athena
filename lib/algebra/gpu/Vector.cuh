#pragma once

#include <std/cuda/span>

namespace hyc::gpu
{
template <typename T>
__global__ void addVectors(cuda::span<T> result, cuda::span<const T> a, cuda::span<const T> b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < result.size())
    {
        result[idx] = a[idx] + b[idx];
    }
}
};  // namespace hyc::gpu