#pragma once

#include <cstddef>

namespace gpu
{
__global__ void directTangentCompute(const float* angles, float* tangents, std::size_t count,
                                     unsigned workRepeats);

__global__ void computeTangentFromLUT(const float* sin_lut, const float* cos_lut, float* tangents,
                                      std::size_t count, unsigned workRepeats);
}  // namespace gpu
