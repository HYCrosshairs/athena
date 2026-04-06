#include "LoadVsDirectCompute.hpp"

namespace gpu
{
__global__ void directTangentCompute(const float* angles, float* tangents, std::size_t count,
                                     unsigned workRepeats)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    float a   = angles[idx];
    float acc = 0.f;
    for (unsigned r = 0; r < workRepeats; ++r)
    {
        float s = sinf(a);
        float c = cosf(a);
        acc += s / c;
    }
    tangents[idx] = acc;
}

__global__ void computeTangentFromLUT(const float* sin_lut, const float* cos_lut, float* tangents,
                                      std::size_t count, unsigned workRepeats)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    float acc = 0.f;
    for (unsigned r = 0; r < workRepeats; ++r)
    {
        std::size_t li = static_cast<std::size_t>(r) * count + idx;
        acc += sin_lut[li] / cos_lut[li];
    }
    tangents[idx] = acc;
}
}  // namespace gpu
