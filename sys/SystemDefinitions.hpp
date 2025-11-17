#pragma once

namespace sys
{
enum class Target
{
    cpu,
    gpu
};

struct CPU {};
struct GPU {};

struct SwBuildSettings
{
#if defined(DEVICE_CPU)
    static constexpr Target target = Target::cpu;
#elif defined(DEVICE_GPU)
    static constexpr Target target = Target::gpu;
#endif
};

template <Target target>
bool isBuildFor();

template<>
inline bool isBuildFor<Target::cpu>()
{
    return SwBuildSettings::target == Target::cpu;
}

template<>
inline bool isBuildFor<Target::gpu>()
{
    return SwBuildSettings::target == Target::gpu;
}
} // namespace sys