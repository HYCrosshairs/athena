#pragma once

namespace sys
{
enum class Target
{
    cpu,
    gpu
};

struct SwBuildSettings
{
#if defined(CPU)
    static constexpr Target target = Target::cpu;
#elif defined(GPU)
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