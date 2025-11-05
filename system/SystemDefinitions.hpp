#pragma once

namespace lib::system
{
enum class Target
{
    cpu,
    gpu
};

struct SwBuildSettings
{
#if defined(CPU)
    static constexpr Target target = SocTarget::cpu;
#elif defined(GPU)
    static constexpr Target target = SocTarget::gpu;
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
} // namespace lib::system