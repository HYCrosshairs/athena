#pragma once

#include <type_traits>

namespace hw::lib::manager
{
enum class ProcessingUnit : int
{
    cpu,
    gpu
};

enum class Arch : int
{
    x86,
    arm
};

template<ProcessingUnit>
struct checkBuild;

template <>
struct checkBuild<ProcessingUnit::cpu>
{
    static void train()
    {
        // GPU is not availabe, use CPU by default
    }
};

template <>
struct checkBuild<ProcessingUnit::gpu>
{
    static void train()
    {
        // GPU is available, use GPU for better performances
    }
};

  
} // namespace hw::lib::
