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
  
} // namespace hw::lib::
