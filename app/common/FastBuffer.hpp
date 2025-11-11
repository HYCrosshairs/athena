#pragma once

//#include <cstddef>

namespace hyc::memory
{
template <typename T, size_t Size>
struct FastBuffer
{
    alignas(alignof(T)) unsigned char buffer[sizeof(T) * Size];

    T* data()
    {
        return reinterpret_cast<T*>(buffer);
    }

    const T* data() const
    {
        return reinterpret_cast<const T*>(buffer);
    }

    T& operator[](size_t index)
    {
        return data()[index];
    }

    const T& operator[](size_t index) const
    {
        return data()[index];
    }
};

} // namespace hyc::memory