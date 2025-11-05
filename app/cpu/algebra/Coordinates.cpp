#include "Coordinates.hpp"

#include <cmath>

Point Coordinates::sphereToCartesian(float azimuth, float elevation, float r) const
{
    const float x = r * std::sin(elevation) * std::cos(azimuth);
    const float y = r * std::sin(elevation) * std::sin(azimuth);
    const float z = r * std::cos(elevation);

    return Point(x, y, z);
}