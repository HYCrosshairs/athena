#pragma once
#include "Point.hpp"

struct Coordinates
{
    Point sphereToCartesian(float azimuth, float elevation, float r) const;
};
