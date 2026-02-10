#ifndef INTERSECT_DATA_H
#define INTERSECT_DATA_H

#include "point.h"
#include "vec3.h"
#include "../object.h"

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

struct IntersectData {
    Point hit_point; // point of intersection
    Vec3 normal; // surface normal at hit point
    Vec3 incoming; // direction of incoming ray
    float t; // distance along ray to hit point
    const Object* object; // pointer to intersected object (includes material info)
    bool hit; // did intersection occur

    CUDA_CALLABLE IntersectData() : hit_point(Point()), normal(Vec3()), incoming(Vec3()), t(0), object(nullptr), hit(false) {}
};

#endif // INTERSECT_DATA_H