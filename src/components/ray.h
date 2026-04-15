#ifndef RAYTRACER_RAY_H
#define RAYTRACER_RAY_H

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

#include "vec3.h"
#include "point.h"

class Ray {
    private:
        Point origin;
        Vec3 direction;
    
    public:
        // default constructor
        CUDA_CALLABLE Ray() : origin(Point()), direction(Vec3(0, 0, 1)) {}

        // two parameter constructor
        CUDA_CALLABLE Ray(const Point &orig, const Vec3 &dir)
            : origin(orig), direction(dir) {}

        // copy constructor
        CUDA_CALLABLE Ray(const Ray &r)
            : origin(r.origin), direction(r.direction) {}

        // getters
        CUDA_CALLABLE Point getOrigin() const { return origin; }
        CUDA_CALLABLE Vec3 getDirection() const { return direction; }

        // setters
        CUDA_CALLABLE void setOrigin(const Point &orig) { origin = orig; }
        CUDA_CALLABLE void setDirection(const Vec3 &dir) { direction = dir; }
};

#undef CUDA_CALLABLE

#endif // RAYTRACER_RAY_H