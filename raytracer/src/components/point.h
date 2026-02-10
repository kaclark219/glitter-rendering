#ifndef RAYTRACER_POINT_H
#define RAYTRACER_POINT_H

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

#include <cmath>

// mat4 is cpu only
#ifndef __CUDACC__
#include "mat4.h"
#endif

// forward declaration for Vec3 to avoid circular dependency
class Vec3;

class Point {
    private:
        float x, y, z;
    public:
        // default constructor
        CUDA_CALLABLE Point() : x(0), y(0), z(0) {}

        // three parameter constructor
        CUDA_CALLABLE Point(float x_val, float y_val, float z_val) : x(x_val), y(y_val), z(z_val) {}

        // use default copy constructor/assignment
        CUDA_CALLABLE Point(const Point &p) = default;
        CUDA_CALLABLE Point& operator=(const Point &p) = default;

        // getters
        CUDA_CALLABLE float getX() const { return x; }
        CUDA_CALLABLE float getY() const { return y; }
        CUDA_CALLABLE float getZ() const { return z; }

        // setters
        CUDA_CALLABLE void setX(float x_val) { x = x_val; }
        CUDA_CALLABLE void setY(float y_val) { y = y_val; }
        CUDA_CALLABLE void setZ(float z_val) { z = z_val; }
        
        // distance between two points
        CUDA_CALLABLE float distance(const Point &p) const {
            float dx = x - p.x;
            float dy = y - p.y;
            float dz = z - p.z;

        #ifdef __CUDACC__
            return sqrtf(dx * dx + dy * dy + dz * dz);
        #else
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        #endif
        }

        // subtraction operator to get direction vector from two points
        CUDA_CALLABLE Vec3 operator-(const Point& other) const;

#ifndef __CUDACC__
        // transform point by a 4x4 matrix
        Point transform(const Mat4 &m) const {
            float tx = m.get(0,0) * x + m.get(0,1) * y + m.get(0,2) * z + m.get(0,3);
            float ty = m.get(1,0) * x + m.get(1,1) * y + m.get(1,2) * z + m.get(1,3);
            float tz = m.get(2,0) * x + m.get(2,1) * y + m.get(2,2) * z + m.get(2,3);
            return Point(tx, ty, tz);
        }
#endif
};

#undef CUDA_CALLABLE

// include Vec3 after Point definition to avoid circular dependency
#include "vec3.h"

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

// implement Point subtraction operator
inline CUDA_CALLABLE Vec3 Point::operator-(const Point& other) const {
    return Vec3(x - other.x, y - other.y, z - other.z);
}

#undef CUDA_CALLABLE

#endif // RAYTRACER_POINT_H