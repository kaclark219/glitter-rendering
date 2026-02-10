#ifndef RAYTRACER_VEC3_H
#define RAYTRACER_VEC3_H

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

#include <cmath>

class Vec3 {
private:
    float x, y, z;

public:
    // default constructor
    CUDA_CALLABLE Vec3() : x(0), y(0), z(0) {}

    // three parameter constructor
    CUDA_CALLABLE Vec3(float x_val, float y_val, float z_val) : x(x_val), y(y_val), z(z_val) {}

    // copy/assign constructors
    CUDA_CALLABLE Vec3(const Vec3& v) = default;
    CUDA_CALLABLE Vec3& operator=(const Vec3& v) = default;

    // getters
    CUDA_CALLABLE float getX() const { return x; }
    CUDA_CALLABLE float getY() const { return y; }
    CUDA_CALLABLE float getZ() const { return z; }

    // setters
    CUDA_CALLABLE void setX(float v) { x = v; }
    CUDA_CALLABLE void setY(float v) { y = v; }
    CUDA_CALLABLE void setZ(float v) { z = v; }

    // dot product
    CUDA_CALLABLE float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    // cross product
    CUDA_CALLABLE Vec3 cross(const Vec3& other) const {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    // length
    CUDA_CALLABLE float length() const {
    #ifdef __CUDACC__
        return sqrtf(x*x + y*y + z*z);
    #else
        return std::sqrt(x*x + y*y + z*z);
    #endif
    }

    // normalize
    CUDA_CALLABLE void normalize() {
        float len = length();
        if (len > 0.0f) {
            float inv = 1.0f / len;
            x *= inv; y *= inv; z *= inv;
        }
    }

    // subtraction operator
    CUDA_CALLABLE Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    // scalar multiplication
    CUDA_CALLABLE Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
};

#undef CUDA_CALLABLE

#endif // RAYTRACER_VEC3_H