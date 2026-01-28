#ifndef RAYTRACER_VEC3_H
#define RAYTRACER_VEC3_H

#include <cmath>
#include "mat4.h"

class Vec3 {
    private:
        float x, y, z;

    public:
        // default constructor
        Vec3() { x = y = z = 0; }

        // three parameter constructor (does NOT normalize)
        Vec3(float x_val, float y_val, float z_val) {
            x = x_val;
            y = y_val;
            z = z_val;
        }

        // use default copy constructor/assignment
        Vec3(const Vec3 &v) = default;
        Vec3& operator=(const Vec3 &v) = default;

        // getters
        float getX() const { return x; }
        float getY() const { return y; }
        float getZ() const { return z; }

        // setters
        void setX(float x_val) { x = x_val; }
        void setY(float y_val) { y = y_val; }
        void setZ(float z_val) { z = z_val; }

        // add two vectors
        Vec3 add(const Vec3 v1, const Vec3 v2) const {
            return Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
        }

        // subtract two vectors
        Vec3 subtract(const Vec3 v1, const Vec3 v2) const {
            return Vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
        }

        // normalize the vector
        void normalize() {
            float length = std::sqrt(x*x + y*y + z*z);
            if (length != 0) {
                x /= length;
                y /= length;
                z /= length;
            }
        }

        // cross product
        Vec3 cross(const Vec3 v1, const Vec3 v2) const {
            return Vec3(
                v1.y * v2.z - v1.z * v2.y,
                v1.z * v2.x - v1.x * v2.z,
                v1.x * v2.y - v1.y * v2.x
            );
        }

        // dot product
        float dot(const Vec3 v1, const Vec3 v2) const {
            return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        }

        // length of the vector
        float length() const {
            return std::sqrt(x*x + y*y + z*z);
        }

        // transform the vector by a 4x4 matrix
        Vec3 transform(const Mat4 &m) const {
            float tx = m.get(0,0) * x + m.get(0,1) * y + m.get(0,2) * z + m.get(0,3);
            float ty = m.get(1,0) * x + m.get(1,1) * y + m.get(1,2) * z + m.get(1,3);
            float tz = m.get(2,0) * x + m.get(2,1) * y + m.get(2,2) * z + m.get(2,3);
            return Vec3(tx, ty, tz);
        }
};

#endif // RAYTRACER_VEC3_H