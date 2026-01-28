#ifndef RAYTRACER_POINT_H
#define RAYTRACER_POINT_H

#include <cmath>
#include "mat4.h"

class Point {
    private:
        float x, y, z;
    public:
        // default constructor
        Point() : x(0), y(0), z(0) {}

        // three parameter constructor
        Point(float x_val, float y_val, float z_val) : x(x_val), y(y_val), z(z_val) {}

        // use default copy constructor/assignment
        Point(const Point &p) = default;
        Point& operator=(const Point &p) = default;

        // getters
        float getX() const { return x; }
        float getY() const { return y; }
        float getZ() const { return z; }

        // setters
        void setX(float x_val) { x = x_val; }
        void setY(float y_val) { y = y_val; }
        void setZ(float z_val) { z = z_val; }
        
        // distance between two points
        float distance(const Point &p) const {
            return std::sqrt((x - p.x) * (x - p.x) +
                             (y - p.y) * (y - p.y) +
                             (z - p.z) * (z - p.z));
        }

        // transform point by a 4x4 matrix
        Point transform(const Mat4 &m) const {
            float tx = m.get(0,0) * x + m.get(0,1) * y + m.get(0,2) * z + m.get(0,3);
            float ty = m.get(1,0) * x + m.get(1,1) * y + m.get(1,2) * z + m.get(1,3);
            float tz = m.get(2,0) * x + m.get(2,1) * y + m.get(2,2) * z + m.get(2,3);
            return Point(tx, ty, tz);
        }
};

#endif // RAYTRACER_POINT_H