#ifndef RAYTRACER_RAY_H
#define RAYTRACER_RAY_H

#include "vec3.h"
#include "point.h"

class Ray {
    private:
        Point origin;
        Vec3 direction;
    
    public:
        // default constructor
        Ray() : origin(Point()), direction(Vec3(0, 0, 1)) {}

        // two parameter constructor
        Ray(const Point &orig, const Vec3 &dir) : origin(orig), direction(dir) {}

        // copy constructor
        Ray(const Ray &r) : origin(r.origin), direction(r.direction) {}

        // getters
        Point getOrigin() const { return origin; }
        Vec3 getDirection() const { return direction; }

        // setters
        void setOrigin(const Point &orig) { origin = orig; }
        void setDirection(const Vec3 &dir) { direction = dir; }
};

#endif // RAYTRACER_RAY_h