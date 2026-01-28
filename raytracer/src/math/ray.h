#include "vec3.h"

class Ray {
    private:
        Vec3 origin;
        Vec3 direction;
    
    public:
        // default constructor
        Ray() : origin(Vec3()), direction(Vec3(0, 0, 1)) {}

        // two parameter constructor
        Ray(const Vec3 &orig, const Vec3 &dir) : origin(orig), direction(dir) {}

        // copy constructor
        Ray(const Ray &r) : origin(r.origin), direction(r.direction) {}

        // getters
        Vec3 getOrigin() const { return origin; }
        Vec3 getDirection() const { return direction; }

        // setters
        void setOrigin(const Vec3 &orig) { origin = orig; }
        void setDirection(const Vec3 &dir) { direction = dir; }
};