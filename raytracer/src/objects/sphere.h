#ifndef RAYTRACER_SPHERE_H
#define RAYTRACER_SPHERE_H

#include "../object.h"
#include "../math/point.h"
#include "../math/vec3.h"
#include "../math/ray.h"

#include <cmath>

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

class Sphere : public Object {
    private:
        Point center;
        float radius;

    public:
        // default constructor
        Sphere() : Object(), center(Point()), radius(1.0f) {}

        // constructor with center, radius, optional material index and color
        Sphere(const Point& c, float r, int mat = -1, const Color &col = Color())
            : Object(mat, col), center(c), radius(r) {}

        // copy constructor
        Sphere(const Sphere &s) : Object(s), center(s.center), radius(s.radius) {}

        // getters
        Point getCenter() const { return center; }
        float getRadius() const { return radius; }

        // setters
        void setCenter(const Point& c) { center = c; }
        void setRadius(float r) { radius = r; }

        // intersection with ray
        bool intersect(const Ray& ray, float& t) const override {
            // compute ray-sphere quadratic using scalar arithmetic to avoid Vec3 normalization issues
            float ox = ray.getOrigin().getX() - center.getX();
            float oy = ray.getOrigin().getY() - center.getY();
            float oz = ray.getOrigin().getZ() - center.getZ();

            float dx = ray.getDirection().getX();
            float dy = ray.getDirection().getY();
            float dz = ray.getDirection().getZ();

            float a = dx*dx + dy*dy + dz*dz;
            float b = 2.0f * (ox*dx + oy*dy + oz*dz);
            float c = ox*ox + oy*oy + oz*oz - radius * radius;

            float discriminant = b * b - 4 * a * c;
            if (discriminant < 0.0f) return false;

            float sqrtD = std::sqrt(discriminant);
            float t0 = (-b - sqrtD) / (2.0f * a);
            float t1 = (-b + sqrtD) / (2.0f * a);
            const float EPS = 1e-6f;
            if (t0 > EPS) { t = t0; return true; }
            if (t1 > EPS) { t = t1; return true; }
            return false;
        }

        Vec3 normal(const Point& p) const override {
            Vec3 n(p.getX() - center.getX(), p.getY() - center.getY(), p.getZ() - center.getZ());
            return n; // Vec3 normalizes
        }
};

// cuda version
#ifdef __CUDACC__

struct SphereGPU {
    Point center;
    float radius;
    Color color;
    int materialIndex;
};

CUDA_CALLABLE inline bool intersectSphereGPU(const SphereGPU& s, const Ray& ray, float& t) {
    // same math as your CPU version (scalar arithmetic)
    float ox = ray.getOrigin().getX() - s.center.getX();
    float oy = ray.getOrigin().getY() - s.center.getY();
    float oz = ray.getOrigin().getZ() - s.center.getZ();

    float dx = ray.getDirection().getX();
    float dy = ray.getDirection().getY();
    float dz = ray.getDirection().getZ();

    float a = dx*dx + dy*dy + dz*dz;
    float b = 2.0f * (ox*dx + oy*dy + oz*dz);
    float c = ox*ox + oy*oy + oz*oz - s.radius * s.radius;

    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0.0f) return false;

    float sqrtD = sqrtf(discriminant);
    float t0 = (-b - sqrtD) / (2.0f * a);
    float t1 = (-b + sqrtD) / (2.0f * a);
    const float EPS = 1e-6f;
    if (t0 > EPS) { t = t0; return true; }
    if (t1 > EPS) { t = t1; return true; }
    return false;
}

#endif // __CUDACC__

#undef CUDA_CALLABLE

#endif // RAYTRACER_SPHERE_H