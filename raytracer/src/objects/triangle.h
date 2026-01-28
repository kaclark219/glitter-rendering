#ifndef RAYTRACER_TRIANGLE_H
#define RAYTRACER_TRIANGLE_H

#include "../object.h"
#include "../math/point.h"
#include "../math/vec3.h"
#include "../math/ray.h"

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

class Triangle : public Object {
    private:
        Point points[3];

    public:
        // default constructor
        Triangle() : Object(), points{Point(), Point(), Point()} {}

        // construct from three points with optional material index
        Triangle(const Point &p0, const Point &p1, const Point &p2, int mat = -1) : Object(mat) {
            points[0] = p0;
            points[1] = p1;
            points[2] = p2;
        }

        // getter
        const Point& getVertex(int i) const { return points[i]; }

        // setter
        void setVertex(int i, const Point &p) { points[i] = p; }

        // intersection with ray
        bool intersect(const Ray& ray, float& t) const override {
            // moller-trumbore intersection algorithm
            const float EPSILON = 1e-8f;

            Vec3 edge1 = Vec3(
                points[1].getX() - points[0].getX(),
                points[1].getY() - points[0].getY(),
                points[1].getZ() - points[0].getZ()
            );
            Vec3 edge2 = Vec3(
                points[2].getX() - points[0].getX(),
                points[2].getY() - points[0].getY(),
                points[2].getZ() - points[0].getZ()
            );

            Vec3 h = ray.getDirection().cross(edge2);
            float a = edge1.dot(h);
            if (a > -EPSILON && a < EPSILON)
                return false;

            float f = 1.0f / a;
            Vec3 s = Vec3(
                ray.getOrigin().getX() - points[0].getX(),
                ray.getOrigin().getY() - points[0].getY(),
                ray.getOrigin().getZ() - points[0].getZ()
            );
            float u = f * s.dot(h);
            if (u < 0.0f || u > 1.0f)
                return false;

            Vec3 q = s.cross(edge1);
            float v = f * ray.getDirection().dot(q);
            if (v < 0.0f || u + v > 1.0f)
                return false;

            t = f * edge2.dot(q);
            if (t > EPSILON)
                return true;
            else
                return false;
        }

        Vec3 normal(const Point& p) const override {
            Vec3 edge1 = Vec3(
                points[1].getX() - points[0].getX(),
                points[1].getY() - points[0].getY(),
                points[1].getZ() - points[0].getZ()
            );
            Vec3 edge2 = Vec3(
                points[2].getX() - points[0].getX(),
                points[2].getY() - points[0].getY(),
                points[2].getZ() - points[0].getZ()
            );
            return edge1.cross(edge2);
        }
};

// cuda friendly
#ifdef __CUDACC__

struct TriangleGPU {
    Point points[3];
    Color color;
    int materialIndex;
};

CUDA_CALLABLE inline bool intersectTriangleGPU(const TriangleGPU& tri, const Ray& ray, float& t) {
    // moller-trumbore intersection algorithm
    const float EPSILON = 1e-8f;

    Vec3 edge1 = Vec3(
        tri.points[1].getX() - tri.points[0].getX(),
        tri.points[1].getY() - tri.points[0].getY(),
        tri.points[1].getZ() - tri.points[0].getZ()
    );
    Vec3 edge2 = Vec3(
        tri.points[2].getX() - tri.points[0].getX(),
        tri.points[2].getY() - tri.points[0].getY(),
        tri.points[2].getZ() - tri.points[0].getZ()
    );

    Vec3 h = ray.getDirection().cross(edge2);
    float a = edge1.dot(h);
    if (a > -EPSILON && a < EPSILON)
        return false;

    float f = 1.0f / a;
    Vec3 s = Vec3(
        ray.getOrigin().getX() - tri.points[0].getX(),
        ray.getOrigin().getY() - tri.points[0].getY(),
        ray.getOrigin().getZ() - tri.points[0].getZ()
    );
    float u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f)
        return false;

    Vec3 q = s.cross(edge1);
    float v = f * ray.getDirection().dot(q);
    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = f * edge2.dot(q);
    if (t > EPSILON)
        return true;
    else
        return false;
}

#endif // __CUDACC__

#undef CUDA_CALLABLE

#endif // RAYTRACER_TRIANGLE_H