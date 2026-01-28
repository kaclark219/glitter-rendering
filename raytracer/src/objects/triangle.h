#ifndef RAYTRACER_TRIANGLE_H
#define RAYTRACER_TRIANGLE_H

#include "../object.h"
#include "../math/point.h"
#include "../math/vec3.h"
#include "../math/ray.h"

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
            const float EPSILON = 1e-8;
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

            Vec3 h = ray.getDirection().cross(ray.getDirection(), edge2);
            float a = edge1.dot(edge1, h);
            if (a > -EPSILON && a < EPSILON)
                return false;

            float f = 1.0f / a;
            Vec3 s = Vec3(
                ray.getOrigin().getX() - points[0].getX(),
                ray.getOrigin().getY() - points[0].getY(),
                ray.getOrigin().getZ() - points[0].getZ()
            );
            float u = f * s.dot(s, h);
            if (u < 0.0f || u > 1.0f)
                return false;

            Vec3 q = s.cross(s, edge1);
            float v = f * ray.getDirection().dot(ray.getDirection(), q);
            if (v < 0.0f || u + v > 1.0f)
                return false;

            t = f * edge2.dot(edge2, q);
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
            return edge1.cross(edge1, edge2);
        }
};

#endif // RAYTRACER_TRIANGLE_H