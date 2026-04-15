#ifndef RAYTRACER_CUBE_H
#define RAYTRACER_CUBE_H

#include "../object.h"
#include "../components/point.h"
#include "../components/vec3.h"
#include "../components/ray.h"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

class Cube : public Object {
    private:
        Point center;
        Vec3 size;
        float yawRadians;

        CUDA_CALLABLE Vec3 worldToLocal(const Vec3& v) const {
            float c = std::cos(yawRadians);
            float s = std::sin(yawRadians);
            return Vec3(
                c * v.getX() - s * v.getZ(),
                v.getY(),
                s * v.getX() + c * v.getZ()
            );
        }

        CUDA_CALLABLE Vec3 localToWorld(const Vec3& v) const {
            float c = std::cos(yawRadians);
            float s = std::sin(yawRadians);
            return Vec3(
                c * v.getX() + s * v.getZ(),
                v.getY(),
                -s * v.getX() + c * v.getZ()
            );
        }

        CUDA_CALLABLE Point minCorner() const {
            return Point(
                center.getX() - size.getX() * 0.5f,
                center.getY() - size.getY() * 0.5f,
                center.getZ() - size.getZ() * 0.5f
            );
        }

        CUDA_CALLABLE Point maxCorner() const {
            return Point(
                center.getX() + size.getX() * 0.5f,
                center.getY() + size.getY() * 0.5f,
                center.getZ() + size.getZ() * 0.5f
            );
        }

    public:
        Cube() : Object(), center(Point()), size(Vec3(1.0f, 1.0f, 1.0f)), yawRadians(0.0f) {}

        Cube(const Point& c, float sideLength)
            : Object(), center(c), size(Vec3(sideLength, sideLength, sideLength)), yawRadians(0.0f) {}

        Cube(const Point& c, const Vec3& dimensions)
            : Object(), center(c), size(dimensions), yawRadians(0.0f) {}

        Cube(const Cube& other) : Object(other), center(other.center), size(other.size), yawRadians(other.yawRadians) {}

        Point getCenter() const { return center; }
        Vec3 getSize() const { return size; }
        float getYawRadians() const { return yawRadians; }

        void setCenter(const Point& c) { center = c; }
        void setSize(float sideLength) { size = Vec3(sideLength, sideLength, sideLength); }
        void setSize(const Vec3& dimensions) { size = dimensions; }
        void setYawRadians(float radians) { yawRadians = radians; }

        bool intersect(const Ray& ray, float& t) const override {
            const Point bmin(
                -size.getX() * 0.5f,
                -size.getY() * 0.5f,
                -size.getZ() * 0.5f
            );
            const Point bmax(
                size.getX() * 0.5f,
                size.getY() * 0.5f,
                size.getZ() * 0.5f
            );
            Vec3 originOffset(
                ray.getOrigin().getX() - center.getX(),
                ray.getOrigin().getY() - center.getY(),
                ray.getOrigin().getZ() - center.getZ()
            );
            Vec3 originLocal = worldToLocal(originOffset);
            Point origin(originLocal.getX(), originLocal.getY(), originLocal.getZ());
            const Vec3 dir = worldToLocal(ray.getDirection());
            const float EPS = 1e-6f;

            float tMin = -std::numeric_limits<float>::infinity();
            float tMax = std::numeric_limits<float>::infinity();

            auto updateSlab = [&](float originCoord, float dirCoord, float minCoord, float maxCoord) -> bool {
                if (std::fabs(dirCoord) < EPS) {
                    return originCoord >= minCoord && originCoord <= maxCoord;
                }

                float invDir = 1.0f / dirCoord;
                float t0 = (minCoord - originCoord) * invDir;
                float t1 = (maxCoord - originCoord) * invDir;
                if (t0 > t1) std::swap(t0, t1);

                tMin = std::max(tMin, t0);
                tMax = std::min(tMax, t1);
                return tMax >= tMin;
            };

            if (!updateSlab(origin.getX(), dir.getX(), bmin.getX(), bmax.getX())) return false;
            if (!updateSlab(origin.getY(), dir.getY(), bmin.getY(), bmax.getY())) return false;
            if (!updateSlab(origin.getZ(), dir.getZ(), bmin.getZ(), bmax.getZ())) return false;

            if (tMax < EPS) return false;
            t = (tMin > EPS) ? tMin : tMax;
            return true;
        }

        Vec3 normal(const Point& p) const override {
            const Point bmin(
                -size.getX() * 0.5f,
                -size.getY() * 0.5f,
                -size.getZ() * 0.5f
            );
            const Point bmax(
                size.getX() * 0.5f,
                size.getY() * 0.5f,
                size.getZ() * 0.5f
            );
            const float EPS = 1e-3f;
            Vec3 offset(
                p.getX() - center.getX(),
                p.getY() - center.getY(),
                p.getZ() - center.getZ()
            );
            Vec3 localPoint = worldToLocal(offset);
            Point pl(localPoint.getX(), localPoint.getY(), localPoint.getZ());

            if (std::fabs(pl.getX() - bmin.getX()) < EPS) return localToWorld(Vec3(-1.0f, 0.0f, 0.0f));
            if (std::fabs(pl.getX() - bmax.getX()) < EPS) return localToWorld(Vec3(1.0f, 0.0f, 0.0f));
            if (std::fabs(pl.getY() - bmin.getY()) < EPS) return localToWorld(Vec3(0.0f, -1.0f, 0.0f));
            if (std::fabs(pl.getY() - bmax.getY()) < EPS) return localToWorld(Vec3(0.0f, 1.0f, 0.0f));
            if (std::fabs(pl.getZ() - bmin.getZ()) < EPS) return localToWorld(Vec3(0.0f, 0.0f, -1.0f));
            if (std::fabs(pl.getZ() - bmax.getZ()) < EPS) return localToWorld(Vec3(0.0f, 0.0f, 1.0f));

            float ax = std::fabs(localPoint.getX() / (size.getX() * 0.5f));
            float ay = std::fabs(localPoint.getY() / (size.getY() * 0.5f));
            float az = std::fabs(localPoint.getZ() / (size.getZ() * 0.5f));

            if (ax >= ay && ax >= az) return localToWorld(Vec3((localPoint.getX() >= 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f));
            if (ay >= ax && ay >= az) return localToWorld(Vec3(0.0f, (localPoint.getY() >= 0.0f) ? 1.0f : -1.0f, 0.0f));
            return localToWorld(Vec3(0.0f, 0.0f, (localPoint.getZ() >= 0.0f) ? 1.0f : -1.0f));
        }

        Point getUV(const Point& p) const override {
            const Point bmin(
                -size.getX() * 0.5f,
                -size.getY() * 0.5f,
                -size.getZ() * 0.5f
            );
            const Point bmax(
                size.getX() * 0.5f,
                size.getY() * 0.5f,
                size.getZ() * 0.5f
            );
            const Vec3 n = normal(p);
            Vec3 offset(
                p.getX() - center.getX(),
                p.getY() - center.getY(),
                p.getZ() - center.getZ()
            );
            Vec3 localPoint = worldToLocal(offset);
            Point pl(localPoint.getX(), localPoint.getY(), localPoint.getZ());

            float u = 0.0f;
            float v = 0.0f;

            if (std::fabs(n.getX()) > 0.5f) {
                u = (pl.getZ() - bmin.getZ()) / std::max(bmax.getZ() - bmin.getZ(), 1e-6f);
                v = (pl.getY() - bmin.getY()) / std::max(bmax.getY() - bmin.getY(), 1e-6f);
                if (n.getX() < 0.0f) {
                    u = 1.0f - u;
                }
            } else if (std::fabs(n.getY()) > 0.5f) {
                u = (pl.getX() - bmin.getX()) / std::max(bmax.getX() - bmin.getX(), 1e-6f);
                v = (pl.getZ() - bmin.getZ()) / std::max(bmax.getZ() - bmin.getZ(), 1e-6f);
                if (n.getY() < 0.0f) {
                    v = 1.0f - v;
                }
            } else {
                u = (pl.getX() - bmin.getX()) / std::max(bmax.getX() - bmin.getX(), 1e-6f);
                v = (pl.getY() - bmin.getY()) / std::max(bmax.getY() - bmin.getY(), 1e-6f);
                if (n.getZ() < 0.0f) {
                    u = 1.0f - u;
                }
            }

            u = std::clamp(u, 0.0f, 1.0f);
            v = std::clamp(v, 0.0f, 1.0f);
            return Point(u, v, 0.0f);
        }
};

#ifdef __CUDACC__

struct CubeGPU {
    Point center;
    Vec3 size;
    Color color;
    int materialIndex;
};

CUDA_CALLABLE inline bool intersectCubeGPU(const CubeGPU& cube, const Ray& ray, float& t) {
    Point bmin(
        cube.center.getX() - cube.size.getX() * 0.5f,
        cube.center.getY() - cube.size.getY() * 0.5f,
        cube.center.getZ() - cube.size.getZ() * 0.5f
    );
    Point bmax(
        cube.center.getX() + cube.size.getX() * 0.5f,
        cube.center.getY() + cube.size.getY() * 0.5f,
        cube.center.getZ() + cube.size.getZ() * 0.5f
    );

    Point origin = ray.getOrigin();
    Vec3 dir = ray.getDirection();
    const float EPS = 1e-6f;

    float tMin = -INFINITY;
    float tMax = INFINITY;

    auto updateSlab = [&](float originCoord, float dirCoord, float minCoord, float maxCoord) -> bool {
        if (fabsf(dirCoord) < EPS) {
            return originCoord >= minCoord && originCoord <= maxCoord;
        }

        float invDir = 1.0f / dirCoord;
        float t0 = (minCoord - originCoord) * invDir;
        float t1 = (maxCoord - originCoord) * invDir;
        if (t0 > t1) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        tMin = fmaxf(tMin, t0);
        tMax = fminf(tMax, t1);
        return tMax >= tMin;
    };

    if (!updateSlab(origin.getX(), dir.getX(), bmin.getX(), bmax.getX())) return false;
    if (!updateSlab(origin.getY(), dir.getY(), bmin.getY(), bmax.getY())) return false;
    if (!updateSlab(origin.getZ(), dir.getZ(), bmin.getZ(), bmax.getZ())) return false;

    if (tMax < EPS) return false;
    t = (tMin > EPS) ? tMin : tMax;
    return true;
}

#endif // __CUDACC__

#undef CUDA_CALLABLE

#endif // RAYTRACER_CUBE_H
