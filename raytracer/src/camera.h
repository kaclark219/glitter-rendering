#ifndef RAYTRACER_CAMERA_H
#define RAYTRACER_CAMERA_H

#include "math/vec3.h"
#include "math/point.h"

#ifndef CUDA_CALLABLE
    #ifdef __CUDACC__
        #define CUDA_CALLABLE __host__ __device__
    #else
        #define CUDA_CALLABLE
    #endif
#endif

class Camera {
    private:
        Point position;
        Vec3 lookAt;
        Vec3 up;
        float fov;

    public:
        // default constructor
        CUDA_CALLABLE Camera()
            : position(Point(0, 0, 0)),
            lookAt(Vec3(0, 0, -1)),
            up(Vec3(0, 1, 0)),
            fov(90.0f) {}

        // four parameter constructor
        CUDA_CALLABLE Camera(const Point& pos, const Vec3& look, const Vec3& upVec, float fieldOfView)
            : position(pos), lookAt(look), up(upVec), fov(fieldOfView) {}

        // copy constructor
        CUDA_CALLABLE Camera(const Camera &c)
            : position(c.position), lookAt(c.lookAt), up(c.up), fov(c.fov) {}

        // getters
        CUDA_CALLABLE Point getPosition() const { return position; }
        CUDA_CALLABLE Vec3 getLookAt() const { return lookAt; }
        CUDA_CALLABLE Vec3 getUp() const { return up; }
        CUDA_CALLABLE float getFov() const { return fov; }

        // setters
        CUDA_CALLABLE void setPosition(const Point& pos) { position = pos; }
        CUDA_CALLABLE void setLookAt(const Vec3& look) { lookAt = look; }
        CUDA_CALLABLE void setUp(const Vec3& upVec) { up = upVec; }
        CUDA_CALLABLE void setFov(float fieldOfView) { fov = fieldOfView; }
};

#endif // RAYTRACER_CAMERA_H