#ifndef RAYTRACER_CAMERA_H
#define RAYTRACER_CAMERA_H

#include "math/vec3.h"
#include "math/point.h"

class Camera {
    private:
        Point position;
        Vec3 lookAt;
        Vec3 up;
        float fov;
    
    public:
        // default constructor
        Camera() : position(Point(0, 0, 0)), lookAt(Vec3(0, 0, -1)), up(Vec3(0, 1, 0)), fov(90.0f) {}

        // four parameter constructor
        Camera(const Point& pos, const Vec3& look, const Vec3& upVec, float fieldOfView)
            : position(pos), lookAt(look), up(upVec), fov(fieldOfView) {}
            
        // copy constructor
        Camera(const Camera &c) : position(c.position), lookAt(c.lookAt), up(c.up), fov(c.fov) {}

        // getters
        Point getPosition() const { return position; }
        Vec3 getLookAt() const { return lookAt; }
        Vec3 getUp() const { return up; }
        float getFov() const { return fov; }

        // setters
        void setPosition(const Point& pos) { position = pos; }
        void setLookAt(const Vec3& look) { lookAt = look; }
        void setUp(const Vec3& upVec) { up = upVec; }
        void setFov(float fieldOfView) { fov = fieldOfView; }
};

#endif // RAYTRACER_CAMERA_H