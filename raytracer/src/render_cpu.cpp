#include "image/image.h"
#include "objects/sphere.h"
#include "objects/triangle.h"
#include "object.h"
#include "camera.h"
#include "math/point.h"
#include "math/vec3.h"

#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

using std::vector;
using std::unique_ptr;
using std::make_unique;

// helper functions
static inline float dot3(const Vec3& a, const Vec3& b) {
    return a.getX() * b.getX() + a.getY() * b.getY() + a.getZ() * b.getZ();
}
static inline Point world_to_cam_pt(const Point& P, const Point& camPos, const Vec3& right, const Vec3& up, const Vec3& forward) {
    Vec3 v(P.getX() - camPos.getX(), P.getY() - camPos.getY(), P.getZ() - camPos.getZ());
    return Point(dot3(v, right), dot3(v, up), dot3(v, forward));
}

// main render function
int run_render_cpu() {
    const int W = 800;
    const int H = 600;

    const float FOV_DEG = 90.0f;
    const float fov = FOV_DEG * 3.14159265358979323846f / 180.0f;
    const float aspect = (float)W / (float)H;

    // camera from specifications.txt
    Camera camera(
        Point(0.033089f, 0.765843f, -0.331214f),
        Vec3(0.033089f, 0.765843f, -1.331214f),
        Vec3(0.0f, 1.0f, 0.0f),
        60.0f
    );
    Point camPos = camera.getPosition();
    Point camLook(0.033089f, 0.765843f, -1.331214f);

    // camera basis in world space
    Vec3 forward(camLook.getX() - camPos.getX(), camLook.getY() - camPos.getY(), camLook.getZ() - camPos.getZ());
    forward.normalize();
    Vec3 worldUp(0.0f, 1.0f, 0.0f);
    Vec3 right = right.cross(forward, worldUp);
    right.normalize();
    Vec3 up = up.cross(right, forward);
    up.normalize();

    // build scene in world coords
    // sphere #1
    Point s1c_world(0.498855f, 0.393785f, -1.932619f);
    float s1r = 0.36358747f;

    // sphere #2
    Point s2c_world(0.026044f, 0.864156f, -1.366522f);
    float s2r = 0.38744035f;

    // floor as two triangles
    Point floorCenter(1.991213f, -0.257648f, -2.878398f);
    float fx = 6.148293f * 0.5f;
    float fz = 5.984314f * 0.5f;
    float fy = floorCenter.getY();
    Point f00_world(floorCenter.getX() - fx, fy, floorCenter.getZ() - fz);
    Point f10_world(floorCenter.getX() + fx, fy, floorCenter.getZ() - fz);
    Point f01_world(floorCenter.getX() - fx, fy, floorCenter.getZ() + fz);
    Point f11_world(floorCenter.getX() + fx, fy, floorCenter.getZ() + fz);

    // convert scene to camera space
    vector<unique_ptr<Object>> sceneCam;

    // transform centers to camera space
    Point s1c_cam = world_to_cam_pt(s1c_world, camPos, right, up, forward);
    sceneCam.push_back(make_unique<Sphere>(s1c_cam, s1r, -1, Color(255, 255, 0))); // yellow sphere

    Point s2c_cam = world_to_cam_pt(s2c_world, camPos, right, up, forward);
    sceneCam.push_back(make_unique<Sphere>(s2c_cam, s2r, -1, Color(200, 200, 200))); // grey sphere

    // ftransform vertices to camera space
    Point f00_cam = world_to_cam_pt(f00_world, camPos, right, up, forward);
    Point f10_cam = world_to_cam_pt(f10_world, camPos, right, up, forward);
    Point f01_cam = world_to_cam_pt(f01_world, camPos, right, up, forward);
    Point f11_cam = world_to_cam_pt(f11_world, camPos, right, up, forward);

    auto t1 = make_unique<Triangle>(f00_cam, f10_cam, f11_cam, -1);
    t1->setColor(Color(255, 0, 0)); // red
    sceneCam.push_back(std::move(t1));
    auto t2 = make_unique<Triangle>(f00_cam, f11_cam, f01_cam, -1);
    t2->setColor(Color(255, 0, 0)); // red
    sceneCam.push_back(std::move(t2));

    // prep img with sky blue background
    Image img(W, H, Color(135, 206, 235));

    // ray trace in camera coords
    float scale = std::tan(fov * 0.5f);
    Point rayOriginCam(0.0f, 0.0f, 0.0f);

    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            float ndc_x = ((i + 0.5f) / W) * 2.0f - 1.0f;
            float ndc_y = 1.0f - ((j + 0.5f) / H) * 2.0f;

            float px = ndc_x * aspect * scale;
            float py = ndc_y * scale;

            Vec3 rayDir(px, py, 1.0f);
            Ray ray(rayOriginCam, rayDir);

            float nearestT = std::numeric_limits<float>::infinity();
            Object* hitObj = nullptr;

            for (const auto& obj : sceneCam) {
                float t;
                if (obj->intersect(ray, t) && t < nearestT) {
                    nearestT = t;
                    hitObj = obj.get();
                }
            }

            if (hitObj)
                img.setPixel(i, j, hitObj->getColor());
        }
    }

    std::string out = "output.ppm";
    if (!img.writePPM(out)) return 1;
    return 0;
}
