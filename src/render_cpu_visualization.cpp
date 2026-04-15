#include "image/image.h"
#include "components/vec3.h"
#include "camera.h"
#include "components/point.h"
#include "components/ray.h"
#include "components/material.h"
#include "components/light.h"
#include "components/illumination.h"
#include "components/intersect_data.h"
#include "world.h"
#include "object.h"
#include "objects/cube.h"
#include "objects/sphere.h"
#include "textures/glitter.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using std::make_unique;
using std::unique_ptr;
using std::vector;

// global constants
const int W = 800;
const int H = 600;
const float FOV_DEG = 40.0f;
const int MAX_DEPTH = 3;

// helper functions
static inline Point worldToCam(const Point& P, const Point& cam_pos, const Vec3& right, const Vec3& up, const Vec3& forward) {
    Vec3 v(P.getX() - cam_pos.getX(), P.getY() - cam_pos.getY(), P.getZ() - cam_pos.getZ());
    return Point(v.dot(right), v.dot(up), v.dot(forward));
}

static Material tintedMatte(int r, int g, int b) {
    Material material = Material::Matte();
    material.setAmbient(Color(r / 4, g / 4, b / 4));
    material.setDiffuse(Color(r, g, b));
    material.setSpecular(Color(10, 10, 10));
    material.setShininess(10.0f);
    material.setReflectivity(0.0f);
    return material;
}

static Color traceRay(
    const Ray& ray,
    int depth,
    const PhongIllumination& phong,
    const std::vector<std::unique_ptr<Light>>& lights,
    const std::vector<std::unique_ptr<Object>>& objects,
    const Color& backgroundColor
) {
    if (depth >= MAX_DEPTH) {
        return Color(0, 0, 0);
    }

    float nearest = std::numeric_limits<float>::infinity();
    Object* obj_hit = nullptr;
    for (const auto& obj : objects) {
        float t;
        if (obj->intersect(ray, t) && t < nearest) {
            nearest = t;
            obj_hit = obj.get();
        }
    }

    if (!obj_hit) {
        return backgroundColor;
    }

    Vec3 ray_dir = ray.getDirection();
    Point hit_point(
        ray.getOrigin().getX() + nearest * ray_dir.getX(),
        ray.getOrigin().getY() + nearest * ray_dir.getY(),
        ray.getOrigin().getZ() + nearest * ray_dir.getZ()
    );

    Vec3 normal = obj_hit->normal(hit_point);
    normal.normalize();

    Vec3 view_dir = ray_dir * -1.0f;
    view_dir.normalize();

    IntersectData data;
    data.hit_point = hit_point;
    data.normal = normal;
    data.incoming = ray_dir;
    data.t = nearest;
    data.object = obj_hit;
    data.hit = true;
    data.uv_coords = obj_hit->getUV(hit_point);

    Color localColor = phong.computeLocalIllumination(
        data,
        lights,
        objects,
        obj_hit->getMaterial(),
        view_dir,
        obj_hit->getTexture()
    );

    float kr = obj_hit->getMaterial().getReflectivity();
    if (kr > 0.0f && depth < MAX_DEPTH) {
        Vec3 I = ray_dir;
        I.normalize();
        float dotIN = I.dot(normal);

        Vec3 reflect_dir = I - (normal * (2.0f * dotIN));
        reflect_dir.normalize();

        const float EPS = 1e-4f;
        float offsetSign = (reflect_dir.dot(normal) >= 0.0f) ? 1.0f : -1.0f;
        Point reflect_origin(
            hit_point.getX() + normal.getX() * EPS * offsetSign,
            hit_point.getY() + normal.getY() * EPS * offsetSign,
            hit_point.getZ() + normal.getZ() * EPS * offsetSign
        );

        Ray reflectedRay(reflect_origin, reflect_dir);
        Color reflectedColor = traceRay(reflectedRay, depth + 1, phong, lights, objects, backgroundColor);
        localColor = localColor + (reflectedColor * kr);
    }

    localColor.clamp();
    return localColor;
}

int renderAllVisualizations() {
    const float fov = FOV_DEG * 3.14159265358979323846f / 180.0f;
    const float aspect = (float)W / (float)H;

    // build in world coords, transform to camera coords, ray trace in camera space
    Camera camera(Point(0.0f, 2.25f, -10.0f), Vec3(0.0f, 1.15f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), FOV_DEG);
    Point cam_pos = camera.getPosition();
    Point cam_look(0.0f, 1.15f, 0.0f);

    Vec3 forward(cam_look.getX() - cam_pos.getX(), cam_look.getY() - cam_pos.getY(), cam_look.getZ() - cam_pos.getZ());
    forward.normalize();
    Vec3 world_up(0.0f, 1.0f, 0.0f);
    Vec3 right = forward.cross(world_up);
    right.normalize();
    Vec3 up = right.cross(forward);
    up.normalize();

    Material matObjA = tintedMatte(215, 215, 215);
    Material matObjB = tintedMatte(170, 210, 255);
    Material matObjC = tintedMatte(255, 200, 170);

    matObjA.setSpecular(Color(235, 235, 235));
    matObjA.setShininess(72.0f);
    matObjA.setReflectivity(0.10f);

    matObjB.setSpecular(Color(235, 235, 235));
    matObjB.setShininess(64.0f);
    matObjB.setReflectivity(0.06f);

    matObjC.setSpecular(Color(235, 235, 235));
    matObjC.setShininess(64.0f);
    matObjC.setReflectivity(0.04f);

    World world;
    world.setAmbientLight(Color(10, 10, 10));
    world.addLight(make_unique<PointLight>(
        worldToCam(Point(-5.5f, 7.5f, -6.0f), cam_pos, right, up, forward),
        Color(255, 255, 255),
        0.75f
    ));
    world.addLight(make_unique<PointLight>(
        worldToCam(Point(5.75f, 6.75f, -4.0f), cam_pos, right, up, forward),
        Color(255, 255, 255),
        0.65f
    ));
    world.addLight(make_unique<PointLight>(
        worldToCam(Point(0.0f, 4.5f, 6.5f), cam_pos, right, up, forward),
        Color(255, 255, 255),
        0.55f
    ));

    PhongIllumination phong(world.getAmbientLight());

    GlitterTexture glitterFine(Color(218, 223, 230), 18.0f);
    GlitterTexture glitterCoarse(Color(218, 223, 230), 42.0f);
    glitterCoarse.setTintColor(Color(200, 180, 255));
    glitterFine.setVisualizationMode(0);
    glitterCoarse.setVisualizationMode(0);

    vector<unique_ptr<Object>> scene_cam;

    // fun scene layout!
    {
        Point platformWorld(-1.9f, 0.35f, 2.1f);
        Point platformCam = worldToCam(platformWorld, cam_pos, right, up, forward);
        auto platform = make_unique<Cube>(platformCam, Vec3(2.6f, 0.7f, 2.6f));
        platform->setYawRadians(0.35f);
        platform->setMaterial(matObjC);
        platform->setTexture(&glitterCoarse);
        scene_cam.push_back(std::move(platform));

        Point sphereAWorld(-1.9f, 1.45f, 2.1f);
        Point sphereACam = worldToCam(sphereAWorld, cam_pos, right, up, forward);
        auto sphereA = make_unique<Sphere>(sphereACam, 0.95f);
        sphereA->setMaterial(matObjA);
        sphereA->setTexture(&glitterFine);
        scene_cam.push_back(std::move(sphereA));

        Point cubeBigWorld(2.2f, 0.95f, 1.4f);
        Point cubeBigCam = worldToCam(cubeBigWorld, cam_pos, right, up, forward);
        auto cubeBig = make_unique<Cube>(cubeBigCam, 2.2f);
        cubeBig->setYawRadians(-0.90f);
        cubeBig->setMaterial(matObjA);
        cubeBig->setTexture(&glitterFine);
        scene_cam.push_back(std::move(cubeBig));

        Point sphereBWorld(-3.2f, 0.65f, -0.1f);
        Point sphereBCam = worldToCam(sphereBWorld, cam_pos, right, up, forward);
        auto sphereB = make_unique<Sphere>(sphereBCam, 0.65f);
        sphereB->setMaterial(matObjB);
        sphereB->setTexture(&glitterCoarse);
        scene_cam.push_back(std::move(sphereB));

        Point cubeSmallWorld(-2.6f, 0.30f, -2.2f);
        Point cubeSmallCam = worldToCam(cubeSmallWorld, cam_pos, right, up, forward);
        auto cubeSmall = make_unique<Cube>(cubeSmallCam, 0.8f);
        cubeSmall->setYawRadians(1.35f);
        cubeSmall->setMaterial(matObjB);
        cubeSmall->setTexture(&glitterFine);
        scene_cam.push_back(std::move(cubeSmall));

        Point cubeTallWorld(3.3f, 1.55f, 3.6f);
        Point cubeTallCam = worldToCam(cubeTallWorld, cam_pos, right, up, forward);
        auto cubeTall = make_unique<Cube>(cubeTallCam, Vec3(0.9f, 3.1f, 0.9f));
        cubeTall->setYawRadians(0.95f);
        cubeTall->setMaterial(matObjC);
        cubeTall->setTexture(&glitterCoarse);
        scene_cam.push_back(std::move(cubeTall));

        Point sphereCWorld(2.7f, 0.85f, -0.8f);
        Point sphereCCam = worldToCam(sphereCWorld, cam_pos, right, up, forward);
        auto sphereC = make_unique<Sphere>(sphereCCam, 0.85f);
        sphereC->setMaterial(matObjC);
        sphereC->setTexture(&glitterFine);
        scene_cam.push_back(std::move(sphereC));

        Point sphereDWorld(0.0f, 0.40f, 4.6f);
        Point sphereDCam = worldToCam(sphereDWorld, cam_pos, right, up, forward);
        auto sphereD = make_unique<Sphere>(sphereDCam, 0.40f);
        sphereD->setMaterial(matObjB);
        sphereD->setTexture(&glitterFine);
        scene_cam.push_back(std::move(sphereD));
    }

    float scale = std::tan(fov * 0.5f);
    Point ray_origin(0.0f, 0.0f, 0.0f);

    const int SAMPLES_PER_PIXEL = 4;
    const int SAMPLES_AXIS = 2;

    const Color backgroundColor(0, 0, 0);
    const int modes[4] = {0, 1, 2, 3};
    const std::string filenames[4] = {
        "output_viz_normal.ppm",
        "output_viz_mask.ppm",
        "output_viz_height.ppm",
        "output_viz_rotation.ppm"
    };

    for (int mode_idx = 0; mode_idx < 4; ++mode_idx) {
        const int visMode = modes[mode_idx];
        glitterFine.setVisualizationMode(visMode);
        glitterCoarse.setVisualizationMode(visMode);

        Image img(W, H, Color(0, 0, 0));

        for (int j = 0; j < H; ++j) {
            for (int i = 0; i < W; ++i) {
                float accumR = 0.0f;
                float accumG = 0.0f;
                float accumB = 0.0f;

                for (int sy = 0; sy < SAMPLES_AXIS; ++sy) {
                    for (int sx = 0; sx < SAMPLES_AXIS; ++sx) {
                        float u = (i + (sx + 0.5f) / SAMPLES_AXIS) / (float)W;
                        float v = (j + (sy + 0.5f) / SAMPLES_AXIS) / (float)H;

                        float ndc_x = u * 2.0f - 1.0f;
                        float ndc_y = 1.0f - v * 2.0f;

                        float px = ndc_x * aspect * scale;
                        float py = ndc_y * scale;

                        Vec3 ray_dir(px, py, 1.0f);
                        Ray ray(ray_origin, ray_dir);
                        Color sampleColor = traceRay(ray, 1, phong, world.getLights(), scene_cam, backgroundColor);

                        accumR += sampleColor.r;
                        accumG += sampleColor.g;
                        accumB += sampleColor.b;
                    }
                }

                float invSamples = 1.0f / (float)SAMPLES_PER_PIXEL;
                Color finalColor(
                    (int)(accumR * invSamples),
                    (int)(accumG * invSamples),
                    (int)(accumB * invSamples)
                );
                finalColor.clamp();
                img.setPixel(i, j, finalColor);
            }
        }

        if (!img.writePPM(filenames[mode_idx])) {
            return 1;
        }
    }

    return 0;
}

int main() {
    return renderAllVisualizations();
}
