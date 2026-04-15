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
#include "objects/triangle.h"
#include "textures/checkerboard.h"
#include "textures/noise.h"
#include "textures/glitter.h"

// each object is heap allocated & stored in a vector
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
using std::vector;
using std::unique_ptr;
using std::make_unique;

// global constants
const int W = 600;
const int H = 600;
const float FOV_DEG = 40.0f;
const int MAX_DEPTH = 3;

// helper functions
// change of basis into camera space
static inline Point worldToCam(const Point& P, const Point& cam_pos, const Vec3& right, const Vec3& up, const Vec3& forward) {
    Vec3 v(P.getX() - cam_pos.getX(), P.getY() - cam_pos.getY(), P.getZ() - cam_pos.getZ());
    return Point(v.dot(right), v.dot(up), v.dot(forward));
}

static void addQuad(
    std::vector<std::unique_ptr<Object>>& scene,
    Point p0,
    Point p1,
    Point p2,
    Point p3,
    const Material& material,
    const Point& cam_pos,
    const Vec3& right,
    const Vec3& up,
    const Vec3& forward,
    const Vec3& normalHint,
    Texture* texture = nullptr
) {
    Vec3 edge1 = p1 - p0;
    Vec3 edge2 = p2 - p0;
    Vec3 normal = edge1.cross(edge2);
    if (normal.dot(normalHint) < 0.0f) {
        std::swap(p1, p3);
    }

    Point c0 = worldToCam(p0, cam_pos, right, up, forward);
    Point c1 = worldToCam(p1, cam_pos, right, up, forward);
    Point c2 = worldToCam(p2, cam_pos, right, up, forward);
    Point c3 = worldToCam(p3, cam_pos, right, up, forward);

    auto t1 = make_unique<Triangle>(c0, c1, c2);
    t1->setMaterial(material);
    t1->setTexture(texture);
    scene.push_back(std::move(t1));

    auto t2 = make_unique<Triangle>(c0, c2, c3);
    t2->setMaterial(material);
    t2->setTexture(texture);
    scene.push_back(std::move(t2));
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

static Material ceilingMatte() {
    Material material = Material::Matte();
    material.setAmbient(Color(150, 138, 120));
    material.setDiffuse(Color(245, 232, 210));
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

// main render function
int renderCPU() {
    const float fov = FOV_DEG * 3.14159265358979323846f / 180.0f;
    const float aspect = (float)W / (float)H;

    Camera camera(Point(278.0f, 273.0f, -800.0f), Vec3(278.0f, 273.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), FOV_DEG);
    Point cam_pos = camera.getPosition();
    Point cam_look(278.0f, 273.0f, 0.0f);

    // camera basis in world space
    Vec3 forward(cam_look.getX() - cam_pos.getX(), cam_look.getY() - cam_pos.getY(), cam_look.getZ() - cam_pos.getZ());
    forward.normalize();
    Vec3 world_up(0.0f, 1.0f, 0.0f);
    Vec3 right = forward.cross(world_up);
    right.normalize();
    Vec3 up = right.cross(forward);
    up.normalize();

    Material matFloor = tintedMatte(255, 255, 255);
    Material matWhite = tintedMatte(230, 230, 230);
    Material matCeiling = ceilingMatte();
    Material matRed = tintedMatte(225, 85, 75);
    Material matGreen = tintedMatte(100, 225, 115);
    const Material matObject = tintedMatte(188, 202, 220);
    const Color warmAmberLight(255, 188, 112);
    const Color warmGoldLight(255, 156, 72);
    const Color warmFillLight(255, 198, 128);
    Material matLight = matCeiling;
    Material matGlitterObject = matObject;
    matGlitterObject.setSpecular(Color(196, 150, 92));
    matGlitterObject.setShininess(34.0f);
    matGlitterObject.setAmbient(Color(56, 64, 76));

    matFloor.setAmbient(Color(70, 70, 70));
    matWhite.setAmbient(Color(125, 125, 125));
    matCeiling.setAmbient(Color(145, 140, 135));
    matRed.setAmbient(Color(95, 35, 30));
    matGreen.setAmbient(Color(35, 95, 40));

    // cornell box scene setup
    World world;
    world.setAmbientLight(Color(38, 24, 14));
    world.addLight(make_unique<PointLight>(
        worldToCam(Point(110.0f, 255.0f, 170.0f), cam_pos, right, up, forward),
        warmAmberLight,
        1.05f
    ));
    world.addLight(make_unique<PointLight>(
        worldToCam(Point(455.0f, 240.0f, 345.0f), cam_pos, right, up, forward),
        warmGoldLight,
        0.98f
    ));
    world.addLight(make_unique<PointLight>(
        worldToCam(Point(420.0f, 115.0f, -120.0f), cam_pos, right, up, forward),
        warmFillLight,
        0.42f
    ));

    // create illumination model
    PhongIllumination phong(world.getAmbientLight());

    GlitterTexture glitter(Color(198, 214, 235), 22.0f);
    glitter.setTintColor(Color(210, 226, 245));

    // build scene in camera space .. converted from world space using camera basis
    vector<unique_ptr<Object>> scene_cam;

    addQuad(scene_cam,
        Point(552.8f, 0.0f, 0.0f), Point(0.0f, 0.0f, 0.0f), Point(0.0f, 0.0f, 559.2f), Point(549.6f, 0.0f, 559.2f),
        matFloor, cam_pos, right, up, forward, Vec3(0.0f, 1.0f, 0.0f));

    addQuad(scene_cam,
        Point(556.0f, 548.8f, 0.0f), Point(556.0f, 548.8f, 559.2f), Point(343.0f, 548.8f, 559.2f), Point(343.0f, 548.8f, 0.0f),
        matCeiling, cam_pos, right, up, forward, Vec3(0.0f, -1.0f, 0.0f));
    addQuad(scene_cam,
        Point(213.0f, 548.8f, 0.0f), Point(213.0f, 548.8f, 559.2f), Point(0.0f, 548.8f, 559.2f), Point(0.0f, 548.8f, 0.0f),
        matCeiling, cam_pos, right, up, forward, Vec3(0.0f, -1.0f, 0.0f));
    addQuad(scene_cam,
        Point(343.0f, 548.8f, 0.0f), Point(343.0f, 548.8f, 227.0f), Point(213.0f, 548.8f, 227.0f), Point(213.0f, 548.8f, 0.0f),
        matCeiling, cam_pos, right, up, forward, Vec3(0.0f, -1.0f, 0.0f));
    addQuad(scene_cam,
        Point(343.0f, 548.8f, 332.0f), Point(343.0f, 548.8f, 559.2f), Point(213.0f, 548.8f, 559.2f), Point(213.0f, 548.8f, 332.0f),
        matCeiling, cam_pos, right, up, forward, Vec3(0.0f, -1.0f, 0.0f));

    addQuad(scene_cam,
        Point(353.0f, 548.79f, 217.0f), Point(353.0f, 548.79f, 342.0f), Point(203.0f, 548.79f, 342.0f), Point(203.0f, 548.79f, 217.0f),
        matLight, cam_pos, right, up, forward, Vec3(0.0f, -1.0f, 0.0f));

    addQuad(scene_cam,
        Point(549.6f, 0.0f, 559.2f), Point(0.0f, 0.0f, 559.2f), Point(0.0f, 548.8f, 559.2f), Point(556.0f, 548.8f, 559.2f),
        matWhite, cam_pos, right, up, forward, Vec3(0.0f, 0.0f, -1.0f));
    addQuad(scene_cam,
        Point(0.0f, 0.0f, 559.2f), Point(0.0f, 0.0f, 0.0f), Point(0.0f, 548.8f, 0.0f), Point(0.0f, 548.8f, 559.2f),
        matGreen, cam_pos, right, up, forward, Vec3(1.0f, 0.0f, 0.0f));
    addQuad(scene_cam,
        Point(552.8f, 0.0f, 0.0f), Point(549.6f, 0.0f, 559.2f), Point(556.0f, 548.8f, 559.2f), Point(556.0f, 548.8f, 0.0f),
        matRed, cam_pos, right, up, forward, Vec3(-1.0f, 0.0f, 0.0f));

    Point cubeCenterWorld(186.0f, 82.0f, 169.0f);
    Point cubeCenterCam = worldToCam(cubeCenterWorld, cam_pos, right, up, forward);
    auto cube = make_unique<Cube>(cubeCenterCam, 164.0f);
    cube->setYawRadians(0.55f);
    cube->setMaterial(matGlitterObject);
    cube->setTexture(&glitter);
    scene_cam.push_back(std::move(cube));

    Point sphereCenterWorld(369.0f, 130.0f, 351.0f);
    Point sphereCenterCam = worldToCam(sphereCenterWorld, cam_pos, right, up, forward);
    auto sphere = make_unique<Sphere>(sphereCenterCam, 130.0f);
    sphere->setMaterial(matGlitterObject);
    sphere->setTexture(&glitter);
    scene_cam.push_back(std::move(sphere));

    Image img(W, H, Color(0, 0, 0));

    // ray trace in camera coords
    float scale = std::tan(fov * 0.5f);
    Point ray_origin(0.0f, 0.0f, 0.0f);

    const int SAMPLES_PER_PIXEL = 4; // 2x2 stratified
    const int SAMPLES_AXIS = 2;

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
                    Color sampleColor = traceRay(ray, 1, phong, world.getLights(), scene_cam, Color(0, 0, 0));

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

    std::string out = "output_img.ppm";
    if (!img.writePPM(out)) return 1;
    return 0;
}
