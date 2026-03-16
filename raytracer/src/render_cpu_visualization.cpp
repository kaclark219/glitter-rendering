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
#include "objects/sphere.h"
#include "objects/triangle.h"
#include "textures/checkerboard.h"
#include "textures/glitter.h"

#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

using std::vector;
using std::unique_ptr;
using std::make_unique;

// global constants
const int W = 800;
const int H = 600;
const float FOV_DEG = 90.0f;

// helper functions
static inline Point worldToCam(const Point& P, const Point& cam_pos, const Vec3& right, const Vec3& up, const Vec3& forward) {
    Vec3 v(P.getX() - cam_pos.getX(), P.getY() - cam_pos.getY(), P.getZ() - cam_pos.getZ());
    return Point(v.dot(right), v.dot(up), v.dot(forward));
}

int renderAllVisualizations() {
    const float fov = FOV_DEG * 3.14159265358979323846f / 180.0f;
    const float aspect = (float)W / (float)H;

    // camera from specifications.txt
    Camera camera(Point(0.033089f, 0.765843f, -0.331214f), Vec3(0.033089f, 0.765843f, -1.331214f), Vec3(0.0f, 1.0f, 0.0f), FOV_DEG);
    Point cam_pos = camera.getPosition();
    Point cam_look(0.033089f, 0.765843f, -1.331214f);

    // camera basis in world space
    Vec3 forward(cam_look.getX() - cam_pos.getX(), cam_look.getY() - cam_pos.getY(), cam_look.getZ() - cam_pos.getZ());
    forward.normalize();
    Vec3 world_up(0.0f, 1.0f, 0.0f);
    Vec3 right = forward.cross(world_up);
    right.normalize();
    Vec3 up = right.cross(forward);
    up.normalize();

    // create materials
    Material matYellow(Color(20, 20, 0), Color(150, 150, 0), Color(30, 30, 30), 20.0f, 0.0f);
    Material matGrey(Color(40, 40, 40), Color(190, 190, 190), Color(130, 130, 130), 50.0f, 0.0f);
    Material matRed(Color(20, 0, 0), Color(150, 0, 0), Color(10, 10, 10), 5.0f, 0.0f);

    // create world and add lights
    World world;
    world.setAmbientLight(Color(15, 15, 15));
    world.addLight(make_unique<PointLight>(
        worldToCam(Point(0.262f, 2.8f, -1.2f), cam_pos, right, up, forward),
        Color(255, 255, 255),
        0.9f
    ));

    // create illumination model
    PhongIllumination phong(world.getAmbientLight());

    // create glitter texture for floor
    GlitterTexture glitterTexture(Color(200, 200, 200), 96.0f);

    // build scene in world coords
    Point s1c_world(0.498855f, 0.393785f, -1.932619f);
    float s1r = 0.36358747f;

    Point s2c_world(0.026044f, 0.864156f, -1.366522f);
    float s2r = 0.38744035f;

    Point floorCenter(1.991213f, -0.257648f, -2.878398f);
    float fx = 6.148293f * 0.5f;
    float fz = 5.984314f * 0.5f;
    float fy = floorCenter.getY();
    Point f00_world(floorCenter.getX() - fx, fy, floorCenter.getZ() - fz);
    Point f10_world(floorCenter.getX() + fx, fy, floorCenter.getZ() - fz);
    Point f01_world(floorCenter.getX() - fx, fy, floorCenter.getZ() + fz);
    Point f11_world(floorCenter.getX() + fx, fy, floorCenter.getZ() + fz);

    // convert scene to camera space
    vector<unique_ptr<Object>> scene_cam;

    Point s1c_cam = worldToCam(s1c_world, cam_pos, right, up, forward);
    auto sphere1 = make_unique<Sphere>(s1c_cam, s1r);
    sphere1->setMaterial(matYellow);
    sphere1->setTexture(&glitterTexture);
    scene_cam.push_back(std::move(sphere1));

    Point s2c_cam = worldToCam(s2c_world, cam_pos, right, up, forward);
    auto sphere2 = make_unique<Sphere>(s2c_cam, s2r);
    sphere2->setMaterial(matGrey);
    sphere2->setTexture(&glitterTexture);
    scene_cam.push_back(std::move(sphere2));

    // transform vertices to camera space
    Point f00_cam = worldToCam(f00_world, cam_pos, right, up, forward);
    Point f10_cam = worldToCam(f10_world, cam_pos, right, up, forward);
    Point f01_cam = worldToCam(f01_world, cam_pos, right, up, forward);
    Point f11_cam = worldToCam(f11_world, cam_pos, right, up, forward);

    auto t1 = make_unique<Triangle>(f00_cam, f10_cam, f11_cam);
    t1->setMaterial(matRed);
    t1->setTexture(&glitterTexture);
    scene_cam.push_back(std::move(t1));
    auto t2 = make_unique<Triangle>(f00_cam, f11_cam, f01_cam);
    t2->setMaterial(matRed);
    t2->setTexture(&glitterTexture);
    scene_cam.push_back(std::move(t2));

    // ray trace helper
    float scale = std::tan(fov * 0.5f);
    Point ray_origin(0.0f, 0.0f, 0.0f);

    const int SAMPLES_PER_PIXEL = 4;
    const int SAMPLES_AXIS = 2;

    // === Render 4 visualization modes ===
    const int modes[4] = {0, 1, 2, 3};
    const std::string filenames[4] = {
        "output_viz_normal.ppm",
        "output_viz_mask.ppm",
        "output_viz_height.ppm",
        "output_viz_rotation.ppm"
    };

    for (int mode_idx = 0; mode_idx < 4; ++mode_idx) {
        int vis_mode = modes[mode_idx];
        glitterTexture.setVisualizationMode(vis_mode);

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

                        float nearest = std::numeric_limits<float>::infinity();
                        Object* obj_hit = nullptr;

                        for (const auto& obj : scene_cam) {
                            float t;
                            if (obj->intersect(ray, t) && t < nearest) {
                                nearest = t;
                                obj_hit = obj.get();
                            }
                        }

                        Color sampleColor = Color(0, 0, 0);
                        if (obj_hit) {
                            Point hit_point(
                                ray_origin.getX() + nearest * ray_dir.getX(),
                                ray_origin.getY() + nearest * ray_dir.getY(),
                                ray_origin.getZ() + nearest * ray_dir.getZ()
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
                            
                            const Sphere* sphere = dynamic_cast<const Sphere*>(obj_hit);
                            if (sphere) {
                                data.uv_coords = sphere->getUV(hit_point);
                            }
                            
                            sampleColor = phong.illuminate(data, world.getLights(), scene_cam, obj_hit->getMaterial(), view_dir, obj_hit->getTexture());
                        }

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
