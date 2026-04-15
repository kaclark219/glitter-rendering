#include <cuda_runtime.h>
#include "image/image.h"

#include "components/material.h"
#include "components/illumination.h"
#include "objects/sphere.h"
#include "objects/triangle.h"
#include "components/color.h"
#include "components/point.h"
#include "components/vec3.h"
#include "components/ray.h"
#include "components/light.h"
#include "components/intersect_data.h"
#include "textures/checkerboard.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>

// global constants
const int W = 800;
const int H = 600;
const float FOV_DEG = 90.0f;
const int MAX_DEPTH = 3;

// cuda error checking
static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

#ifdef __CUDACC__
    #define CUDA_DEVICE __device__
#else
    #define CUDA_DEVICE
#endif

// helper functions
// change of basis from world to camera space
static inline Point worldToCam(const Point& P, const Point& cam_pos, const Vec3& right, const Vec3& up, const Vec3& forward) {
    Vec3 v(P.getX() - cam_pos.getX(), P.getY() - cam_pos.getY(), P.getZ() - cam_pos.getZ());
    return Point(v.dot(right), v.dot(up), v.dot(forward));
}

CUDA_DEVICE Color traceRayGPU(
    const Ray& primaryRay,
    const SphereGPU* spheres,
    int nSpheres,
    const TriangleGPU* tris,
    int nTris,
    const Material* materials,
    int numMaterials,
    const LightData* lights,
    int numLights,
    const CheckerboardTextureData* floorTexture,
    const Color& ambientLight,
    const Color& background
) {
    Ray currentRay = primaryRay;
    float pathWeight = 1.0f;
    Color result(0, 0, 0);

    for (int depth = 1; depth < MAX_DEPTH; ++depth) {
        float nearest = 1e30f;
        int hitType = -1; // 0 = sphere, 1 = triangle
        int hitIndex = -1;

        for (int s = 0; s < nSpheres; ++s) {
            float t;
            if (intersectSphereGPU(spheres[s], currentRay, t) && t < nearest) {
                nearest = t;
                hitType = 0;
                hitIndex = s;
            }
        }
        for (int tIdx = 0; tIdx < nTris; ++tIdx) {
            float t;
            if (intersectTriangleGPU(tris[tIdx], currentRay, t) && t < nearest) {
                nearest = t;
                hitType = 1;
                hitIndex = tIdx;
            }
        }

        if (hitType == -1) {
            result = result + (background * pathWeight);
            break;
        }

        Vec3 ray_dir = currentRay.getDirection();
        Point hit_point(
            currentRay.getOrigin().getX() + nearest * ray_dir.getX(),
            currentRay.getOrigin().getY() + nearest * ray_dir.getY(),
            currentRay.getOrigin().getZ() + nearest * ray_dir.getZ()
        );

        Vec3 normal;
        int matIndex = 0;
        if (hitType == 0) {
            normal = Vec3(
                hit_point.getX() - spheres[hitIndex].center.getX(),
                hit_point.getY() - spheres[hitIndex].center.getY(),
                hit_point.getZ() - spheres[hitIndex].center.getZ()
            );
            matIndex = spheres[hitIndex].materialIndex;
        } else {
            Vec3 edge1 = Vec3(
                tris[hitIndex].points[1].getX() - tris[hitIndex].points[0].getX(),
                tris[hitIndex].points[1].getY() - tris[hitIndex].points[0].getY(),
                tris[hitIndex].points[1].getZ() - tris[hitIndex].points[0].getZ()
            );
            Vec3 edge2 = Vec3(
                tris[hitIndex].points[2].getX() - tris[hitIndex].points[0].getX(),
                tris[hitIndex].points[2].getY() - tris[hitIndex].points[0].getY(),
                tris[hitIndex].points[2].getZ() - tris[hitIndex].points[0].getZ()
            );
            normal = edge1.cross(edge2);
            matIndex = tris[hitIndex].materialIndex;
        }
        normal.normalize();

        if (matIndex < 0 || matIndex >= numMaterials) {
            matIndex = 0;
        }

        Vec3 view_dir = ray_dir * -1.0f;
        view_dir.normalize();

        Color localColor = materials[matIndex].getAmbient() * ambientLight;
        const float EPS = 1e-4f;
        for (int li = 0; li < numLights; ++li) {
            Vec3 L = lights[li].position - hit_point;
            float lightDist = L.length();
            L.normalize();

            float NdotL = normal.dot(L);
            if (NdotL < 0.0f) NdotL = 0.0f;

            Point shadow_origin(
                hit_point.getX() + normal.getX() * EPS,
                hit_point.getY() + normal.getY() * EPS,
                hit_point.getZ() + normal.getZ() * EPS
            );
            Ray shadow_ray(shadow_origin, L);
            bool inShadow = false;
            for (int s = 0; s < nSpheres; ++s) {
                float tShadow;
                if (intersectSphereGPU(spheres[s], shadow_ray, tShadow) && tShadow > EPS && tShadow < lightDist) {
                    inShadow = true;
                    break;
                }
            }
            if (!inShadow) {
                for (int tIdx = 0; tIdx < nTris; ++tIdx) {
                    float tShadow;
                    if (intersectTriangleGPU(tris[tIdx], shadow_ray, tShadow) && tShadow > EPS && tShadow < lightDist) {
                        inShadow = true;
                        break;
                    }
                }
            }
            if (inShadow) {
                Color lightColor = lights[li].color * lights[li].intensity;
                Color diffuseColor = materials[matIndex].getDiffuse();
                if (matIndex == 2 && floorTexture != nullptr) {
                    diffuseColor = sampleCheckerboardGPU(*floorTexture, hit_point);
                }
                const float SHADOW_BOUNCE = 0.10f;
                localColor = localColor + (diffuseColor * lightColor * NdotL * SHADOW_BOUNCE);
                continue;
            }

            Vec3 R = (normal * (2.0f * NdotL)) - L;
            R.normalize();
            float RdotV = R.dot(view_dir);
            if (RdotV < 0.0f) RdotV = 0.0f;
            float specularFactor = (RdotV > 0.0f) ? powf(RdotV, materials[matIndex].getShininess()) : 0.0f;

            Color lightColor = lights[li].color * lights[li].intensity;
            Color diffuseColor = materials[matIndex].getDiffuse();
            if (matIndex == 2 && floorTexture != nullptr) {
                diffuseColor = sampleCheckerboardGPU(*floorTexture, hit_point);
            }

            localColor = localColor + (diffuseColor * lightColor * NdotL);
            localColor = localColor + (materials[matIndex].getSpecular() * lightColor * specularFactor);
        }

        localColor.clamp();
        result = result + (localColor * pathWeight);

        float kr = materials[matIndex].getReflectivity();
        if (kr <= 0.0f || depth + 1 >= MAX_DEPTH) {
            break;
        }

        Vec3 I = ray_dir;
        I.normalize();
        float dotIN = I.dot(normal);
        Vec3 reflect_dir = I - (normal * (2.0f * dotIN));
        reflect_dir.normalize();

        float offsetSign = (reflect_dir.dot(normal) >= 0.0f) ? 1.0f : -1.0f;
        Point reflect_origin(
            hit_point.getX() + normal.getX() * EPS * offsetSign,
            hit_point.getY() + normal.getY() * EPS * offsetSign,
            hit_point.getZ() + normal.getZ() * EPS * offsetSign
        );

        currentRay = Ray(reflect_origin, reflect_dir);
        pathWeight *= kr;
    }

    result.clamp();
    return result;
}

// one thread per pixel
__global__ void renderKernel(Color* fb, int w, int h, float aspect, float scale,
    const SphereGPU* spheres, int nSpheres,
    const TriangleGPU* tris, int nTris,
    const Material* materials, int numMaterials,
    const LightData* lights, int numLights,
    const CheckerboardTextureData* floorTexture,
    Color ambientLight, Color background) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h) return;

    float ndc_x = ((i + 0.5f) / (float)w) * 2.0f - 1.0f;
    float ndc_y = 1.0f - ((j + 0.5f) / (float)h) * 2.0f;

    float px = ndc_x * aspect * scale;
    float py = ndc_y * scale;

    Point ray_origin(0.0f, 0.0f, 0.0f);
    Vec3 ray_dir(px, py, 1.0f);
    Ray ray(ray_origin, ray_dir);

    Color result = traceRayGPU(
        ray,
        spheres, nSpheres,
        tris, nTris,
        materials, numMaterials,
        lights, numLights,
        floorTexture,
        ambientLight,
        background
    );

    fb[(size_t)j * (size_t)w + (size_t)i] = result;
}

// main render function
int renderCUDA() {
    const float PI = 3.14159265358979323846f;
    const float fov = FOV_DEG * PI / 180.0f;
    const float aspect = (float)W / (float)H;
    const float scale = tanf(fov * 0.5f);

    // camera values from specifications.txt
    Point cam_pos(0.033089f, 0.765843f, -0.331214f);
    Point cam_look(0.033089f, 0.765843f, -1.331214f);

    // camera basis in world space
    Vec3 forward(cam_look.getX() - cam_pos.getX(), cam_look.getY() - cam_pos.getY(), cam_look.getZ() - cam_pos.getZ());
    forward.normalize();

    Vec3 world_up(0.0f, 1.0f, 0.0f);

    Vec3 right = forward.cross(world_up);
    right.normalize();

    Vec3 up = right.cross(forward);
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
    Point s1c_cam = worldToCam(s1c_world, cam_pos, right, up, forward);
    Point s2c_cam = worldToCam(s2c_world, cam_pos, right, up, forward);

    Point f00_cam = worldToCam(f00_world, cam_pos, right, up, forward);
    Point f10_cam = worldToCam(f10_world, cam_pos, right, up, forward);
    Point f01_cam = worldToCam(f01_world, cam_pos, right, up, forward);
    Point f11_cam = worldToCam(f11_world, cam_pos, right, up, forward);

    // materials
    Material hMats[3];
    hMats[0] = Material::Matte();
    hMats[1] = Material::Mirror();
    hMats[2] = Material(Color(20, 0, 0), Color(150, 0, 0), Color(10, 10, 10), 5.0f, 0.0f);

    // lights
    LightData hLights[1];
    hLights[0] = LightData(
        worldToCam(Point(0.10f, 2.2f, -0.9f), cam_pos, right, up, forward),
        Color(255, 255, 255),
        0.9f
    );
    Color ambientLight(25, 25, 25);

    // create checkerboard texture for floor
    CheckerboardTextureData hCheckboard(Color(255, 0, 0), Color(255, 255, 0), 1.5f);

    // host gpu-scene arrays
    SphereGPU hSpheres[2];
    hSpheres[0].center = s1c_cam;
    hSpheres[0].radius = s1r;
    hSpheres[0].materialIndex = 0;
    hSpheres[0].color = Color(255, 255, 255); // unused in shading

    hSpheres[1].center = s2c_cam;
    hSpheres[1].radius = s2r;
    hSpheres[1].materialIndex = 1;
    hSpheres[1].color = Color(255, 255, 255); // unused in shading

    TriangleGPU hTris[2];
    hTris[0].points[0] = f00_cam;
    hTris[0].points[1] = f10_cam;
    hTris[0].points[2] = f11_cam;
    hTris[0].materialIndex = 2;
    hTris[0].color = Color(255, 0, 0); // unused in shading

    hTris[1].points[0] = f00_cam;
    hTris[1].points[1] = f11_cam;
    hTris[1].points[2] = f01_cam;
    hTris[1].materialIndex = 2;
    hTris[1].color = Color(255, 0, 0); // unused in shading

    // allocate device memory
    Color* dFB = nullptr;
    SphereGPU* dSpheres = nullptr;
    TriangleGPU* dTris = nullptr;
    Material* dMats = nullptr;
    LightData* dLights = nullptr;
    CheckerboardTextureData* dCheckboard = nullptr;

    size_t fbBytes = (size_t)W * (size_t)H * sizeof(Color);

    CUDA_CHECK(cudaMalloc(&dFB, fbBytes));
    CUDA_CHECK(cudaMalloc(&dSpheres, 2 * sizeof(SphereGPU)));
    CUDA_CHECK(cudaMalloc(&dTris, 2 * sizeof(TriangleGPU)));
    CUDA_CHECK(cudaMalloc(&dMats, 3 * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&dLights, 1 * sizeof(LightData)));
    CUDA_CHECK(cudaMalloc(&dCheckboard, sizeof(CheckerboardTextureData)));

    CUDA_CHECK(cudaMemcpy(dSpheres, hSpheres, 2 * sizeof(SphereGPU), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTris, hTris, 2 * sizeof(TriangleGPU), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dMats, hMats, 3 * sizeof(Material), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dLights, hLights, 1 * sizeof(LightData), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCheckboard, &hCheckboard, sizeof(CheckerboardTextureData), cudaMemcpyHostToDevice));

    // launch kernel
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    Color bg(135, 206, 235); // sky blue background
    renderKernel<<<grid, block>>>(dFB, W, H, aspect, scale,
        dSpheres, 2, dTris, 2,
        dMats, 3,
        dLights, 1,
        dCheckboard,
        ambientLight, bg);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy framebuffer back
    std::vector<Color> hFB((size_t)W * (size_t)H);
    CUDA_CHECK(cudaMemcpy(hFB.data(), dFB, fbBytes, cudaMemcpyDeviceToHost));

    // cleanup device memory
    CUDA_CHECK(cudaFree(dFB));
    CUDA_CHECK(cudaFree(dSpheres));
    CUDA_CHECK(cudaFree(dTris));
    CUDA_CHECK(cudaFree(dMats));
    CUDA_CHECK(cudaFree(dLights));
    CUDA_CHECK(cudaFree(dCheckboard));

    // write to image class
    Image img(W, H, bg);
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            img.setPixel(i, j, hFB[(size_t)j * (size_t)W + (size_t)i]);
        }
    }

    std::string out = "output_img_gpu.ppm";
    if (!img.writePPM(out)) return 1;

    return 0;
}

#undef CUDA_DEVICE