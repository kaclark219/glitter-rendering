#include <cuda_runtime.h>
#include "image/image.h"

#include "objects/sphere.h"
#include "objects/triangle.h"

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

// cuda error checking
static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

// helper functions
// change of basis from world to camera space
static inline Point worldToCam(const Point& P, const Point& cam_pos, const Vec3& right, const Vec3& up, const Vec3& forward) {
    Vec3 v(P.getX() - cam_pos.getX(), P.getY() - cam_pos.getY(), P.getZ() - cam_pos.getZ());
    return Point(v.dot(right), v.dot(up), v.dot(forward));
}

// one thread per pixel
__global__ void renderKernel(Color* fb, int w, int h, float aspect, float scale, const SphereGPU* spheres, int nSpheres, const TriangleGPU* tris, int nTris, Color background) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y
    if (i >= w || j >= h) return;

    float ndc_x = ((i + 0.5f) / (float)w) * 2.0f - 1.0f;
    float ndc_y = 1.0f - ((j + 0.5f) / (float)h) * 2.0f;

    float px = ndc_x * aspect * scale;
    float py = ndc_y * scale;

    Point ray_origin(0.0f, 0.0f, 0.0f);
    Vec3 ray_dir(px, py, 1.0f);
    Ray ray(ray_origin, ray_dir);

    float nearest = 1e30f;
    bool hit = false;
    Color hitCol = background;

    // spheres
    for (int s = 0; s < nSpheres; ++s) {
        float t;
        if (intersectSphereGPU(spheres[s], ray, t) && t < nearest) {
            nearest = t;
            hit = true;
            hitCol = spheres[s].color;
        }
    }
    // triangles
    for (int tIdx = 0; tIdx < nTris; ++tIdx) {
        float t;
        if (intersectTriangleGPU(tris[tIdx], ray, t) && t < nearest) {
            nearest = t;
            hit = true;
            hitCol = tris[tIdx].color;
        }
    }

    fb[(size_t)j * (size_t)w + (size_t)i] = hit ? hitCol : background;
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

    // host gpu-scene arrays
    SphereGPU hSpheres[2];
    hSpheres[0].center = s1c_cam;
    hSpheres[0].radius = s1r;
    hSpheres[0].materialIndex = -1;
    hSpheres[0].color = Color(255, 255, 0); // yellow sphere

    hSpheres[1].center = s2c_cam;
    hSpheres[1].radius = s2r;
    hSpheres[1].materialIndex = -1;
    hSpheres[1].color = Color(200, 200, 200); // grey sphere

    TriangleGPU hTris[2];
    hTris[0].points[0] = f00_cam;
    hTris[0].points[1] = f10_cam;
    hTris[0].points[2] = f11_cam;
    hTris[0].materialIndex = -1;
    hTris[0].color = Color(255, 0, 0); // red

    hTris[1].points[0] = f00_cam;
    hTris[1].points[1] = f11_cam;
    hTris[1].points[2] = f01_cam;
    hTris[1].materialIndex = -1;
    hTris[1].color = Color(255, 0, 0); // red

    // allocate device memory
    Color* dFB = nullptr;
    SphereGPU* dSpheres = nullptr;
    TriangleGPU* dTris = nullptr;

    size_t fbBytes = (size_t)W * (size_t)H * sizeof(Color);

    CUDA_CHECK(cudaMalloc(&dFB, fbBytes));
    CUDA_CHECK(cudaMalloc(&dSpheres, 2 * sizeof(SphereGPU)));
    CUDA_CHECK(cudaMalloc(&dTris, 2 * sizeof(TriangleGPU)));

    CUDA_CHECK(cudaMemcpy(dSpheres, hSpheres, 2 * sizeof(SphereGPU), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTris, hTris, 2 * sizeof(TriangleGPU), cudaMemcpyHostToDevice));

    // launch kernel
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    Color bg(135, 206, 235); // sky blue background
    renderKernel<<<grid, block>>>(dFB, W, H, aspect, scale, dSpheres, 2, dTris, 2, bg);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy framebuffer back
    std::vector<Color> hFB((size_t)W * (size_t)H);
    CUDA_CHECK(cudaMemcpy(hFB.data(), dFB, fbBytes, cudaMemcpyDeviceToHost));

    // cleanup device memory
    CUDA_CHECK(cudaFree(dFB));
    CUDA_CHECK(cudaFree(dSpheres));
    CUDA_CHECK(cudaFree(dTris));

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