#include <cuda_runtime.h>
#include "image/image.h"

#include "components/material.h"
#include "components/illumination.h"
#include "objects/cube.h"
#include "objects/sphere.h"
#include "objects/triangle.h"
#include "components/color.h"
#include "components/point.h"
#include "components/vec3.h"
#include "components/ray.h"
#include "components/light.h"
#include "components/intersect_data.h"
#include "textures/glitter.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>

const int W = 600;
const int H = 600;
const float FOV_DEG = 40.0f;
const int MAX_DEPTH = 3;

static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

#ifdef __CUDACC__
    #define CUDA_DEVICE __device__
    #define CUDA_CALLABLE_LOCAL __host__ __device__
#else
    #define CUDA_DEVICE
    #define CUDA_CALLABLE_LOCAL
#endif

static inline Point worldToCam(const Point& P, const Point& cam_pos, const Vec3& right, const Vec3& up, const Vec3& forward) {
    Vec3 v(P.getX() - cam_pos.getX(), P.getY() - cam_pos.getY(), P.getZ() - cam_pos.getZ());
    return Point(v.dot(right), v.dot(up), v.dot(forward));
}

struct GlitterData {
    GlitterParams params;
    Color baseColor;
    Color tintColor;
};

struct RotatedCubeGPU {
    Point center;
    Vec3 size;
    float yawRadians;
    int materialIndex;
    int useGlitter;
};

CUDA_CALLABLE_LOCAL inline Vec3 rotateWorldToLocal(const Vec3& v, float yawRadians) {
    float c = std::cos(yawRadians);
    float s = std::sin(yawRadians);
    return Vec3(
        c * v.getX() - s * v.getZ(),
        v.getY(),
        s * v.getX() + c * v.getZ()
    );
}

CUDA_CALLABLE_LOCAL inline Vec3 rotateLocalToWorld(const Vec3& v, float yawRadians) {
    float c = std::cos(yawRadians);
    float s = std::sin(yawRadians);
    return Vec3(
        c * v.getX() + s * v.getZ(),
        v.getY(),
        -s * v.getX() + c * v.getZ()
    );
}

CUDA_CALLABLE_LOCAL inline bool intersectRotatedCubeGPU(const RotatedCubeGPU& cube, const Ray& ray, float& t) {
    const Point bmin(
        -cube.size.getX() * 0.5f,
        -cube.size.getY() * 0.5f,
        -cube.size.getZ() * 0.5f
    );
    const Point bmax(
        cube.size.getX() * 0.5f,
        cube.size.getY() * 0.5f,
        cube.size.getZ() * 0.5f
    );
    Vec3 originOffset(
        ray.getOrigin().getX() - cube.center.getX(),
        ray.getOrigin().getY() - cube.center.getY(),
        ray.getOrigin().getZ() - cube.center.getZ()
    );
    Vec3 originLocal = rotateWorldToLocal(originOffset, cube.yawRadians);
    Point origin(originLocal.getX(), originLocal.getY(), originLocal.getZ());
    Vec3 dir = rotateWorldToLocal(ray.getDirection(), cube.yawRadians);
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

CUDA_CALLABLE_LOCAL inline Vec3 normalRotatedCubeGPU(const RotatedCubeGPU& cube, const Point& p) {
    const Point bmin(
        -cube.size.getX() * 0.5f,
        -cube.size.getY() * 0.5f,
        -cube.size.getZ() * 0.5f
    );
    const Point bmax(
        cube.size.getX() * 0.5f,
        cube.size.getY() * 0.5f,
        cube.size.getZ() * 0.5f
    );
    const float EPS = 1e-3f;
    Vec3 offset(
        p.getX() - cube.center.getX(),
        p.getY() - cube.center.getY(),
        p.getZ() - cube.center.getZ()
    );
    Vec3 localPoint = rotateWorldToLocal(offset, cube.yawRadians);
    Point pl(localPoint.getX(), localPoint.getY(), localPoint.getZ());

    if (fabsf(pl.getX() - bmin.getX()) < EPS) return rotateLocalToWorld(Vec3(-1.0f, 0.0f, 0.0f), cube.yawRadians);
    if (fabsf(pl.getX() - bmax.getX()) < EPS) return rotateLocalToWorld(Vec3(1.0f, 0.0f, 0.0f), cube.yawRadians);
    if (fabsf(pl.getY() - bmin.getY()) < EPS) return rotateLocalToWorld(Vec3(0.0f, -1.0f, 0.0f), cube.yawRadians);
    if (fabsf(pl.getY() - bmax.getY()) < EPS) return rotateLocalToWorld(Vec3(0.0f, 1.0f, 0.0f), cube.yawRadians);
    if (fabsf(pl.getZ() - bmin.getZ()) < EPS) return rotateLocalToWorld(Vec3(0.0f, 0.0f, -1.0f), cube.yawRadians);
    if (fabsf(pl.getZ() - bmax.getZ()) < EPS) return rotateLocalToWorld(Vec3(0.0f, 0.0f, 1.0f), cube.yawRadians);

    float ax = fabsf(localPoint.getX() / (cube.size.getX() * 0.5f));
    float ay = fabsf(localPoint.getY() / (cube.size.getY() * 0.5f));
    float az = fabsf(localPoint.getZ() / (cube.size.getZ() * 0.5f));

    if (ax >= ay && ax >= az) return rotateLocalToWorld(Vec3((localPoint.getX() >= 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f), cube.yawRadians);
    if (ay >= ax && ay >= az) return rotateLocalToWorld(Vec3(0.0f, (localPoint.getY() >= 0.0f) ? 1.0f : -1.0f, 0.0f), cube.yawRadians);
    return rotateLocalToWorld(Vec3(0.0f, 0.0f, (localPoint.getZ() >= 0.0f) ? 1.0f : -1.0f), cube.yawRadians);
}

CUDA_CALLABLE_LOCAL inline Point cubeUVRotatedGPU(const RotatedCubeGPU& cube, const Point& p, const Vec3& worldNormal) {
    const Point bmin(
        -cube.size.getX() * 0.5f,
        -cube.size.getY() * 0.5f,
        -cube.size.getZ() * 0.5f
    );
    const Point bmax(
        cube.size.getX() * 0.5f,
        cube.size.getY() * 0.5f,
        cube.size.getZ() * 0.5f
    );
    Vec3 offset(
        p.getX() - cube.center.getX(),
        p.getY() - cube.center.getY(),
        p.getZ() - cube.center.getZ()
    );
    Vec3 localPoint = rotateWorldToLocal(offset, cube.yawRadians);
    Point pl(localPoint.getX(), localPoint.getY(), localPoint.getZ());

    float u = 0.0f;
    float v = 0.0f;

    if (fabsf(worldNormal.getX()) > 0.5f) {
        u = (pl.getZ() - bmin.getZ()) / fmaxf(bmax.getZ() - bmin.getZ(), 1e-6f);
        v = (pl.getY() - bmin.getY()) / fmaxf(bmax.getY() - bmin.getY(), 1e-6f);
        if (worldNormal.getX() < 0.0f) u = 1.0f - u;
    } else if (fabsf(worldNormal.getY()) > 0.5f) {
        u = (pl.getX() - bmin.getX()) / fmaxf(bmax.getX() - bmin.getX(), 1e-6f);
        v = (pl.getZ() - bmin.getZ()) / fmaxf(bmax.getZ() - bmin.getZ(), 1e-6f);
        if (worldNormal.getY() < 0.0f) v = 1.0f - v;
    } else {
        u = (pl.getX() - bmin.getX()) / fmaxf(bmax.getX() - bmin.getX(), 1e-6f);
        v = (pl.getY() - bmin.getY()) / fmaxf(bmax.getY() - bmin.getY(), 1e-6f);
        if (worldNormal.getZ() < 0.0f) u = 1.0f - u;
    }

    u = fminf(fmaxf(u, 0.0f), 1.0f);
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    return Point(u, v, 0.0f);
}

CUDA_CALLABLE_LOCAL inline Point sphereUVGPU(const SphereGPU& sphere, const Point& p) {
    Vec3 normal(
        p.getX() - sphere.center.getX(),
        p.getY() - sphere.center.getY(),
        p.getZ() - sphere.center.getZ()
    );
    normal.normalize();

    float u = atan2f(normal.getZ(), normal.getX()) / (2.0f * 3.14159265358979323846f) + 0.5f;
    float v = acosf(fminf(fmaxf(normal.getY(), -1.0f), 1.0f)) / 3.14159265358979323846f;
    return Point(u, v, 0.0f);
}

CUDA_CALLABLE_LOCAL inline UV selectGlitterUVGPU(const Point& world_pos, const Point& uv_coords, bool use_uv) {
    if (use_uv) {
        return UV(uv_coords.getX(), uv_coords.getY());
    }
    return UV(world_pos.getX(), world_pos.getZ());
}

CUDA_CALLABLE_LOCAL inline FlakeSample sampleProjectedFlakeGPU(
    const GlitterData& glitter,
    const Point& world_pos,
    const Point& uv_coords,
    const Vec3& normal
) {
    bool use_uv = (uv_coords.getX() != 0.0f || uv_coords.getY() != 0.0f);
    if (use_uv && fabsf(normal.getY()) > 0.92f) {
        UV uv = selectGlitterUVGPU(world_pos, uv_coords, true);
        return sample_glitter(uv, glitter.params);
    }
    return sample_glitter_triplanar(world_pos, normal, glitter.params);
}

CUDA_CALLABLE_LOCAL inline float sampleGlitterHeightGPU(
    const GlitterData& glitter,
    const Point& world_pos,
    const Point& uv_coords,
    const Vec3& normal
) {
    return sampleProjectedFlakeGPU(glitter, world_pos, uv_coords, normal).height;
}

CUDA_CALLABLE_LOCAL inline Vec3 bumpNormalGPU(
    const GlitterData& glitter,
    const Point& world_pos,
    const Point& uv_coords,
    const Vec3& geom_normal,
    float strength = 1.0f,
    float eps = (1.0f / 512.0f)
) {
    Vec3 T, B;
    build_tangent_basis(geom_normal, T, B);
    Point pu(
        world_pos.getX() + T.getX() * eps,
        world_pos.getY() + T.getY() * eps,
        world_pos.getZ() + T.getZ() * eps
    );
    Point mu(
        world_pos.getX() - T.getX() * eps,
        world_pos.getY() - T.getY() * eps,
        world_pos.getZ() - T.getZ() * eps
    );
    Point pv(
        world_pos.getX() + B.getX() * eps,
        world_pos.getY() + B.getY() * eps,
        world_pos.getZ() + B.getZ() * eps
    );
    Point mv(
        world_pos.getX() - B.getX() * eps,
        world_pos.getY() - B.getY() * eps,
        world_pos.getZ() - B.getZ() * eps
    );

    float dhdu = (sampleGlitterHeightGPU(glitter, pu, uv_coords, geom_normal) - sampleGlitterHeightGPU(glitter, mu, uv_coords, geom_normal)) / (2.0f * eps);
    float dhdv = (sampleGlitterHeightGPU(glitter, pv, uv_coords, geom_normal) - sampleGlitterHeightGPU(glitter, mv, uv_coords, geom_normal)) / (2.0f * eps);

    Vec3 grad = add3(mul3(T, dhdu * strength), mul3(B, dhdv * strength));
    return normalize_safe3(sub3(geom_normal, grad));
}

CUDA_CALLABLE_LOCAL inline Color sampleGlitterColorGPU(
    const GlitterData& glitter,
    const Point& world_pos,
    const Point& uv_coords,
    const Vec3& normal
) {
    FlakeSample flake = sampleProjectedFlakeGPU(glitter, world_pos, uv_coords, normal);
    float baseStrength = lerp(glitter.params.base_mix, glitter.params.base_mix + 0.25f, clamp01(flake.shade));
    float flakeCore = clamp01(flake.mask * flake.mask);
    float sparkleStrength = flakeCore * flake.sparkle * 0.62f;
    Color metallicBase = (glitter.baseColor * 0.28f) + (glitter.tintColor * 0.92f);
    Color baseLayer = metallicBase * baseStrength;

    float hueShift = (flake.theta * (1.0f / 6.28318530717958647692f) - 0.5f) * glitter.params.hue_variation;
    float value = clamp01(0.90f + flake.sparkle * glitter.params.value_variation);
    Vec3 flakeTint = hsv_to_rgb(0.58f + hueShift, 0.08f, value);
    Color silverGlint(
        static_cast<int>(flakeTint.getX() * 255.0f),
        static_cast<int>(flakeTint.getY() * 255.0f),
        static_cast<int>(flakeTint.getZ() * 255.0f)
    );
    Color tintedGlint = glitter.tintColor * 0.88f;
    Color flakeLayer = (tintedGlint * 0.74f) + (silverGlint * 0.26f);

    return baseLayer + (flakeLayer * sparkleStrength);
}

CUDA_DEVICE Color traceRayGPU(
    const Ray& primaryRay,
    const SphereGPU* spheres,
    int nSpheres,
    const RotatedCubeGPU* cubes,
    int nCubes,
    const TriangleGPU* tris,
    int nTris,
    const Material* materials,
    int numMaterials,
    const LightData* lights,
    int numLights,
    const GlitterData& glitter,
    const Color& ambientLight,
    const Color& background
) {
    Ray currentRay = primaryRay;
    float pathWeight = 1.0f;
    Color result(0, 0, 0);

    for (int depth = 1; depth < MAX_DEPTH; ++depth) {
        float nearest = 1e30f;
        int hitType = -1; // 0 = sphere, 1 = triangle, 2 = cube
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
        for (int c = 0; c < nCubes; ++c) {
            float t;
            if (intersectRotatedCubeGPU(cubes[c], currentRay, t) && t < nearest) {
                nearest = t;
                hitType = 2;
                hitIndex = c;
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
        Point uv_coords(0.0f, 0.0f, 0.0f);
        bool useGlitter = false;
        int matIndex = 0;

        if (hitType == 0) {
            normal = Vec3(
                hit_point.getX() - spheres[hitIndex].center.getX(),
                hit_point.getY() - spheres[hitIndex].center.getY(),
                hit_point.getZ() - spheres[hitIndex].center.getZ()
            );
            matIndex = spheres[hitIndex].materialIndex;
            uv_coords = sphereUVGPU(spheres[hitIndex], hit_point);
            useGlitter = true;
        } else if (hitType == 1) {
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
        } else {
            normal = normalRotatedCubeGPU(cubes[hitIndex], hit_point);
            matIndex = cubes[hitIndex].materialIndex;
            uv_coords = cubeUVRotatedGPU(cubes[hitIndex], hit_point, normal);
            useGlitter = cubes[hitIndex].useGlitter != 0;
        }
        normal.normalize();

        if (matIndex < 0 || matIndex >= numMaterials) {
            matIndex = 0;
        }

        Vec3 view_dir = ray_dir * -1.0f;
        view_dir.normalize();

        Color diffuseColor = materials[matIndex].getDiffuse();
        Vec3 effectiveNormal = normal;
        if (useGlitter) {
            Color textureSample = sampleGlitterColorGPU(glitter, hit_point, uv_coords, normal);
            diffuseColor = materials[matIndex].getDiffuse() * textureSample;
            effectiveNormal = bumpNormalGPU(glitter, hit_point, Point(0.0f, 0.0f, 0.0f), normal, 0.15f);
        }

        Color localColor = materials[matIndex].getAmbient() * ambientLight;
        const float EPS = 1e-4f;
        for (int li = 0; li < numLights; ++li) {
            Vec3 L = lights[li].position - hit_point;
            float lightDist = L.length();
            L.normalize();
            float NdotL = effectiveNormal.dot(L);
            if (NdotL < 0.0f) NdotL = 0.0f;

            Point shadow_origin(
                hit_point.getX() + effectiveNormal.getX() * EPS,
                hit_point.getY() + effectiveNormal.getY() * EPS,
                hit_point.getZ() + effectiveNormal.getZ() * EPS
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
                for (int c = 0; c < nCubes; ++c) {
                    float tShadow;
                    if (intersectRotatedCubeGPU(cubes[c], shadow_ray, tShadow) && tShadow > EPS && tShadow < lightDist) {
                        inShadow = true;
                        break;
                    }
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
                const float SHADOW_BOUNCE = 0.0f;
                localColor = localColor + (diffuseColor * lightColor * NdotL * SHADOW_BOUNCE);
                continue;
            }

            Vec3 R = (effectiveNormal * (2.0f * NdotL)) - L;
            R.normalize();
            float RdotV = R.dot(view_dir);
            if (RdotV < 0.0f) RdotV = 0.0f;
            float specularFactor = (RdotV > 0.0f) ? powf(RdotV, materials[matIndex].getShininess()) : 0.0f;

            Color lightColor = lights[li].color * lights[li].intensity;
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

__global__ void renderKernel(Color* fb, int w, int h, float aspect, float scale,
    const SphereGPU* spheres, int nSpheres,
    const RotatedCubeGPU* cubes, int nCubes,
    const TriangleGPU* tris, int nTris,
    const Material* materials, int numMaterials,
    const LightData* lights, int numLights,
    GlitterData glitter,
    Color ambientLight, Color background) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h) return;

    float accumR = 0.0f;
    float accumG = 0.0f;
    float accumB = 0.0f;
    const int SAMPLES_AXIS = 2;
    const float invSamples = 0.25f;

    for (int sy = 0; sy < SAMPLES_AXIS; ++sy) {
        for (int sx = 0; sx < SAMPLES_AXIS; ++sx) {
            float u = (i + (sx + 0.5f) / SAMPLES_AXIS) / (float)w;
            float v = (j + (sy + 0.5f) / SAMPLES_AXIS) / (float)h;

            float ndc_x = u * 2.0f - 1.0f;
            float ndc_y = 1.0f - v * 2.0f;

            float px = ndc_x * aspect * scale;
            float py = ndc_y * scale;

            Point ray_origin(0.0f, 0.0f, 0.0f);
            Vec3 ray_dir(px, py, 1.0f);
            Ray ray(ray_origin, ray_dir);

            Color sample = traceRayGPU(
                ray,
                spheres, nSpheres,
                cubes, nCubes,
                tris, nTris,
                materials, numMaterials,
                lights, numLights,
                glitter,
                ambientLight,
                background
            );
            accumR += sample.r;
            accumG += sample.g;
            accumB += sample.b;
        }
    }

    Color result(accumR * invSamples, accumG * invSamples, accumB * invSamples);
    result.clamp();
    fb[(size_t)j * (size_t)w + (size_t)i] = result;
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

static void addTriangleGPU(
    std::vector<TriangleGPU>& tris,
    const Point& p0,
    const Point& p1,
    const Point& p2,
    int materialIndex
) {
    TriangleGPU tri{};
    tri.points[0] = p0;
    tri.points[1] = p1;
    tri.points[2] = p2;
    tri.materialIndex = materialIndex;
    tri.color = Color(255, 255, 255);
    tris.push_back(tri);
}

static void addQuadGPU(
    std::vector<TriangleGPU>& tris,
    Point p0,
    Point p1,
    Point p2,
    Point p3,
    int materialIndex,
    const Point& cam_pos,
    const Vec3& right,
    const Vec3& up,
    const Vec3& forward,
    const Vec3& normalHint
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

    addTriangleGPU(tris, c0, c1, c2, materialIndex);
    addTriangleGPU(tris, c0, c2, c3, materialIndex);
}

int renderCUDA() {
    const float PI = 3.14159265358979323846f;
    const float fov = FOV_DEG * PI / 180.0f;
    const float aspect = (float)W / (float)H;
    const float scale = tanf(fov * 0.5f);

    Point cam_pos(278.0f, 273.0f, -800.0f);
    Point cam_look(278.0f, 273.0f, 0.0f);

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
    Material matLight(Color(255, 255, 255), Color(255, 255, 255), Color(0, 0, 0), 1.0f, 0.0f);
    Material matGlitterObject = tintedMatte(215, 215, 215);
    matGlitterObject.setSpecular(Color(225, 225, 225));
    matGlitterObject.setShininess(72.0f);

    matFloor.setAmbient(Color(70, 70, 70));
    matWhite.setAmbient(Color(125, 125, 125));
    matCeiling.setAmbient(Color(145, 140, 135));
    matRed.setAmbient(Color(95, 35, 30));
    matGreen.setAmbient(Color(35, 95, 40));

    std::vector<Material> hMats = {
        matFloor,
        matWhite,
        matCeiling,
        matRed,
        matGreen,
        matLight,
        matGlitterObject
    };

    std::vector<TriangleGPU> hTris;
    hTris.reserve(18);

    addQuadGPU(hTris,
        Point(552.8f, 0.0f, 0.0f), Point(0.0f, 0.0f, 0.0f), Point(0.0f, 0.0f, 559.2f), Point(549.6f, 0.0f, 559.2f),
        0, cam_pos, right, up, forward, Vec3(0.0f, 1.0f, 0.0f));
    addQuadGPU(hTris,
        Point(556.0f, 548.8f, 0.0f), Point(556.0f, 548.8f, 559.2f), Point(343.0f, 548.8f, 559.2f), Point(343.0f, 548.8f, 0.0f),
        2, cam_pos, right, up, forward, Vec3(0.0f, -1.0f, 0.0f));
    addQuadGPU(hTris,
        Point(213.0f, 548.8f, 0.0f), Point(213.0f, 548.8f, 559.2f), Point(0.0f, 548.8f, 559.2f), Point(0.0f, 548.8f, 0.0f),
        2, cam_pos, right, up, forward, Vec3(0.0f, -1.0f, 0.0f));
    addQuadGPU(hTris,
        Point(343.0f, 548.8f, 0.0f), Point(343.0f, 548.8f, 227.0f), Point(213.0f, 548.8f, 227.0f), Point(213.0f, 548.8f, 0.0f),
        2, cam_pos, right, up, forward, Vec3(0.0f, -1.0f, 0.0f));
    addQuadGPU(hTris,
        Point(343.0f, 548.8f, 332.0f), Point(343.0f, 548.8f, 559.2f), Point(213.0f, 548.8f, 559.2f), Point(213.0f, 548.8f, 332.0f),
        2, cam_pos, right, up, forward, Vec3(0.0f, -1.0f, 0.0f));
    addQuadGPU(hTris,
        Point(353.0f, 548.79f, 217.0f), Point(353.0f, 548.79f, 342.0f), Point(203.0f, 548.79f, 342.0f), Point(203.0f, 548.79f, 217.0f),
        5, cam_pos, right, up, forward, Vec3(0.0f, -1.0f, 0.0f));
    addQuadGPU(hTris,
        Point(549.6f, 0.0f, 559.2f), Point(0.0f, 0.0f, 559.2f), Point(0.0f, 548.8f, 559.2f), Point(556.0f, 548.8f, 559.2f),
        1, cam_pos, right, up, forward, Vec3(0.0f, 0.0f, -1.0f));
    addQuadGPU(hTris,
        Point(0.0f, 0.0f, 559.2f), Point(0.0f, 0.0f, 0.0f), Point(0.0f, 548.8f, 0.0f), Point(0.0f, 548.8f, 559.2f),
        4, cam_pos, right, up, forward, Vec3(1.0f, 0.0f, 0.0f));
    addQuadGPU(hTris,
        Point(552.8f, 0.0f, 0.0f), Point(549.6f, 0.0f, 559.2f), Point(556.0f, 548.8f, 559.2f), Point(556.0f, 548.8f, 0.0f),
        3, cam_pos, right, up, forward, Vec3(-1.0f, 0.0f, 0.0f));

    std::vector<SphereGPU> hSpheres(1);
    hSpheres[0].center = worldToCam(Point(369.0f, 130.0f, 351.0f), cam_pos, right, up, forward);
    hSpheres[0].radius = 130.0f;
    hSpheres[0].materialIndex = 6;
    hSpheres[0].color = Color(255, 255, 255);

    std::vector<RotatedCubeGPU> hCubes(1);
    hCubes[0].center = worldToCam(Point(186.0f, 82.0f, 169.0f), cam_pos, right, up, forward);
    hCubes[0].size = Vec3(164.0f, 164.0f, 164.0f);
    hCubes[0].yawRadians = 0.55f;
    hCubes[0].materialIndex = 6;
    hCubes[0].useGlitter = 1;

    std::vector<LightData> hLights = {
        LightData(worldToCam(Point(240.0f, 545.0f, 245.0f), cam_pos, right, up, forward), Color(240, 240, 235), 1.7f),
        LightData(worldToCam(Point(316.0f, 545.0f, 245.0f), cam_pos, right, up, forward), Color(240, 240, 235), 1.7f),
        LightData(worldToCam(Point(240.0f, 545.0f, 314.0f), cam_pos, right, up, forward), Color(240, 240, 235), 1.7f),
        LightData(worldToCam(Point(316.0f, 545.0f, 314.0f), cam_pos, right, up, forward), Color(240, 240, 235), 1.7f),
        LightData(worldToCam(Point(278.0f, 430.0f, -650.0f), cam_pos, right, up, forward), Color(235, 235, 230), 0.20f),
        LightData(worldToCam(Point(278.0f, 360.0f, -500.0f), cam_pos, right, up, forward), Color(235, 235, 230), 0.04f)
    };
    Color ambientLight(75, 75, 75);

    GlitterData glitter{};
    glitter.params.scale = 22.0f;
    glitter.params.radius_min_a = 0.40f;
    glitter.params.radius_max_a = 0.72f;
    glitter.params.radius_min_b = 0.70f;
    glitter.params.radius_max_b = 1.18f;
    glitter.params.style_b_frequency = 0.30f;
    glitter.params.feather = 0.07f;
    glitter.params.shade_min = 0.78f;
    glitter.params.shade_max = 1.10f;
    glitter.params.sparkle_min = 1.00f;
    glitter.params.sparkle_max = 2.55f;
    glitter.params.base_mix = 0.55f;
    glitter.params.jitter = 1.35f;
    glitter.params.hue_variation = 0.03f;
    glitter.params.value_variation = 0.28f;
    glitter.params.hex_ratio_a = 0.92f;
    glitter.params.hex_ratio_b = 0.60f;
    glitter.params.height_min = 0.008f;
    glitter.params.height_max = 0.04f;
    glitter.params.seed = 12345u;
    glitter.baseColor = Color(198, 214, 235);
    glitter.tintColor = Color(210, 226, 245);

    Color* dFB = nullptr;
    SphereGPU* dSpheres = nullptr;
    RotatedCubeGPU* dCubes = nullptr;
    TriangleGPU* dTris = nullptr;
    Material* dMats = nullptr;
    LightData* dLights = nullptr;

    size_t fbBytes = (size_t)W * (size_t)H * sizeof(Color);

    CUDA_CHECK(cudaMalloc(&dFB, fbBytes));
    CUDA_CHECK(cudaMalloc(&dSpheres, hSpheres.size() * sizeof(SphereGPU)));
    CUDA_CHECK(cudaMalloc(&dCubes, hCubes.size() * sizeof(RotatedCubeGPU)));
    CUDA_CHECK(cudaMalloc(&dTris, hTris.size() * sizeof(TriangleGPU)));
    CUDA_CHECK(cudaMalloc(&dMats, hMats.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&dLights, hLights.size() * sizeof(LightData)));

    CUDA_CHECK(cudaMemcpy(dSpheres, hSpheres.data(), hSpheres.size() * sizeof(SphereGPU), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCubes, hCubes.data(), hCubes.size() * sizeof(RotatedCubeGPU), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTris, hTris.data(), hTris.size() * sizeof(TriangleGPU), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dMats, hMats.data(), hMats.size() * sizeof(Material), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dLights, hLights.data(), hLights.size() * sizeof(LightData), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    Color bg(0, 0, 0);
    renderKernel<<<grid, block>>>(dFB, W, H, aspect, scale,
        dSpheres, static_cast<int>(hSpheres.size()),
        dCubes, static_cast<int>(hCubes.size()),
        dTris, static_cast<int>(hTris.size()),
        dMats, static_cast<int>(hMats.size()),
        dLights, static_cast<int>(hLights.size()),
        glitter,
        ambientLight, bg);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Color> hFB((size_t)W * (size_t)H);
    CUDA_CHECK(cudaMemcpy(hFB.data(), dFB, fbBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dFB));
    CUDA_CHECK(cudaFree(dSpheres));
    CUDA_CHECK(cudaFree(dCubes));
    CUDA_CHECK(cudaFree(dTris));
    CUDA_CHECK(cudaFree(dMats));
    CUDA_CHECK(cudaFree(dLights));

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
#undef CUDA_CALLABLE_LOCAL
