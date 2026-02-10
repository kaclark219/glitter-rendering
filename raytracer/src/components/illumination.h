#ifndef ILLUMINATION_H
#define ILLUMINATION_H

#include "intersect_data.h"
#include "light.h"
#include "material.h"
#include "object.h"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

// FOR CPU RENDERING
class Illumination {
    public:
        virtual ~Illumination() = default;

        // virtual method to compute illumination
        virtual Color illuminate(const IntersectData& data, const std::vector<std::unique_ptr<Light>>& lights,
            const std::vector<std::unique_ptr<Object>>& objects,
            const Material& material, const Vec3& view_dir) const = 0;
};

class PhongIllumination : public Illumination {
    private:
        Color ambientLight; // global ambient light color/intensity
    
    public:
        // constructor with ambient light
        PhongIllumination(const Color& ambient = Color(50, 50, 50)) : ambientLight(ambient) {}

        Color illuminate(const IntersectData& data, const std::vector<std::unique_ptr<Light>>& lights,
            const std::vector<std::unique_ptr<Object>>& objects,
            const Material& material, const Vec3& view_dir) const override {
            Color result = material.getAmbient() * ambientLight;
            // compute diffuse and specular contributions from each light
            for (const auto& light : lights) {
                Vec3 L = light->getPosition() - data.hit_point;
                float lightDist = L.length();
                L.normalize();
                Vec3 N = data.normal;
                float NdotL = std::max(N.dot(L), 0.0f);

                // shadow ray: check if light is blocked
                const float EPS = 1e-4f;
                Point shadow_origin(
                    data.hit_point.getX() + N.getX() * EPS,
                    data.hit_point.getY() + N.getY() * EPS,
                    data.hit_point.getZ() + N.getZ() * EPS
                );
                Ray shadow_ray(shadow_origin, L);
                bool inShadow = false;
                for (const auto& obj : objects) {
                    float tShadow;
                    if (obj->intersect(shadow_ray, tShadow) && tShadow > EPS && tShadow < lightDist) {
                        inShadow = true;
                        break;
                    }
                }
                if (inShadow) {
                    continue;
                }

                Vec3 R = (N * (2.0f * NdotL)) - L;
                R.normalize();
                float RdotV = std::max(R.dot(view_dir), 0.0f);
                float specularFactor = std::pow(RdotV, material.getShininess());
                Color lightColor = light->getColor() * light->getIntensity();
                result = result + (material.getDiffuse() * lightColor * NdotL);
                result = result + (material.getSpecular() * lightColor * specularFactor);
            }
            result.clamp(); // make sure color values are within valid range
            return result;
        }
};

// FOR GPU RENDERING
struct LightData {
    Point position;
    Color color;
    float intensity;
    
    CUDA_CALLABLE LightData() : position(Point()), color(Color(255, 255, 255)), intensity(1.0f) {}
    CUDA_CALLABLE LightData(const Point& pos, const Color& col, float inten) 
        : position(pos), color(col), intensity(inten) {}
};

CUDA_CALLABLE inline Color computePhongIllumination(
    const Point& hit_point,
    const Vec3& normal,
    const Vec3& view_dir,
    const Material& material,
    const Color& ambientLight,
    const LightData* lights,
    int numLights
) {
    Color result = material.getAmbient() * ambientLight;
    for (int i = 0; i < numLights; i++) {
        Vec3 L = lights[i].position - hit_point;
        L.normalize();
        Vec3 N = normal;
        float NdotL = N.dot(L);
        if (NdotL < 0.0f) NdotL = 0.0f;
        Vec3 R = (N * (2.0f * NdotL)) - L;
        R.normalize();
        float RdotV = R.dot(view_dir);
        if (RdotV < 0.0f) RdotV = 0.0f;
        float specularFactor = 1.0f;
        if (RdotV > 0.0f) {
            #ifdef __CUDACC__
                specularFactor = powf(RdotV, material.getShininess());
            #else
                specularFactor = std::pow(RdotV, material.getShininess());
            #endif
        }
        Color lightColor = lights[i].color * lights[i].intensity;
        result = result + (material.getDiffuse() * lightColor * NdotL);
        result = result + (material.getSpecular() * lightColor * specularFactor);
    }
    result.clamp();
    return result;
}

#undef CUDA_CALLABLE

#endif // ILLUMINATION_H