#ifndef NOISE_H
#define NOISE_H

#include "texture.h"
#include "../components/color.h"
#include "../components/point.h"
#include "../components/vec3.h"
#include "checkerboard.h"  // for PerlinNoise 
#include <cmath>

// FOR CPU RENDERING
class NoiseTexture : public Texture {
    private:
        float scale;
        Color color1;  // color for low noise values
        Color color2;  // color for high noise values

    public:
        // default to black and white, scale of 1
        NoiseTexture(float s = 1.0f, const Color& c1 = Color(0, 0, 0), const Color& c2 = Color(255, 255, 255))
            : scale(s), color1(c1), color2(c2) {}

        // override sample method in texture.h
        Color sample(const Point& world_pos, const Point& uv_coords, const Vec3& normal) const override {
            // sample perlin noise at scaled world position
            float noise = PerlinNoise::perlin(world_pos.getX() * scale, world_pos.getZ() * scale);
            
            // interpolate between color1 and color2 based on noise value
            int r = (int)(color1.r + (color2.r - color1.r) * noise);
            int g = (int)(color1.g + (color2.g - color1.g) * noise);
            int b = (int)(color1.b + (color2.b - color1.b) * noise);
            
            Color result(r, g, b);
            result.clamp();
            return result;
        }

        // getters
        float getScale() const { return scale; }
        const Color& getColor1() const { return color1; }
        const Color& getColor2() const { return color2; }

        // setters
        void setScale(float s) { scale = s; }
        void setColor1(const Color& c) { color1 = c; }
        void setColor2(const Color& c) { color2 = c; }
};

// FOR GPU RENDERING
struct NoiseTextureData : public TextureData {
    float scale;
    Color color1;
    Color color2;
    
    NoiseTextureData() : TextureData(TEX_NOISE), scale(1.0f), color1(Color(0, 0, 0)), color2(Color(255, 255, 255)) {}
    
    NoiseTextureData(float s, const Color& c1, const Color& c2): 
        TextureData(TEX_NOISE), scale(s), color1(c1), color2(c2) {}
};

// sampling function for noise texture on gpu
#ifdef __CUDACC__
// GPU permutation table for Perlin noise (must be in constant memory for device access)
__device__ __constant__ static const int perm_gpu_noise[512] = {
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 206, 39, 142, 112, 91, 149, 97, 42, 145, 170, 163, 78,
    112, 76, 86, 45, 89, 212, 73, 78, 76, 76, 79, 99, 155, 140, 163, 130,
    141, 137, 137, 111, 75, 61, 89, 241, 6, 241, 7, 40, 7, 33, 33, 62, 0, 141,
    24, 12, 12, 63, 0, 69, 8, 98, 6, 99, 12, 30, 36, 68, 25, 7, 26, 4, 3, 4,
    65, 91, 193, 233, 206, 70, 55, 61, 216, 185, 89, 124, 50, 80, 61, 110, 63,
    196, 35, 151, 65, 30, 58, 138, 200, 55, 198, 89, 110, 141, 41, 141, 137,
    111, 181, 212, 10, 44, 173, 204, 46, 211, 100, 159, 42, 190, 228, 203, 184,
    98, 194, 37, 13, 30, 180, 6, 129, 62, 253, 236, 110, 76, 127, 198, 128,
    158, 129, 226, 174, 76, 120, 48, 47, 40, 78, 128, 11, 200, 87, 244, 21,
    6, 85, 104, 73, 119, 142, 175, 195, 167, 142, 200, 40, 129, 185, 137, 141,
    137, 111, 75, 61, 89, 241, 6, 241, 7, 40, 7, 33, 33, 62, 0, 141, 24, 12,
    12, 63, 0, 69, 8, 98, 6, 99, 12, 30, 36, 68, 25, 7, 26, 4, 3, 4, 65, 91,
    193, 233, 206, 70, 55, 61, 216, 185, 89, 124, 50, 80, 61, 110, 63, 196,
    35, 151, 65, 30, 58, 138, 200, 55, 198, 89, 110, 141, 41, 141, 137, 111,
    181, 212, 10, 44, 173, 204, 46, 211, 100, 159, 42, 190, 228, 203, 184, 98,
    194, 37, 13, 30, 180, 6, 129, 62, 253, 236, 110, 76, 127, 198, 128, 158,
    129, 226, 174, 76, 120, 48, 47, 40, 78, 128, 11, 200, 87, 244, 21, 6, 85,
    104, 73, 119, 142, 175, 195, 167, 142, 200, 40, 129, 185, 137, 141, 137,
    111, 75, 61, 89, 241, 6, 241, 7, 40, 7, 33, 33, 62, 0, 141, 24, 12, 12, 63,
    0, 69, 8, 98
};

// GPU versions of Perlin noise helper functions
__host__ __device__ inline float fadeGPUNoise(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

__host__ __device__ inline float lerpGPUNoise(float t, float a, float b) {
    return a + t * (b - a);
}

__host__ __device__ inline float gradGPUNoise(int hash, float x, float y) {
    int h = hash & 15;
    float u = (h < 8) ? x : y;
    float v = (h < 8) ? y : x;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

__host__ __device__ inline float perlinGPUNoise(float x, float y) {
    int xi = (int)floorf(x) & 255;
    int yi = (int)floorf(y) & 255;
    
    float xf = x - floorf(x);
    float yf = y - floorf(y);
    
    float u = fadeGPUNoise(xf);
    float v = fadeGPUNoise(yf);
    
    int p00 = perm_gpu_noise[xi + perm_gpu_noise[yi]];
    int p10 = perm_gpu_noise[xi + 1 + perm_gpu_noise[yi]];
    int p01 = perm_gpu_noise[xi + perm_gpu_noise[yi + 1]];
    int p11 = perm_gpu_noise[xi + 1 + perm_gpu_noise[yi + 1]];
    
    float n00 = gradGPUNoise(p00, xf, yf);
    float n10 = gradGPUNoise(p10, xf - 1, yf);
    float n01 = gradGPUNoise(p01, xf, yf - 1);
    float n11 = gradGPUNoise(p11, xf - 1, yf - 1);
    
    float nx0 = lerpGPUNoise(u, n00, n10);
    float nx1 = lerpGPUNoise(u, n01, n11);
    float nxy = lerpGPUNoise(v, nx0, nx1);
    
    return (nxy + 1.0f) * 0.5f;
}

inline __host__ __device__ Color sampleNoiseGPU(
    const NoiseTextureData& tex,
    const Point& world_pos)
{
    // sample perlin noise at scaled world position
    float noise = perlinGPUNoise(world_pos.getX() * tex.scale, world_pos.getZ() * tex.scale);
    
    // interpolate between color1 and color2 based on noise value
    int r = (int)(tex.color1.r + (tex.color2.r - tex.color1.r) * noise);
    int g = (int)(tex.color1.g + (tex.color2.g - tex.color1.g) * noise);
    int b = (int)(tex.color1.b + (tex.color2.b - tex.color1.b) * noise);
    
    Color result(r, g, b);
    result.clamp();
    return result;
}
#endif // __CUDACC__

#endif // NOISE_H
