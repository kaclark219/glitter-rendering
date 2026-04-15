#ifndef CHECKERBOARD_H
#define CHECKERBOARD_H

#include "texture.h"
#include "../components/color.h"
#include "../components/point.h"
#include "../components/vec3.h"
#include <cmath>

// perlin noise implementation
namespace PerlinNoise {
    // permutation table for Perlin noise
    static const int perm[512] = {
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
    
    // helper functions for perlin noise
    inline float fade(float t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }
    inline float lerp(float t, float a, float b) {
        return a + t * (b - a);
    }
    inline float grad(int hash, float x, float y) {
        int h = hash & 15;
        float u = (h < 8) ? x : y;
        float v = (h < 8) ? y : x;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }
    
    // 2d perlin noise
    inline float perlin(float x, float y) {
        int xi = (int)floorf(x) & 255;
        int yi = (int)floorf(y) & 255;
        
        float xf = x - floorf(x);
        float yf = y - floorf(y);
        
        float u = fade(xf);
        float v = fade(yf);
        
        int p00 = perm[xi + perm[yi]];
        int p10 = perm[xi + 1 + perm[yi]];
        int p01 = perm[xi + perm[yi + 1]];
        int p11 = perm[xi + 1 + perm[yi + 1]];
        
        float n00 = grad(p00, xf, yf);
        float n10 = grad(p10, xf - 1, yf);
        float n01 = grad(p01, xf, yf - 1);
        float n11 = grad(p11, xf - 1, yf - 1);
        
        float nx0 = lerp(u, n00, n10);
        float nx1 = lerp(u, n01, n11);
        float nxy = lerp(v, nx0, nx1);
        
        return (nxy + 1.0f) * 0.5f;
    }
}

// FOR CPU RENDERING
class CheckerboardTexture : public Texture {
    private:
        Color color1;
        Color color2;
        float scale;
        bool use_discolor;
        float discolor_amount;

    public:
        // default colors are red & yellow, default scale is 1, discolor off by default
        CheckerboardTexture(const Color& c1 = Color(255, 0, 0), const Color& c2 = Color(255, 255, 0), float s = 1.0f)
            : color1(c1), color2(c2), scale(s), use_discolor(false), discolor_amount(0.2f) {}

        // override sample method in texture.h
        Color sample(const Point& world_pos, const Point& uv_coords, const Vec3& normal) const override {
            // determine checkerboard based on world pos & scale
            int x = static_cast<int>(std::floor(world_pos.getX() * scale));
            int z = static_cast<int>(std::floor(world_pos.getZ() * scale));
            
            Color result = ((x + z) % 2 == 0) ? color1 : color2;
            
            // apply discoloration if enabled - sample once per tile
            if (use_discolor) {
                // Use tile coordinates for noise so entire tile has same variation
                // Sample noise at different offsets for each channel to create hue shifts
                float noise_r = PerlinNoise::perlin((float)x * 0.5f, (float)z * 0.5f);
                float noise_g = PerlinNoise::perlin((float)x * 0.5f + 100.0f, (float)z * 0.5f);
                float noise_b = PerlinNoise::perlin((float)x * 0.5f, (float)z * 0.5f + 100.0f);
                
                // Add/subtract based on noise to shift hue (centered around 0)
                int shift_amount = (int)(discolor_amount * 120.0f); // scale to strong color shift
                int r_shift = (int)((noise_r - 0.5f) * 2.0f * shift_amount);
                int g_shift = (int)((noise_g - 0.5f) * 2.0f * shift_amount);
                int b_shift = (int)((noise_b - 0.5f) * 2.0f * shift_amount);
                
                result = Color(result.r + r_shift, result.g + g_shift, result.b + b_shift);
                result.clamp();
            }
            
            return result;
        }

        // getters
        const Color& getColor1() const { return color1; }
        const Color& getColor2() const { return color2; }
        float getScale() const { return scale; }
        bool getUseDiscolor() const { return use_discolor; }
        float getDiscolorAmount() const { return discolor_amount; }

        // setters
        void setColor1(const Color& c) { color1 = c; }
        void setColor2(const Color& c) { color2 = c; }
        void setScale(float s) { scale = s; }
        void setUseDiscolor(bool enable) { use_discolor = enable; }
        void setDiscolorAmount(float amount) { discolor_amount = amount; }  // 0.0-1.0
};

// FOR GPU RENDERING
struct CheckerboardTextureData : public TextureData {
    Color color1;
    Color color2;
    float scale;
    bool use_discolor;
    float discolor_amount;
    
    CheckerboardTextureData() : TextureData(TEX_CHECKERBOARD), color1(Color(255, 0, 0)), color2(Color(255, 255, 0)), scale(1.0f), use_discolor(false), discolor_amount(0.2f) {}
    
    CheckerboardTextureData(const Color& c1, const Color& c2, float s, bool discolor = false, float discolor_amt = 0.2f): 
        TextureData(TEX_CHECKERBOARD), color1(c1), color2(c2), scale(s), use_discolor(discolor), discolor_amount(discolor_amt) {}
};

// sampling function for checkerboard texture on gpu
#ifdef __CUDACC__
// gpu permutation table for perlin noise (must be in constant memory for device access)
__device__ __constant__ static const int perm_gpu[512] = {
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

// gpu vers of perlin noise helper functions
__host__ __device__ inline float fadeGPU(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

__host__ __device__ inline float lerpGPU(float t, float a, float b) {
    return a + t * (b - a);
}

__host__ __device__ inline float gradGPU(int hash, float x, float y) {
    int h = hash & 15;
    float u = (h < 8) ? x : y;
    float v = (h < 8) ? y : x;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

__host__ __device__ inline float perlinGPU(float x, float y) {
    int xi = (int)floorf(x) & 255;
    int yi = (int)floorf(y) & 255;
    
    float xf = x - floorf(x);
    float yf = y - floorf(y);
    
    float u = fadeGPU(xf);
    float v = fadeGPU(yf);
    
    int p00 = perm_gpu[xi + perm_gpu[yi]];
    int p10 = perm_gpu[xi + 1 + perm_gpu[yi]];
    int p01 = perm_gpu[xi + perm_gpu[yi + 1]];
    int p11 = perm_gpu[xi + 1 + perm_gpu[yi + 1]];
    
    float n00 = gradGPU(p00, xf, yf);
    float n10 = gradGPU(p10, xf - 1, yf);
    float n01 = gradGPU(p01, xf, yf - 1);
    float n11 = gradGPU(p11, xf - 1, yf - 1);
    
    float nx0 = lerpGPU(u, n00, n10);
    float nx1 = lerpGPU(u, n01, n11);
    float nxy = lerpGPU(v, nx0, nx1);
    
    return (nxy + 1.0f) * 0.5f;
}

inline __host__ __device__ Color sampleCheckerboardGPU(
    const CheckerboardTextureData& tex,
    const Point& world_pos)
{
    int x = static_cast<int>(floorf(world_pos.getX() * tex.scale));
    int z = static_cast<int>(floorf(world_pos.getZ() * tex.scale));
    
    Color result = ((x + z) % 2 == 0) ? tex.color1 : tex.color2;
    
    // apply discoloration if enabled - sample once per tile
    if (tex.use_discolor) {
        // Use tile coordinates for noise so entire tile has same variation
        // Sample noise at different offsets for each channel to create hue shifts
        float noise_r = perlinGPU((float)x * 0.5f, (float)z * 0.5f);
        float noise_g = perlinGPU((float)x * 0.5f + 100.0f, (float)z * 0.5f);
        float noise_b = perlinGPU((float)x * 0.5f, (float)z * 0.5f + 100.0f);
        
        // Add/subtract based on noise to shift hue (centered around 0)
        int shift_amount = (int)(tex.discolor_amount * 120.0f); // scale to strong color shift
        int r_shift = (int)((noise_r - 0.5f) * 2.0f * shift_amount);
        int g_shift = (int)((noise_g - 0.5f) * 2.0f * shift_amount);
        int b_shift = (int)((noise_b - 0.5f) * 2.0f * shift_amount);
        
        result = Color(
            result.r + r_shift,
            result.g + g_shift,
            result.b + b_shift
        );
        result.clamp();
    }
    
    return result;
}
#endif // __CUDACC__

#endif // CHECKERBOARD_H
