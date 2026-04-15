#ifndef TEXTURE_H
#define TEXTURE_H

#include "../components/color.h"
#include "../components/point.h"
#include "../components/vec3.h"

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

// FOR CPU RENDERING
class Texture {
    public:
        virtual ~Texture() = default;
        // virtual method to get color from texture
        virtual Color sample(const Point& world_pos, const Point& uv_coords, const Vec3& normal) const = 0;
};

// FOR GPU RENDERING
enum TextureType {
    TEX_CHECKERBOARD = 0,
    TEX_IMAGE = 1,
    TEX_NOISE = 2,
};

struct TextureData {
    TextureType type;
    
    CUDA_CALLABLE TextureData() : type(TEX_CHECKERBOARD) {}
    CUDA_CALLABLE TextureData(TextureType t) : type(t) {}
};

// helper function to get color from texture on gpu
CUDA_CALLABLE inline Color sampleTextureGPU(
    const TextureData* texture_data,
    const Point& world_pos,
    const Point& uv_coords,
    const Vec3& normal);

#endif // TEXTURE_H