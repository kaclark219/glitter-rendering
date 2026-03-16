#ifndef GLITTER_H
#define GLITTER_H

#include "../components/point.h"
#include "../components/vec3.h"
#include "../components/color.h"
#include "texture.h"

#include <cmath>
#include <cstdint>
#include <utility>

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

using UV = std::pair<float, float>;

struct Vec2 {
    float x;
    float y;

    CUDA_CALLABLE Vec2() : x(0.0f), y(0.0f) {}
    CUDA_CALLABLE Vec2(float px, float py) : x(px), y(py) {}

    CUDA_CALLABLE Vec2 operator+(const Vec2& v) const { return Vec2(x + v.x, y + v.y); }
    CUDA_CALLABLE Vec2 operator-(const Vec2& v) const { return Vec2(x - v.x, y - v.y); }
    CUDA_CALLABLE Vec2 operator*(float s) const { return Vec2(x * s, y * s); }
};

struct IVec2 {
    int x;
    int y;

    CUDA_CALLABLE IVec2() : x(0), y(0) {}
    CUDA_CALLABLE IVec2(int px, int py) : x(px), y(py) {}
};

struct GlitterParams {
    float scale;
    float radius_min;
    float radius_max;
    float feather;
    float shade_min;
    float shade_max;
    float height_min;
    float height_max;
    unsigned int seed;
};

struct FlakeSample {
    float mask; // 0..1
    float height; // height-field value (for bump)
    float shade; // multiplier on base_color
    float theta; // per-flake rotation (debug/use later)
};

// math helpers

CUDA_CALLABLE inline Vec2 floor2(const Vec2& p) {
    return Vec2(std::floor(p.x), std::floor(p.y));
}

CUDA_CALLABLE inline Vec2 frac2(const Vec2& p) {
    return Vec2(p.x - std::floor(p.x), p.y - std::floor(p.y));
}

CUDA_CALLABLE inline float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

CUDA_CALLABLE inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

CUDA_CALLABLE inline float smoothstep(float edge0, float edge1, float x) {
    if (x <= edge0) return 0.0f;
    if (x >= edge1) return 1.0f;
    float t = (x - edge0) / (edge1 - edge0);
    return t * t * (3.0f - 2.0f * t);
}

CUDA_CALLABLE inline Vec2 rotate2d(const Vec2& p, float theta) {
    float c = std::cos(theta);
    float s = std::sin(theta);
    return Vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

CUDA_CALLABLE inline float dot3(const Vec3& a, const Vec3& b) {
    return a.getX() * b.getX() + a.getY() * b.getY() + a.getZ() * b.getZ();
}

CUDA_CALLABLE inline Vec3 cross3(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.getY() * b.getZ() - a.getZ() * b.getY(),
        a.getZ() * b.getX() - a.getX() * b.getZ(),
        a.getX() * b.getY() - a.getY() * b.getX()
    );
}

CUDA_CALLABLE inline Vec3 mul3(const Vec3& v, float s) {
    return Vec3(v.getX() * s, v.getY() * s, v.getZ() * s);
}

CUDA_CALLABLE inline Vec3 add3(const Vec3& a, const Vec3& b) {
    return Vec3(a.getX() + b.getX(), a.getY() + b.getY(), a.getZ() + b.getZ());
}

CUDA_CALLABLE inline Vec3 sub3(const Vec3& a, const Vec3& b) {
    return Vec3(a.getX() - b.getX(), a.getY() - b.getY(), a.getZ() - b.getZ());
}

CUDA_CALLABLE inline Vec3 normalize_safe3(const Vec3& v) {
    float len2 = dot3(v, v);
    if (len2 <= 1e-20f) return Vec3(0.0f, 1.0f, 0.0f);
    return mul3(v, 1.0f / std::sqrt(len2));
}

CUDA_CALLABLE inline void build_tangent_basis(const Vec3& n, Vec3& T, Vec3& B) {
    // Pick a helper axis not parallel to n
    Vec3 a = (std::fabs(n.getY()) < 0.999f) ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
    T = normalize_safe3(cross3(n, a));
    B = normalize_safe3(cross3(n, T));
}

CUDA_CALLABLE inline uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

CUDA_CALLABLE inline float rand01(const IVec2& cell, uint32_t seed) {
    // stable even for negative coords (wraps in uint32)
    uint32_t ux = static_cast<uint32_t>(cell.x) ^ 0xa24baedcu;
    uint32_t uy = static_cast<uint32_t>(cell.y) ^ 0x9fb21c65u;

    uint32_t h = ux * 0x45d9f3bu;
    h ^= uy * 0x2710f99du;
    h ^= seed * 0x94d049bbu;
    h ^= 0x9e3779b9u;

    h = hash_u32(h);
    // [0,1)
    return static_cast<float>(h) * (1.0f / 4294967296.0f);
}


CUDA_CALLABLE inline float diamond_sdf(const Vec2& p, float radius) {
    // diamond centered at origin: |x| + |y| = radius
    return std::abs(p.x) + std::abs(p.y) - radius;
}

CUDA_CALLABLE inline FlakeSample eval_cell_flake(const Vec2& ps, const IVec2& cell, const GlitterParams& params) {
    const float TWO_PI = 6.28318530717958647692f;

    // ps is in lattice space; cell coords are in that same space
    Vec2 center(static_cast<float>(cell.x) + 0.5f, static_cast<float>(cell.y) + 0.5f);
    Vec2 f = ps - center; // local coords around the cell center

    // per-cell rotation
    float theta = rand01(cell, params.seed + 0u) * TWO_PI;
    Vec2 fr = rotate2d(f, theta);

    // per-cell radius
    float radius_t = rand01(cell, params.seed + 1u);
    float radius = lerp(params.radius_min, params.radius_max, radius_t);

    // sdf + soft mask
    float d = diamond_sdf(fr, radius);
    float mask = 1.0f - smoothstep(0.0f, params.feather, d);
    mask = clamp01(mask);

    // per-cell shade (greyscale multiplier)
    float shade_t = rand01(cell, params.seed + 2u);
    float shade = lerp(params.shade_min, params.shade_max, shade_t);

    // height field (inside-only ramp), good for finite-difference bump
    float height_t = rand01(cell, params.seed + 3u);
    float base_height = lerp(params.height_min, params.height_max, height_t);

    // inside ramp: t=1 deeper inside, t=0 at boundary/outside
    float t = clamp01((-d) / radius);
    // slightly sharper profile
    float profile = t * t;

    float height = base_height * profile; // already 0 outside

    FlakeSample s;
    s.mask = mask;
    s.height = height;
    s.shade = shade;
    s.theta = theta;
    return s;
}

CUDA_CALLABLE inline FlakeSample sample_glitter(const UV& uv, const GlitterParams& params) {
    Vec2 p(uv.first * params.scale, uv.second * params.scale);

    Vec2 ps(p.x + 0.5f * p.y, 0.8660254f * p.y);

    Vec2 cellF = floor2(ps);
    IVec2 cell(static_cast<int>(cellF.x), static_cast<int>(cellF.y));

    FlakeSample best;
    best.mask = 0.0f;
    best.height = 0.0f;
    best.shade = 1.0f;
    best.theta = 0.0f;

    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            IVec2 ncell(cell.x + ox, cell.y + oy);
            FlakeSample s = eval_cell_flake(ps, ncell, params);
            if (s.mask > best.mask) best = s;
        }
    }

    return best;
}

CUDA_CALLABLE inline FlakeSample sample_glitter_default(const UV& uv) {
    GlitterParams params;
    params.scale = 96.0f;
    params.radius_min = 0.48f;
    params.radius_max = 0.60f;
    params.feather = 0.01f;
    params.shade_min = 0.75f;
    params.shade_max = 1.00f;
    params.height_min = 0.005f;
    params.height_max = 0.03f;
    params.seed = 12345u;

    return sample_glitter(uv, params);
}


CUDA_CALLABLE inline Vec3 render_flake_mask(const UV& uv, const GlitterParams& params) {
    FlakeSample s = sample_glitter(uv, params);
    return Vec3(s.mask, s.mask, s.mask);
}

CUDA_CALLABLE inline Vec3 render_flake_height(const UV& uv, const GlitterParams& params) {
    FlakeSample s = sample_glitter(uv, params);
    float denom = (params.height_max > 0.0f) ? params.height_max : 1.0f;
    float h = clamp01(s.height / denom);
    return Vec3(h, h, h);
}

CUDA_CALLABLE inline Vec3 render_dominant_rotation_hue(const UV& uv, const GlitterParams& params) {
    FlakeSample s = sample_glitter(uv, params);
    float hue = s.theta * (1.0f / 6.28318530717958647692f);
    
    float h = hue * 6.0f;
    float x = 1.0f - std::abs(std::fmod(h, 2.0f) - 1.0f);
    
    float r = 0.0f, g = 0.0f, b = 0.0f;
    if (h < 1.0f) { r = 1.0f; g = x; }
    else if (h < 2.0f) { r = x; g = 1.0f; }
    else if (h < 3.0f) { g = 1.0f; b = x; }
    else if (h < 4.0f) { g = x; b = 1.0f; }
    else if (h < 5.0f) { r = x; b = 1.0f; }
    else { r = 1.0f; b = x; }
    
    return Vec3(r, g, b);
}


class GlitterTexture : public Texture {
private:
    GlitterParams params;
    Color base_color;
    int vis_mode; // 0=normal, 1=mask, 2=height, 3=rotation_hue

    CUDA_CALLABLE UV select_uv(const Point& world_pos, const Point& uv_coords, bool use_uv) const {
        if (use_uv) {
            return UV(uv_coords.getX(), uv_coords.getY());
        }
        // planar fallback mapping
        return UV(world_pos.getX(), world_pos.getZ());
    }

public:
    GlitterTexture(const Color& c = Color(200, 200, 200), float scale = 96.0f)
        : base_color(c), vis_mode(0)
    {
        params.scale = scale;
        params.radius_min = 0.48f;
        params.radius_max = 0.60f;
        params.feather = 0.01f;
        params.shade_min = 0.75f;
        params.shade_max = 1.00f;
        params.height_min = 0.005f;
        params.height_max = 0.03f;
        params.seed = 12345u;
    }

    CUDA_CALLABLE FlakeSample sampleFlake(const Point& world_pos, const Point& uv_coords, bool use_uv) const {
        UV uv = select_uv(world_pos, uv_coords, use_uv);
        return sample_glitter(uv, params);
    }

    CUDA_CALLABLE const GlitterParams& getParams() const {
        return params;
    }

    void setVisualizationMode(int mode) {
        vis_mode = mode;
    }

    CUDA_CALLABLE float sampleHeight(const Point& world_pos, const Point& uv_coords, bool use_uv) const {
        UV uv = select_uv(world_pos, uv_coords, use_uv);
        return sample_glitter(uv, params).height;
    }

    CUDA_CALLABLE Vec3 bump_normal(const Point& world_pos, const Point& uv_coords, const Vec3& geom_normal, float strength = 1.0f, float eps = (1.0f / 512.0f)) const {
        bool use_uv = (uv_coords.getX() != 0.0f || uv_coords.getY() != 0.0f);

        UV uv0 = select_uv(world_pos, uv_coords, use_uv);

        UV uvp(uv0.first + eps, uv0.second);
        UV uvm(uv0.first - eps, uv0.second);
        UV vvp(uv0.first, uv0.second + eps);
        UV vvm(uv0.first, uv0.second - eps);

        float hup = sample_glitter(uvp, params).height;
        float hum = sample_glitter(uvm, params).height;
        float hvp = sample_glitter(vvp, params).height;
        float hvm = sample_glitter(vvm, params).height;

        float dhdu = (hup - hum) / (2.0f * eps);
        float dhdv = (hvp - hvm) / (2.0f * eps);

        Vec3 T, B;
        build_tangent_basis(geom_normal, T, B);

        // n' = normalize(n - strength*(dh/du*T + dh/dv*B))
        Vec3 grad = add3(mul3(T, dhdu * strength), mul3(B, dhdv * strength));
        Vec3 bumped = normalize_safe3(sub3(geom_normal, grad));
        return bumped;
    }
    
    Color sample(const Point& world_pos, const Point& uv_coords, const Vec3& normal) const override {
        (void)normal;
        bool use_uv = (uv_coords.getX() != 0.0f || uv_coords.getY() != 0.0f);
        UV uv = select_uv(world_pos, uv_coords, use_uv);
        
        switch (vis_mode) {
            case 1: {
                // VIS_MASK: show flake coverage
                Vec3 c = render_flake_mask(uv, params);
                return Color(static_cast<int>(c.getX() * 255.0f),
                            static_cast<int>(c.getY() * 255.0f),
                            static_cast<int>(c.getZ() * 255.0f));
            }
            case 2: {
                // VIS_HEIGHT: show height field (for bump mapping)
                Vec3 c = render_flake_height(uv, params);
                return Color(static_cast<int>(c.getX() * 255.0f),
                            static_cast<int>(c.getY() * 255.0f),
                            static_cast<int>(c.getZ() * 255.0f));
            }
            case 3: {
                // VIS_ROTATION_HUE: show per-flake rotation as color
                Vec3 c = render_dominant_rotation_hue(uv, params);
                return Color(static_cast<int>(c.getX() * 255.0f),
                            static_cast<int>(c.getY() * 255.0f),
                            static_cast<int>(c.getZ() * 255.0f));
            }
            case 0:
            default: {
                // VIS_NORMAL: actual glitter appearance with bump mapping
                FlakeSample flake = sampleFlake(world_pos, uv_coords, use_uv);
                float s = clamp01(flake.shade);
                return base_color * s;
            }
        }
    }
};

#endif // GLITTER_H