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
    float radius_min_a;
    float radius_max_a;
    float radius_min_b;
    float radius_max_b;
    float style_b_frequency;
    float feather;
    float shade_min;
    float shade_max;
    float sparkle_min;
    float sparkle_max;
    float base_mix;
    float jitter;
    float hue_variation;
    float value_variation;
    float hex_ratio_a;
    float hex_ratio_b;
    float height_min;
    float height_max;
    unsigned int seed;
};

struct FlakeSample {
    float mask; // 0..1
    float height; // height-field value (for bump)
    float shade; // multiplier on base_color
    float sparkle; // bright flake highlight multiplier
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

CUDA_CALLABLE inline float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

CUDA_CALLABLE inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

CUDA_CALLABLE inline float fractf(float x) {
    return x - std::floor(x);
}

CUDA_CALLABLE inline float smoothstep(float edge0, float edge1, float x) {
    if (x <= edge0) return 0.0f;
    if (x >= edge1) return 1.0f;
    float t = (x - edge0) / (edge1 - edge0);
    return t * t * (3.0f - 2.0f * t);
}

CUDA_CALLABLE inline Color color_from_unit(float r, float g, float b) {
    return Color(
        static_cast<int>(clamp01(r) * 255.0f),
        static_cast<int>(clamp01(g) * 255.0f),
        static_cast<int>(clamp01(b) * 255.0f)
    );
}

CUDA_CALLABLE inline Vec3 hsv_to_rgb(float h, float s, float v) {
    float hh = fractf(h) * 6.0f;
    float c = v * s;
    float x = c * (1.0f - std::fabs(std::fmod(hh, 2.0f) - 1.0f));
    float m = v - c;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    if (hh < 1.0f) { r = c; g = x; b = 0.0f; }
    else if (hh < 2.0f) { r = x; g = c; b = 0.0f; }
    else if (hh < 3.0f) { r = 0.0f; g = c; b = x; }
    else if (hh < 4.0f) { r = 0.0f; g = x; b = c; }
    else if (hh < 5.0f) { r = x; g = 0.0f; b = c; }
    else { r = c; g = 0.0f; b = x; }

    return Vec3(r + m, g + m, b + m);
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

CUDA_CALLABLE inline float hash11(float x) {
    return fractf(std::sin(x) * 43758.5453123f);
}

CUDA_CALLABLE inline Vec2 warp2(const Vec2& p, unsigned int seed, float amount) {
    float wx = hash11(p.x * 12.9898f + p.y * 78.233f + static_cast<float>(seed));
    float wy = hash11(p.x * 39.3468f + p.y * 11.135f + static_cast<float>(seed) * 1.37f);
    return Vec2(
        p.x + (wx - 0.5f) * amount,
        p.y + (wy - 0.5f) * amount
    );
}


CUDA_CALLABLE inline float circle_sdf(const Vec2& p, float radius) {
    return std::sqrt(p.x * p.x + p.y * p.y) - radius;
}

CUDA_CALLABLE inline float hexagon_sdf(const Vec2& p, float radius) {
    // Regular hexagon SDF adapted for a point-centered flake mask.
    const float kx = -0.8660254f;
    const float ky = 0.5f;
    const float kz = 0.57735027f;

    float px = std::fabs(p.x);
    float py = std::fabs(p.y);
    float dotp = kx * px + ky * py;
    float m = (dotp < 0.0f) ? dotp : 0.0f;
    px -= 2.0f * m * kx;
    py -= 2.0f * m * ky;
    px -= clampf(px, -kz * radius, kz * radius);
    py -= radius;

    float len = std::sqrt(px * px + py * py);
    float sign = (py > 0.0f) ? 1.0f : -1.0f;
    return len * sign;
}

CUDA_CALLABLE inline FlakeSample eval_cell_flake(const Vec2& ps, const IVec2& cell, const GlitterParams& params) {
    const float TWO_PI = 6.28318530717958647692f;

    // ps is in lattice space; cell coords are in that same space
    float jitterX = (rand01(cell, params.seed + 10u) - 0.5f) * params.jitter;
    float jitterY = (rand01(cell, params.seed + 11u) - 0.5f) * params.jitter;
    Vec2 center(
        static_cast<float>(cell.x) + 0.5f + jitterX * 0.45f,
        static_cast<float>(cell.y) + 0.5f + jitterY * 0.45f
    );
    Vec2 f = ps - center; // local coords around the cell center

    // per-cell rotation
    float theta = rand01(cell, params.seed + 0u) * TWO_PI;
    Vec2 fr = rotate2d(f, theta);

    // per-cell radius
    float stylePick = rand01(cell, params.seed + 12u);
    bool useStyleB = stylePick < params.style_b_frequency;
    float radius_t = rand01(cell, params.seed + 1u);
    float radius = useStyleB
        ? lerp(params.radius_min_b, params.radius_max_b, radius_t)
        : lerp(params.radius_min_a, params.radius_max_a, radius_t);
    float hexRatio = useStyleB ? params.hex_ratio_b : params.hex_ratio_a;

    // blend between round and hexagonal flakes
    float circleD = circle_sdf(fr, radius);
    float hexD = hexagon_sdf(fr, radius);
    float d = lerp(circleD, hexD, hexRatio);
    float mask = 1.0f - smoothstep(0.0f, params.feather, d);
    mask = clamp01(mask);

    // per-cell shade (greyscale multiplier)
    float shade_t = rand01(cell, params.seed + 2u);
    float shade = lerp(params.shade_min, params.shade_max, shade_t);

    float sparkle_t = rand01(cell, params.seed + 4u);
    float sparkle = lerp(params.sparkle_min, params.sparkle_max, sparkle_t);

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
    s.sparkle = sparkle;
    s.theta = theta;
    return s;
}

CUDA_CALLABLE inline FlakeSample blend_flake_samples(const FlakeSample& a, float wa, const FlakeSample& b, float wb, const FlakeSample& c, float wc) {
    float sum = wa + wb + wc;
    if (sum <= 1e-6f) {
        return a;
    }

    float inv = 1.0f / sum;
    FlakeSample out;
    out.mask = (a.mask * wa + b.mask * wb + c.mask * wc) * inv;
    out.height = (a.height * wa + b.height * wb + c.height * wc) * inv;
    out.shade = (a.shade * wa + b.shade * wb + c.shade * wc) * inv;
    out.sparkle = (a.sparkle * wa + b.sparkle * wb + c.sparkle * wc) * inv;
    out.theta = (a.theta * wa + b.theta * wb + c.theta * wc) * inv;
    return out;
}

CUDA_CALLABLE inline FlakeSample sample_glitter(const UV& uv, const GlitterParams& params) {
    Vec2 p(uv.first * params.scale, uv.second * params.scale);
    p = warp2(p, params.seed + 20u, 0.65f);

    Vec2 ps(p.x + 0.5f * p.y, 0.8660254f * p.y);

    Vec2 cellF = floor2(ps);
    IVec2 cell(static_cast<int>(cellF.x), static_cast<int>(cellF.y));

    FlakeSample best;
    best.mask = 0.0f;
    best.height = 0.0f;
    best.shade = 1.0f;
    best.sparkle = 0.0f;
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

CUDA_CALLABLE inline FlakeSample sample_glitter_triplanar(const Point& world_pos, const Vec3& normal, const GlitterParams& params) {
    Vec3 nabs(std::fabs(normal.getX()), std::fabs(normal.getY()), std::fabs(normal.getZ()));
    float wx = std::pow(nabs.getX(), 3.0f) + 1e-4f;
    float wy = std::pow(nabs.getY(), 3.0f) + 1e-4f;
    float wz = std::pow(nabs.getZ(), 3.0f) + 1e-4f;

    FlakeSample sx = sample_glitter(
        UV(world_pos.getY() * 0.0105f + 17.31f, world_pos.getZ() * 0.0105f - 9.73f),
        params
    );
    FlakeSample sy = sample_glitter(
        UV(world_pos.getX() * 0.0105f - 5.17f, world_pos.getZ() * 0.0105f + 13.91f),
        params
    );
    FlakeSample sz = sample_glitter(
        UV(world_pos.getX() * 0.0105f + 21.43f, world_pos.getY() * 0.0105f - 14.29f),
        params
    );

    return blend_flake_samples(sx, wx, sy, wy, sz, wz);
}

CUDA_CALLABLE inline FlakeSample sample_glitter_default(const UV& uv) {
    GlitterParams params;
    params.scale = 22.0f;
    params.radius_min_a = 0.40f;
    params.radius_max_a = 0.72f;
    params.radius_min_b = 0.70f;
    params.radius_max_b = 1.18f;
    params.style_b_frequency = 0.30f;
    params.feather = 0.06f;
    params.shade_min = 0.86f;
    params.shade_max = 1.05f;
    params.sparkle_min = 0.85f;
    params.sparkle_max = 1.40f;
    params.base_mix = 0.52f;
    params.jitter = 1.0f;
    params.hue_variation = 0.02f;
    params.value_variation = 0.18f;
    params.hex_ratio_a = 0.75f;
    params.hex_ratio_b = 0.35f;
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
    Color tint_color;
    int vis_mode; // 0=normal, 1=mask, 2=height, 3=rotation_hue

    CUDA_CALLABLE UV select_uv(const Point& world_pos, const Point& uv_coords, bool use_uv) const {
        if (use_uv) {
            return UV(uv_coords.getX(), uv_coords.getY());
        }
        // planar fallback mapping
        return UV(world_pos.getX(), world_pos.getZ());
    }

    CUDA_CALLABLE FlakeSample sampleProjectedFlake(const Point& world_pos, const Point& uv_coords, const Vec3& normal) const {
        bool use_uv = (uv_coords.getX() != 0.0f || uv_coords.getY() != 0.0f);
        if (use_uv && std::fabs(normal.getY()) > 0.92f) {
            UV uv = select_uv(world_pos, uv_coords, true);
            return sample_glitter(uv, params);
        }
        return sample_glitter_triplanar(world_pos, normal, params);
    }

public:
    GlitterTexture(const Color& c = Color(218, 223, 230), float scale = 22.0f)
        : base_color(c), tint_color(c), vis_mode(0)
    {
        params.scale = scale;
        params.radius_min_a = 0.40f;
        params.radius_max_a = 0.72f;
        params.radius_min_b = 0.70f;
        params.radius_max_b = 1.18f;
        params.style_b_frequency = 0.30f;
        params.feather = 0.07f;
        params.shade_min = 0.78f;
        params.shade_max = 1.10f;
        params.sparkle_min = 1.00f;
        params.sparkle_max = 2.55f;
        params.base_mix = 0.55f;
        params.jitter = 1.35f;
        params.hue_variation = 0.03f;
        params.value_variation = 0.28f;
        params.hex_ratio_a = 0.92f;
        params.hex_ratio_b = 0.60f;
        params.height_min = 0.008f;
        params.height_max = 0.04f;
        params.seed = 12345u;
    }

    CUDA_CALLABLE FlakeSample sampleFlake(const Point& world_pos, const Point& uv_coords, const Vec3& normal) const {
        return sampleProjectedFlake(world_pos, uv_coords, normal);
    }

    CUDA_CALLABLE const GlitterParams& getParams() const {
        return params;
    }

    void setBaseColor(const Color& c) {
        base_color = c;
    }

    void setTintColor(const Color& c) {
        tint_color = c;
    }

    CUDA_CALLABLE const Color& getBaseColor() const {
        return base_color;
    }

    CUDA_CALLABLE const Color& getTintColor() const {
        return tint_color;
    }

    void setVisualizationMode(int mode) {
        vis_mode = mode;
    }

    CUDA_CALLABLE float sampleHeight(const Point& world_pos, const Point& uv_coords, const Vec3& normal) const {
        return sampleProjectedFlake(world_pos, uv_coords, normal).height;
    }

    CUDA_CALLABLE Vec3 bump_normal(const Point& world_pos, const Point& uv_coords, const Vec3& geom_normal, float strength = 1.0f, float eps = (1.0f / 512.0f)) const {
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

        float dhdu = (sampleHeight(pu, uv_coords, geom_normal) - sampleHeight(mu, uv_coords, geom_normal)) / (2.0f * eps);
        float dhdv = (sampleHeight(pv, uv_coords, geom_normal) - sampleHeight(mv, uv_coords, geom_normal)) / (2.0f * eps);

        // n' = normalize(n - strength*(dh/du*T + dh/dv*B))
        Vec3 grad = add3(mul3(T, dhdu * strength), mul3(B, dhdv * strength));
        Vec3 bumped = normalize_safe3(sub3(geom_normal, grad));
        return bumped;
    }
    
    Color sample(const Point& world_pos, const Point& uv_coords, const Vec3& normal) const override {
        switch (vis_mode) {
            case 1: {
                // VIS_MASK: dark background, bright flake interiors, warm rims for boundaries
                FlakeSample flake = sampleProjectedFlake(world_pos, uv_coords, normal);
                float fill = clamp01(flake.mask);
                float rim = smoothstep(0.08f, 0.35f, fill) * (1.0f - smoothstep(0.55f, 0.92f, fill));
                float core = smoothstep(0.70f, 0.98f, fill);
                float base = 0.05f + fill * 0.18f;

                return color_from_unit(
                    base + rim * 0.75f + core * 0.25f,
                    base + rim * 0.38f + core * 0.88f,
                    base + core * 0.95f
                );
            }
            case 2: {
                // VIS_HEIGHT: blue lowlands to bright peaks for bump profile
                float denom = (params.height_max > 0.0f) ? params.height_max : 1.0f;
                FlakeSample flake = sampleProjectedFlake(world_pos, uv_coords, normal);
                float h = clamp01(flake.height / denom);
                float plateau = smoothstep(0.55f, 0.95f, h);
                float ridge = smoothstep(0.12f, 0.65f, h) * (1.0f - plateau);

                return color_from_unit(
                    0.06f + h * 0.92f,
                    0.10f + ridge * 0.78f + plateau * 0.82f,
                    0.16f + (1.0f - h) * 0.62f + ridge * 0.18f
                );
            }
            case 3: {
                // VIS_ROTATION_HUE: hue = orientation, value = sparkle strength, masked to flakes
                FlakeSample flake = sampleProjectedFlake(world_pos, uv_coords, normal);
                float hue = flake.theta * (1.0f / 6.28318530717958647692f);
                float sat = 0.78f;
                float sparkleNorm = clamp01(
                    (flake.sparkle - params.sparkle_min) /
                    std::max(params.sparkle_max - params.sparkle_min, 1e-6f)
                );
                float value = 0.28f + sparkleNorm * 0.72f;
                Vec3 hueColor = hsv_to_rgb(hue, sat, value);
                float mask = smoothstep(0.10f, 0.85f, clamp01(flake.mask));
                float bg = 0.05f;

                return color_from_unit(
                    bg + hueColor.getX() * mask,
                    bg + hueColor.getY() * mask,
                    bg + hueColor.getZ() * mask
                );
            }
            case 0:
            default: {
                // VIS_NORMAL: tint-driven glitter base with silver glints on top
                FlakeSample flake = sampleProjectedFlake(world_pos, uv_coords, normal);
                float baseStrength = lerp(params.base_mix, params.base_mix + 0.25f, clamp01(flake.shade));
                float flakeCore = clamp01(flake.mask * flake.mask);
                float sparkleStrength = flakeCore * flake.sparkle * 0.62f;
                Color metallicBase = (base_color * 0.28f) + (tint_color * 0.92f);
                Color baseLayer = metallicBase * baseStrength;

                float hueShift = (flake.theta * (1.0f / 6.28318530717958647692f) - 0.5f) * params.hue_variation;
                float value = clamp01(0.90f + flake.sparkle * params.value_variation);
                Vec3 flakeTint = hsv_to_rgb(0.58f + hueShift, 0.08f, value);
                Color silverGlint(
                    static_cast<int>(flakeTint.getX() * 255.0f),
                    static_cast<int>(flakeTint.getY() * 255.0f),
                    static_cast<int>(flakeTint.getZ() * 255.0f)
                );
                Color tintedGlint = tint_color * 0.88f;
                Color flakeLayer = (tintedGlint * 0.74f) + (silverGlint * 0.26f);

                return baseLayer + (flakeLayer * sparkleStrength);
            }
        }
    }
};

#endif // GLITTER_H
