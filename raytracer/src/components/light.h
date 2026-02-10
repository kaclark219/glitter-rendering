#ifndef LIGHT_H
#define LIGHT_H

#include "point.h"
#include "vec3.h"
#include "color.h"

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

class Light {
    private: 
        Point position; 
        Color color; 
        float intensity;

    public:
        // default constructor
        CUDA_CALLABLE Light() 
            : position(Point()), color(Color(255, 255, 255)), intensity(1.0f) {}

        // parameterized constructor
        CUDA_CALLABLE Light(const Point &pos, const Color &col, float inten)
            : position(pos), color(col), intensity(inten) {}
        
        // getters
        CUDA_CALLABLE Point getPosition() const { return position; }
        CUDA_CALLABLE Color getColor() const { return color; }
        CUDA_CALLABLE float getIntensity() const { return intensity; }

        // setters
        CUDA_CALLABLE void setPosition(const Point &pos) { position = pos; }
        CUDA_CALLABLE void setColor(const Color &col) { color = col; }
        CUDA_CALLABLE void setIntensity(float inten) { intensity = inten; }

        // virtual method to compute lighting at a point
        CUDA_CALLABLE virtual Color computeLighting(const Point &point, const Vec3 &normal, const Vec3 &view_dir) const {
            int r = static_cast<int>(color.r * intensity);
            int g = static_cast<int>(color.g * intensity);
            int b = static_cast<int>(color.b * intensity);
            return Color(r > 255 ? 255 : r, g > 255 ? 255 : g, b > 255 ? 255 : b);
        }
};

class PointLight : public Light {
    public:
        // default constructor
        CUDA_CALLABLE PointLight() : Light() {}

        // parameterized constructor
        CUDA_CALLABLE PointLight(const Point &pos, const Color &col, float inten)
            : Light(pos, col, inten) {}

        // override computeLighting to include distance attenuation (if needed)
        CUDA_CALLABLE Color computeLighting(const Point &point, const Vec3 &normal, const Vec3 &view_dir) const override {
            // Basic implementation without attenuation
            return Light::computeLighting(point, normal, view_dir);
        }
};

class DirectionalLight : public Light {
    private:
        Point direction; // normalized direction vector

    public:
        // default constructor
        CUDA_CALLABLE DirectionalLight() 
            : Light(), direction(Point(0.0f, -1.0f, 0.0f)) {}

        // parameterized constructor
        CUDA_CALLABLE DirectionalLight(const Point &dir, const Color &col, float inten)
            : Light(Point(), col, inten), direction(dir) {}

        // getter for direction
        CUDA_CALLABLE Point getDirection() const { return direction; }

        // setter for direction
        CUDA_CALLABLE void setDirection(const Point &dir) { direction = dir; }

        // override computeLighting to ignore position
        CUDA_CALLABLE Color computeLighting(const Point &point, const Vec3 &normal, const Vec3 &view_dir) const override {
            return Light::computeLighting(point, normal, view_dir);
        }
};

#endif // LIGHT_H