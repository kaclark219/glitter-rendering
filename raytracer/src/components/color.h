#ifndef RAYTRACER_COLOR_H
#define RAYTRACER_COLOR_H

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

class Color {
    public:
        float r, g, b;

        // default constructor
        CUDA_CALLABLE Color() : r(0), g(0), b(0) {}

        // three parameter constructor
        CUDA_CALLABLE Color(int red, int green, int blue)
            : r(static_cast<float>(red)), g(static_cast<float>(green)), b(static_cast<float>(blue)) {}

        // use default copy constructor/assignment
        CUDA_CALLABLE Color(const Color &c) = default;
        CUDA_CALLABLE Color& operator=(const Color &c) = default;

        // multiplication operator for scaling color by a float
        CUDA_CALLABLE Color operator*(float scalar) const {
            return Color(static_cast<int>(r * scalar), static_cast<int>(g * scalar), static_cast<int>(b * scalar));
        }

        // addition operator for adding two colors
        CUDA_CALLABLE Color operator+(const Color &other) const {
            return Color(static_cast<int>(r + other.r), static_cast<int>(g + other.g), static_cast<int>(b + other.b));
        }

        // clamp method to ensure color values are within 0-255
        CUDA_CALLABLE void clamp() {
            r = (r > 255.0f) ? 255.0f : (r < 0.0f) ? 0.0f : r;
            g = (g > 255.0f) ? 255.0f : (g < 0.0f) ? 0.0f : g;
            b = (b > 255.0f) ? 255.0f : (b < 0.0f) ? 0.0f : b;
        }

        // multiplication operator for component-wise color multiplication
        CUDA_CALLABLE Color operator*(const Color &other) const {
            // Normalize to 0-1, multiply, then scale back to 0-255
            float nr = (r / 255.0f) * (other.r / 255.0f) * 255.0f;
            float ng = (g / 255.0f) * (other.g / 255.0f) * 255.0f;
            float nb = (b / 255.0f) * (other.b / 255.0f) * 255.0f;
            return Color(static_cast<int>(nr), static_cast<int>(ng), static_cast<int>(nb));
        }
};

#undef CUDA_CALLABLE

#endif // RAYTRACER_COLOR_H