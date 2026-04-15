#ifndef RAYTRACER_IMAGE_H
#define RAYTRACER_IMAGE_H

#include <vector>
#include <string>
#include <cstdint>
#include "../components/color.h"

class Image {
    private:
        int width;
        int height;
        std::vector<Color> pixels;

    public:
        // default constructor
        Image(int w = 0, int h = 0) : width(w), height(h), pixels(w * h) {}

        // constructor with three parameters
        Image(int w, int h, const Color &bgColor) : width(w), height(h), pixels(w * h, bgColor) {}

        // getters
        Color getPixel(int x, int y) const {
            if (x < 0 || x >= width || y < 0 || y >= height) return Color();
            return pixels[y * width + x];
        }
        int getWidth() const { return width; }
        int getHeight() const { return height; }

        // setters
        void setPixel(int x, int y, const Color &c) {
            if (x < 0 || x >= width || y < 0 || y >= height) return;
            pixels[y * width + x] = c;
        }

        bool writePPM(const std::string &filename) const;
};

#endif //RAYTRACER_IMAGE_H