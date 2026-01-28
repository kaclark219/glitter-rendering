#include "image.h"
#include <fstream>
#include <algorithm>

static inline unsigned char clamp_u8(int v) {
    if (v <= 0) return 0;
    if (v >= 255) return 255;
    return static_cast<unsigned char>(v);
}

bool Image::writePPM(const std::string &filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) return false;

    ofs << "P6\n" << width << " " << height << "\n255\n";

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const Color &c = pixels[y * width + x];
            unsigned char rgb[3] = { clamp_u8(c.r), clamp_u8(c.g), clamp_u8(c.b) };
            ofs.write(reinterpret_cast<char *>(rgb), 3);
        }
    }

    return true;
}
