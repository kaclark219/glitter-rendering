#include "image/image.h"
#include <string>
#include <cstdlib>
#include <cmath>

int run_render_cpu_example() {
	const int W = 800;
	const int H = 600;
	Image img(W, H);

	Color white(1.0f, 1.0f, 1.0f);
	for (int y = 0; y < H; ++y) {
		for (int x = 0; x < W; ++x) {
			img.setPixel(x, y, white);
		}
	}

	std::string out = "output.ppm";
	if (!img.writePPM(out)) return 1;
	return 0;
}

