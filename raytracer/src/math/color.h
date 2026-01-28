class Color {
    public:
        float r, g, b;

        // default constructor
        Color() : r(0), g(0), b(0) {}

        // three parameter constructor
        Color(float red, float green, float blue) : r(red), g(green), b(blue) {}

        // copy constructor
        Color(const Color &c) : r(c.r), g(c.g), b(c.b) {}
};