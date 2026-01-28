#ifndef RAYTRACER_OBJECT_H
#define RAYTRACER_OBJECT_H

#include "math/point.h"
#include "math/vec3.h"
#include "math/color.h"
#include "math/ray.h"

class Object {
    protected:
        int materialIndex;
        Color color;
    public:
        Object() : materialIndex(-1), color(Color()) {}
        Object(int matIndex, const Color &col = Color()) : materialIndex(matIndex), color(col) {}
        virtual ~Object() = default;

        // getters
        int getMaterialIndex() const { return materialIndex; }
        const Color& getColor() const { return color; }

        // setters
        void setMaterialIndex(int matIndex) { materialIndex = matIndex; }
        void setColor(const Color &col) { color = col; }

        // intersect ray (origin as Point, direction as Vec3), returns hit distance in t
        virtual bool intersect(const Ray& ray, float& t) const = 0;

        // surface normal (unit) at surface point p
        virtual Vec3 normal(const Point& p) const = 0;

        // build polygons from triangles
        
};

#endif // RAYTRACER_OBJECT_H