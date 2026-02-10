#ifndef RAYTRACER_OBJECT_H
#define RAYTRACER_OBJECT_H

#include "components/point.h"
#include "components/vec3.h"
#include "components/color.h"
#include "components/ray.h"
#include "components/material.h"

class Object {
    protected:
        Material mat;
        Color color;
    public:
        Object() : mat(Material()), color(Color()) {}
        Object(const Material &material, const Color &col = Color()) : mat(material), color(col) {}
        virtual ~Object() = default;

        // getters
        const Material& getMaterial() const { return mat; }
        const Color& getColor() const { return color; }

        // setters
        void setMaterial(const Material &material) { mat = material; }
        void setColor(const Color &col) { color = col; }

        // intersect ray (origin as Point, direction as Vec3), returns hit distance in t
        virtual bool intersect(const Ray& ray, float& t) const = 0;

        // surface normal (unit) at surface point p
        virtual Vec3 normal(const Point& p) const = 0;

        // build polygons from triangles
        
};

#endif // RAYTRACER_OBJECT_H

// need to add object list for k-d trees