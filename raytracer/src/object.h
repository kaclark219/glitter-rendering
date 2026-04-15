#ifndef RAYTRACER_OBJECT_H
#define RAYTRACER_OBJECT_H

#include "components/point.h"
#include "components/vec3.h"
#include "components/color.h"
#include "components/ray.h"
#include "components/material.h"
#include "textures/texture.h"

// forward declaration for gpu texture data struct
#ifdef __CUDACC__
    struct TextureData;
#endif

class Object {
    protected:
        Material mat;
        Color color;
        Texture* texture;  // cpu texture pointer
        
    public:
        Object() : mat(Material()), color(Color()), texture(nullptr) {}
        Object(const Material &material, const Color &col = Color()) : mat(material), color(col), texture(nullptr) {}
        virtual ~Object() = default;

        // getters
        const Material& getMaterial() const { return mat; }
        const Color& getColor() const { return color; }
        const Texture* getTexture() const { return texture; }

        // setters
        void setMaterial(const Material &material) { mat = material; }
        void setColor(const Color &col) { color = col; }
        void setTexture(Texture* tex) { texture = tex; }

        // intersect ray (origin as point, direction as vec3), returns hit distance in t
        virtual bool intersect(const Ray& ray, float& t) const = 0;

        // surface normal (unit) at surface point p
        virtual Vec3 normal(const Point& p) const = 0;

        // texture coordinates at surface point p
        virtual Point getUV(const Point& p) const {
            (void)p;
            return Point(0.0f, 0.0f, 0.0f);
        }

        // build polygons from triangles
        
};

#endif // RAYTRACER_OBJECT_H
