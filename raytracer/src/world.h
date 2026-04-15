#ifndef WORLD_H
#define WORLD_H

#include <vector>
#include <memory>
#include "components/color.h"
#include "components/material.h"
#include "components/light.h"
#include "components/illumination.h"
#include "object.h"
#include "objects/cube.h"
#include "objects/sphere.h"
#include "objects/triangle.h"

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

// forward declarations for GPU structs
#ifdef __CUDACC__
struct SphereGPU;
struct CubeGPU;
struct TriangleGPU;
#endif

// FOR CPU RENDERING
class World {
    private:
        std::vector<std::unique_ptr<Object>> objectList; // list of objs
        std::vector<std::unique_ptr<Light>> lightList; // list of lights
        Color ambientLight; // ambient light color/intensity

    public:
        // default constructor
        World() : ambientLight(Color(0, 0, 0)) {}

        // add object to world
        void addObject(std::unique_ptr<Object> obj) {
            objectList.push_back(std::move(obj));
        }

        // add light to world
        void addLight(std::unique_ptr<Light> light) {
            lightList.push_back(std::move(light));
        }

        // set ambient light
        void setAmbientLight(const Color &ambient) {
            ambientLight = ambient;
        }

        // getters
        const std::vector<std::unique_ptr<Object>>& getObjects() const { return objectList; }
        const std::vector<std::unique_ptr<Light>>& getLights() const { return lightList; }
        const Color& getAmbientLight() const { return ambientLight; }
};

// FOR GPU RENDERING (only available when compiling with CUDA)
#ifdef __CUDACC__
struct WorldGPU {
    Material* materials;
    int numMaterials;
    LightData* lights;
    int numLights;
    Color ambientLight;
    
    // GPU object arrays
    SphereGPU* spheres;
    int numSpheres;
    CubeGPU* cubes;
    int numCubes;
    TriangleGPU* triangles;
    int numTriangles;
    
    CUDA_CALLABLE WorldGPU() 
        : materials(nullptr), numMaterials(0), 
        lights(nullptr), numLights(0), 
        ambientLight(Color(0, 0, 0)),
        spheres(nullptr), numSpheres(0),
        cubes(nullptr), numCubes(0),
        triangles(nullptr), numTriangles(0) {}
};
#endif

#undef CUDA_CALLABLE

#endif // WORLD_H