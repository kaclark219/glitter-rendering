#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

class Material {
    private:
        Color ka; // ambient color
        Color kd; // diffuse color
        Color ks; // specular color
        float shininess; // shininess coefficient
        float reflectivity; // reflectivity coefficient
        float transmission; // transmission coefficient (for refraction)

    public:
        // default constructor
        CUDA_CALLABLE Material()
            : ka(Color(0, 0, 0)), kd(Color(0, 0, 0)), ks(Color(0, 0, 0)), shininess(0.0f), reflectivity(0.0f), transmission(0.0f) {}

        // Parameterized constructor
        CUDA_CALLABLE Material(const Color &ambient, const Color &diffuse, const Color &specular, float shiny, float reflect, float transmit = 0.0f)
            : ka(ambient), kd(diffuse), ks(specular), shininess(shiny), reflectivity(reflect), transmission(transmit) {}

        // getters
        CUDA_CALLABLE Color getAmbient() const { return ka; }
        CUDA_CALLABLE Color getDiffuse() const { return kd; }
        CUDA_CALLABLE Color getSpecular() const { return ks; }
        CUDA_CALLABLE float getShininess() const { return shininess; }
        CUDA_CALLABLE float getReflectivity() const { return reflectivity; }
        CUDA_CALLABLE float getTransmission() const { return transmission; }

        // setters
        CUDA_CALLABLE void setAmbient(const Color &ambient) { ka = ambient; }
        CUDA_CALLABLE void setDiffuse(const Color &diffuse) { kd = diffuse; }
        CUDA_CALLABLE void setSpecular(const Color &specular) { ks = specular; }
        CUDA_CALLABLE void setShininess(float shiny) { shininess = shiny; }
        CUDA_CALLABLE void setReflectivity(float reflect) { reflectivity = reflect; }
        CUDA_CALLABLE void setTransmission(float transmit) { transmission = transmit; }

        // preset materials
        static CUDA_CALLABLE Material Matte() {
            return Material(Color(50, 50, 50), Color(200, 200, 200), Color(10, 10, 10), 10.0f, 0.0f, 0.0f);
        }
        static CUDA_CALLABLE Material Plastic() {
            return Material(Color(30, 30, 30), Color(150, 150, 150), Color(100, 100, 100), 50.0f, 0.2f, 0.0f);
        }
        static CUDA_CALLABLE Material Metal() {
            return Material(Color(20, 20, 20), Color(100, 100, 100), Color(200, 200, 200), 100.0f, 0.5f, 0.0f);
        }
        static CUDA_CALLABLE Material Mirror() {
            return Material(Color(5, 5, 5), Color(20, 20, 20), Color(255, 255, 255), 300.0f, 0.95f, 0.0f);
        }
        static CUDA_CALLABLE Material Glass() {
            return Material(Color(10, 10, 10), Color(50, 50, 50), Color(150, 150, 150), 200.0f, 0.05f, 0.0f); // change later
        }
};

#endif // MATERIAL_H