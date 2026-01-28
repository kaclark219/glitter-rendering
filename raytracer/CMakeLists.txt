cmake_minimum_required(VERSION 3.24)
project(raytracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CPU executable ---
add_executable(raytracer_cpu
    src/main.cpp
    src/render_cpu.cpp
    src/image/image_ppm.cpp
)
target_include_directories(raytracer_cpu PRIVATE src)

# --- CUDA executable ---
# Expect these files to exist:
#   src/main_cuda.cu        (calls renderCUDA())
#   src/render_cuda.cu      (your GPU renderer)
add_executable(raytracer_cuda
    src/main_cuda.cu
    src/render_cuda.cu
    src/image/image_ppm.cpp
)
target_include_directories(raytracer_cuda PRIVATE src)

# Make sure NVCC is used properly and can link device code if needed
set_target_properties(raytracer_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Optional: pick a reasonable default GPU arch if none is set
# (You can also set this via -DCMAKE_CUDA_ARCHITECTURES=86 etc.)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if (MSVC)
    target_compile_options(raytracer_cpu PRIVATE /W4)
    target_compile_options(raytracer_cuda PRIVATE /W4)
else()
    target_compile_options(raytracer_cpu PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(raytracer_cuda PRIVATE -Wall -Wextra -Wpedantic)
endif()

# to build:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
# run:
#   ./build/raytracer_cpu
#   ./build/raytracer_cuda