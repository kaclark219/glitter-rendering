## implementation notes
- dealing w/ back sphere
- recursive ray tracing by considering reflection
    - reflection vs. transmission constants (k_t = 0 for this checkpoint)
    - specify a max depth
- after shadow ray, go until max depth
    - spawn reflection ray, get the color with the new illumination addition
        - ``if (k_r > 0)``
    - do the same w/ the transmission ray
        - ``if (k_t > 0)`` 
    - ... repeat
- start depth at 1! be careful w/ depth .. start with 4-5
- r = S + 2a = I - 2*(S dot n / ||n||^2)*n
    - need vector of incoming ray in opposite direction

- make one sphere reflective (k_r != 0.0), the other non-reflective, the floor non-reflective, & all non-transparent
- for extra points, spawn multiple reflection rays or implement  kajiya's method (path tracing w/o refraction .. *will* get a noisy image)
