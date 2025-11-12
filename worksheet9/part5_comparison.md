# Worksheet 9 - Part 5: Comparison of Projection Shadows vs Shadow Mapping

## Projection Shadows (Part 2)

### Advantages:
1. **Simple to implement** - Only requires basic matrix math (shadow projection matrix)
2. **Fast computation** - Single rendering pass, no extra textures needed
3. **Low memory usage** - No shadow map texture storage required
4. **Minimal shader complexity** - Simple vertex transformation with projection matrix
5. **Works well for planar receivers** - Perfect for flat ground planes

### Disadvantages:
1. **No self-shadowing** - Objects cannot cast shadows on themselves
2. **Limited to planar surfaces** - Only works for flat shadow receivers (like ground)
3. **No soft shadows** - Sharp, hard-edged shadows only
4. **Depth fighting artifacts** - Requires epsilon offsets and special depth testing (greater)
5. **Not perspective correct** - Shadows can look distorted at angles
6. **Cannot handle complex geometry** - Fails with curved or non-planar receivers
7. **Single receiver limitation** - Shadows only appear on one designated plane

## Shadow Mapping (Parts 3-4)

### Advantages:
1. **Handles self-shadowing** - Objects can correctly shadow themselves
2. **Works on any geometry** - Shadows appear on curved surfaces, complex meshes
3. **Multiple shadow receivers** - Any object in the scene can receive shadows
4. **More realistic appearance** - Proper depth-based occlusion
5. **Supports multiple lights** - Can create multiple shadow maps for different lights
6. **Hardware support** - Built-in GPU features (comparison samplers, PCF filtering)
7. **Extensible** - Can add soft shadows, cascaded shadow maps, etc.

### Disadvantages:
1. **Higher complexity** - Requires two rendering passes and coordinate transformations
2. **Memory overhead** - Needs large depth texture (e.g., 2048x2048 or higher)
3. **Resolution dependent** - Shadow quality limited by shadow map resolution
4. **Shadow acne artifacts** - Requires bias values to prevent self-shadowing errors
5. **Peter Panning** - Too much bias can cause shadows to detach from objects
6. **Performance cost** - Two render passes instead of one
7. **Aliasing issues** - Can show jagged edges, especially at grazing angles
8. **Limited shadow distance** - Shadow map covers finite area (frustum)

## Summary

**Projection shadows** are ideal for simple scenes with flat ground planes where performance and simplicity are priorities, and self-shadowing is not needed.

**Shadow mapping** is the industry-standard technique for realistic shadows in 3D graphics, handling complex geometry and self-shadowing at the cost of increased complexity and memory usage.

For most modern 3D applications, shadow mapping is preferred despite its disadvantages because it provides much more realistic and flexible shadow rendering.
