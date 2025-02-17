import taichi as ti
import numpy as np
import colorsys



@ti.data_oriented
class FluidVisualizer:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny))


    def compute_fluid_colors(self,
                             density: ti.template(),
                             temperature: ti.template(),
                             velocity: ti.template(),
                             trails: ti.template(),
                             color_mode: str):
        for i, j in self.color_buffer:
            color = ti.Vector([0.0, 0.0, 0.0])

            # Base color from density
            d = density[i, j]
            if color_mode == "blue":
                color = ti.Vector([d * 0.8, d * 0.9, ti.min(0.4 + d * 0.6, 1.0)])
            elif color_mode == "fire":
                color = ti.Vector([ti.min(d * 2, 1.0), d * d, d * 0.5])
            elif color_mode == "rainbow":
                hue = d
                # Convert HSV to RGB (continued)
                if hue < 1 / 6:
                    color = ti.Vector([1.0, 6.0 * hue, 0.0])
                elif hue < 2 / 6:
                    color = ti.Vector([2.0 - 6.0 * hue, 1.0, 0.0])
                elif hue < 3 / 6:
                    color = ti.Vector([0.0, 1.0, 6.0 * hue - 2.0])
                elif hue < 4 / 6:
                    color = ti.Vector([0.0, 4.0 - 6.0 * hue, 1.0])
                elif hue < 5 / 6:
                    color = ti.Vector([6.0 * hue - 4.0, 0.0, 1.0])
                else:
                    color = ti.Vector([1.0, 0.0, 6.0 - 6.0 * hue])

            # Add temperature effect
            temp = temperature[i, j]
            if temp > 0:
                color += ti.Vector([0.3, 0.0, 0.0]) * temp

            # Add velocity highlights
            vel = velocity[i, j]
            speed = ti.sqrt(vel[0] * vel[0] + vel[1] * vel[1])
            color += ti.Vector([0.1, 0.1, 0.2]) * speed

            # Blend with trails
            trail = trails[i, j]
            color = color * (1.0 - trail) + ti.Vector([1.0, 1.0, 1.0]) * trail

            # Ensure colors stay in valid range
            self.color_buffer[i, j] = ti.min(ti.max(color, 0.0), 1.0)
