import taichi as ti
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
import colorsys
from visualization import FluidVisualizer
from particle_system import ParticleSystem
from ui_system import FluidUI


@dataclass
class SimulationParams:
    dt: float = 0.1
    viscosity: float = 0.0001
    omega: float = 1.0
    density_multiplier: float = 5.0
    velocity_multiplier: float = 2.0
    diffusion_rate: float = 0.05
    color_mode: str = 'blue'
    brush_size: int = 3
    vorticity: float = 0.1
    temperature: float = 0.0
    ball_radius: int = 10
    ball_interaction_strength: float = 3.0
    trail_length: int = 15


@ti.data_oriented
class FluidSimulation:
    def __init__(self, nx: int = 256, ny: int = 256, method: str = 'eulerian'):
        # Initialize Taichi with GPU support
        ti.init(arch=ti.gpu, debug=True)

        self.params = SimulationParams()
        self.nx, self.ny = nx, ny
        self.method = method

        # Add new components
        self.visualizer = FluidVisualizer(nx, ny)
        self.particle_system = ParticleSystem(self.num_particles, nx, ny)
        self.ui = FluidUI(self.window)

        # Create window and canvas
        self.window = ti.ui.Window("Fluid Simulation", (800, 800))
        self.canvas = self.window.get_canvas()

        # Initialize fields for simulation
        self.velocity = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.pressure = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.density = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.temperature = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vorticity = ti.field(dtype=ti.f32, shape=(nx, ny))

        # Fields for LBM method
        if method == 'lbm':
            self.f = ti.field(dtype=ti.f32, shape=(9, nx, ny))
            self.feq = ti.field(dtype=ti.f32, shape=(9, nx, ny))

            # LBM parameters
            self.e = ti.Vector.field(2, dtype=ti.i32, shape=9)
            self.w = ti.field(dtype=ti.f32, shape=9)
            self.initialize_lbm_parameters()

        # Particle system
        self.num_particles = nx * ny // 4
        self.particles = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles)
        self.particle_velocities = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles)
        self.particle_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)
        self.particle_ages = ti.field(dtype=ti.f32, shape=self.num_particles)

        # Trail system
        self.trails = ti.field(dtype=ti.f32, shape=(nx, ny))

        # Ball attributes
        self.ball_position = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.ball_velocity = ti.Vector.field(2, dtype=ti.f32, shape=())

        # GUI state
        self.show_controls = True
        self.visualization_mode = 'fluid'  # 'fluid', 'pressure', 'velocity'
        self.current_screen = 'selector'  # 'selector', 'simulation'

        # Initialize simulation
        self.initialize_simulation()

    @ti.kernel
    def initialize_simulation(self):
        # Initialize particles
        for i in range(self.num_particles):
            self.particles[i] = ti.Vector([ti.random() * self.nx, ti.random() * self.ny])
            self.particle_colors[i] = ti.Vector([0.1, 0.2, 0.8])  # Base blue color
            self.particle_ages[i] = ti.random() * self.params.trail_length

        # Initialize ball position
        self.ball_position[None] = ti.Vector([self.nx // 2, self.ny // 2])
        self.ball_velocity[None] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def initialize_lbm_parameters(self):
        # Initialize LBM lattice velocities and weights
        # D2Q9 lattice
        directions = ti.static([
            ti.Vector([0, 0]),
            ti.Vector([1, 0]), ti.Vector([0, 1]),
            ti.Vector([-1, 0]), ti.Vector([0, -1]),
            ti.Vector([1, 1]), ti.Vector([-1, 1]),
            ti.Vector([-1, -1]), ti.Vector([1, -1])
        ])

        weights = ti.static([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)

        for i in range(9):
            self.e[i] = directions[i]
            self.w[i] = weights[i]

    @ti.kernel
    def eulerian_step(self):
        # Advection
        for i, j in self.velocity:
            pos = ti.Vector([i, j]) - self.velocity[i, j] * self.params.dt
            pos.x = ti.min(ti.max(pos.x, 0.0), float(self.nx - 1))
            pos.y = ti.min(ti.max(pos.y, 0.0), float(self.ny - 1))
            self.velocity[i, j] = self.sample_field(self.velocity, pos)

        # Diffusion
        self.diffuse()

        # Projection
        self.project()

    @ti.func
    def sample_field(self, field: ti.template(), pos: ti.template()) -> ti.Vector:
        """Bilinear interpolation for vector fields"""
        i, j = int(pos.x), int(pos.y)
        fx, fy = pos.x - i, pos.y - j

        a = field[i, j]
        b = field[min(i + 1, self.nx - 1), j]
        c = field[i, min(j + 1, self.ny - 1)]
        d = field[min(i + 1, self.nx - 1), min(j + 1, self.ny - 1)]

        return (1 - fx) * (1 - fy) * a + fx * (1 - fy) * b + (1 - fx) * fy * c + fx * fy * d

    @ti.kernel
    def project(self):
        # Compute divergence
        for i, j in self.pressure:
            if i > 0 and i < self.nx - 1 and j > 0 and j < self.ny - 1:
                self.pressure[i, j] = -0.5 * (
                        self.velocity[i + 1, j][0] - self.velocity[i - 1, j][0] +
                        self.velocity[i, j + 1][1] - self.velocity[i, j - 1][1]
                )

        # Jacobi iteration
        for _ in range(20):
            for i, j in self.pressure:
                if i > 0 and i < self.nx - 1 and j > 0 and j < self.ny - 1:
                    self.pressure[i, j] = (
                                                  self.pressure[i + 1, j] + self.pressure[i - 1, j] +
                                                  self.pressure[i, j + 1] + self.pressure[i, j - 1]
                                          ) * 0.25

        # Apply pressure gradient
        for i, j in self.velocity:
            if i > 0 and i < self.nx - 1 and j > 0 and j < self.ny - 1:
                self.velocity[i, j] -= 0.5 * ti.Vector([
                    self.pressure[i + 1, j] - self.pressure[i - 1, j],
                    self.pressure[i, j + 1] - self.pressure[i, j - 1]
                ])

    @ti.kernel
    def diffuse(self):
        # Apply viscosity diffusion
        alpha = self.params.viscosity * self.params.dt
        for i, j in self.velocity:
            if i > 0 and i < self.nx - 1 and j > 0 and j < self.ny - 1:
                self.velocity[i, j] += alpha * (
                        self.velocity[i + 1, j] + self.velocity[i - 1, j] +
                        self.velocity[i, j + 1] + self.velocity[i, j - 1] -
                        4.0 * self.velocity[i, j]
                )

    @ti.kernel
    def lbm_step(self):
        # Collision step
        for i, j in ti.ndrange(self.nx, self.ny):
            rho = 0.0
            u = ti.Vector([0.0, 0.0])

            # Calculate macroscopic quantities
            for k in ti.static(range(9)):
                rho += self.f[k, i, j]
                u += self.e[k] * self.f[k, i, j]
            u /= rho

            # Calculate equilibrium distribution
            for k in ti.static(range(9)):
                eu = self.e[k].dot(u)
                usqr = u.dot(u)
                self.feq[k, i, j] = self.w[k] * rho * (
                        1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usqr
                )

            # Collision
            for k in ti.static(range(9)):
                self.f[k, i, j] = self.f[k, i, j] - self.params.omega * (
                        self.f[k, i, j] - self.feq[k, i, j]
                )

        # Streaming step
        for i, j, k in ti.ndrange(self.nx, self.ny, 9):
            ni = i + self.e[k][0]
            nj = j + self.e[k][1]

            if 0 <= ni < self.nx and 0 <= nj < self.ny:
                self.f[k, ni, nj] = self.f[k, i, j]

    @ti.kernel
    def add_force(self, x: float, y: float, dx: float, dy: float):
        # Add force at mouse position
        force = ti.Vector([dx, dy]) * self.params.velocity_multiplier

        for i, j in ti.ndrange((-self.params.brush_size, self.params.brush_size + 1),
                               (-self.params.brush_size, self.params.brush_size + 1)):
            pos_x = int(x + i)
            pos_y = int(y + j)

            if 0 <= pos_x < self.nx and 0 <= pos_y < self.ny:
                r2 = i * i + j * j
                if r2 <= self.params.brush_size * self.params.brush_size:
                    weight = ti.exp(-r2 / (self.params.brush_size * self.params.brush_size))
                    self.velocity[pos_x, pos_y] += force * weight
                    self.density[pos_x, pos_y] += self.params.density_multiplier * weight

    def update(self):
        # Handle input
        mouse_pos = self.window.get_cursor_pos()
        if self.window.is_pressed(ti.ui.LMB):
            grid_x = int(mouse_pos[0] * self.nx)
            grid_y = int(mouse_pos[1] * self.ny)

            if hasattr(self, 'last_mouse_pos'):
                dx = grid_x - self.last_mouse_pos[0]
                dy = grid_y - self.last_mouse_pos[1]
                self.add_force(grid_x, grid_y, dx, dy)

            self.last_mouse_pos = (grid_x, grid_y)
        else:
            self.last_mouse_pos = None

        # Update simulation
        if self.method == 'eulerian':
            self.eulerian_step()
        else:
            self.lbm_step()

        # Update particles and trails
        self.update_particles()
        self.update_trails()

        # Update ball physics
        self.update_ball()

    def render(self):
        if self.current_screen == 'selector':
            method = self.ui.draw_selector()
            if method:
                self.method = method
                self.current_screen = 'simulation'
                self.initialize_simulation()
        else:
            # Update visualization
            self.visualizer.compute_fluid_colors(
                self.density,
                self.temperature,
                self.velocity,
                self.trails,
                self.params.color_mode
            )

            # Draw fluid
            self.canvas.set_image(self.visualizer.color_buffer)

            # Draw particles
            self.canvas.circles(
                self.particle_system.positions.to_numpy() / [self.nx, self.ny],
                radius=0.002,
                color=self.particle_system.colors.to_numpy()
            )

            # Draw UI
            if self.show_controls:
                new_mode = self.ui.draw_controls(self.params)
                if new_mode:
                    self.visualization_mode = new_mode

        self.window.show()
    def num_particles(self):
        print('I am being used')

    def run(self):
        while self.window.running:
            # Update simulation state
            if self.current_screen == 'simulation':
                self.update()

            # Render frame
            self.render()

            # Handle keyboard input
            if self.window.get_event(ti.ui.PRESS):
                if self.window.event.key == 'escape':
                    break
                elif self.window.event.key == 'r':
                    self.initialize_simulation()
                elif self.window.event.key == 'c':
                    self.show_controls = not self.show_controls
                elif self.window.event.key == 'v':
                    self.cycle_visualization_mode()


if __name__ == "__main__":
    sim = FluidSimulation(method='lbm')
    sim.run()