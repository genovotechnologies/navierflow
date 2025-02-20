import taichi as ti
import numpy as np
from dataclasses import dataclass

# Initialize Taichi with GPU support
ti.init(arch=ti.vulkan)


@dataclass
class SimulationParams:
    dt: float = 0.1
    viscosity: float = 0.1
    omega: float = 1.0
    density_multiplier: float = 5.0
    velocity_multiplier: float = 2.0
    diffusion_rate: float = 0.05
    color_mode: str = 'default'
    brush_size: int = 2
    vorticity: float = 0.3
    temperature: float = 0.5
    buoyancy: float = 1.0
    num_particles: int = 1000000
    particle_life: float = 10
    ball_radius: int = 20
    ball_interaction_strength: float = 2.0
    smoke_rise_speed: float = 0.5
    ball_drag: float = 0.1  # New parameter for ball drag
    ball_mass: float = 5.0  # New parameter for ball mass
    trail_length: int = 20
    ball_pos: tuple = (0.0, 0.0)
    trail_opacity: float = 0.3
    particle_speed_multiplier: float = 1.0
    color_scheme: str = 'velocity'  # Options: 'temperature', 'velocity', 'density'

    # LBM specific
    collision_frequency: float = 0.1

    # Visualization
    background_color: tuple = (1, 1, 1)
    particle_color: tuple = (0.8, 0.9, 1.0)
    trail_color: tuple = (0.7, 0.8, 0.9)


@ti.data_oriented
class ParticleTrailSystem:
    def __init__(self, num_particles, nx, ny, trail_length=20):
        self.num_particles = num_particles
        self.nx = nx
        self.ny = ny
        self.trail_length = trail_length

        # Particle properties
        self.positions = ti.Vector.field(2, dtype=ti.f32, shape=(num_particles, trail_length))
        self.velocities = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
        self.trail_alphas = ti.field(dtype=ti.f32, shape=(num_particles, trail_length))

    @ti.kernel
    def initialize(self):
        for i in range(self.num_particles):
            for j in range(self.trail_length):
                self.positions[i, j] = ti.Vector([
                    ti.random() * self.nx,
                    ti.random() * self.ny
                ])
                self.trail_alphas[i, j] = 1.0 - j / self.trail_length

            self.velocities[i] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def update(self, velocity_field: ti.template(), dt: ti.f32):
        # Update trail positions and velocities
        for i in range(self.num_particles):
            # Save current position
            curr_pos = self.positions[i, 0]

            # Shift trail positions
            for j in range(self.trail_length - 1):
                next_idx = self.trail_length - 1 - j
                prev_idx = next_idx - 1
                if prev_idx >= 0:
                    self.positions[i, next_idx] = self.positions[i, prev_idx]

            # Update velocity based on field
            pos = curr_pos
            x, y = int(pos[0]), int(pos[1])

            if 0 <= x < self.nx and 0 <= y < self.ny:
                self.velocities[i] = velocity_field[x, y]
                new_pos = pos + self.velocities[i] * dt

                # Boundary conditions
                new_pos[0] = ti.min(ti.max(new_pos[0], 0.0), float(self.nx - 1))
                new_pos[1] = ti.min(ti.max(new_pos[1], 0.0), float(self.ny - 1))

                self.positions[i, 0] = new_pos

@ti.data_oriented
class FluidSimulation:
    def __init__(self, nx=256, ny=256, method='eulerian'):
        self.params = SimulationParams()
        self.nx = nx
        self.ny = ny
        self.method = method

        # Fields
        self.velocity = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.density = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.temperature = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vorticity = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.pressure = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.divergence = ti.field(dtype=ti.f32, shape=(nx, ny))

        # New fields for smoke effect
        self.smoke_density = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.prev_smoke_density = ti.field(dtype=ti.f32, shape=(nx, ny))

        # Initialize ball-related fields regardless of method
        self.ball_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.ball_vel = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.ball_force = ti.Vector.field(2, dtype=ti.f32, shape=())

        # Mouse interaction
        self.prev_mouse_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.curr_mouse_pos = ti.Vector.field(2, dtype=ti.f32, shape=())

        # Particle system
        self.particles = ParticleTrailSystem(self.params.num_particles, nx, ny)

        # LBM specific fields
        if method == 'lbm':
            self.f = ti.field(dtype=ti.f32, shape=(nx, ny, 9))
            self.feq = ti.field(dtype=ti.f32, shape=(nx, ny, 9))
            self.w = ti.field(dtype=ti.f32, shape=9)
            self.e = ti.Matrix([
                [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                [1, 1], [-1, 1], [-1, -1], [1, -1]
            ], dt=ti.f32)
            self._initialize_lbm()

        self.initialize_fields()

    @ti.kernel
    def compute_vorticity(self):
        for i, j in self.velocity:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                # Compute velocity gradients
                du_dy = (self.velocity[i, j + 1][0] - self.velocity[i, j - 1][0]) * 0.5
                dv_dx = (self.velocity[i + 1, j][1] - self.velocity[i - 1, j][1]) * 0.5
                # Vorticity is the curl of velocity in 2D
                self.vorticity[i, j] = du_dy - dv_dx

    @ti.kernel
    def apply_vorticity_confinement(self):
        for i, j in self.velocity:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                # Compute vorticity gradient
                vort_grad_x = (self.vorticity[i + 1, j] - self.vorticity[i - 1, j]) * 0.5
                vort_grad_y = (self.vorticity[i, j + 1] - self.vorticity[i, j - 1]) * 0.5

                # Normalize gradient
                length = ti.sqrt(vort_grad_x * vort_grad_x + vort_grad_y * vort_grad_y) + 1e-9
                vort_grad_x /= length
                vort_grad_y /= length

                # Apply force
                force_x = vort_grad_y * self.vorticity[i, j] * self.params.vorticity
                force_y = -vort_grad_x * self.vorticity[i, j] * self.params.vorticity

                self.velocity[i, j][0] += force_x * self.params.dt
                self.velocity[i, j][1] += force_y * self.params.dt

    @ti.kernel
    def diffuse(self, field: ti.template(), diffusion_rate: float):
        for i, j in field:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                a = self.params.dt * diffusion_rate * (self.nx - 2) * (self.ny - 2)
                inv_denom = 1.0 / (1 + 4 * a)

                # Common computation for Vector field and Scalar field
                field[i, j] = (field[i, j] + a * (
                        field[i + 1, j] + field[i - 1, j] + field[i, j + 1] + field[i, j - 1])) * inv_denom

    @ti.kernel
    def compute_divergence(self):
        for i, j in self.divergence:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                self.divergence[i, j] = 0.5 * (
                        self.velocity[i + 1, j][0] - self.velocity[i - 1, j][0] +
                        self.velocity[i, j + 1][1] - self.velocity[i, j - 1][1]
                )


    def project(self):
        # Split the projection step into separate kernel calls
        self.compute_divergence()

        # Solve pressure Poisson equation


        # Apply pressure gradient


    @ti.kernel
    def advect(self):
        # Advect velocity field
        for i, j in self.velocity:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                pos = ti.Vector([float(i), float(j)]) - self.velocity[i, j] * self.params.dt
                pos = ti.max(ti.Vector([0.5, 0.5]), ti.min(ti.Vector([self.nx - 1.5, self.ny - 1.5]), pos))

                i0, j0 = int(pos[0]), int(pos[1])
                i1, j1 = i0 + 1, j0 + 1
                s1, t1 = pos[0] - i0, pos[1] - j0
                s0, t0 = 1 - s1, 1 - t1

                self.velocity[i, j] = (
                        s0 * (t0 * self.velocity[i0, j0] + t1 * self.velocity[i0, j1]) +
                        s1 * (t0 * self.velocity[i1, j0] + t1 * self.velocity[i1, j1])
                )

        # Advect density field
        for i, j in self.density:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                pos = ti.Vector([float(i), float(j)]) - self.velocity[i, j] * self.params.dt
                pos = ti.max(ti.Vector([0.5, 0.5]), ti.min(ti.Vector([self.nx - 1.5, self.ny - 1.5]), pos))

                i0, j0 = int(pos[0]), int(pos[1])
                i1, j1 = i0 + 1, j0 + 1
                s1, t1 = pos[0] - i0, pos[1] - j0
                s0, t0 = 1 - s1, 1 - t1

                self.density[i, j] = (
                        s0 * (t0 * self.density[i0, j0] + t1 * self.density[i0, j1]) +
                        s1 * (t0 * self.density[i1, j0] + t1 * self.density[i1, j1])
                )
    @ti.kernel
    def initialize_fields(self):
        for i, j in self.velocity:
            self.velocity[i, j] = [0, 0]
            self.density[i, j] = 0
            self.temperature[i, j] = 0
            self.vorticity[i, j] = 0

    @ti.kernel
    def _initialize_lbm(self):
        # Initialize LBM weights
        weights = ti.Vector([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                             1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0])
        for i in range(9):
            self.w[i] = weights[i]

        # Initialize distribution functions
        for i, j, k in self.f:
            self.f[i, j, k] = self.w[k]
            self.feq[i, j, k] = self.w[k]

    @ti.kernel
    def apply_temperature_buoyancy(self):
        for i, j in self.velocity:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                buoyant_force = self.temperature[i, j] * self.params.buoyancy
                self.velocity[i, j][1] += buoyant_force * self.params.dt

    @ti.kernel
    def render_with_trails(self, pixels: ti.types.ndarray()):
        # Clear buffer with background color
        for i, j in ti.ndrange(self.nx, self.ny):
            pixels[i, j, 0] = 1.0
            pixels[i, j, 1] = 1.0
            pixels[i, j, 2] = 1.0
            pixels[i, j, 3] = 1.0  # Alpha channel

        # Render fluid density
        for i, j in self.density:
            density_val = self.density[i, j]
            temperature_val = self.temperature[i, j]

            # Color mapping based on temperature
            r = ti.min(1.0, temperature_val * 2.0)
            g = ti.max(0.0, 1.0 - abs(temperature_val - 0.5) * 2.0)
            b = ti.max(0.0, 1.0 - temperature_val * 2.0)

            # Blend with existing color
            alpha = density_val * 0.5
            pixels[i, j, 0] = pixels[i, j, 0] * (1 - alpha) + r * alpha
            pixels[i, j, 1] = pixels[i, j, 1] * (1 - alpha) + g * alpha
            pixels[i, j, 2] = pixels[i, j, 2] * (1 - alpha) + b * alpha

        # Render particle trails
        for i, j in ti.ndrange(self.particles.num_particles, self.particles.trail_length):
            pos = self.particles.positions[i, j]
            alpha = self.particles.trail_alphas[i, j] * 0.3

            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.nx and 0 <= y < self.ny:
                # Blend trail with existing color
                pixels[x, y, 0] = pixels[x, y, 0] * (1 - alpha) + 0.8 * alpha
                pixels[x, y, 1] = pixels[x, y, 1] * (1 - alpha) + 0.9 * alpha
                pixels[x, y, 2] = pixels[x, y, 2] * (1 - alpha) + 1.0 * alpha

    # 3. Enhanced LBM Implementation
    @ti.kernel
    def lbm_stream_collide(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            # Compute macroscopic quantities
            rho = 0.0
            u = ti.Vector([0.0, 0.0])

            for k in ti.static(range(9)):
                rho += self.f[i, j, k]
                u += self.e[k] * self.f[i, j, k]

            u /= rho

            # Compute equilibrium distribution
            for k in ti.static(range(9)):
                eu = self.e[k].dot(u)
                usqr = u.dot(u)
                self.feq[i, j, k] = self.w[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usqr)

            # Collision step
            omega = self.params.omega
            for k in ti.static(range(9)):
                self.f[i, j, k] = self.f[i, j, k] * (1 - omega) + self.feq[i, j, k] * omega

            # Update macroscopic fields
            self.density[i, j] = rho
            if rho > 0:
                self.velocity[i, j] = u / rho

    @ti.kernel
    def update_ball(self):
        # Update ball position
        self.ball_pos[None] += self.ball_vel[None] * self.params.dt

        # Boundary conditions
        for i in ti.static(range(2)):
            if self.ball_pos[None][i] < self.params.ball_radius:
                self.ball_pos[None][i] = self.params.ball_radius
                self.ball_vel[None][i] *= -0.8
            elif self.ball_pos[None][i] > (self.nx if i == 0 else self.ny) - self.params.ball_radius:
                self.ball_pos[None][i] = (self.nx if i == 0 else self.ny) - self.params.ball_radius
                self.ball_vel[None][i] *= -0.8

        # Ball-fluid interaction
        pos = self.ball_pos[None].cast(int)
        radius = self.params.ball_radius
        for i, j in ti.ndrange((-radius, radius + 1), (-radius, radius + 1)):
            x, y = pos[0] + i, pos[1] + j
            if 0 <= x < self.nx and 0 <= y < self.ny:
                r2 = i * i + j * j
                if r2 <= radius * radius:
                    intensity = 1.0 - ti.sqrt(float(r2)) / radius
                    self.density[x, y] += intensity * 0.1
                    self.temperature[x, y] += intensity * 0.1

    def step(self):
        self.compute_vorticity()
        self.apply_vorticity_confinement()
        self.diffuse(self.velocity, self.params.viscosity)
        self.project()
        self.advect()
        self.project()
        self.diffuse(self.density, 0.05)


    def add_density_velocity(self, x, y, dx=0, dy=0):
        radius = self.params.brush_size
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                px, py = x + i, y + j
                if 0 <= px < self.nx and 0 <= py < self.ny:
                    r2 = i * i + j * j
                    if r2 <= radius * radius:
                        intensity = 1.0 - ti.sqrt(float(r2)) / radius
                        self.density[px, py] += self.params.density_multiplier * intensity
                        self.temperature[px, py] += self.params.temperature * intensity
                        if dx != 0 or dy != 0:
                            self.velocity[px, py][0] += dx * self.params.velocity_multiplier * intensity
                            self.velocity[px, py][1] += dy * self.params.velocity_multiplier * intensity

    @ti.kernel
    def apply_smoke_effect(self):
        for i, j in self.smoke_density:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                # Apply smoke rising effect
                self.velocity[i, j][1] += self.smoke_density[i, j] * self.params.smoke_rise_speed

                # Diffuse smoke
                self.prev_smoke_density[i, j] = self.smoke_density[i, j]
                self.smoke_density[i, j] = (
                        self.prev_smoke_density[i, j] +
                        self.params.diffusion_rate * (
                                self.prev_smoke_density[i + 1, j] +
                                self.prev_smoke_density[i - 1, j] +
                                self.prev_smoke_density[i, j + 1] +
                                self.prev_smoke_density[i, j - 1] -
                                4 * self.prev_smoke_density[i, j]
                        )
                )

    @ti.kernel
    def update_ball_physics(self, mouse_x: float, mouse_y: float, is_dragging: int):
        if is_dragging:
            # Mouse-based movement
            target = ti.Vector([mouse_x * self.nx, mouse_y * self.ny])
            self.ball_force[None] = (target - self.ball_pos[None]) * 50.0
        else:
            # Apply fluid forces
            pos = self.ball_pos[None].cast(int)
            fluid_force = ti.Vector([0.0, 0.0])

            for i, j in ti.ndrange((-1, 2), (-1, 2)):
                x, y = pos[0] + i, pos[1] + j
                if 0 <= x < self.nx and 0 <= y < self.ny:
                    fluid_force += self.velocity[x, y] * self.params.ball_interaction_strength

            self.ball_force[None] = fluid_force

        # Update velocity and position with physics
        self.ball_vel[None] += self.ball_force[None] * self.params.dt / self.params.ball_mass
        self.ball_vel[None] *= self.params.ball_drag
        self.ball_pos[None] += self.ball_vel[None] * self.params.dt

        # Boundary conditions
        for i in ti.static(range(2)):
            if self.ball_pos[None][i] < self.params.ball_radius:
                self.ball_pos[None][i] = self.params.ball_radius
                self.ball_vel[None][i] *= -0.8
            elif self.ball_pos[None][i] > (self.nx if i == 0 else self.ny) - self.params.ball_radius:
                self.ball_pos[None][i] = (self.nx if i == 0 else self.ny) - self.params.ball_radius
                self.ball_vel[None][i] *= -0.8
    @ti.kernel
    def solve_navier_stokes(self):
        # Improved Navier-Stokes solver
        for i, j in self.velocity:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                # Compute viscous forces
                laplacian = (self.velocity[i + 1, j] + self.velocity[i - 1, j] +
                             self.velocity[i, j + 1] + self.velocity[i, j - 1] -
                             4.0 * self.velocity[i, j])

                # Update velocity using Navier-Stokes equation
                self.velocity[i, j] += (
                        self.params.viscosity * laplacian * self.params.dt +  # Viscosity term
                        self.ball_force[None] * 0.1 * self.params.dt  # External forces
                )

    @ti.kernel
    def render(self, pixels: ti.types.ndarray()):
        # White background
        for i, j in ti.ndrange(self.nx, self.ny):
            pixels[i, j, 0] = 1.0
            pixels[i, j, 1] = 1.0
            pixels[i, j, 2] = 1.0

        # Render smoke and fluid
        for i, j in self.density:
            smoke_intensity = self.density[i, j] * 0.5
            pixels[i, j, 0] *= (1.0 - smoke_intensity)
            pixels[i, j, 1] *= (1.0 - smoke_intensity)
            pixels[i, j, 2] *= (1.0 - smoke_intensity)

        # Render ball with gradient
        ball_pos = self.ball_pos[None].cast(int)
        for i, j in ti.ndrange((-self.params.ball_radius, self.params.ball_radius + 1),
                               (-self.params.ball_radius, self.params.ball_radius + 1)):
            x, y = ball_pos[0] + i, ball_pos[1] + j
            if 0 <= x < self.nx and 0 <= y < self.ny:
                r2 = i * i + j * j
                if r2 <= self.params.ball_radius * self.params.ball_radius:
                    gradient = 1.0 - ti.sqrt(float(r2)) / self.params.ball_radius
                    pixels[x, y, 0] = 1.0 - gradient * 0.5
                    pixels[x, y, 1] = 1.0 - gradient * 0.5
                    pixels[x, y, 2] = 1.0 - gradient * 0.5

    def run_simulation(self):
        window = ti.ui.Window("Enhanced Fluid Simulation", (self.nx, self.ny))
        canvas = window.get_canvas()
        gui = window.get_gui()
        pixels = np.zeros((self.nx, self.ny, 3), dtype=np.float32)

        is_dragging = False

        # Create pixel array for rendering
        pixels = np.zeros((self.nx, self.ny, 3), dtype=np.float32)

        # Main loop
        while window.running:
            # Enhanced UI controls
            with gui.sub_window("Controls", 0.02, 0.02, 0.25, 0.98):
                gui.text("=== Simulation Parameters ===")

                # Fluid dynamics controls
                gui.text("Fluid Dynamics")
                self.params.viscosity = gui.slider_float("Viscosity", self.params.viscosity, 0.0, 1.0)
                self.params.vorticity = gui.slider_float("Vorticity", self.params.vorticity, 0.0, 2.0)

                # Temperature and buoyancy
                gui.text("\nTemperature Effects")
                self.params.temperature = gui.slider_float("Temperature", self.params.temperature, 0.0, 2.0)
                self.params.buoyancy = gui.slider_float("Buoyancy", self.params.buoyancy, 0.0, 3.0)

                # Smoke effects
                gui.text("\nSmoke Parameters")
                self.params.smoke_rise_speed = gui.slider_float("Smoke Rise", self.params.smoke_rise_speed, 0.0, 2.0)
                self.params.density_multiplier = gui.slider_float("Smoke Density", self.params.density_multiplier, 1.0,
                                                                  10.0)

                # Interaction controls
                gui.text("\nInteraction")
                self.params.brush_size = gui.slider_int("Brush Size", self.params.brush_size, 1, 20)

                if self.method == 'lbm':
                    gui.text("\nBall Parameters")
                    self.params.ball_radius = gui.slider_int("Ball Radius", self.params.ball_radius, 2, 30)
                    self.params.ball_interaction_strength = gui.slider_float("Ball Interaction",
                                                                             self.params.ball_interaction_strength, 0.0,
                                                                             5.0)

            # Handle mouse interaction
            mouse_pos = window.get_cursor_pos()
            if window.is_pressed(ti.ui.LMB):
                is_dragging = True
                self.curr_mouse_pos[None] = ti.Vector([mouse_pos[0], mouse_pos[1]])
            else:
                is_dragging = False

            # Update simulation
            self.update_ball_physics(mouse_pos[0], mouse_pos[1], int(is_dragging))
            self.solve_navier_stokes()

            # Update particles - fixed line using correct attribute name
            self.particles.update(self.velocity, self.params.dt)

            # Render
            self.render(pixels)
            canvas.set_image(pixels)
            window.show()


def create_splash_screen():
    splash_window = ti.ui.Window("Fluid Simulation Setup", (600, 400))
    canvas = splash_window.get_canvas()
    gui = splash_window.get_gui()

    method = 'eulerian'
    size = 256
    preset = 'default'

    presets = {
        'default': SimulationParams(),
        'smoke': SimulationParams(
            viscosity=0.05,
            temperature=0.8,
            buoyancy=1.5,
            smoke_rise_speed=1.0
        ),
        'water': SimulationParams(
            viscosity=0.2,
            temperature=0.1,
            buoyancy=0.3,
            smoke_rise_speed=0.1
        )
    }

    while splash_window.running:
        with gui.sub_window("Setup", 0.1, 0.1, 0.9, 0.9):
            gui.text("=== Fluid Simulation Setup ===")

            gui.text("\nSimulation Method:")
            if gui.button("Eulerian Method (Smoke & Fluid)"):
                method = 'eulerian'
                splash_window.running = False
            if gui.button("Lattice Boltzmann Method (with Interactive Ball)"):
                method = 'lbm'
                splash_window.running = False

            gui.text("\nGrid Size:")
            size = gui.slider_int("Resolution", size, 128, 512)

            gui.text("\nPresets:")
            if gui.button("Default"):
                preset = 'default'
            if gui.button("Smoke Simulation"):
                preset = 'smoke'
            if gui.button("Water Simulation"):
                preset = 'water'

            gui.text("\nControls:")
            gui.text("- Left click: Add fluid/smoke")
            gui.text("- WASD: Control fluid direction")
            gui.text("- UI sliders: Adjust parameters")

        splash_window.show()

    return method, size, presets[preset]


if __name__ == "__main__":
    # Show enhanced splash screen
    method, size, preset_params = create_splash_screen()

    # Start simulation with selected parameters
    sim = FluidSimulation(nx=size, ny=size, method=method)
    sim.params = preset_params  # Apply preset parameters
    sim.run_simulation()
