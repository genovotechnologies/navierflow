import taichi as ti
import numpy as np
from dataclasses import dataclass

ti.init(arch=ti.vulkan)


@dataclass
class SimulationParams:
    dt: float = 0.1
    viscosity: float = 0.1
    density_multiplier: float = 5.0
    velocity_multiplier: float = 2.0
    brush_size: int = 2
    vorticity: float = 0.3
    temperature: float = 0.5
    color_mode: str = 'smoke'
    display_velocity: bool = False
    display_vorticity: bool = False
    display_temperature: bool = False
    num_particles: int = 10000
    particle_color: ti.Vector = ti.Vector([1.0, 0.5, 0.2])
    particle_life: float = 2.0
    particle_size: float = 1.0
    omega: float = 1.0


@ti.data_oriented
class ParticleSystem:
    def __init__(self, num_particles, nx, ny):
        self.num_particles = num_particles
        self.nx = nx
        self.ny = ny

        # Particle properties
        self.positions = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
        self.velocities = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
        self.life = ti.field(dtype=ti.f32, shape=num_particles)

        self.initialize_particles()

    @ti.kernel
    def initialize_particles(self):
        for i in self.positions:
            self.positions[i] = ti.Vector([
                ti.random() * self.nx,
                ti.random() * self.ny
            ])
            self.velocities[i] = ti.Vector([0.0, 0.0])
            self.life[i] = ti.random() * 2.0

    @ti.kernel
    def update(self, velocity_field: ti.template(), dt: float):
        for i in self.positions:
            pos = self.positions[i]
            x, y = int(pos[0]), int(pos[1])

            if 0 <= x < self.nx and 0 <= y < self.ny:
                # Sample velocity from the grid
                self.velocities[i] = velocity_field[x, y]

                # Update position
                self.positions[i] += self.velocities[i] * dt

                # Update life and reset if needed
                self.life[i] -= dt
                if self.life[i] <= 0 or not (0 <= pos[0] < self.nx and 0 <= pos[1] < self.ny):
                    self.positions[i] = ti.Vector([
                        ti.random() * self.nx,
                        ti.random() * self.ny
                    ])
                    self.life[i] = 2.0

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

        # Initialize particle system
        self.particles = ParticleSystem(self.params.num_particles, nx, ny)

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

            # Ball properties (LBM only)
            self.ball_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
            self.ball_vel = ti.Vector.field(2, dtype=ti.f32, shape=())
            self.ball_pos[None] = [nx / 2, ny / 2]
            self.ball_vel[None] = [0, 0]


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

    @ti.kernel
    def compute_divergence(self):
        for i, j in self.divergence:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                self.divergence[i, j] = 0.5 * (
                        self.velocity[i + 1, j][0] - self.velocity[i - 1, j][0] +
                        self.velocity[i, j + 1][1] - self.velocity[i, j - 1][1]
                )

    @ti.kernel
    def solve_pressure(self):
        for i, j in self.pressure:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                self.pressure[i, j] = (
                                              self.pressure[i + 1, j] + self.pressure[i - 1, j] +
                                              self.pressure[i, j + 1] + self.pressure[i, j - 1] -
                                              self.divergence[i, j]
                                      ) * 0.25

    @ti.kernel
    def apply_pressure(self):
        for i, j in self.velocity:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                self.velocity[i, j][0] -= 0.5 * (self.pressure[i + 1, j] - self.pressure[i - 1, j])
                self.velocity[i, j][1] -= 0.5 * (self.pressure[i, j + 1] - self.pressure[i, j - 1])

    def project(self):
        # Split the projection step into separate kernel calls
        self.compute_divergence()

        # Solve pressure Poisson equation
        for _ in range(50):  # Jacobi iteration
            self.solve_pressure()

        # Apply pressure gradient
        self.apply_pressure()

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
    def add_density_velocity(self, x: int, y: int, dx: float, dy: float):
        for i, j in ti.ndrange((-self.params.brush_size, self.params.brush_size + 1),
                               (-self.params.brush_size, self.params.brush_size + 1)):
            pos_x, pos_y = x + i, y + j
            if 0 <= pos_x < self.nx and 0 <= pos_y < self.ny:
                # Add density
                self.density[pos_x, pos_y] += self.params.density_multiplier
                # Add velocity
                self.velocity[pos_x, pos_y][0] += dx * self.params.velocity_multiplier
                self.velocity[pos_x, pos_y][1] += dy * self.params.velocity_multiplier

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

    @ti.kernel
    def render_particles(self, canvas: ti.template()):
        for i, j in self.density:
            # Start with the density field
            color = ti.Vector([self.density[i, j], self.density[i, j], self.density[i, j]])
            canvas.set_pixel(i, j, color)

        # Render particles
        for p in range(self.params.num_particles):
            pos = self.particles.positions[p]
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.nx and 0 <= y < self.ny:
                life_factor = self.particles.life[p] / self.params.particle_life
                particle_color = self.params.particle_color * life_factor
                canvas.set_pixel(x, y, particle_color)


    def step(self):
        self.compute_vorticity()
        self.apply_vorticity_confinement()
        self.diffuse(self.velocity, self.params.viscosity)
        self.project()
        self.advect()
        self.project()
        self.diffuse(self.density, 0.05)

    def run_simulation(self):
        window = ti.ui.Window("Fluid Simulation", (self.nx, self.ny))
        canvas = window.get_canvas()
        gui = window.get_gui()
        last_mouse_pos = None

        while window.running:
            # GUI controls
            with gui.sub_window("Controls", 0.05, 0.05, 0.3, 0.95):
                gui.text("Simulation Parameters")
                self.params.viscosity = gui.slider_float("Viscosity", self.params.viscosity, 0.0, 1.0)
                self.params.vorticity = gui.slider_float("Vorticity", self.params.vorticity, 0.0, 1.0)
                self.params.density_multiplier = gui.slider_float("Density", self.params.density_multiplier, 0.0, 10.0)
                self.params.velocity_multiplier = gui.slider_float("Velocity", self.params.velocity_multiplier, 0.0,
                                                                   5.0)
                self.params.brush_size = gui.slider_int("Brush Size", self.params.brush_size, 1, 10)

                gui.text("Particle Parameters")
                self.params.particle_size = gui.slider_float("Particle Size", self.params.particle_size, 0.5, 3.0)
                self.params.particle_life = gui.slider_float("Particle Life", self.params.particle_life, 0.5, 5.0)

            # Mouse interaction
            if window.is_pressed(ti.ui.LMB):
                mouse_x, mouse_y = window.get_cursor_pos()
                x, y = int(mouse_x * self.nx), int(mouse_y * self.ny)

                if last_mouse_pos is not None:
                    dx = x - last_mouse_pos[0]
                    dy = y - last_mouse_pos[1]
                    self.add_density_velocity(x, y, dx, dy)
                last_mouse_pos = (x, y)
            else:
                last_mouse_pos = None

            # Update simulation
            self.step()
            self.particles.update(self.velocity, self.params.dt)

            # Visualization
            self.render_particles(canvas)
            window.show()


if __name__ == "__main__":
    sim = FluidSimulation(nx=256, ny=256)
    sim.run_simulation()