import taichi as ti
import numpy as np
from dataclasses import dataclass

@dataclass
class SimulationParams:
    dt: float = 0.1
    viscosity: float = 0.1
    omega: float = 1.0
    density_multiplier: float = 5.0
    velocity_multiplier: float = 2.0
    diffusion_rate: float = 0.05
    brush_size: int = 2
    vorticity: float = 0.3
    temperature: float = 0.5
    buoyancy: float = 1.0
    num_particles: int = 10000
    particle_speed_multiplier: float = 1.0
    ball_radius: int = 15
    ball_mass: float = 1.0
    ball_drag: float = 0.95
    ball_interaction_strength: float = 2.0
    cfl_number: float = 0.5  # For timestep control
    artificial_viscosity: float = 0.1  # For numerical stability
    lbm_relaxation_time: float = 0.6  # For LBM collision
    grid_spacing: float = 1.0  # For fixed reference frame
    mac_iterations: int = 3  # For MacCormack advection

# Initialize Taichi once at the start
ti.init(arch=ti.vulkan)


@ti.data_oriented
class FluidSimulation:
    def __init__(self, nx=256, ny=256, method='eulerian'):
        self.params = SimulationParams()
        self.nx = nx
        self.ny = ny
        self.method = method
        self.dx = self.params.grid_spacing
        self.inv_dx = 1.0 / self.dx

        # Fixed reference frame grid
        self.cell_centers = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))

        # Common fields
        self.velocity = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.density = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.pressure = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.temperature = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vorticity = ti.field(dtype=ti.f32, shape=(nx, ny))

        # Conservation fields
        self.mass_flux = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.momentum = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.energy = ti.field(dtype=ti.f32, shape=(nx, ny))

        # MacCormack advection fields
        self.phi_n = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.phi_star = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.phi_corrected = ti.field(dtype=ti.f32, shape=(nx, ny))

        # Mouse interaction
        self.prev_mouse_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.curr_mouse_pos = ti.Vector.field(2, dtype=ti.f32, shape=())

        # Enhanced LBM fields
        if method == 'lbm':
            # LBM distribution functions
            self.f = ti.field(dtype=ti.f32, shape=(nx, ny, 9))
            self.f_next = ti.field(dtype=ti.f32, shape=(nx, ny, 9))  # Double buffer
            self.feq = ti.field(dtype=ti.f32, shape=(nx, ny, 9))
            self.w = ti.field(dtype=ti.f32, shape=9)
            self.e = ti.Matrix.field(2, 9, dtype=ti.f32, shape=())

            # LBM collision matrix for MRT
            self.collision_matrix = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())

            # Initialize LBM specific fields
            self._initialize_lbm()
        else:
            self.particles = ti.Vector.field(2, dtype=ti.f32, shape=self.params.num_particles)
            self.particle_velocities = ti.Vector.field(2, dtype=ti.f32, shape=self.params.num_particles)
            self._initialize_particles()

        self.initialize_fields()

    @ti.kernel
    def initialize_grid(self):
        for i, j in self.cell_centers:
            self.cell_centers[i, j] = ti.Vector([
                (i + 0.5) * self.dx,
                (j + 0.5) * self.dx
            ])

    @ti.func
    def get_cell_index(self, pos):
        index = ti.cast(pos * self.inv_dx - 0.5, ti.i32)
        return ti.Vector([
            ti.max(0, ti.min(index[0], self.nx - 1)),
            ti.max(0, ti.min(index[1], self.ny - 1))
        ])

    @ti.kernel
    def compute_conservation_fluxes(self):
        for i, j in self.mass_flux:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                vel = self.velocity[i, j]
                rho = self.density[i, j]
                p = self.pressure[i, j]

                # Mass flux
                self.mass_flux[i, j] = rho * vel

                # Momentum
                self.momentum[i, j] = rho * vel

                # Energy (internal + kinetic)
                self.energy[i, j] = p / (0.4) + 0.5 * rho * vel.dot(vel)

    @ti.kernel
    def maccormack_advect(self, field: ti.template(), velocity: ti.template()):
        # Forward step
        for i, j in field:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                pos = ti.Vector([float(i), float(j)]) - velocity[i, j] * self.params.dt
                self.phi_star[i, j] = self.interpolate(field, pos)

        # Backward step
        for i, j in field:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                pos = ti.Vector([float(i), float(j)]) + velocity[i, j] * self.params.dt
                self.phi_corrected[i, j] = self.interpolate(self.phi_star, pos)

        # Correction with flux limiting
        for i, j in field:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                correction = (field[i, j] - self.phi_corrected[i, j]) * 0.5
                field[i, j] = self.phi_star[i, j] + correction

                # Apply flux limiter
                min_val = ti.min(
                    field[i - 1, j], field[i + 1, j],
                    field[i, j - 1], field[i, j + 1]
                )
                max_val = ti.max(
                    field[i - 1, j], field[i + 1, j],
                    field[i, j - 1], field[i, j + 1]
                )
                field[i, j] = ti.min(ti.max(field[i, j], min_val), max_val)

    @ti.func
    def interpolate(self, field: ti.template(), pos: ti.template()):
        i0 = ti.cast(pos[0], ti.i32)
        j0 = ti.cast(pos[1], ti.i32)
        i0 = ti.max(1, ti.min(i0, self.nx - 2))
        j0 = ti.max(1, ti.min(j0, self.ny - 2))

        fx = pos[0] - i0
        fy = pos[1] - j0

        return (1 - fx) * (1 - fy) * field[i0, j0] + \
            fx * (1 - fy) * field[i0 + 1, j0] + \
            (1 - fx) * fy * field[i0, j0 + 1] + \
            fx * fy * field[i0 + 1, j0 + 1]
    @ti.kernel
    def _initialize_particles(self):
        for i in self.particles:
            self.particles[i] = ti.Vector([ti.random() * self.nx, ti.random() * self.ny])
            self.particle_velocities[i] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def update_particles(self):
        for i in self.particles:
            pos = self.particles[i]
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.nx and 0 <= y < self.ny:
                self.particle_velocities[i] = self.velocity[x, y]
                new_pos = pos + self.particle_velocities[i] * self.params.dt

                # Boundary conditions
                new_pos[0] = ti.min(ti.max(new_pos[0], 0.0), float(self.nx - 1))
                new_pos[1] = ti.min(ti.max(new_pos[1], 0.0), float(self.ny - 1))

                self.particles[i] = new_pos

    def step(self):
        if self.method == 'eulerian':
            # Compute conservation quantities
            self.compute_conservation_fluxes()

            # Advection using MacCormack
            self.maccormack_advect(self.velocity, self.velocity)
            self.maccormack_advect(self.density, self.velocity)

            # Continue with existing steps
            self.diffuse(self.velocity, self.params.viscosity)
            self.project()
            self.update_particles()
            self.compute_vorticity()
            self.apply_vorticity_confinement()
            self.apply_temperature_buoyancy()
        else:
            # Enhanced LBM method
            self.lbm_stream_collide()
            self.update_ball_physics(
                self.curr_mouse_pos[None][0],
                self.curr_mouse_pos[None][1],
                0
            )

    # Add Navier-Stokes solver for LBM

    @ti.kernel
    def render(self, pixels: ti.types.ndarray()):
        # White background
        for i, j in ti.ndrange(self.nx, self.ny):
            pixels[i, j, 0] = 1.0
            pixels[i, j, 1] = 1.0
            pixels[i, j, 2] = 1.0

            # Render density for both methods
            density_val = self.density[i, j]
            pixels[i, j, 0] *= (1.0 - density_val * 0.3)
            pixels[i, j, 1] *= (1.0 - density_val * 0.3)
            pixels[i, j, 2] *= (1.0 - density_val * 0.3)

        # Method-specific rendering
        if ti.static(self.method == 'eulerian'):
            # Render particles
            for i in range(self.params.num_particles):
                pos = self.particles[i]
                x = ti.cast(pos[0], ti.i32)
                y = ti.cast(pos[1], ti.i32)
                if 0 <= x < self.nx and 0 <= y < self.ny:
                    pixels[x, y, 0] = 0.2
                    pixels[x, y, 1] = 0.2
                    pixels[x, y, 2] = 0.8
        else:  # LBM method
            # Get ball position
            pos_x = ti.cast(self.ball_pos[None][0], ti.i32)
            pos_y = ti.cast(self.ball_pos[None][1], ti.i32)
            radius = self.params.ball_radius

            # Render ball
            for i, j in ti.ndrange((-radius, radius + 1), (-radius, radius + 1)):
                x = pos_x + i
                y = pos_y + j
                if 0 <= x < self.nx and 0 <= y < self.ny:
                    r2 = i * i + j * j
                    if r2 <= radius * radius:
                        pixels[x, y, 0] = 0.2
                        pixels[x, y, 1] = 0.2
                        pixels[x, y, 2] = 0.2

    def run_simulation(self):
        window = ti.ui.Window("Fluid Simulation", (self.nx, self.ny))
        canvas = window.get_canvas()
        gui = window.get_gui()
        pixels = np.zeros((self.nx, self.ny, 3), dtype=np.float32)

        # Initialize smoke source for LBM
        if self.method == 'lbm':
            self.add_density_velocity(self.nx // 8, self.ny // 2, 1, 0)

        while window.running:
            # GUI controls
            with gui.sub_window("Controls", 0.02, 0.02, 0.25, 0.98):
                gui.text("=== Simulation Parameters ===")
                self.params.viscosity = gui.slider_float("Viscosity", self.params.viscosity, 0.0, 1.0)

                if self.method == 'eulerian':
                    self.params.brush_size = gui.slider_int("Brush Size", self.params.brush_size, 1, 20)
                else:
                    self.params.ball_interaction_strength = gui.slider_float("Ball-Fluid Interaction",
                                                                             self.params.ball_interaction_strength, 0.0,
                                                                             5.0)

            # Handle mouse interaction
            mouse_pos = window.get_cursor_pos()
            if window.is_pressed(ti.ui.LMB):
                x, y = int(mouse_pos[0] * self.nx), int(mouse_pos[1] * self.ny)
                if self.method == 'eulerian':
                    self.add_density_velocity(x, y,
                                              mouse_pos[0] - self.prev_mouse_pos[None][0],
                                              mouse_pos[1] - self.prev_mouse_pos[None][1])
                else:
                    self.ball_pos[None] = ti.Vector([float(x), float(y)])

            self.prev_mouse_pos[None] = ti.Vector([mouse_pos[0], mouse_pos[1]])

            # Update simulation
            self.step()

            # Render
            self.render(pixels)
            canvas.set_image(pixels)
            window.show()

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
    def jacobi_iteration(self):
        for i, j in self.pressure:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                self.pressure[i, j] = (
                        (self.pressure[i + 1, j] + self.pressure[i - 1, j] +
                         self.pressure[i, j + 1] + self.pressure[i, j - 1] -
                         self.divergence[i, j]) / 4.0
                )

    @ti.kernel
    def init_pressure(self):
        for i, j in self.pressure:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                self.pressure[i, j] = 0.0

    def solve_pressure_poisson(self):
        # Initialize pressure field
        self.init_pressure()

        # Jacobi iteration for pressure Poisson equation
        for _ in range(50):  # Number of iterations
            self.jacobi_iteration()

    @ti.kernel
    def apply_pressure_gradient(self):
        for i, j in self.velocity:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                self.velocity[i, j][0] -= 0.5 * (self.pressure[i + 1, j] - self.pressure[i - 1, j])
                self.velocity[i, j][1] -= 0.5 * (self.pressure[i, j + 1] - self.pressure[i, j - 1])

    def project(self):
        self.compute_divergence()
        self.solve_pressure_poisson()
        self.apply_pressure_gradient()

    @ti.kernel
    def eulerian_step(self):
        # Store previous velocity for semi-Lagrangian advection
        for i, j in self.velocity:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                # Apply external forces (gravity, buoyancy)
                gravity = -9.81 * self.params.dt
                buoyancy = self.temperature[i, j] * self.params.buoyancy * self.params.dt
                self.velocity[i, j][1] += gravity + buoyancy

                # Apply vorticity confinement
                if self.params.vorticity > 0:
                    vort = self.vorticity[i, j]
                    vort_grad_x = (self.vorticity[i + 1, j] - self.vorticity[i - 1, j]) * 0.5
                    vort_grad_y = (self.vorticity[i, j + 1] - self.vorticity[i, j - 1]) * 0.5

                    # Normalize vorticity gradient
                    length = ti.sqrt(vort_grad_x * vort_grad_x + vort_grad_y * vort_grad_y) + 1e-10
                    vort_grad_x /= length
                    vort_grad_y /= length

                    # Apply vorticity force
                    self.velocity[i, j][0] += self.params.vorticity * vort * vort_grad_y * self.params.dt
                    self.velocity[i, j][1] -= self.params.vorticity * vort * vort_grad_x * self.params.dt

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
    def apply_temperature_buoyancy(self):
        for i, j in self.velocity:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                buoyant_force = self.temperature[i, j] * self.params.buoyancy
                self.velocity[i, j][1] += buoyant_force * self.params.dt


    # 3. Enhanced LBM Implementation
    @ti.kernel
    def _initialize_lbm(self):
        # Initialize weights
        w = ti.Vector([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                       1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0])
        for i in range(9):
            self.w[i] = w[i]

        # Initialize lattice velocities
        self.e.from_numpy(np.array([
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1]
        ], dtype=np.float32).T)

        # Initialize MRT collision matrix (D2Q9 model)
        M = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [-4, -1, -1, -1, -1, 2, 2, 2, 2],
            [4, -2, -2, -2, -2, 1, 1, 1, 1],
            [0, 1, 0, -1, 0, 1, -1, -1, 1],
            [0, -2, 0, 2, 0, 1, -1, -1, 1],
            [0, 0, 1, 0, -1, 1, 1, -1, -1],
            [0, 0, -2, 0, 2, 1, 1, -1, -1],
            [0, 1, -1, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, 1, -1]
        ], dtype=np.float32)
        self.collision_matrix.from_numpy(M)

        # Initialize distribution functions
        for i, j, k in self.f:
            self.f[i, j, k] = self.w[k]
            self.f_next[i, j, k] = self.w[k]
            self.feq[i, j, k] = self.w[k]

    @ti.kernel
    def lbm_stream_collide(self):
        # Compute macroscopic quantities and equilibrium
        for i, j in ti.ndrange(self.nx, self.ny):
            rho = 0.0
            u = ti.Vector([0.0, 0.0])

            # Calculate density and velocity
            for k in ti.static(range(9)):
                f = self.f[i, j, k]
                rho += f
                u += ti.Vector([
                    self.e[None][0, k],
                    self.e[None][1, k]
                ]) * f

            if rho > 1e-6:
                u /= rho

            # Store macroscopic quantities
            self.density[i, j] = rho
            self.velocity[i, j] = u

            # Compute equilibrium distributions
            usqr = u.dot(u)
            for k in ti.static(range(9)):
                eu = (self.e[None][0, k] * u[0] + self.e[None][1, k] * u[1])
                self.feq[i, j, k] = self.w[k] * rho * (
                        1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usqr
                )

        # MRT collision and streaming
        for i, j in ti.ndrange(self.nx, self.ny):
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
                # Transform to moment space
                m = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                meq = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

                for k in ti.static(range(9)):
                    f = self.f[i, j, k]
                    feq = self.feq[i, j, k]
                    for n in ti.static(range(9)):
                        m[n] += self.collision_matrix[n, k] * f
                        meq[n] += self.collision_matrix[n, k] * feq

                # Relax different moments
                s = ti.Vector([
                    1.0, 1.4, 1.4, 1.0, 1.2, 1.0, 1.2, self.params.omega, self.params.omega
                ])

                for k in ti.static(range(9)):
                    m[k] = m[k] - s[k] * (m[k] - meq[k])

                # Back to velocity space and stream
                for k in ti.static(range(9)):
                    sum_f = 0.0
                    for n in ti.static(range(9)):
                        sum_f += self.collision_matrix[k, n] * m[n] / 9.0

                    ni = i - int(self.e[None][0, k])
                    nj = j - int(self.e[None][1, k])

                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        self.f_next[ni, nj, k] = sum_f

        # Update distribution functions
        for i, j, k in self.f:
            self.f[i, j, k] = self.f_next[i, j, k]

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


def main():
    # Create a single window for method selection
    window = ti.ui.Window("Fluid Simulation", (400, 200))
    gui = window.get_gui()

    method = 'eulerian'
    size = 256
    start_simulation = False

    while window.running and not start_simulation:
        with gui.sub_window("Setup", 0.1, 0.1, 0.9, 0.9):
            gui.text("Choose simulation method:")
            if gui.button("Eulerian (Smoke simulation)"):
                method = 'eulerian'
                start_simulation = True
            if gui.button("Lattice Boltzmann (Ball in fluid)"):
                method = 'lbm'
                start_simulation = True
        window.show()

    # Close the selection window
    window.destroy()

    if start_simulation:
        # Create new window for simulation
        sim = FluidSimulation(nx=size, ny=size, method=method)
        sim_window = ti.ui.Window("Fluid Simulation", (size, size))
        canvas = sim_window.get_canvas()
        gui = sim_window.get_gui()
        pixels = np.zeros((size, size, 3), dtype=np.float32)

        # Initialize smoke source for LBM
        if method == 'lbm':
            sim.add_density_velocity(size // 8, size // 2, 1, 0)

        while sim_window.running:
            # GUI controls
            with gui.sub_window("Controls", 0.02, 0.02, 0.25, 0.98):
                gui.text("=== Simulation Parameters ===")
                sim.params.viscosity = gui.slider_float("Viscosity", sim.params.viscosity, 0.0, 1.0)

                if method == 'eulerian':
                    sim.params.brush_size = gui.slider_int("Brush Size", sim.params.brush_size, 1, 20)
                else:
                    sim.params.ball_interaction_strength = gui.slider_float(
                        "Ball-Fluid Interaction",
                        sim.params.ball_interaction_strength,
                        0.0,
                        5.0
                    )

            # Handle mouse interaction
            mouse_pos = sim_window.get_cursor_pos()
            if sim_window.is_pressed(ti.ui.LMB):
                x, y = int(mouse_pos[0] * size), int(mouse_pos[1] * size)
                if method == 'eulerian':
                    sim.add_density_velocity(
                        x, y,
                        mouse_pos[0] - sim.prev_mouse_pos[None][0],
                        mouse_pos[1] - sim.prev_mouse_pos[None][1]
                    )
                else:
                    sim.ball_pos[None] = ti.Vector([float(x), float(y)])

            sim.prev_mouse_pos[None] = ti.Vector([mouse_pos[0], mouse_pos[1]])

            # Update simulation
            sim.step()

            # Render
            sim.render(pixels)
            canvas.set_image(pixels)
            sim_window.show()

        # Clean up
        sim_window.destroy()


if __name__ == "__main__":
    main()
