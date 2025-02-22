import taichi as ti
import numpy as np
from datetime import datetime

# Metadata
__author__ = "tafolabi009"
__created__ = "2025-02-22 07:25:51"
__version__ = "2.4.0"

ti.init(arch=ti.vulkan, default_fp=ti.f32)


class StartScreen:
    def __init__(self, window):
        self.window = window
        self.gui = window.get_gui()
        self.selected_method = "eulerian"
        self.start_simulation = False
        self.brush_size = 20.0  # Default brush size

    def render(self):
        # Center the start menu
        with self.gui.sub_window("Start Menu", 0.3, 0.3, 0.4, 0.4):
            self.gui.text("Welcome to Fluid Simulation")
            self.gui.text("Select Simulation Method:")

            if self.gui.button("Eulerian Method"):
                self.selected_method = "eulerian"
            if self.gui.button("Lattice Boltzmann Method (LBM)"):
                self.selected_method = "lbm"

            self.gui.text("")  # Spacing
            self.gui.text("Initial Brush Size:")
            self.brush_size = self.gui.slider_float("", self.brush_size, 5.0, 50.0)

            self.gui.text("")  # Spacing
            if self.gui.button("Start Simulation"):
                self.start_simulation = True


@ti.data_oriented
class EulerianSolver:
    def __init__(self, width, height, brush_size=20.0):
        self.width = width
        self.height = height
        self.force_radius = brush_size  # Initialize with brush size

        # Core fluid fields
        self.velocity = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))
        self.velocity_tmp = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))
        self.pressure = ti.field(dtype=ti.f32, shape=(width, height))
        self.divergence = ti.field(dtype=ti.f32, shape=(width, height))
        self.density = ti.field(dtype=ti.f32, shape=(width, height))

        # Mouse interaction
        self.prev_mouse_pos = ti.Vector([0.0, 0.0])

        # Simulation parameters
        self.dt = 0.05
        self.num_iterations = 50
        self.velocity_dissipation = 0.999
        self.density_dissipation = 0.995
        self.force_radius = 25.0
        self.force_strength = 70.0

        self.initialize_fields()

    @ti.kernel
    def initialize_fields(self):
        for i, j in self.velocity:
            self.velocity[i, j] = ti.Vector([0.0, 0.0])
            self.pressure[i, j] = 0.0
            self.density[i, j] = 0.0

    @ti.kernel
    def add_force_and_density(self, pos_x: float, pos_y: float, vel_x: float, vel_y: float):
        # Smooth force application using gaussian distribution
        for i, j in ti.ndrange((-20, 20), (-20, 20)):
            x = int(pos_x + i)
            y = int(pos_y + j)

            if 0 <= x < self.width and 0 <= y < self.height:
                dx = float(i)
                dy = float(j)
                d2 = dx * dx + dy * dy
                factor = ti.exp(-d2 / self.force_radius)

                # Add velocity
                self.velocity[x, y] += factor * self.force_strength * ti.Vector([vel_x, vel_y])

                # Add density with velocity-dependent intensity
                vel_magnitude = ti.sqrt(vel_x * vel_x + vel_y * vel_y)
                self.density[x, y] = ti.min(self.density[x, y] + factor * vel_magnitude * 0.1, 1.0)

    @ti.kernel
    def advect(self, field: ti.template(), dissipation: float):
        for i, j in field:
            pos = ti.Vector([float(i), float(j)])
            vel = self.velocity[i, j]
            pos_back = pos - vel * self.dt

            # Clamp positions
            pos_back[0] = ti.max(0.5, ti.min(float(self.width - 1.5), pos_back[0]))
            pos_back[1] = ti.max(0.5, ti.min(float(self.height - 1.5), pos_back[1]))

            field[i, j] = self.sample_field(field, pos_back) * dissipation

    @ti.func
    def sample_field(self, field: ti.template(), pos: ti.template()):
        x0, y0 = int(pos[0]), int(pos[1])
        x1, y1 = x0 + 1, y0 + 1
        fx, fy = pos[0] - x0, pos[1] - y0

        return (field[x0, y0] * (1 - fx) * (1 - fy) +
                field[x1, y0] * fx * (1 - fy) +
                field[x0, y1] * (1 - fx) * fy +
                field[x1, y1] * fx * fy)

    @ti.kernel
    def compute_divergence(self):
        for i, j in self.divergence:
            vl = self.velocity[max(0, i - 1), j].x
            vr = self.velocity[min(self.width - 1, i + 1), j].x
            vb = self.velocity[i, max(0, j - 1)].y
            vt = self.velocity[i, min(self.height - 1, j + 1)].y
            self.divergence[i, j] = (vr - vl + vt - vb) * 0.5

    @ti.kernel
    def solve_pressure(self):
        for i, j in self.pressure:
            pl = self.pressure[max(0, i - 1), j]
            pr = self.pressure[min(self.width - 1, i + 1), j]
            pb = self.pressure[i, max(0, j - 1)]
            pt = self.pressure[i, min(self.height - 1, j + 1)]
            div = self.divergence[i, j]
            self.pressure[i, j] = (pl + pr + pb + pt - div) * 0.25

    @ti.kernel
    def apply_pressure(self):
        for i, j in self.velocity:
            pl = self.pressure[max(0, i - 1), j]
            pr = self.pressure[min(self.width - 1, i + 1), j]
            pb = self.pressure[i, max(0, j - 1)]
            pt = self.pressure[i, min(self.height - 1, j + 1)]
            v = self.velocity[i, j]
            v -= ti.Vector([pr - pl, pt - pb]) * 0.5
            self.velocity[i, j] = v

    def update_brush_size(self, size):
        self.force_radius = size

    def step(self, mouse_pos=None, mouse_down=False):
        if mouse_down and mouse_pos is not None:
            # Calculate mouse velocity for more natural interaction
            current_pos = ti.Vector([mouse_pos[0], mouse_pos[1]])
            velocity = (current_pos - self.prev_mouse_pos) * 0.5
            self.add_force_and_density(mouse_pos[0], mouse_pos[1], velocity[0], velocity[1])
            self.prev_mouse_pos = current_pos
        else:
            self.prev_mouse_pos = ti.Vector([mouse_pos[0], mouse_pos[1]]) if mouse_pos is not None else ti.Vector(
                [0.0, 0.0])

        self.advect(self.velocity, self.velocity_dissipation)
        self.advect(self.density, self.density_dissipation)

        self.compute_divergence()
        for _ in range(self.num_iterations):
            self.solve_pressure()
        self.apply_pressure()


@ti.data_oriented
class LBMSolver:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # LBM parameters
        self.tau = 0.6  # Relaxation time
        self.omega = 1.0 / self.tau

        # Ball parameters
        self.ball_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.ball_vel = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.ball_radius = 15.0
        self.ball_mass = 1.0
        self.gravity = ti.Vector([0.0, -0.1])
        self.restitution = 0.8

        # Lattice velocities (D2Q9)
        self.c = ti.Vector.field(2, dtype=ti.f32, shape=9)
        self.w = ti.field(dtype=ti.f32, shape=9)

        # Distribution functions
        self.f = ti.field(dtype=ti.f32, shape=(width, height, 9))
        self.f_temp = ti.field(dtype=ti.f32, shape=(width, height, 9))

        # Macroscopic quantities
        self.rho = ti.field(dtype=ti.f32, shape=(width, height))
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))

        self.initialize_lattice()
        self.initialize_fields()
        self.initialize_ball()

    @ti.kernel
    def initialize_ball(self):
        self.ball_pos[None] = ti.Vector([self.width * 0.5, self.height * 0.8])
        self.ball_vel[None] = ti.Vector([2.0, 0.0])  # Initial horizontal velocity

    @ti.kernel
    def initialize_lattice(self):
        # D2Q9 lattice velocities
        self.c[0] = ti.Vector([0, 0])
        self.c[1] = ti.Vector([1, 0])
        self.c[2] = ti.Vector([0, 1])
        self.c[3] = ti.Vector([-1, 0])
        self.c[4] = ti.Vector([0, -1])
        self.c[5] = ti.Vector([1, 1])
        self.c[6] = ti.Vector([-1, 1])
        self.c[7] = ti.Vector([-1, -1])
        self.c[8] = ti.Vector([1, -1])

        # Lattice weights
        self.w[0] = 4.0 / 9.0
        for i in ti.static(range(1, 5)):
            self.w[i] = 1.0 / 9.0
        for i in ti.static(range(5, 9)):
            self.w[i] = 1.0 / 36.0

    @ti.kernel
    def initialize_fields(self):
        # Initialize with fluid at rest
        for i, j in ti.ndrange(self.width, self.height):
            self.rho[i, j] = 1.0
            self.vel[i, j] = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                self.f[i, j, k] = self.w[k]

    @ti.kernel
    def update_ball(self):
        # Update ball position and velocity
        pos = self.ball_pos[None]
        vel = self.ball_vel[None]

        # Apply gravity
        vel += self.gravity

        # Update position
        pos += vel

        # Boundary collisions with damping
        if pos[0] < self.ball_radius:
            pos[0] = self.ball_radius
            vel[0] *= -self.restitution
        if pos[0] > self.width - self.ball_radius:
            pos[0] = self.width - self.ball_radius
            vel[0] *= -self.restitution
        if pos[1] < self.ball_radius:
            pos[1] = self.ball_radius
            vel[1] *= -self.restitution
        if pos[1] > self.height - self.ball_radius:
            pos[1] = self.height - self.ball_radius
            vel[1] *= -self.restitution

        # Fluid interaction - approximate drag force
        ball_i = int(pos[0])
        ball_j = int(pos[1])
        if 0 <= ball_i < self.width and 0 <= ball_j < self.height:
            fluid_vel = self.vel[ball_i, ball_j]
            rel_vel = vel - fluid_vel
            drag = -0.1 * rel_vel  # Simple drag model
            vel += drag

        # Update ball state
        self.ball_pos[None] = pos
        self.ball_vel[None] = vel

    @ti.kernel
    def apply_ball_boundary(self):
        # Apply no-slip boundary condition around the ball
        pos = self.ball_pos[None]
        radius = self.ball_radius

        for i, j in ti.ndrange((-int(radius) - 1, int(radius) + 2), (-int(radius) - 1, int(radius) + 2)):
            x = int(pos[0]) + i
            y = int(pos[1]) + j

            if 0 <= x < self.width and 0 <= y < self.height:
                dx = float(i)
                dy = float(j)
                d2 = dx * dx + dy * dy

                if d2 <= radius * radius:
                    # Inside ball - set velocity to ball velocity
                    self.vel[x, y] = self.ball_vel[None]
                    # Adjust density for incompressibility
                    self.rho[x, y] = 1.0

    @ti.kernel
    def collide(self):
        for i, j in ti.ndrange(self.width, self.height):
            # Calculate macroscopic quantities
            rho = 0.0
            vel = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                rho += self.f[i, j, k]
                vel += self.c[k] * self.f[i, j, k]

            if rho > 0:
                vel /= rho

            # Store macroscopic quantities
            self.rho[i, j] = rho
            self.vel[i, j] = vel

            # Collision
            for k in ti.static(range(9)):
                cu = self.c[k].dot(vel)
                usqr = vel.dot(vel)
                feq = self.w[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr)
                self.f_temp[i, j, k] = self.f[i, j, k] - self.omega * (self.f[i, j, k] - feq)

    @ti.kernel
    def stream(self):
        for i, j, k in self.f:
            ni = (i + int(self.c[k][0])) % self.width
            nj = (j + int(self.c[k][1])) % self.height
            self.f[ni, nj, k] = self.f_temp[i, j, k]

    def step(self):
        # Update ball position
        self.update_ball()

        # LBM steps
        self.collide()
        self.stream()

        # Apply ball boundary conditions
        self.apply_ball_boundary()


class SimulationManager:
    def __init__(self, width, height, initial_brush_size=20.0):
        self.width = width
        self.height = height
        self.method = "eulerian"
        self.view_mode = "density"
        self.brush_size = initial_brush_size

        self.eulerian_solver = EulerianSolver(width, height, self.brush_size)
        self.lbm_solver = LBMSolver(width, height)

    def update_brush_size(self, size):
        self.brush_size = size
        if self.method == "eulerian":
            self.eulerian_solver.update_brush_size(size)

    def update(self, mouse_pos=None, mouse_down=False):
        if self.method == "eulerian":
            self.eulerian_solver.step(mouse_pos, mouse_down)
        else:
            self.lbm_solver.step()

    def get_display_field(self):
        if self.method == "eulerian":
            if self.view_mode == "density":
                return self.eulerian_solver.density.to_numpy()
            elif self.view_mode == "pressure":
                return self.eulerian_solver.pressure.to_numpy()
            else:  # velocity
                vel = self.eulerian_solver.velocity.to_numpy()
                return np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)
        else:
            return self.lbm_solver.rho.to_numpy()

    def get_ball_info(self):
        if self.method == "lbm":
            return {
                'pos': self.lbm_solver.ball_pos[None],
                'radius': self.lbm_solver.ball_radius,
                'velocity': self.lbm_solver.ball_vel[None]
            }
        return None


class GUIManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.window = ti.ui.Window("Fluid Simulation", (width, height), vsync=True)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.start_screen = StartScreen(self.window)

    def render(self, simulation):
        # Create GUI controls
        with self.gui.sub_window("Controls", 0.02, 0.02, 0.2, 0.25):
            if self.gui.button("Eulerian"):
                simulation.method = "eulerian"
            if self.gui.button("LBM"):
                simulation.method = "lbm"

            if simulation.method == "eulerian":
                if self.gui.button("Density"):
                    simulation.view_mode = "density"
                if self.gui.button("Pressure"):
                    simulation.view_mode = "pressure"
                if self.gui.button("Velocity"):
                    simulation.view_mode = "velocity"

                # Add brush size control
                new_brush_size = self.gui.slider_float("Brush Size", simulation.brush_size, 5.0, 70.0)
                if new_brush_size != simulation.brush_size:
                    simulation.update_brush_size(new_brush_size)

        # Get field to display
        field = simulation.get_display_field()

        # Normalize field for display
        if field is not None:
            field = (field - field.min()) / (field.max() - field.min() + 1e-8)

            # Create RGB image
            if simulation.method == "lbm":
                # White background with grey smoke for LBM
                img = np.ones((self.height, self.width, 3), dtype=np.float32)
                img *= (1.0 - field[:, :, np.newaxis] * 0.5)  # Lighter grey effect

                # Draw ball
                ball_info = simulation.get_ball_info()
                if ball_info:
                    ball_pos = ball_info['pos']
                    radius = ball_info['radius']
                    velocity = ball_info['velocity']

                    # Draw ball
                    x, y = int(ball_pos[0]), int(ball_pos[1])
                    r = int(radius)

                    # Draw ball with a simple circle
                    for i in range(-r, r + 1):
                        for j in range(-r, r + 1):
                            if i * i + j * j <= r * r:
                                px, py = x + i, y + j
                                if 0 <= px < self.width and 0 <= py < self.height:
                                    # Red ball with shading
                                    img[py, px] = [1.0, 0.2, 0.2]  # Red color

                    # Add velocity indicator
                    vel_scale = 10.0
                    end_x = int(x + velocity[0] * vel_scale)
                    end_y = int(y + velocity[1] * vel_scale)
                    self.draw_line(img, x, y, end_x, end_y, [0.0, 0.0, 1.0])

            else:
                # Blue tint for Eulerian
                img = np.zeros((self.height, self.width, 3), dtype=np.float32)
                img[:, :, 2] = field  # Blue channel

            self.canvas.set_image(img)

        self.window.show()

    def draw_line(self, img, x0, y0, x1, y1, color):
        """Draw a line on the image using Bresenham's algorithm"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if 0 <= x < self.width and 0 <= y < self.height:
                    img[y, x] = color
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if 0 <= x < self.width and 0 <= y < self.height:
                    img[y, x] = color
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy


def main():
    print(f"Enhanced Fluid Simulation v{__version__}")
    width, height = 800, 800

    gui_manager = GUIManager(width, height)
    start_screen = gui_manager.start_screen
    sim_manager = None

    while gui_manager.window.running:
        if not start_screen.start_simulation:
            # Show start screen
            start_screen.render()
            gui_manager.window.show()
        else:
            # Initialize simulation if not done yet
            if sim_manager is None:
                sim_manager = SimulationManager(width, height, start_screen.brush_size)
                sim_manager.method = start_screen.selected_method

            # Handle input
            mouse_pos = gui_manager.window.get_cursor_pos()
            mouse_pos = (mouse_pos[0] * width, mouse_pos[1] * height)
            mouse_down = gui_manager.window.is_pressed(ti.ui.LMB)

            # Update simulation
            sim_manager.update(mouse_pos, mouse_down)

            # Render
            gui_manager.render(sim_manager)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")