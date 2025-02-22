import taichi as ti  # Remove duplicate import
import numpy as np
from datetime import datetime

# Metadata
__author__ = "tafolabi009"
__created__ = "2025-02-22 07:25:51"
__version__ = "2.3.1"

# Initialize Taichi with better performance settings
ti.init(arch=ti.gpu,  # Use GPU if available
        debug=False,  # Disable debug mode for better performance
        default_fp=ti.f32,  # Use float32 by default
        kernel_profiler=True)  # Enable profiling


@ti.data_oriented
class EulerianSolver:
    def __init__(self, width, height):
        # Reduce dimensions for better performance
        self.scale_factor = 2  # Downscale factor for internal simulation
        self.width = width
        self.height = height

        # Internal simulation dimensions
        self.sim_width = width // self.scale_factor
        self.sim_height = height // self.scale_factor

        print(f"Initializing simulation with dimensions: {self.sim_width}x{self.sim_height}")

        # Core fluid fields with reduced size
        self.velocity = ti.Vector.field(2, dtype=ti.f32, shape=(self.sim_width, self.sim_height))
        self.velocity_tmp = ti.Vector.field(2, dtype=ti.f32, shape=(self.sim_width, self.sim_height))
        self.pressure = ti.field(dtype=ti.f32, shape=(self.sim_width, self.sim_height))
        self.divergence = ti.field(dtype=ti.f32, shape=(self.sim_width, self.sim_height))
        self.density = ti.field(dtype=ti.f32, shape=(self.sim_width, self.sim_height))

        # Mouse interaction
        self.prev_mouse_pos = ti.Vector([0.0, 0.0])

        # Optimized simulation parameters
        self.dt = 0.033  # Larger timestep for smoother simulation
        self.num_iterations = 10  # Reduced pressure iterations
        self.velocity_dissipation = 0.999
        self.density_dissipation = 0.995
        self.force_radius = 10.0  # Reduced force radius
        self.force_strength = 15.0  # Reduced force

        # Frame timing
        self.frame_time = 0.0
        self.frame_count = 0

        self.initialize_fields()

    @ti.kernel
    def initialize_fields(self):
        for i, j in ti.ndrange(self.sim_width, self.sim_height):
            self.velocity[i, j] = ti.Vector([0.0, 0.0])
            self.pressure[i, j] = 0.0
            self.density[i, j] = 0.0

    @ti.kernel
    def add_force_and_density(self, pos_x: float, pos_y: float, vel_x: float, vel_y: float, brush_size: float):
        # Scale input coordinates to simulation dimensions
        pos_x = pos_x / self.scale_factor
        pos_y = pos_y / self.scale_factor

        radius = ti.max(5.0, brush_size * 5.0)  # Reduced radius for better performance

        for i, j in ti.ndrange((-int(radius), int(radius)), (-int(radius), int(radius))):
            x = int(pos_x + i)
            y = int(pos_y + j)

            if 0 <= x < self.sim_width and 0 <= y < self.sim_height:
                dx = float(i)
                dy = float(j)
                d2 = dx * dx + dy * dy
                factor = ti.exp(-d2 / (radius * 0.5))

                # Add velocity with stronger clamping
                force = factor * self.force_strength * ti.Vector([vel_x, vel_y])
                self.velocity[x, y] += ti.Vector([
                    ti.min(ti.max(force[0], -50.0), 50.0),
                    ti.min(ti.max(force[1], -50.0), 50.0)
                ])

                # Add density with smoother accumulation
                vel_magnitude = ti.sqrt(vel_x * vel_x + vel_y * vel_y)
                self.density[x, y] = ti.min(self.density[x, y] + factor * vel_magnitude * 0.05, 1.0)

    @ti.kernel
    def advect(self, field: ti.template(), dissipation: float):
        for i, j in field:
            pos = ti.Vector([float(i), float(j)])
            vel = self.velocity[i, j]
            pos_back = pos - vel * self.dt

            # Improved boundary handling
            pos_back[0] = ti.max(0.5, ti.min(float(self.sim_width - 1.5), pos_back[0]))
            pos_back[1] = ti.max(0.5, ti.min(float(self.sim_height - 1.5), pos_back[1]))

            if ti.static(field.n == 2):
                field[i, j] = self.sample_vector_field(field, pos_back) * dissipation
            else:
                field[i, j] = self.sample_scalar_field(field, pos_back) * dissipation

    @ti.func
    def sample_scalar_field(self, field: ti.template(), pos: ti.template()):
        x0, y0 = int(pos[0]), int(pos[1])
        x1, y1 = x0 + 1, y0 + 1
        fx, fy = pos[0] - x0, pos[1] - y0

        return (field[x0, y0] * (1 - fx) * (1 - fy) +
                field[x1, y0] * fx * (1 - fy) +
                field[x0, y1] * (1 - fx) * fy +
                field[x1, y1] * fx * fy)

    @ti.func
    def sample_vector_field(self, field: ti.template(), pos: ti.template()):
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
            vr = self.velocity[min(self.sim_width - 1, i + 1), j].x
            vb = self.velocity[i, max(0, j - 1)].y
            vt = self.velocity[i, min(self.sim_height - 1, j + 1)].y
            self.divergence[i, j] = (vr - vl + vt - vb) * 0.5

    def step(self, mouse_pos, mouse_down, brush_size):
        try:
            if mouse_down and mouse_pos is not None:
                current_pos = ti.Vector([mouse_pos[0], mouse_pos[1]])
                velocity = (current_pos - self.prev_mouse_pos) * 0.5
                self.add_force_and_density(mouse_pos[0], mouse_pos[1], velocity[0], velocity[1], brush_size)
                self.prev_mouse_pos = current_pos
            else:
                self.prev_mouse_pos = ti.Vector([mouse_pos[0], mouse_pos[1]]) if mouse_pos is not None else ti.Vector(
                    [0.0, 0.0])

            # Simulation steps
            self.advect(self.velocity, self.velocity_dissipation)
            self.advect(self.density, self.density_dissipation)
            self.compute_divergence()

            # Reduced pressure iterations
            for _ in range(self.num_iterations):
                self.solve_pressure()
            self.apply_pressure()

            # Performance monitoring
            self.frame_count += 1
            if self.frame_count % 60 == 0:
                ti.profiler.print_kernel_profiler_info()
                print(f"Average frame time: {self.frame_time / 60:.3f}ms")
                self.frame_time = 0.0

        except Exception as e:
            print(f"Error in simulation step: {e}")


class SimulationManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        print(f"Initializing simulation manager with dimensions: {width}x{height}")

        self.method = "eulerian"  # Start with Eulerian only
        self.view_mode = "density"
        self.brush_size = 1.0
        self.paused = False  # Add pause functionality

        # Initialize only Eulerian solver
        self.eulerian_solver = EulerianSolver(width, height)

        # Performance monitoring
        self.last_time = datetime.now()
        self.frame_count = 0

    def update(self, mouse_pos=None, mouse_down=False):
        if self.paused:
            return

        try:
            # Performance monitoring
            self.frame_count += 1
            if self.frame_count % 60 == 0:
                current_time = datetime.now()
                delta = (current_time - self.last_time).total_seconds()
                fps = 60 / delta if delta > 0 else 0
                print(f"FPS: {fps:.1f}")
                self.last_time = current_time

            # Update simulation
            if self.method == "eulerian":
                self.eulerian_solver.step(mouse_pos, mouse_down, self.brush_size)

        except Exception as e:
            print(f"Error in update: {e}")

    def get_display_field(self):
        try:
            if self.method == "eulerian":
                solver = self.eulerian_solver
                field = None

                if self.view_mode == "density":
                    field = solver.density.to_numpy()
                elif self.view_mode == "pressure":
                    field = solver.pressure.to_numpy()
                else:  # velocity
                    vel = solver.velocity.to_numpy()
                    field = np.clip(np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2), 0, 1)

                # Upscale the field to display resolution
                return np.repeat(np.repeat(field, solver.scale_factor, axis=0),
                                 solver.scale_factor, axis=1)

            return np.zeros((self.width, self.height))

        except Exception as e:
            print(f"Error in get_display_field: {e}")
            return np.zeros((self.width, self.height))


class GUIManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        try:
            self.window = ti.ui.Window("Fluid Simulation", (width, height),
                                       vsync=True)  # Enable vsync
            self.canvas = self.window.get_canvas()
            self.gui = self.window.get_gui()
            self.show_start_screen = True
            self.show_about = False

            # GUI state
            self.frame_time = 0.0
            self.frame_count = 0
            self.last_time = datetime.now()

        except Exception as e:
            print(f"Error initializing GUI: {e}")
            raise

    def render(self, simulation):
        try:
            # Performance monitoring
            self.frame_count += 1
            current_time = datetime.now()
            if self.frame_count % 60 == 0:
                delta = (current_time - self.last_time).total_seconds()
                fps = 60 / delta if delta > 0 else 0
                print(f"GUI FPS: {fps:.1f}")
                self.last_time = current_time

            if self.show_start_screen:
                self.render_start_screen()
                if self.show_about:
                    self.render_about()
                return

            # Display the field
            field = simulation.get_display_field()
            self.canvas.set_image(field)

            # Simplified GUI controls
            with self.gui.sub_window("Controls", 0.02, 0.02, 0.2, 0.15):
                if simulation.method == "eulerian":
                    # View mode cycling
                    if self.gui.button("View: " + simulation.view_mode):
                        modes = ["density", "pressure", "velocity"]
                        curr_idx = modes.index(simulation.view_mode)
                        simulation.view_mode = modes[(curr_idx + 1) % len(modes)]

                    # Brush size control
                    _, simulation.brush_size = self.gui.slider("Brush",
                                                               simulation.brush_size,
                                                               0.1, 3.0)

                    # Pause button
                    if self.gui.button("Pause" if not simulation.paused else "Resume"):
                        simulation.paused = not simulation.paused

            self.window.show()

        except Exception as e:
            print(f"Error in render: {e}")


def main():
    print(f"Enhanced Fluid Simulation v{__version__}")

    # Use reasonable dimensions
    width = 512
    height = 512

    try:
        sim_manager = SimulationManager(width, height)
        gui_manager = GUIManager(width, height)

        while gui_manager.window.running:
            try:
                # Handle input
                mouse_pos = gui_manager.window.get_cursor_pos()
                mouse_pos = (mouse_pos[0] * width, mouse_pos[1] * height)
                mouse_down = gui_manager.window.is_pressed(ti.ui.LMB)

                # Update and render
                if not gui_manager.show_start_screen:
                    sim_manager.update(mouse_pos, mouse_down)
                gui_manager.render(sim_manager)

            except Exception as e:
                print(f"Error in main loop: {e}")
                continue  # Continue running even if there's an error

    except Exception as e:
        print(f"Fatal error in main: {e}")


if __name__ == "__main__":
    main()