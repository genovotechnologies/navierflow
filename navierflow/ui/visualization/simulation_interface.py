import taichi as ti
import numpy as np
from datetime import datetime
from navierflow.core import CoreEulerianSolver, CoreLBMSolver

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
        self.brush_size = 20.0
        self.show_tutorial = False
        self.tutorial_system = TutorialSystem()
        self.simulation_mode = "educational"  # or "research"
        self.selected_preset = "basic"
        self.presets = {
            "basic": {"viscosity": 0.1, "iterations": 50},
            "research": {"viscosity": 0.05, "iterations": 100},
            "high_accuracy": {"viscosity": 0.01, "iterations": 200}
        }

    def render(self):
        # Center the start menu
        with self.gui.sub_window("NavierFlow", 0.25, 0.2, 0.5, 0.6):
            self.gui.text("Welcome to NavierFlow")
            self.gui.text("Advanced Fluid Dynamics Simulation Engine")
            self.gui.text("")

            self.gui.text("Select Mode:")
            if self.gui.button("Educational Mode"):
                self.simulation_mode = "educational"
            if self.gui.button("Research Mode"):
                self.simulation_mode = "research"
            
            self.gui.text("")
            self.gui.text("Select Simulation Method:")
            if self.gui.button("Eulerian (Navier-Stokes)"):
                self.selected_method = "eulerian"
            if self.gui.button("Lattice Boltzmann Method (LBM)"):
                self.selected_method = "lbm"

            self.gui.text("")
            self.gui.text("Simulation Preset:")
            if self.gui.button("Basic"):
                self.selected_preset = "basic"
            if self.gui.button("Research Grade"):
                self.selected_preset = "research"
            if self.gui.button("High Accuracy"):
                self.selected_preset = "high_accuracy"

            self.gui.text("")
            self.gui.text("Initial Brush Size:")
            self.brush_size = self.gui.slider_float("", self.brush_size, 5.0, 50.0)

            if self.simulation_mode == "educational":
                self.gui.text("")
                self.gui.text("Tutorial Options:")
                if self.gui.button("Enable Tutorial"):
                    self.show_tutorial = True
                    self.tutorial_system.start_tutorial("basics")

            self.gui.text("")
            if self.gui.button("Start Simulation"):
                self.start_simulation = True

            # Version info
            self.gui.text(f"\nVersion {__version__}")
            self.gui.text("For educational and research applications")


class SimulationManager:
    def __init__(self, width, height, initial_brush_size=20.0):
        self.width = width
        self.height = height
        self.method = "eulerian"
        self.view_mode = "density"
        self.brush_size = initial_brush_size

        # Initialize solvers with configuration
        self.eulerian_solver = CoreEulerianSolver(
            width=width,
            height=height,
            config={
                'force_radius': initial_brush_size,
                'force_strength': 70.0
            }
        )
        self.lbm_solver = CoreLBMSolver(
            width=width,
            height=height,
            config={
                'force_radius': 5.0,
                'force_magnitude': 0.01
            }
        )

    def update_brush_size(self, size):
        self.brush_size = size
        if self.method == "eulerian":
            self.eulerian_solver.update_config({'force_radius': size})

    def update(self, mouse_pos=None, mouse_down=False):
        if self.method == "eulerian":
            self.eulerian_solver.step(mouse_pos, mouse_down)
        else:
            self.lbm_solver.step(mouse_pos, mouse_down)

    def get_display_field(self):
        if self.method == "eulerian":
            state = self.eulerian_solver.get_state()
            if self.view_mode == "density":
                return state['density']
            elif self.view_mode == "pressure":
                return state['pressure']
            else:  # velocity
                vel = state['velocity']
                return np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)
        else:  # LBM method
            state = self.lbm_solver.get_state()
            # Get both density and velocity for better visualization
            rho = state['density']
            vel = state['velocity']
            # Combine density and velocity magnitude for smoke effect
            vel_mag = np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)
            return 1.0 - np.clip(vel_mag * 0.5 + (rho - 1.0) * 0.2, 0, 1)

    def get_ball_info(self):
        if self.method == "lbm":
            state = self.lbm_solver.get_state()
            return {
                'pos': state['ball_position'],
                'radius': self.lbm_solver.config['ball_radius'],
                'velocity': state['ball_velocity']
            }
        return None


class GUIManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.window = ti.ui.Window("NavierFlow", (width, height), vsync=True)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.start_screen = StartScreen(self.window)
        self.show_tutorial = False
        self.show_help = False
        self.show_analytics = False
        self.current_tutorial_step = 0
        self.color_mode = "blue"  # blue, fire, rainbow, grayscale
        self.show_vectors = False

    def render(self, simulation):
        # Main control panel
        with self.gui.sub_window("NavierFlow Controls", 0.02, 0.02, 0.2, 0.4):
            # Mode indicator
            self.gui.text(f"Mode: {'Educational' if self.start_screen.simulation_mode == 'educational' else 'Research'}")
            
            # Method selection
            if self.gui.button("Eulerian (Navier-Stokes)"):
                simulation.method = "eulerian"
            if self.gui.button("Lattice Boltzmann (LBM)"):
                simulation.method = "lbm"

            # Visualization controls
            self.gui.text("\nVisualization:")
            if simulation.method == "eulerian":
                if self.gui.button("Density Field"):
                    simulation.view_mode = "density"
                if self.gui.button("Pressure Field"):
                    simulation.view_mode = "pressure"
                if self.gui.button("Velocity Field"):
                    simulation.view_mode = "velocity"

            # Color mode selection
            self.gui.text("\nColor Scheme:")
            if self.gui.button("Blue Gradient"):
                self.color_mode = "blue"
            if self.gui.button("Fire Effect"):
                self.color_mode = "fire"
            if self.gui.button("Rainbow"):
                self.color_mode = "rainbow"
            if self.gui.button("Grayscale"):
                self.color_mode = "grayscale"

            # Simulation controls
            self.gui.text("\nSimulation Controls:")
            new_brush_size = self.gui.slider_float("Brush Size", simulation.brush_size, 5.0, 70.0)
            if new_brush_size != simulation.brush_size:
                simulation.update_brush_size(new_brush_size)

            # Toggle vector visualization
            if self.gui.button("Toggle Vectors"):
                self.show_vectors = not self.show_vectors

            # Educational features
            if self.start_screen.simulation_mode == "educational":
                self.gui.text("\nEducational Tools:")
                if self.gui.button("Show Tutorial"):
                    self.show_tutorial = not self.show_tutorial
                if self.gui.button("Show Help"):
                    self.show_help = not self.show_help

            # Research features
            if self.start_screen.simulation_mode == "research":
                self.gui.text("\nResearch Tools:")
                if self.gui.button("Show Analytics"):
                    self.show_analytics = not self.show_analytics

        # Tutorial window
        if self.show_tutorial:
            with self.gui.sub_window("Tutorial", 0.25, 0.7, 0.5, 0.28):
                self.gui.text("NavierFlow Tutorial")
                # Tutorial content here
                if self.gui.button("Next Step"):
                    self.current_tutorial_step += 1

        # Help window
        if self.show_help:
            with self.gui.sub_window("Help", 0.25, 0.4, 0.5, 0.28):
                self.gui.text("NavierFlow Help")
                self.gui.text("- Click and drag to add fluid")
                self.gui.text("- Use controls to change visualization")
                self.gui.text("- Press ESC to exit")

        # Analytics window
        if self.show_analytics:
            with self.gui.sub_window("Analytics", 0.75, 0.02, 0.23, 0.3):
                self.gui.text("Simulation Analytics")
                if simulation.method == "eulerian":
                    vel = simulation.eulerian_solver.velocity.to_numpy()
                    avg_vel = np.mean(np.sqrt(vel[:, :, 0]**2 + vel[:, :, 1]**2))
                    self.gui.text(f"Average Velocity: {avg_vel:.4f}")
                    self.gui.text(f"Iterations: {simulation.eulerian_solver.num_iterations}")

        # Display simulation field with enhanced visualization
        field = simulation.get_display_field()
        if field is not None:
            # Normalize field
            field_min = np.min(field)
            field_max = np.max(field)
            if field_max > field_min:
                field = (field - field_min) / (field_max - field_min)
            else:
                field = np.zeros_like(field)

            # Apply color scheme
            img = np.zeros((self.height, self.width, 3), dtype=np.float32)
            if self.color_mode == "blue":
                img[:, :, 2] = field
            elif self.color_mode == "fire":
                img[:, :, 0] = field
                img[:, :, 1] = field * 0.5
            elif self.color_mode == "rainbow":
                for i in range(3):
                    img[:, :, i] = np.sin(field * np.pi * (i + 1) / 2)
            else:  # grayscale
                img[:, :] = field[:, :, np.newaxis]

            # Draw velocity vectors if enabled
            if self.show_vectors and simulation.method == "eulerian":
                vel = simulation.eulerian_solver.velocity.to_numpy()
                step = 20  # Vector field resolution
                scale = 50.0  # Vector length scale
                for i in range(0, self.height, step):
                    for j in range(0, self.width, step):
                        vx = vel[i, j, 0] * scale
                        vy = vel[i, j, 1] * scale
                        self.draw_line(img, j, i, j + int(vx), i + int(vy), np.array([1.0, 1.0, 1.0]))

            self.canvas.set_image(img)
        self.window.show()

    def draw_line(self, img, x0, y0, x1, y1, color):
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
            start_screen.render()
            gui_manager.window.show()
        else:
            if sim_manager is None:
                sim_manager = SimulationManager(width, height, start_screen.brush_size)
                sim_manager.method = start_screen.selected_method

            mouse_pos = gui_manager.window.get_cursor_pos()
            mouse_pos = (mouse_pos[0] * width, mouse_pos[1] * height)
            mouse_down = gui_manager.window.is_pressed(ti.ui.LMB)

            sim_manager.update(mouse_pos, mouse_down)
            gui_manager.render(sim_manager)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
