import pygame
import numpy as np
from dataclasses import dataclass
import sys

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
BLUE = (0, 0, 255)
RED = (255, 0, 0)


class EulerianFluid:
    def __init__(self, nx, ny, dt, viscosity):
        self.nx = nx
        self.ny = ny
        self.dt = 0.01
        self.viscosity = 0.001

        # Initialize fields
        self.u = np.zeros((nx, ny))  # x-velocity
        self.v = np.zeros((nx, ny))  # y-velocity
        self.density = np.zeros((nx, ny))
        self.pressure = np.zeros((nx, ny))

        # Precompute constants
        self.dx = 1.0
        self.dy = 1.0

        # Add CFL condition check
        self.max_velocity = 10.0  # Maximum allowed velocity

    def enforce_cfl_condition(self):
        """Enforce CFL condition by limiting velocities"""
        velocity_magnitude = np.sqrt(self.u ** 2 + self.v ** 2)
        scale_factor = np.minimum(1.0, self.max_velocity / (velocity_magnitude + 1e-6))
        self.u *= scale_factor
        self.v *= scale_factor

    def solve_pressure_poisson(self, max_iter=50, tolerance=1e-3):
        """Improved pressure Poisson solver with better convergence"""
        omega = 1.7  # Over-relaxation factor for faster convergence

        for _ in range(max_iter):
            previous_p = self.pressure.copy()

            # Calculate divergence
            div = (
                    (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2 * self.dx) +
                    (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * self.dy)
            )

            # SOR iteration
            self.pressure[1:-1, 1:-1] = (1 - omega) * previous_p[1:-1, 1:-1] + \
                                        omega * 0.25 * (
                                                previous_p[2:, 1:-1] +
                                                previous_p[:-2, 1:-1] +
                                                previous_p[1:-1, 2:] +
                                                previous_p[1:-1, :-2] -
                                                self.dx * self.dy / self.dt * div
                                        )

            # Neumann boundary conditions
            self.pressure[0, :] = self.pressure[1, :]
            self.pressure[-1, :] = self.pressure[-2, :]
            self.pressure[:, 0] = self.pressure[:, 1]
            self.pressure[:, -1] = self.pressure[:, -2]

            if np.max(np.abs(previous_p - self.pressure)) < tolerance:
                break

    def update(self):
        """Updated solver with improved stability"""
        # Store previous velocities
        u_prev = np.clip(self.u.copy(), -self.max_velocity, self.max_velocity)
        v_prev = np.clip(self.v.copy(), -self.max_velocity, self.max_velocity)

        # Solve pressure
        self.solve_pressure_poisson()

        # Helper function for safe division
        def safe_div(a, b, fallback=0.0):
            return np.where(np.abs(b) > 1e-6, a / b, fallback)

        # Update u velocity (x-direction) with limited gradients
        du_dx = safe_div(u_prev[1:-1, 1:-1] - u_prev[:-2, 1:-1], self.dx)
        du_dy = safe_div(u_prev[1:-1, 1:-1] - u_prev[1:-1, :-2], self.dy)

        self.u[1:-1, 1:-1] = u_prev[1:-1, 1:-1] - \
                             np.clip(u_prev[1:-1, 1:-1] * self.dt * du_dx, -1, 1) - \
                             np.clip(v_prev[1:-1, 1:-1] * self.dt * du_dy, -1, 1) - \
                             safe_div(
                                 self.dt * (self.pressure[2:, 1:-1] - self.pressure[1:-1, 1:-1]),
                                 self.dx * np.maximum(self.density[1:-1, 1:-1], 0.1)
                             )

        # Add diffusion term with limited coefficients
        self.u[1:-1, 1:-1] += self.viscosity * self.dt * (
                safe_div(u_prev[2:, 1:-1] - 2 * u_prev[1:-1, 1:-1] + u_prev[:-2, 1:-1], self.dx ** 2) +
                safe_div(u_prev[1:-1, 2:] - 2 * u_prev[1:-1, 1:-1] + u_prev[1:-1, :-2], self.dy ** 2)
        )

        # Update v velocity (y-direction) with similar improvements
        dv_dx = safe_div(v_prev[1:-1, 1:-1] - v_prev[:-2, 1:-1], self.dx)
        dv_dy = safe_div(v_prev[1:-1, 1:-1] - v_prev[1:-1, :-2], self.dy)

        self.v[1:-1, 1:-1] = v_prev[1:-1, 1:-1] - \
                             np.clip(u_prev[1:-1, 1:-1] * self.dt * dv_dx, -1, 1) - \
                             np.clip(v_prev[1:-1, 1:-1] * self.dt * dv_dy, -1, 1) - \
                             safe_div(
                                 self.dt * (self.pressure[1:-1, 2:] - self.pressure[1:-1, 1:-1]),
                                 self.dy * np.maximum(self.density[1:-1, 1:-1], 0.1)
                             )

        # Add diffusion term with limited coefficients
        self.v[1:-1, 1:-1] += self.viscosity * self.dt * (
                safe_div(v_prev[2:, 1:-1] - 2 * v_prev[1:-1, 1:-1] + v_prev[:-2, 1:-1], self.dx ** 2) +
                safe_div(v_prev[1:-1, 2:] - 2 * v_prev[1:-1, 1:-1] + v_prev[1:-1, :-2], self.dy ** 2)
        )

        # Enforce CFL condition
        self.enforce_cfl_condition()

        # Boundary conditions
        self.u[0, :] = 0
        self.u[-1, :] = 0
        self.u[:, 0] = 0
        self.u[:, -1] = 0

        self.v[0, :] = 0
        self.v[-1, :] = 0
        self.v[:, 0] = 0
        self.v[:, -1] = 0

        # Update density with improved advection-diffusion
        density_grad_x = np.gradient(self.density, axis=0)
        density_grad_y = np.gradient(self.density, axis=1)

        self.density = self.density - self.dt * np.clip(
            u_prev * density_grad_x + v_prev * density_grad_y,
            -1, 1
        )

        # Add diffusion with stability limits
        self.density += np.clip(
            self.viscosity * self.dt * (
                    np.gradient(np.gradient(self.density, axis=0), axis=0) +
                    np.gradient(np.gradient(self.density, axis=1), axis=1)
            ),
            -0.1, 0.1
        )

        # Ensure density stays in valid range
        self.density = np.clip(self.density, 0, 1)


class LBMFluid:
    def __init__(self, nx, ny, tau):
        self.nx = nx
        self.ny = ny
        self.tau = tau  # Relaxation time

        # D2Q9 lattice velocities
        self.c = np.array([
            [0, 0],
            [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1]
        ])

        # Lattice weights
        self.w = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)

        # Initialize distribution functions
        self.f = np.ones((9, nx, ny)) * self.w.reshape(-1, 1, 1)
        self.f_eq = np.zeros_like(self.f)

        # Initialize macroscopic variables
        self.density = np.ones((nx, ny))
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))

    def equilibrium(self):
        cu = np.zeros((9, self.nx, self.ny))
        usqr = self.u ** 2 + self.v ** 2

        for i in range(9):
            cu[i] = self.c[i, 0] * self.u + self.c[i, 1] * self.v

        for i in range(9):
            self.f_eq[i] = self.w[i] * self.density * (
                    1 + 3 * cu[i] + 4.5 * cu[i] ** 2 - 1.5 * usqr
            )

    def streaming(self):
        for i in range(9):
            self.f[i] = np.roll(
                np.roll(self.f[i], self.c[i, 0], axis=0),
                self.c[i, 1], axis=1
            )

    def collision(self):
        self.density = np.sum(self.f, axis=0)
        self.u = np.sum(self.f * self.c[:, 0].reshape(-1, 1, 1), axis=0) / self.density
        self.v = np.sum(self.f * self.c[:, 1].reshape(-1, 1, 1), axis=0) / self.density

        self.equilibrium()
        self.f += -(1.0 / self.tau) * (self.f - self.f_eq)

    def update(self):
        self.streaming()
        self.collision()

        # Apply bounce-back boundary conditions
        self.f[:, 0, :] = self.f[:, 1, :]
        self.f[:, -1, :] = self.f[:, -2, :]
        self.f[:, :, 0] = self.f[:, :, 1]
        self.f[:, :, -1] = self.f[:, :, -2]


@dataclass
class SimulationParams:
    dt: float = 0.1
    viscosity: float = 0.01
    tau: float = 0.6  # LBM relaxation time
    brush_size: int = 3
    visualization_mode: str = 'fluid'


class StartMenu:
    def __init__(self, screen_width: int, screen_height: int):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Fluid Simulation")
        self.font = pygame.font.Font(None, 36)

        # Button dimensions
        button_width = 400
        button_height = 60
        spacing = 20

        # Calculate positions for centered buttons
        center_x = screen_width // 2
        start_y = screen_height // 2 - button_height - spacing

        self.eulerian_button = pygame.Rect(
            center_x - button_width // 2,
            start_y,
            button_width,
            button_height
        )

        self.boltzmann_button = pygame.Rect(
            center_x - button_width // 2,
            start_y + button_height + spacing,
            button_width,
            button_height
        )

    def run(self) -> str:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if self.eulerian_button.collidepoint(mouse_pos):
                        return 'eulerian'
                    elif self.boltzmann_button.collidepoint(mouse_pos):
                        return 'boltzmann'

            self.screen.fill(WHITE)

            # Draw title
            title = self.font.render("Select Simulation Method", True, BLACK)
            title_rect = title.get_rect(center=(self.screen.get_width() // 2, 100))
            self.screen.blit(title, title_rect)

            # Draw buttons
            pygame.draw.rect(self.screen, LIGHT_GRAY, self.eulerian_button)
            pygame.draw.rect(self.screen, LIGHT_GRAY, self.boltzmann_button)

            eulerian_text = self.font.render("Eulerian Method", True, BLACK)
            boltzmann_text = self.font.render("Lattice Boltzmann Method", True, BLACK)

            self.screen.blit(eulerian_text, eulerian_text.get_rect(center=self.eulerian_button.center))
            self.screen.blit(boltzmann_text, boltzmann_text.get_rect(center=self.boltzmann_button.center))

            pygame.display.flip()


class FluidSimulation:
    def __init__(self, width: int, height: int, method: str):
        self.width = width
        self.height = height
        self.method = method
        self.params = SimulationParams()

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Fluid Simulation - {method.capitalize()} Method")

        # Initialize simulation grid
        self.nx = 300  # Grid resolution
        self.ny = 250

        # Initialize appropriate solver
        if method == 'eulerian':
            self.solver = EulerianFluid(self.nx, self.ny, self.params.dt, self.params.viscosity)
        else:
            self.solver = LBMFluid(self.nx, self.ny, self.params.tau)

        self.font = pygame.font.Font(None, 24)
        self.show_controls = False
        self.setup_ui()

    def setup_ui(self):
        button_height = 30
        button_width = 100
        spacing = 10

        self.ui_buttons = {
            'controls': pygame.Rect(10, 10, button_width, button_height),
            'fluid': pygame.Rect(10, 50, button_width, button_height),
            'pressure': pygame.Rect(10, 90, button_width, button_height),
            'velocity': pygame.Rect(10, 130, button_width, button_height),
        }

    def draw_ui(self):
        for name, rect in self.ui_buttons.items():
            pygame.draw.rect(self.screen, LIGHT_GRAY, rect)
            text = self.font.render(name.capitalize(), True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

        if self.show_controls:
            self.draw_controls()

    def draw_controls(self):
        controls = [
            "Controls:",
            "Left Mouse: Add fluid",
            "Right Mouse: Clear area",
            "1-3: Change brush size",
            "ESC: Exit",
        ]

        surface = pygame.Surface((300, 200))
        surface.fill(WHITE)
        surface.set_alpha(230)
        self.screen.blit(surface, (self.width - 320, 20))

        y = 30
        for line in controls:
            text = self.font.render(line, True, BLACK)
            self.screen.blit(text, (self.width - 310, y))
            y += 30

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                for name, rect in self.ui_buttons.items():
                    if rect.collidepoint(mouse_pos):
                        if name == 'controls':
                            self.show_controls = not self.show_controls
                        else:
                            self.params.visualization_mode = name

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3]:
                    self.params.brush_size = event.key - pygame.K_0

        return True

    def add_fluid(self, x: int, y: int, clear: bool = False):
        # Convert screen coordinates to grid coordinates
        grid_x = int((x / self.width) * self.nx)
        grid_y = int((y / self.height) * self.ny)

        # Apply brush
        radius = self.params.brush_size
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    nx = grid_x + dx
                    ny = grid_y + dy
                    if 0 <= nx < self.nx and 0 <= ny < self.ny:
                        if clear:
                            self.solver.density[nx, ny] = 0
                            if self.method == 'eulerian':
                                self.solver.u[nx, ny] = 0
                                self.solver.v[nx, ny] = 0
                        else:
                            self.solver.density[nx, ny] = 1.0
                            if self.method == 'eulerian':
                                # Add some initial velocity for more interesting flow
                                self.solver.u[nx, ny] = np.random.uniform(-1, 1)
                                self.solver.v[nx, ny] = np.random.uniform(-1, 1)

    def update(self):
        # Handle mouse interaction
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0] or mouse_buttons[2]:  # Left or right mouse button
            mouse_pos = pygame.mouse.get_pos()
            self.add_fluid(mouse_pos[0], mouse_pos[1], mouse_buttons[2])

        # Update fluid simulation
        self.solver.update()

    def render(self):
        self.screen.fill(WHITE)
        surface = pygame.Surface((self.nx, self.ny))

        if self.method == 'eulerian':
            if self.params.visualization_mode == 'fluid':
                # Visualize density field with blue tint
                density_viz = np.clip(self.solver.density * 255, 0, 255).astype(np.uint32)
                color_array = np.zeros((self.nx, self.ny, 3), dtype=np.uint32)
                color_array[..., 2] = density_viz  # Blue channel
                color_array[..., 0] = (density_viz * 0.2).astype(np.uint8)  # Red tint
                color_array[..., 1] = (density_viz * 0.4).astype(np.uint8)  # Green tint

            elif self.params.visualization_mode == 'pressure':
                # Visualize pressure field (red for high, blue for low)
                pressure_viz = np.clip((self.solver.pressure + 1) * 127, 0, 255).astype(np.uint32)
                color_array = np.zeros((self.nx, self.ny, 3), dtype=np.uint8)
                color_array[..., 0] = pressure_viz  # Red for high pressure
                color_array[..., 2] = 255 - pressure_viz  # Blue for low pressure

            elif self.params.visualization_mode == 'velocity':
                # Visualize velocity field magnitude
                velocity_magnitude = np.sqrt(self.solver.u ** 2 + self.solver.v ** 2)
                velocity_viz = np.clip(velocity_magnitude * 255, 0, 255).astype(np.uint32)

                # Create a color wheel effect based on velocity direction
                angle = np.arctan2(self.solver.v, self.solver.u)
                hue = (angle + np.pi) / (2 * np.pi)

                color_array = np.zeros((self.nx, self.ny, 3), dtype=np.uint32)
                for i in range(3):
                    color_array[..., i] = velocity_viz * (1 + np.cos(6 * np.pi * (hue + i / 3))) / 2

        else:  # LBM visualization
            if self.params.visualization_mode == 'fluid':
                # Visualize density field
                density_viz = np.clip(self.solver.density * 255, 0, 255).astype(np.uint32)
                color_array = np.stack([density_viz] * 3, axis=-1)

            elif self.params.visualization_mode == 'velocity':
                # Visualize velocity magnitude
                velocity_magnitude = np.sqrt(self.solver.u ** 2 + self.solver.v ** 2)
                velocity_viz = np.clip(velocity_magnitude * 255, 0, 255).astype(np.uint32)

                angle = np.arctan2(self.solver.v, self.solver.u)
                hue = (angle + np.pi) / (2 * np.pi)

                color_array = np.zeros((self.nx, self.ny, 3), dtype=np.uint32)
                for i in range(3):
                    color_array[..., i] = velocity_viz * (1 + np.cos(6 * np.pi * (hue + i / 3))) / 2

            else:  # Default to fluid visualization for other modes
                density_viz = np.clip(self.solver.density * 255, 0, 255).astype(np.uint32)
                color_array = np.stack([density_viz] * 3, axis=-1)

        # Update pygame surface
        pygame.surfarray.blit_array(surface, color_array)
        scaled = pygame.transform.scale(surface, (self.width, self.height))
        self.screen.blit(scaled, (0, 0))

        self.draw_ui()
        pygame.display.flip()

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            running = self.handle_input()
            self.update()
            self.render()
            clock.tick(60)

        pygame.quit()


def main():
    # Initialize with standard resolution
    width, height = 800, 600

    # Show start menu
    menu = StartMenu(width, height)
    method = menu.run()

    # Create and run simulation
    sim = FluidSimulation(width, height, method)
    sim.run()


if __name__ == "__main__":
    main()