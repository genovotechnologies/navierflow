import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
from typing import Tuple, Optional, List
from dataclasses import dataclass
import colorsys
import time

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


def _create_rainbow_gradient():
    gradient = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        hue = i / 255.0
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        gradient[i] = rgb
    return gradient


def _create_grayscale_gradient():
    gradient = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        t = i / 255.0
        gradient[i] = [t, t, t]
    return gradient


def _render_text(x: int, y: int, text: str):
    """Render text using GLUT bitmap font"""
    gl.glRasterPos2f(x, y)
    for char in text:
        glut.glutBitmapCharacter(glut.GLUT_BITMAP_9_BY_15, ord(char))


def reshape(self, width: int, height: int):
    """Handle window reshape events"""
    if width == 0 or height == 0:
        return  # Window is minimized

    self.window_width = width
    self.window_height = height

    # Maintain aspect ratio
    aspect = self.nx / self.ny
    window_aspect = width / height

    if window_aspect > aspect:
        viewport_w = int(height * aspect)
        viewport_h = height
        viewport_x = (width - viewport_w) // 2
        viewport_y = 0
    else:
        viewport_w = width
        viewport_h = int(width / aspect)
        viewport_x = 0
        viewport_y = (height - viewport_h) // 2

    gl.glViewport(viewport_x, viewport_y, viewport_w, viewport_h)

    # Set up orthographic projection
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-1, 1, -1, 1, -1, 1)
    gl.glMatrixMode(gl.GL_MODELVIEW)


def _create_blue_gradient():
    gradient = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        t = i / 255.0
        gradient[i] = [t * 0.8, t * 0.9, min(0.4 + t * 0.6, 1.0)]
    return gradient


class FluidSimulationOpenGL:
    def __init__(self, nx: int = 128, ny: int = 128, method: str = 'eulerian'):
        self.params = SimulationParams()
        self.nx, self.ny = nx, ny
        self.method = method
        self.fps = 0.0

        self.window_width = 800
        self.window_height = 800
        self.is_fullscreen = False
        self.window_id = None
        self.saved_window_pos = (100, 100)
        self.saved_window_size = (800, 800)

        self._density = np.zeros((nx, ny), dtype=np.float32)

        self.color_schemes = {
            'blue': _create_blue_gradient,
            'fire': self._create_fire_gradient,
            'rainbow': _create_rainbow_gradient,
            'grayscale': _create_grayscale_gradient
        }

        self.velocity = np.zeros((nx, ny, 2), dtype=np.float32)
        self.density = np.zeros((nx, ny), dtype=np.float32)
        self.temperature = np.zeros((nx, ny), dtype=np.float32)
        self.vorticity = np.zeros((nx, ny), dtype=np.float32)

        self.ball_position = np.array([nx // 2, ny // 2], dtype=np.float32)
        self.ball_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.ball_radius = 10

        self.frame_times = []
        self.last_frame_time = time.time()

        self.texture = None
        self.mouse_down = False
        self.mouse_pos = (0, 0)
        self.last_mouse_pos = (0, 0)

        # Call this method to set up color gradient
        self._setup_color_gradient()

        # Initialize simulation stats
        self.stats = {
            'max_velocity': 0.0,
            'total_density': 0.0,
            'avg_temperature': 0.0
        }

        # Enhanced ball/particle attributes
        self.ball_position = np.array([nx // 2, ny // 2], dtype=np.float32)
        self.ball_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.is_ball_selected = False
        self.ball_drag_offset = np.array([0.0, 0.0], dtype=np.float32)

        # Color for the ball
        self.ball_color = [0.8, 0.2, 0.2]  # Reddish color
        self.e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                           [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=object)
        self.w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
                          dtype=np.float32)

    @property
    def density(self):
        """Get the density field"""
        if not hasattr(self, '_density'):
            self._density = np.zeros((self.nx, self.ny), dtype=np.float32)
        return self._density

    @density.setter
    def density(self, value):
        """Set the density field with type checking"""
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float32)
        if value.shape != (self.nx, self.ny):
            raise ValueError(f"Density field must have shape ({self.nx}, {self.ny})")
        self._density = value.astype(np.float32)


    def _create_fire_gradient(self):
        gradient = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            t = i / 255.0
            gradient[i] = [min(t * 2, 1.0), t * t, t * 0.5]
        return gradient

    def _setup_color_gradient(self):
        self.color_gradient = self.color_schemes[self.params.color_mode]()

    def initialize_opengl(self):
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB | glut.GLUT_DEPTH)
        glut.glutInitWindowSize(self.window_width, self.window_height)
        glut.glutInitWindowPosition(100, 100)

        # Create and store window ID
        self.window_id = glut.glutCreateWindow(b"Enhanced Fluid Simulation")

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glEnable(gl.GL_TEXTURE_2D)
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        gl.glClearColor(0.0, 0.0, 0.1, 1.0)

        # Use the class method for reshape
        glut.glutReshapeFunc(self.reshape)

        # Register all callbacks
        glut.glutMouseFunc(self.mouse_button)
        glut.glutMotionFunc(self.mouse_motion)
        glut.glutKeyboardFunc(self.keyboard)
        glut.glutReshapeFunc(self.reshape)  # Use class method for reshape
        glut.glutSpecialFunc(self.special_keys)
        glut.glutDisplayFunc(self.render)

    def mouse_button(self, button: int, state: int, x: int, y: int):
        # Convert screen coordinates to grid coordinates
        grid_x = int((x / 800) * self.nx)
        grid_y = int(((800 - y) / 800) * self.ny)

        # Check if mouse is near the ball
        ball_screen_x = int((self.ball_position[0] / self.nx) * 800)
        ball_screen_y = int(((self.ny - self.ball_position[1]) / self.ny) * 800)

        distance_to_ball = np.sqrt((x - ball_screen_x) ** 2 + (y - ball_screen_y) ** 2)

        if button == glut.GLUT_LEFT_BUTTON:
            if state == glut.GLUT_DOWN:
                # If close to ball, select it
                if distance_to_ball < self.params.ball_radius * (800 / self.nx):
                    self.is_ball_selected = True
                    # Calculate offset to make drag feel natural
                    self.ball_drag_offset = np.array([grid_x - self.ball_position[0],
                                                      grid_y - self.ball_position[1]])
                else:
                    # Normal mouse interaction for fluid
                    self.mouse_down = True
                    self.last_mouse_pos = (grid_x, grid_y)
            else:
                # Release ball or mouse
                self.is_ball_selected = False
                self.mouse_down = False

    def mouse_motion(self, x: int, y: int):
        if self.mouse_down:
            # Convert window coordinates to simulation coordinates
            sim_x = (x / self.window_width) * self.nx
            sim_y = ((self.window_height - y) / self.window_height) * self.ny

            # Update ball position
            new_pos = np.array([sim_x, sim_y], dtype=np.float32)

            # Calculate ball velocity from movement
            if hasattr(self, 'last_ball_pos'):
                self.ball_velocity = (new_pos - self.last_ball_pos) / self.params.dt
            else:
                self.ball_velocity = np.zeros(2, dtype=np.float32)

            # Update positions
            self.ball_position = new_pos
            self.last_ball_pos = new_pos

            # Update mouse position for fluid interaction
            self.mouse_pos = (int(sim_x), int(sim_y))
            if hasattr(self, 'last_mouse_pos'):
                self._add_interaction(
                    int(self.last_mouse_pos[0]), int(self.last_mouse_pos[1]),
                    self.mouse_pos[0], self.mouse_pos[1]
                )
            self.last_mouse_pos = self.mouse_pos

    def special_keys(self, key: int, x: int, y: int):
        if key == glut.GLUT_KEY_UP:
            self.params.viscosity *= 1.1
        elif key == glut.GLUT_KEY_DOWN:
            self.params.viscosity *= 0.9
        elif key == glut.GLUT_KEY_LEFT:
            self.params.brush_size = max(1, self.params.brush_size - 1)
        elif key == glut.GLUT_KEY_RIGHT:
            self.params.brush_size = min(10, self.params.brush_size + 1)
        elif key == glut.GLUT_KEY_PAGE_UP:
            self.ball_radius = min(20, self.ball_radius + 1)
        elif key == glut.GLUT_KEY_PAGE_DOWN:
            self.ball_radius = max(2, self.ball_radius - 1)

    def _add_interaction(self, x1: int, y1: int, x2: int, y2: int):
        """Enhanced interaction with brush size and temperature effects"""
        try:
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

            dx, dy = x2 - x1, y2 - y1
            distance = np.sqrt(dx * dx + dy * dy)

            if distance > 0:
                dx, dy = dx / distance, dy / distance

                brush_radius = max(1, self.params.brush_size)
                y_grid, x_grid = np.ogrid[-brush_radius:brush_radius + 1, -brush_radius:brush_radius + 1]
                mask = x_grid * x_grid + y_grid * y_grid <= brush_radius * brush_radius

                steps = max(int(distance), 1)
                for i in range(steps):
                    cx = int(x1 + dx * i)
                    cy = int(y1 + dy * i)

                    x_start = max(0, cx - brush_radius)
                    x_end = min(self.nx, cx + brush_radius + 1)
                    y_start = max(0, cy - brush_radius)
                    y_end = min(self.ny, cy + brush_radius + 1)

                    mask_x_start = max(0, brush_radius - (cx - x_start))
                    mask_x_end = mask_x_start + (x_end - x_start)
                    mask_y_start = max(0, brush_radius - (cy - y_start))
                    mask_y_end = mask_y_start + (y_end - y_start)

                    local_mask = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

                    y_coords, x_coords = np.meshgrid(
                        np.arange(y_start, y_end),
                        np.arange(x_start, x_end),
                        indexing='ij'
                    )
                    distances = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
                    intensity = np.maximum(0, 1.0 - distances / brush_radius)

                    if local_mask.shape == intensity.shape:
                        self.density[y_start:y_end, x_start:x_end][local_mask] += (
                                self.params.density_multiplier * intensity[local_mask]
                        )

                        self.temperature[y_start:y_end, x_start:x_end][local_mask] += (
                                self.params.temperature * intensity[local_mask]
                        )

                        velocity_contribution = np.array([dx, dy]) * self.params.velocity_multiplier
                        self.velocity[y_start:y_end, x_start:x_end][local_mask] += (
                                velocity_contribution * intensity[local_mask, np.newaxis]
                        )

        except Exception as e:
            print(f"Error in _add_interaction: {str(e)}")

    def _compute_vorticity(self):
        dx = np.roll(self.velocity[..., 1], -1, axis=0) - np.roll(self.velocity[..., 1], 1, axis=0)
        dy = np.roll(self.velocity[..., 0], -1, axis=1) - np.roll(self.velocity[..., 0], 1, axis=1)
        self.vorticity = (dx - dy) * 0.5

    def _apply_vorticity_confinement(self):
        if self.params.vorticity <= 0:
            return

        dx = np.roll(self.vorticity, -1, axis=0) - np.roll(self.vorticity, 1, axis=0)
        dy = np.roll(self.vorticity, -1, axis=1) - np.roll(self.vorticity, 1, axis=1)

        length = np.sqrt(dx * dx + dy * dy) + 1e-10
        dx /= length
        dy /= length

        force_x = dy * self.vorticity * self.params.vorticity
        force_y = -dx * self.vorticity * self.params.vorticity

        self.velocity[..., 0] += force_x * self.params.dt
        self.velocity[..., 1] += force_y * self.params.dt

    def _diffuse(self, field: np.ndarray, diffusion_rate: float, dt: float) -> np.ndarray:
        a = dt * diffusion_rate * self.nx * self.ny
        result = field.copy()

        for k in range(20):
            old_field = result.copy()
            result[1:-1, 1:-1] = (old_field[1:-1, 1:-1] + a * (
                old_field[2:, 1:-1] + old_field[:-2, 1:-1] +
                old_field[1:-1, 2:] + old_field[1:-1, :-2])) / (1 + 4 * a)

            result[0, :] = result[-2, :]
            result[-1, :] = result[1, :]
            result[:, 0] = result[:, -2]
            result[:, -1] = result[:, 1]

        return result

    def _eulerian_step(self):
        self.velocity[..., 0] = self._diffuse(self.velocity[..., 0], self.params.viscosity, self.params.dt)
        self.velocity[..., 1] = self._diffuse(self.velocity[..., 1], self.params.viscosity, self.params.dt)
        self._project()
        self.velocity = self._advect(self.velocity)
        self._project()
        self.density = self._advect_scalar(self.density)
        self._update_ball()

    def _project(self):
        div = np.zeros((self.nx, self.ny))
        h = 1.0 / self.nx
        div[1:-1, 1:-1] = -0.5 * h * (
                self.velocity[2:, 1:-1, 0] - self.velocity[:-2, 1:-1, 0] +
                self.velocity[1:-1, 2:, 1] - self.velocity[1:-1, :-2, 1]
        )

        p = np.zeros_like(div)
        for k in range(20):
            p[1:-1, 1:-1] = (div[1:-1, 1:-1] + (
                p[2:, 1:-1] + p[:-2, 1:-1] +
                p[1:-1, 2:] + p[1:-1, :-2])) / 4

            p[0, :] = p[-2, :]
            p[-1, :] = p[1, :]
            p[:, 0] = p[:, -2]
            p[:, -1] = p[:, 1]

        self.velocity[1:-1, 1:-1, 0] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
        self.velocity[1:-1, 1:-1, 1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h

        self.velocity[0, :] = self.velocity[-2, :]
        self.velocity[-1, :] = self.velocity[1, :]
        self.velocity[:, 0] = self.velocity[:, -2]
        self.velocity[:, -1] = self.velocity[:, 1]

    def _advect(self, field):
        result = np.zeros_like(field)
        dt0 = self.params.dt * self.nx
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                x = i - dt0 * field[i, j, 0]
                y = j - dt0 * field[i, j, 1]
                if x < 0.5: x = 0.5
                if x > self.nx - 1.5: x = self.nx - 1.5
                if y < 0.5: y = 0.5
                if y > self.ny - 1.5: y = self.ny - 1.5
                i0, j0 = int(x), int(y)
                i1, j1 = i0 + 1, j0 + 1
                s1, t1 = x - i0, y - j0
                s0, t0 = 1 - s1, 1 - t1
                result[i, j] = s0 * (t0 * field[i0, j0] + t1 * field[i0, j1]) + s1 * (t0 * field[i1, j0] + t1 * field[i1, j1])
        return result

    def _advect_scalar(self, field):
        result = np.zeros_like(field)
        dt0 = self.params.dt * self.nx
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                x = i - dt0 * self.velocity[i, j, 0]
                y = j - dt0 * self.velocity[i, j, 1]
                if x < 0.5: x = 0.5
                if x > self.nx - 1.5: x = self.nx - 1.5
                if y < 0.5: y = 0.5
                if y > self.ny - 1.5: y = self.ny - 1.5
                i0, j0 = int(x), int(y)
                i1, j1 = i0 + 1, j0 + 1
                s1, t1 = x - i0, y - j0
                s0, t0 = 1 - s1, 1 - t1
                result[i, j] = s0 * (t0 * field[i0, j0] + t1 * field[i0, j1]) + s1 * (t0 * field[i1, j0] + t1 * field[i1, j1])
        return result

    def _apply_buoyancy(self):
        if self.params.temperature != 0:
            buoyancy = self.temperature * 0.1
            self.velocity[..., 1] += buoyancy * self.params.dt

    def update_simulation(self):
        """Enhanced simulation update with new features"""
        self._verify_initialization()
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.frame_times.append(dt)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        self.last_frame_time = current_time

        if self.method == 'eulerian':
            self._compute_vorticity()
            self._apply_vorticity_confinement()
            self._apply_buoyancy()
            self._eulerian_step()
        elif self.method == 'lbm':
            self._lbm_step()

        self.stats['max_velocity'] = np.sqrt(np.sum(self.velocity ** 2, axis=2)).max()
        self.stats['total_density'] = self.density.sum()
        self.stats['avg_temperature'] = self.temperature.mean()

        self.temperature = self._diffuse(self.temperature, 0.1, self.params.dt)

        glut.glutPostRedisplay()

    def keyboard(self, key: bytes, x: int, y: int):
        """Enhanced keyboard controls"""
        try:
            key = key.lower()

            if key == b'i':
                # Increase ball interaction strength
                self.params.ball_interaction_strength = min(2.0,
                                                            self.params.ball_interaction_strength + 0.1)
            elif key == b'k':
                # Decrease ball interaction strength
                self.params.ball_interaction_strength = max(0.0,
                                                            self.params.ball_interaction_strength - 0.1)
            elif key == b'm':
                self.method = 'lbm' if self.method == 'eulerian' else 'eulerian'
                print(f"Switched to {self.method} method")
            elif key == b'r':
                self._reset_simulation()
            elif key == b'c':
                # Cycle through color modes
                modes = list(self.color_schemes.keys())
                current_idx = modes.index(self.params.color_mode)
                next_idx = (current_idx + 1) % len(modes)
                self.params.color_mode = modes[next_idx]
                self._setup_color_gradient()
            elif key == b't':
                # Toggle temperature effect
                self.params.temperature = 0.5 if self.params.temperature == 0 else 0
            elif key == b'f':  # 'F' key for fullscreen toggle
                self.toggle_fullscreen()
            elif key == b'v':
                # Toggle vorticity confinement
                self.params.vorticity = 0.1 if self.params.vorticity == 0 else 0
            elif key == b'h':
                self._print_help()
            elif key == b'\x1b':  # ESC key
                self.handle_escape()
        except Exception as e:
            print(f"Error in keyboard handler: {str(e)}")

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        if not self.is_fullscreen:
            # Save current window position and size
            self.saved_window_pos = (glut.glutGet(glut.GLUT_WINDOW_X),
                                     glut.glutGet(glut.GLUT_WINDOW_Y))
            self.saved_window_size = (self.window_width, self.window_height)

            # Switch to fullscreen
            glut.glutFullScreen()
            self.is_fullscreen = True
        else:
            # Restore windowed mode
            glut.glutReshapeWindow(self.saved_window_size[0], self.saved_window_size[1])
            glut.glutPositionWindow(self.saved_window_pos[0], self.saved_window_pos[1])
            self.is_fullscreen = False

    def handle_minimize(self):
        """Handle window minimization"""
        if self.is_fullscreen:
            self.toggle_fullscreen()
        glut.glutIconifyWindow()

    def _print_help(self):
        """Display help information"""
        print("\nEnhanced Fluid Simulation Controls:")
        print("-----------------------------------")
        print("Mouse drag: Add fluid")
        print("F: Toggle fullscreen")
        print("ESC: Exit fullscreen / Minimize window")
        print("m: Switch method (Eulerian/LBM)")
        print("r: Reset simulation")
        print("c: Cycle color modes")
        print("t: Toggle temperature effect")
        print("v: Toggle vorticity confinement")
        print("Arrow keys: Adjust brush size and viscosity")
        print("h: Show this help message")
        print(f"Current FPS: {getattr(self, 'fps', 0):.1f}")
        print("\nBall/Particle Controls:")
        print("Click and drag ball to move")
        print("Page Up/Down: Resize ball")
        print("i/k: Adjust ball interaction strength")
        print("h: Show this help message")
        print(f"Current FPS: {getattr(self, 'fps', 0):.1f}")

    def _reset_simulation(self):
        """Reset the simulation to its initial state"""
        # Reset all fields to zero
        self.velocity = np.zeros((self.nx, self.ny, 2), dtype=np.float32)
        self.density = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.temperature = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.vorticity = np.zeros((self.nx, self.ny), dtype=np.float32)

        # Reset performance tracking
        self.frame_times = []
        self.last_frame_time = time.time()
        self.fps = 0.0

        # Reset mouse state
        self.mouse_down = False
        self.mouse_pos = (0, 0)
        self.last_mouse_pos = (0, 0)

        # Reset simulation stats
        self.stats = {
            'max_velocity': 0.0,
            'total_density': 0.0,
            'avg_temperature': 0.0
        }

        print("Simulation reset")

    def _initialize_lbm(self):
        """Initialize LBM distribution functions"""
        if not hasattr(self, 'f'):
            # Initialize distribution functions
            self.f = np.ones((self.nx, self.ny, 9)) * self.w.reshape(1, 1, -1)
            # Initialize equilibrium
            self.feq = np.zeros_like(self.f)

    def _compute_equilibrium(self, rho, ux, uy):
        """Compute equilibrium distribution function"""
        # Compute squared velocity terms
        u_sq = ux * ux + uy * uy

        # Initialize equilibrium array
        feq = np.zeros((self.nx, self.ny, 9))

        # For each direction
        for i in range(9):
            # Compute dot product (e_i · u)
            eu = self.e[i, 0] * ux + self.e[i, 1] * uy
            # Compute equilibrium
            feq[:, :, i] = self.w[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)

        return feq

    def _lbm_step(self):
        """Perform one step of the LBM simulation with improved streaming"""
        # Initialize if needed
        self._initialize_lbm()

        # Collision step
        # Calculate macroscopic quantities
        rho = np.sum(self.f, axis=2)
        ux = np.sum(self.f * self.e[:, 0].reshape(1, 1, -1), axis=2) / (rho + 1e-6)
        uy = np.sum(self.f * self.e[:, 1].reshape(1, 1, -1), axis=2) / (rho + 1e-6)

        # Update velocity field for visualization
        self.velocity[..., 0] = ux
        self.velocity[..., 1] = uy

        # Update density field for visualization
        self.density = (rho - 1.0) * 0.1

        # Compute equilibrium
        feq = self._compute_equilibrium(rho, ux, uy)

        # Collision
        omega = self.params.omega
        self.f = self.f * (1.0 - omega) + feq * omega

        # Improved streaming step
        f_temp = self.f.copy()
        for i in range(9):
            # Use numpy roll with explicit axis specification and int32 to handle larger grid sizes
            if self.e[i, 0] != 0:
                f_temp[:, :, i] = np.roll(f_temp[:, :, i], shift=np.int32(self.e[i, 0]), axis=0)
            if self.e[i, 1] != 0:
                f_temp[:, :, i] = np.roll(f_temp[:, :, i], shift=np.int32(self.e[i, 1]), axis=1)

        # Update f
        self.f = f_temp

        # Bounce-back on walls
        self._apply_boundary_conditions()

    def _apply_boundary_conditions(self):
        """Apply bounce-back boundary conditions"""
        # Bounce-back on walls (assuming index pairs for opposite directions)
        opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]  # Opposite direction indices

        # Temporary copy for modifications
        f_boundary = self.f.copy()

        # Top and bottom walls
        for i in range(9):
            # Bottom wall
            f_boundary[0, :, i] = self.f[0, :, opposite[i]]
            # Top wall
            f_boundary[-1, :, i] = self.f[-1, :, opposite[i]]

        # Left and right walls
        for i in range(9):
            # Left wall
            f_boundary[:, 0, i] = self.f[:, 0, opposite[i]]
            # Right wall
            f_boundary[:, -1, i] = self.f[:, -1, opposite[i]]

        # Update the original distribution function
        self.f = f_boundary

    def render(self):
        """Enhanced rendering with statistics overlay"""
        self._verify_initialization()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Save current matrices
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-1, 1, -1, 1, -1, 1)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        # Enhanced visualization
        self.density = self._diffuse(self.density, self.params.diffusion_rate, self.params.dt)
        enhanced_density = np.clip(self.density * 4.0, 0, 1)
        normalized_density = (enhanced_density * 255).astype(int)

        # Add temperature and vorticity effects to visualization
        color_density = self.color_gradient[normalized_density].copy()

        # Add temperature effect (red tint)
        temp_mask = self.temperature > 0
        color_density[temp_mask] += np.array([0.3, 0.0, 0.0]) * self.temperature[temp_mask, np.newaxis]

        # Add vorticity effect (swirl highlights)
        vorticity_magnitude = np.abs(self.vorticity)
        normalized_vorticity = vorticity_magnitude / (np.max(vorticity_magnitude) + 1e-6)
        color_density += np.array([0.1, 0.1, 0.2]) * normalized_vorticity[..., np.newaxis]

        # Ensure colors stay in valid range
        color_density = np.clip(color_density, 0, 1)

        # Enable texturing
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

        # Generate mipmaps for better quality
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.nx, self.ny, 0,
                        gl.GL_RGB, gl.GL_FLOAT, color_density)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        # Draw textured quad
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 0)
        gl.glVertex2f(-1, -1)
        gl.glTexCoord2f(1, 0)
        gl.glVertex2f(1, -1)
        gl.glTexCoord2f(1, 1)
        gl.glVertex2f(1, 1)
        gl.glTexCoord2f(0, 1)
        gl.glVertex2f(-1, 1)
        gl.glEnd()

        # Draw the ball
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glColor3f(1.0, 1.0, 1.0)  # White color for the ball

        # Convert ball position to GL coordinates (-1 to 1)
        ball_x = (self.ball_position[0] / self.nx) * 2 - 1
        ball_y = (self.ball_position[1] / self.ny) * 2 - 1
        radius = (self.ball_radius / self.nx) * 2  # Convert radius to GL coordinates

        # Draw circle
        gl.glBegin(gl.GL_TRIANGLE_FAN)
        gl.glVertex2f(ball_x, ball_y)  # Center point

        # Draw circle points
        segments = 32
        for i in range(segments + 1):
            angle = 2.0 * np.pi * i / segments
            x = ball_x + radius * np.cos(angle)
            y = ball_y + radius * np.sin(angle)
            gl.glVertex2f(x, y)
        gl.glEnd()

        # Render stats with separate matrix stack
        self._render_stats()

        glut.glutSwapBuffers()

    def _render_stats(self):
        """Render simulation statistics overlay"""
        # Set up ortho projection for 2D text rendering
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(0, 800, 0, 800, -1, 1)  # Use window coordinates

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()

        # Disable texturing and set color for text
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glColor3f(1.0, 1.0, 1.0)

        # Render stats (in window coordinates)
        _render_text(10, 780, f"FPS: {self.fps:.1f}")
        _render_text(10, 760, f"Method: {self.method}")
        _render_text(10, 740, f"Brush Size: {self.params.brush_size}")
        _render_text(10, 720, f"Max Velocity: {self.stats['max_velocity']:.2f}")

        # Clean up state
        gl.glEnable(gl.GL_TEXTURE_2D)

        # Restore matrices in reverse order
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()

    def _update_ball(self):
        """Enhanced ball update with vorticity and interactive fluid dynamics"""
        ball_x, ball_y = map(int, self.ball_position)

        # Ensure the ball is within the grid bounds
        ball_x = max(0, min(ball_x, self.nx - 1))
        ball_y = max(0, min(ball_y, self.ny - 1))

        # Calculate ball radius based on simulation parameters
        ball_radius = int(self.params.ball_radius)

        # Create a mask for the ball region
        Y, X = np.ogrid[-ball_radius:ball_radius + 1, -ball_radius:ball_radius + 1]
        ball_mask = X * X + Y * Y <= ball_radius * ball_radius

        # Calculate local vorticity around the ball
        local_vorticity = self.vorticity[
                          max(0, ball_x - ball_radius):min(self.nx, ball_x + ball_radius + 1),
                          max(0, ball_y - ball_radius):min(self.ny, ball_y + ball_radius + 1)
                          ]

        # Calculate velocity divergence
        div_x = np.roll(self.velocity[..., 0], -1, axis=0) - np.roll(self.velocity[..., 0], 1, axis=0)
        div_y = np.roll(self.velocity[..., 1], -1, axis=1) - np.roll(self.velocity[..., 1], 1, axis=1)
        divergence = div_x + div_y

        # Ball-fluid interaction
        interaction_strength = self.params.ball_interaction_strength

        # Apply ball boundary conditions (bounce-back)
        reflected_velocity = -self.velocity[ball_x, ball_y] * 0.8

        # Update ball velocity based on fluid dynamics
        self.ball_velocity += (
                interaction_strength * np.array([
            np.mean(divergence),  # x-direction
            np.mean(local_vorticity)  # y-direction
        ])
        )

        # Add some damping
        self.ball_velocity *= 0.95

        # Update ball position
        self.ball_position += self.ball_velocity * self.params.dt

        # Boundary conditions with energy loss
        for i in range(2):
            if (self.ball_position[i] < ball_radius or
                    self.ball_position[i] > (self.nx if i == 0 else self.ny) - ball_radius):
                self.ball_velocity[i] *= -0.8  # Bounce with energy loss

        # Modify local velocity field around the ball
        x_start = max(0, ball_x - ball_radius)
        x_end = min(self.nx, ball_x + ball_radius + 1)
        y_start = max(0, ball_y - ball_radius)
        y_end = min(self.ny, ball_y + ball_radius + 1)

        # Create distance-based intensity mask
        Y, X = np.ogrid[y_start:y_end, x_start:x_end]
        distances = np.sqrt((X - ball_x) ** 2 + (Y - ball_y) ** 2)
        intensity = np.maximum(0, 1.0 - distances / ball_radius)

        # Apply reflected velocity to fluid around the ball
        self.velocity[y_start:y_end, x_start:x_end][ball_mask] += (
                reflected_velocity * intensity[ball_mask, np.newaxis] * interaction_strength
        )

        # Ensure the ball stays within the simulation boundaries
        self.ball_position[0] = np.clip(self.ball_position[0],
                                        self.ball_radius,
                                        self.nx - self.ball_radius)
        self.ball_position[1] = np.clip(self.ball_position[1],
                                        self.ball_radius,
                                        self.ny - self.ball_radius)

        # Get integer position
        ball_x, ball_y = int(self.ball_position[0]), int(self.ball_position[1])

        # Create circular mask for ball influence
        y, x = np.ogrid[-self.ball_radius:self.ball_radius + 1, -self.ball_radius:self.ball_radius + 1]
        mask = x * x + y * y <= self.ball_radius * self.ball_radius

        # Calculate valid ranges for the ball's influence
        x_start = max(0, ball_x - self.ball_radius)
        x_end = min(self.nx, ball_x + self.ball_radius + 1)
        y_start = max(0, ball_y - self.ball_radius)
        y_end = min(self.ny, ball_y + self.ball_radius + 1)

        # Calculate mask indices
        mask_x_start = max(0, -(ball_x - self.ball_radius))
        mask_x_end = mask_x_start + (x_end - x_start)
        mask_y_start = max(0, -(ball_y - self.ball_radius))
        mask_y_end = mask_y_start + (y_end - y_start)

        # Extract relevant part of the mask
        local_mask = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

        # Create a view of the velocity field we want to modify
        velocity_region = self.velocity[y_start:y_end, x_start:x_end]

        # Update velocity only where the mask is True
        for i in range(2):  # For each velocity component
            velocity_component = velocity_region[..., i]
            velocity_component[local_mask] += self.ball_velocity[i] * 0.1

        # Add some density at the ball's position
        self.density[y_start:y_end, x_start:x_end][local_mask] += 0.1

        # Reset velocity since we're controlling position directly
        self.ball_velocity = np.zeros(2, dtype=np.float32)

    def _verify_initialization(self):
        """Verify that all required attributes are properly initialized"""
        required_attrs = ['density', 'velocity', 'temperature', 'vorticity']
        for attr in required_attrs:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                if attr == 'density':
                    self._density = np.zeros((self.nx, self.ny), dtype=np.float32)
                else:
                    setattr(self, attr, np.zeros((self.nx, self.ny), dtype=np.float32))

    def run(self):
        """Start the enhanced simulation loop"""
        self.initialize_opengl()
        self._print_help()  # Show controls at startup
        glut.glutDisplayFunc(self.render)
        glut.glutIdleFunc(self.update_simulation)
        glut.glutMainLoop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Fluid Simulation')
    parser.add_argument('--method', type=str, default='eulerian',
                        choices=['eulerian', 'lbm'], help='Simulation method')
    parser.add_argument('--size', type=int, default=128,
                        help='Grid size (N x N)')
    parser.add_argument('--color-mode', type=str, default='blue',
                        choices=['blue', 'fire', 'rainbow', 'grayscale'],
                        help='Initial color mode')
    parser.add_argument('--viscosity', type=float, default=0.0001,
                        help='Fluid viscosity')
    parser.add_argument('--vorticity', type=float, default=0.1,
                        help='Vorticity confinement strength')
    args = parser.parse_args()

    sim = FluidSimulationOpenGL(
        nx=args.size,
        ny=args.size,
        method=args.method
    )
    sim.params.viscosity = args.viscosity
    sim.params.vorticity = args.vorticity
    sim.params.color_mode = args.color_mode
    sim._verify_initialization()
    sim.run()
