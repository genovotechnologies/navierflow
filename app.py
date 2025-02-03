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
    color_mode: str = 'blue'  # Added color mode parameter
    brush_size: int = 3  # Added brush size parameter
    vorticity: float = 0.1  # Added vorticity strength
    temperature: float = 0.0  # Added temperature effect

class FluidSimulationOpenGL:
    def __init__(self, nx: int = 128, ny: int = 128, method: str = 'eulerian'):
        self.nx, self.ny = nx, ny
        self.method = method
        self.velocity = np.zeros((nx, ny, 2), dtype=np.float32)
        self.density = np.zeros((nx, ny), dtype=np.float32)
        self.temperature = np.zeros((nx, ny), dtype=np.float32)  # Added temperature field
        self.vorticity = np.zeros((nx, ny), dtype=np.float32)  # Added vorticity field
        self.params = SimulationParams()

        # Performance tracking
        self.frame_times: List[float] = []
        self.last_frame_time = time.time()
        self.fps = 0.0

        # Multiple color schemes
        self.color_schemes = {
            'blue': self._create_blue_gradient,
            'fire': self._create_fire_gradient,
            'rainbow': self._create_rainbow_gradient,
            'grayscale': self._create_grayscale_gradient
        }

        # LBM specific variables
        self.e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                           [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int8)
        self.w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
                          dtype=np.float32)

        # OpenGL state
        self.texture = None
        self.mouse_down = False
        self.mouse_pos = (0, 0)
        self.last_mouse_pos = (0, 0)
        self._setup_color_gradient()

        # Initialize simulation stats
        self.stats = {
            'max_velocity': 0.0,
            'total_density': 0.0,
            'avg_temperature': 0.0
        }

    def _create_blue_gradient(self):
        gradient = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            t = i / 255.0
            gradient[i] = [t * 0.8, t * 0.9, min(0.4 + t * 0.6, 1.0)]
        return gradient

    def _create_fire_gradient(self):
        gradient = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            t = i / 255.0
            gradient[i] = [min(t * 2, 1.0), t * t, t * 0.5]
        return gradient

    def _create_rainbow_gradient(self):
        gradient = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            hue = i / 255.0
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            gradient[i] = rgb
        return gradient

    def _create_grayscale_gradient(self):
        gradient = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            t = i / 255.0
            gradient[i] = [t, t, t]
        return gradient

    def _setup_color_gradient(self):
        """Set up color gradient based on current mode"""
        self.color_gradient = self.color_schemes[self.params.color_mode]()

    def initialize_opengl(self):
        """Initialize OpenGL context with improved settings"""
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB | glut.GLUT_DEPTH)
        glut.glutInitWindowSize(800, 800)
        glut.glutCreateWindow(b"Enhanced Fluid Simulation")

        # Enable blending for smooth rendering
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Enable texturing with improved settings
        gl.glEnable(gl.GL_TEXTURE_2D)
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

        # Use trilinear filtering for better quality
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        # Set dark background
        gl.glClearColor(0.0, 0.0, 0.1, 1.0)

        # Register enhanced callbacks
        glut.glutMouseFunc(self.mouse_button)
        glut.glutMotionFunc(self.mouse_motion)
        glut.glutKeyboardFunc(self.keyboard)
        glut.glutReshapeFunc(self.reshape)
        glut.glutSpecialFunc(self.special_keys)

    def reshape(self, width: int, height: int):
        """Handle window resizing"""
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-1, 1, -1, 1, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def mouse_button(self, button: int, state: int, x: int, y: int):
        """Handle mouse button events"""
        if button == glut.GLUT_LEFT_BUTTON:
            self.mouse_down = state == glut.GLUT_DOWN
            if self.mouse_down:
                # Convert screen coordinates to simulation grid coordinates
                x = int((x / 800) * self.nx)
                y = int(((800 - y) / 800) * self.ny)
                self.mouse_pos = (x, y)
                self.last_mouse_pos = (x, y)

    def mouse_motion(self, x: int, y: int):
        """Handle mouse motion events"""
        if self.mouse_down:
            # Convert screen coordinates to simulation grid coordinates
            x = int((x / 800) * self.nx)
            y = int(((800 - y) / 800) * self.ny)
            self.mouse_pos = (x, y)
            self._add_interaction(self.last_mouse_pos[0], self.last_mouse_pos[1],
                                self.mouse_pos[0], self.mouse_pos[1])
            self.last_mouse_pos = self.mouse_pos

    def special_keys(self, key: int, x: int, y: int):
        """Handle special keys for parameter adjustment"""
        if key == glut.GLUT_KEY_UP:
            self.params.viscosity *= 1.1
        elif key == glut.GLUT_KEY_DOWN:
            self.params.viscosity *= 0.9
        elif key == glut.GLUT_KEY_LEFT:
            self.params.brush_size = max(1, self.params.brush_size - 1)
        elif key == glut.GLUT_KEY_RIGHT:
            self.params.brush_size = min(10, self.params.brush_size + 1)

    def _add_interaction(self, x1: int, y1: int, x2: int, y2: int):
        """Enhanced interaction with brush size and temperature effects"""
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        # Calculate direction and distance
        dx, dy = x2 - x1, y2 - y1
        distance = np.sqrt(dx * dx + dy * dy)

        if distance > 0:
            dx, dy = dx / distance, dy / distance

            # Create brush mask
            brush_radius = self.params.brush_size
            y, x = np.ogrid[-brush_radius:brush_radius + 1, -brush_radius:brush_radius + 1]
            mask = x * x + y * y <= brush_radius * brush_radius

            # Apply along the path
            steps = max(int(distance), 1)
            for i in range(steps):
                x = int(x1 + dx * i)
                y = int(y1 + dy * i)

                # Apply brush effects within bounds
                for ox in range(-brush_radius, brush_radius + 1):
                    for oy in range(-brush_radius, brush_radius + 1):
                        if mask[oy + brush_radius, ox + brush_radius]:
                            px = np.clip(x + ox, 0, self.nx - 1)
                            py = np.clip(y + oy, 0, self.ny - 1)

                            # Add density and temperature
                            intensity = 1.0 - (ox * ox + oy * oy) / (brush_radius * brush_radius)
                            self.density[px, py] += self.params.density_multiplier * intensity
                            self.temperature[px, py] += self.params.temperature * intensity

                            # Add velocity with falloff
                            self.velocity[px, py] += np.array([dx, dy]) * self.params.velocity_multiplier * intensity

    def _compute_vorticity(self):
        """Compute vorticity field"""
        dx = np.roll(self.velocity[..., 1], -1, axis=0) - np.roll(self.velocity[..., 1], 1, axis=0)
        dy = np.roll(self.velocity[..., 0], -1, axis=1) - np.roll(self.velocity[..., 0], 1, axis=1)
        self.vorticity = (dx - dy) * 0.5

    def _apply_vorticity_confinement(self):
        """Apply vorticity confinement force"""
        if self.params.vorticity <= 0:
            return

        # Compute vorticity gradient
        dx = np.roll(self.vorticity, -1, axis=0) - np.roll(self.vorticity, 1, axis=0)
        dy = np.roll(self.vorticity, -1, axis=1) - np.roll(self.vorticity, 1, axis=1)

        # Normalize gradient
        length = np.sqrt(dx * dx + dy * dy) + 1e-10
        dx /= length
        dy /= length

        # Apply force
        force_x = dy * self.vorticity * self.params.vorticity
        force_y = -dx * self.vorticity * self.params.vorticity

        self.velocity[..., 0] += force_x * self.params.dt
        self.velocity[..., 1] += force_y * self.params.dt

    def _diffuse(self, field: np.ndarray, diffusion_rate: float, dt: float) -> np.ndarray:
        """
        Solve the diffusion equation using Gauss-Seidel relaxation.

        Args:
            field: The field to diffuse (density, temperature, etc.)
            diffusion_rate: Rate of diffusion
            dt: Time step

        Returns:
            The diffused field
        """
        a = dt * diffusion_rate * self.nx * self.ny
        result = field.copy()

        # Gauss-Seidel relaxation
        for k in range(20):  # Number of iterations
            old_field = result.copy()
            result[1:-1, 1:-1] = (old_field[1:-1, 1:-1] +
                                  a * (
                                          old_field[2:, 1:-1] +  # right
                                          old_field[:-2, 1:-1] +  # left
                                          old_field[1:-1, 2:] +  # top
                                          old_field[1:-1, :-2]  # bottom
                                  )) / (1 + 4 * a)

            # Handle boundaries - wrap around
            result[0, :] = result[-2, :]  # Top edge
            result[-1, :] = result[1, :]  # Bottom edge
            result[:, 0] = result[:, -2]  # Left edge
            result[:, -1] = result[:, 1]  # Right edge

        return result

    def _eulerian_step(self):
        """
        Perform one step of the Eulerian fluid simulation.
        """
        # Diffuse velocity
        self.velocity[..., 0] = self._diffuse(self.velocity[..., 0], self.params.viscosity, self.params.dt)
        self.velocity[..., 1] = self._diffuse(self.velocity[..., 1], self.params.viscosity, self.params.dt)

        # Project velocity to be divergence-free
        self._project()

        # Advect velocity
        self.velocity = self._advect(self.velocity)

        # Project again
        self._project()

        # Advect density
        self.density = self._advect_scalar(self.density)

    def _project(self):
        """
        Project the velocity field to be divergence-free using Helmholtz-Hodge decomposition.
        """
        # Compute divergence
        div = np.zeros((self.nx, self.ny))
        h = 1.0 / self.nx
        div[1:-1, 1:-1] = -0.5 * h * (
                self.velocity[2:, 1:-1, 0] - self.velocity[:-2, 1:-1, 0] +
                self.velocity[1:-1, 2:, 1] - self.velocity[1:-1, :-2, 1]
        )

        # Solve Poisson equation
        p = np.zeros_like(div)
        for k in range(20):
            p[1:-1, 1:-1] = (div[1:-1, 1:-1] +
                             p[2:, 1:-1] + p[:-2, 1:-1] +
                             p[1:-1, 2:] + p[1:-1, :-2]) / 4.0

            # Handle boundaries
            p[0, :] = p[-2, :]
            p[-1, :] = p[1, :]
            p[:, 0] = p[:, -2]
            p[:, -1] = p[:, 1]

        # Subtract pressure gradient
        self.velocity[1:-1, 1:-1, 0] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
        self.velocity[1:-1, 1:-1, 1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h

        # Handle boundaries
        self.velocity[0, :] = self.velocity[-2, :]
        self.velocity[-1, :] = self.velocity[1, :]
        self.velocity[:, 0] = self.velocity[:, -2]
        self.velocity[:, -1] = self.velocity[:, 1]

    def _advect(self, field):
        """
        Advect a vector field using semi-Lagrangian advection.
        """
        result = np.zeros_like(field)

        dt0 = self.params.dt * self.nx
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                # Trace particle back
                x = i - dt0 * field[i, j, 0]
                y = j - dt0 * field[i, j, 1]

                # Clamp coordinates
                x = max(0.5, min(self.nx - 1.5, x))
                y = max(0.5, min(self.ny - 1.5, y))

                # Get interpolation indices
                i0, i1 = int(x), int(x) + 1
                j0, j1 = int(y), int(y) + 1

                # Get interpolation weights
                s1 = x - i0
                s0 = 1 - s1
                t1 = y - j0
                t0 = 1 - t1

                # Interpolate
                result[i, j] = (
                        t0 * (s0 * field[i0, j0] + s1 * field[i1, j0]) +
                        t1 * (s0 * field[i0, j1] + s1 * field[i1, j1])
                )

        return result

    def _advect_scalar(self, field):
        """
        Advect a scalar field using semi-Lagrangian advection.
        """
        result = np.zeros_like(field)

        dt0 = self.params.dt * self.nx
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                x = i - dt0 * self.velocity[i, j, 0]
                y = j - dt0 * self.velocity[i, j, 1]

                x = max(0.5, min(self.nx - 1.5, x))
                y = max(0.5, min(self.ny - 1.5, y))

                i0, i1 = int(x), int(x) + 1
                j0, j1 = int(y), int(y) + 1

                s1 = x - i0
                s0 = 1 - s1
                t1 = y - j0
                t0 = 1 - t1

                result[i, j] = (
                        t0 * (s0 * field[i0, j0] + s1 * field[i1, j0]) +
                        t1 * (s0 * field[i0, j1] + s1 * field[i1, j1])
                )

        return result

    def _apply_buoyancy(self):
        """Apply temperature-based buoyancy"""
        if self.params.temperature != 0:
            buoyancy = self.temperature * 0.1
            self.velocity[..., 1] += buoyancy * self.params.dt

    def update_simulation(self):
        """Enhanced simulation update with new features"""
        # Track frame time for FPS calculation
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.frame_times.append(dt)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        self.last_frame_time = current_time

        # Update simulation based on method
        if self.method == 'eulerian':
            self._compute_vorticity()
            self._apply_vorticity_confinement()
            self._apply_buoyancy()
            self._eulerian_step()
        elif self.method == 'lbm':
            self._lbm_step()

        # Update simulation statistics
        self.stats['max_velocity'] = np.sqrt(np.sum(self.velocity ** 2, axis=2)).max()
        self.stats['total_density'] = self.density.sum()
        self.stats['avg_temperature'] = self.temperature.mean()

        # Temperature diffusion
        self.temperature = self._diffuse(self.temperature, 0.1, self.params.dt)

        glut.glutPostRedisplay()

    def keyboard(self, key: bytes, x: int, y: int):
        """Enhanced keyboard controls"""
        key = key.lower()
        if key == b'm':
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
        elif key == b'v':
            # Toggle vorticity confinement
            self.params.vorticity = 0.1 if self.params.vorticity == 0 else 0
        elif key == b'h':
            self._print_help()

    def _print_help(self):
        """Display help information"""
        print("\nEnhanced Fluid Simulation Controls:")
        print("-----------------------------------")
        print("Mouse drag: Add fluid")
        print("m: Switch method (Eulerian/LBM)")
        print("r: Reset simulation")
        print("c: Cycle color modes")
        print("t: Toggle temperature effect")
        print("v: Toggle vorticity confinement")
        print("Arrow keys: Adjust brush size and viscosity")
        print("h: Show this help message")
        print(f"Current FPS: {self.fps:.1f}")

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
        """Perform one step of the LBM simulation"""
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

        # Streaming step
        for i in range(9):
            self.f[:, :, i] = np.roll(np.roll(self.f[:, :, i],
                                              self.e[i, 0], axis=0),
                                      self.e[i, 1], axis=1)

        # Bounce-back on walls
        self._apply_boundary_conditions()

    def _apply_boundary_conditions(self):
        """Apply bounce-back boundary conditions"""
        # Bounce-back on walls (assuming index pairs for opposite directions)
        opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]  # Opposite direction indices

        # Top and bottom walls
        for i in range(9):
            # Bottom wall
            self.f[0, :, i] = self.f[0, :, opposite[i]]
            # Top wall
            self.f[-1, :, i] = self.f[-1, :, opposite[i]]

        # Left and right walls
        for i in range(9):
            # Left wall
            self.f[:, 0, i] = self.f[:, 0, opposite[i]]
            # Right wall
            self.f[:, -1, i] = self.f[:, -1, opposite[i]]

    def render(self):
        """Enhanced rendering with statistics overlay"""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

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

        # Generate mipmaps for better quality
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.nx, self.ny, 0,
                        gl.GL_RGB, gl.GL_FLOAT, color_density)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        # Draw textured quad
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 0);
        gl.glVertex2f(-1, -1)
        gl.glTexCoord2f(1, 0);
        gl.glVertex2f(1, -1)
        gl.glTexCoord2f(1, 1);
        gl.glVertex2f(1, 1)
        gl.glTexCoord2f(0, 1);
        gl.glVertex2f(-1, 1)
        gl.glEnd()

        # Render statistics overlay
        self._render_stats()

        glut.glutSwapBuffers()

    def _render_stats(self):
        """Render simulation statistics overlay"""
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(0, self.nx, 0, self.ny, -1, 1)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()

        # Disable texturing for text rendering
        gl.glDisable(gl.GL_TEXTURE_2D)

        # Set text color
        gl.glColor3f(1.0, 1.0, 1.0)

        # Render stats
        self._render_text(10, self.ny - 20, f"FPS: {self.fps:.1f}")
        self._render_text(10, self.ny - 40, f"Method: {self.method}")
        self._render_text(10, self.ny - 60, f"Brush Size: {self.params.brush_size}")
        self._render_text(10, self.ny - 80, f"Max Velocity: {self.stats['max_velocity']:.2f}")

        # Re-enable texturing
        gl.glEnable(gl.GL_TEXTURE_2D)

        # Restore matrices
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def _render_text(self, x: int, y: int, text: str):
        """Render text using GLUT bitmap font"""
        gl.glRasterPos2f(x, y)
        for char in text:
            glut.glutBitmapCharacter(glut.GLUT_BITMAP_9_BY_15, ord(char))

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

    sim = FluidSimulationOpenGL(nx=args.size, ny=args.size, method=args.method)
    sim.params.color_mode = args.color_mode
    sim.params.viscosity = args.viscosity
    sim.params.vorticity = args.vorticity
    sim.run()