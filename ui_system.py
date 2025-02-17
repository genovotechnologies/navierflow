import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL import shaders
import moderngl
import imgui


class FluidSimulation:
    def __init__(self, width=800, height=600):
        # Initialize pygame and OpenGL
        self.particle_texture = None
        self.quad_vao = None
        self.velocity_texture = None
        self.particle_program = None
        self.quad_vbo = None
        self.lbm_program = None
        self.eulerian_program = None
        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        self.ctx = moderngl.create_context()

        # Simulation parameters
        self.width = width
        self.height = height
        self.simulation_method = None
        self.particle_count = 5000
        self.vorticity_strength = 0.5
        self.temperature = 0.3

        # Initialize simulation state
        self.velocity_field = np.zeros((height, width, 4), dtype=np.float32)
        self.particles = np.random.rand(self.particle_count, 4).astype(np.float32)

        # Compile shaders
        self.setup_shaders()

        # Setup buffers and textures
        self.setup_buffers()

        # Initialize ImGui
        imgui.create_context()
        self.imgui_renderer = imgui.core._PygameRenderer()

    def setup_shaders(self):
        # Vertex shader
        vertex_shader = """
            #version 330
            in vec2 position;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        """

        # Eulerian fluid shader
        eulerian_shader = """
            #version 330
            uniform vec2 resolution;
            uniform float time;
            uniform sampler2D velocity;
            uniform float vorticity;
            uniform float temperature;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution;
                vec4 v = texture(velocity, uv);
                vec2 vel = v.xy;
                float p = v.z;
                float temp = v.w;

                // Compute vorticity
                vec2 dx = vec2(1.0 / resolution.x, 0.0);
                vec2 dy = vec2(0.0, 1.0 / resolution.y);

                float curl = texture(velocity, uv + dx).x - texture(velocity, uv - dx).x
                          - texture(velocity, uv + dy).y + texture(velocity, uv - dy).y;

                vec2 force = vec2(curl * vorticity);
                force += vec2(0.0, temperature * temp);

                vel += force * 0.016;
                vel *= 0.995;  // Damping

                fragColor = vec4(vel, p, temp);
            }
        """

        # LBM shader
        lbm_shader = """
            #version 330
            uniform vec2 resolution;
            uniform float time;
            uniform sampler2D velocity;
            uniform vec2 ballPosition;
            uniform float vorticity;
            uniform float temperature;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution;
                vec2 pos = gl_FragCoord.xy;

                // Simple LBM implementation
                vec4 v = texture(velocity, uv);
                vec2 vel = v.xy;

                // Ball interaction
                vec2 ballPos = ballPosition * resolution;
                float d = length(pos - ballPos);
                if(d < 20.0) {
                    vel = normalize(pos - ballPos) * 0.5;
                }

                // Apply temperature and vorticity
                vel.y += temperature * v.w;
                vel += vec2(vorticity * v.z, 0.0);

                // Boundary conditions
                if(pos.x < 2.0 || pos.x > resolution.x - 2.0 ||
                   pos.y < 2.0 || pos.y > resolution.y - 2.0) {
                    vel = vec2(0.0);
                }

                fragColor = vec4(vel, v.z, v.w);
            }
        """

        # Particle shader
        particle_shader = """
            #version 330
            uniform vec2 resolution;
            uniform sampler2D positions;
            uniform sampler2D velocity;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution;
                vec2 pos = texture(positions, uv).xy;
                vec2 vel = texture(velocity, pos).xy;

                pos += vel * 0.016;
                pos = fract(pos);  // Wrap around

                fragColor = vec4(pos, 0.0, 1.0);
            }
        """

        # Compile shaders
        self.eulerian_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=eulerian_shader
        )

        self.lbm_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=lbm_shader
        )

        self.particle_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=particle_shader
        )

    def setup_buffers(self):
        # Vertex buffer for full-screen quad
        vertices = np.array([
            -1.0, -1.0,
            1.0, -1.0,
            -1.0, 1.0,
            1.0, 1.0,
        ], dtype=np.float32)

        self.quad_vbo = self.ctx.buffer(vertices.tobytes())
        self.quad_vao = self.ctx.vertex_array(
            self.eulerian_program,
            [(self.quad_vbo, '2f', 'position')]
        )

        # Create textures with correct filtering
        self.velocity_texture = self.ctx.texture(
            (self.width, self.height), 4, dtype='f4'
        )
        self.velocity_texture.repeat_x = False
        self.velocity_texture.repeat_y = False

        # Create a 1D texture for particles
        # Each particle has 4 components (x, y, vx, vy or other attributes)
        self.particle_texture = self.ctx.texture(
            (self.particle_count, 1), 4, dtype='f4'
        )
        self.particle_texture.repeat_x = False
        self.particle_texture.repeat_y = False

        # Initialize textures with random data
        self.velocity_texture.write(np.random.rand(self.height, self.width, 4).astype('f4'))

        # Initialize particle data
        # Each particle has 4 components (x, y, vx, vy or other attributes)
        self.particles = np.random.rand(self.particle_count, 4).astype(np.float32)
        self.particle_texture.write(self.particles.reshape(-1, 4).astype('f4'))

    def update(self):
        # Clear screen
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Update simulation based on selected method
        if self.simulation_method == 'eulerian':
            self.update_eulerian()
        elif self.simulation_method == 'lbm':
            self.update_lbm()

        # Update particles
        self.update_particles()

    def update_eulerian(self):
        self.velocity_texture.use(0)
        self.eulerian_program['velocity'].value = 0
        self.eulerian_program['resolution'].value = (self.width, self.height)
        self.eulerian_program['time'].value = pygame.time.get_ticks() / 1000.0
        self.eulerian_program['vorticity'].value = self.vorticity_strength
        self.eulerian_program['temperature'].value = self.temperature

        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

    def update_lbm(self):
        self.velocity_texture.use(0)
        self.lbm_program['velocity'].value = 0
        self.lbm_program['resolution'].value = (self.width, self.height)
        self.lbm_program['time'].value = pygame.time.get_ticks() / 1000.0
        self.lbm_program['vorticity'].value = self.vorticity_strength
        self.lbm_program['temperature'].value = self.temperature
        mouse_pos = pygame.mouse.get_pos()
        self.lbm_program['ballPosition'].value = (
            mouse_pos[0] / self.width,
            1.0 - mouse_pos[1] / self.height
        )

        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

    def update_particles(self):
        self.particle_texture.use(0)
        self.velocity_texture.use(1)
        self.particle_program['positions'].value = 0
        self.particle_program['velocity'].value = 1
        self.particle_program['resolution'].value = (self.width, self.height)

        self.quad_vao.render(moderngl.POINTS)

    def render_ui(self):
        imgui.new_frame()

        if self.simulation_method is None:
            imgui.set_next_window_position(
                self.width // 2 - 100,
                self.height // 2 - 50
            )
            imgui.begin("Select Simulation Method", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
            if imgui.button("Eulerian Method", width=200):
                self.simulation_method = 'eulerian'
            if imgui.button("Lattice Boltzmann Method", width=200):
                self.simulation_method = 'lbm'
            imgui.end()

        imgui.set_next_window_position(10, 10)
        imgui.begin("Controls", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

        changed, value = imgui.slider_float(
            "Vorticity", self.vorticity_strength, 0.0, 1.0
        )
        if changed:
            self.vorticity_strength = value

        changed, value = imgui.slider_float(
            "Temperature", self.temperature, 0.0, 1.0
        )
        if changed:
            self.temperature = value

        changed, value = imgui.slider_int(
            "Particles", self.particle_count, 1000, 10000
        )
        if changed:
            self.particle_count = value
            self.particles = np.random.rand(self.particle_count, 4).astype(np.float32)
            self.particle_texture.write(self.particles)

        if imgui.button("Reset Simulation"):
            self.reset()

        imgui.end()

        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

    def reset(self):
        self.velocity_texture.write(np.zeros((self.height, self.width, 4), dtype='f4'))
        self.particles = np.random.rand(self.particle_count, 4).astype(np.float32)
        self.particle_texture.write(self.particles)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.imgui_renderer.process_event(event)

            self.update()
            self.render_ui()
            pygame.display.flip()

        pygame.quit()


if __name__ == '__main__':
    simulation = FluidSimulation()
    simulation.run()