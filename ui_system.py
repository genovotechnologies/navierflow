import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL import shaders
import moderngl
import imgui
from imgui.integrations.pygame import PygameRenderer  # Changed import

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
        self.display_mode = 'velocity'  # Options: 'velocity', 'pressure', 'fluid'
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
        self.mouse_force = 0.5
        self.particle_color = (0.7, 0.7, 0.7)  # Gray particles for LBM

        # Initialize simulation state
        self.velocity_field = np.zeros((height, width, 4), dtype=np.float32)
        self.particles = np.random.rand(self.particle_count, 4).astype(np.float32)

        # Trail system for particles
        self.trail_points = []
        self.max_trail_length = 100
        self.trail_duration = 2.0  # seconds
        self.last_trail_time = 0

        # Initialize ImGui with proper integration
        imgui.create_context()
        self.imgui_renderer = PygameRenderer()  # Updated renderer initialization

        # Setup shaders and buffers
        self.setup_shaders()
        self.setup_buffers()

    def setup_shaders(self):
        # Common vertex shader for quad rendering
        vertex_shader = """
            #version 330
            in vec2 position;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
            """

        # Vertex shader for particles
        particle_vertex_shader = """
            #version 330
            in vec2 position;
            out vec2 particlePosition;

            void main() {
                particlePosition = position;
                gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
                gl_PointSize = 3.0;  // Hardcoded point size
            }
            """

        # Particle shader for visualization
        particle_fragment_shader = """
            #version 330
            uniform sampler2D velocity;
            uniform vec3 particleColor;

            in vec2 particlePosition;
            out vec4 fragColor;

            void main() {
                vec4 velData = texture(velocity, particlePosition);
                vec2 vel = velData.xy;
                float speed = length(vel);
                vec3 color = particleColor * (0.5 + speed * 2.0);
                fragColor = vec4(color, 1.0);
            }
            """

        # Eulerian fluid shader with visualization modes
        eulerian_shader = """
            #version 330
            uniform vec2 resolution;
            uniform float time;
            uniform sampler2D velocity;
            uniform float vorticity;
            uniform float temperature;
            uniform float mouseForce;
            uniform vec2 mousePosition;
            uniform int displayMode;  // 0: velocity, 1: pressure, 2: fluid

            out vec4 fragColor;

            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution;
                vec4 v = texture(velocity, uv);
                vec2 vel = v.xy;
                float p = v.z;
                float temp = v.w;

                // Apply mouse interaction
                vec2 mousePos = mousePosition * resolution;
                vec2 dir = gl_FragCoord.xy - mousePos;
                float dist = length(dir);
                if (dist < 50.0 && dist > 0.1) {
                    dir = normalize(dir);
                    vel += dir * mouseForce * (1.0 - dist/50.0);
                }

                // Compute vorticity
                vec2 dx = vec2(1.0 / resolution.x, 0.0);
                vec2 dy = vec2(0.0, 1.0 / resolution.y);

                float curl = texture(velocity, uv + dx).x - texture(velocity, uv - dx).x
                          - texture(velocity, uv + dy).y + texture(velocity, uv - dy).y;

                vec2 force = vec2(curl * vorticity);
                force += vec2(0.0, temperature * temp);

                vel += force * 0.016;
                vel *= 0.995;  // Damping

                // Store updated values
                vec4 result = vec4(vel, p, temp);

                // Visualization based on display mode
                if (displayMode == 0) {  // Velocity
                    float speed = length(vel);
                    vec3 rgb = hsv2rgb(vec3(0.6 - speed * 2.0, 0.8, min(speed * 5.0, 1.0)));
                    fragColor = vec4(rgb, 1.0);
                } 
                else if (displayMode == 1) {  // Pressure
                    float pressure = p;
                    fragColor = vec4(pressure * 0.5 + 0.5, 0.2, 0.2, 1.0);
                }
                else {  // Fluid (display mode 2)
                    // Fluid-like visualization
                    float density = length(vel) * 0.5 + abs(curl) * 0.3;
                    vec3 fluidColor = mix(vec3(0.95, 0.98, 1.0), vec3(0.1, 0.3, 0.6), density);
                    fragColor = vec4(fluidColor, 1.0);
                }
            }
        """

        # LBM shader with white background and gray smoke
        lbm_shader = """
            #version 330
            uniform vec2 resolution;
            uniform float time;
            uniform sampler2D velocity;
            uniform vec2 ballPosition;
            uniform float vorticity;
            uniform float temperature;
            uniform int displayMode;  // 0: velocity, 1: pressure, 2: fluid

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

                // Store updated values
                vec4 result = vec4(vel, v.z, v.w);

                // Background is white, smoke is gray
                float speed = length(vel);
                float smokeIntensity = 0.0;

                // Calculate smoke intensity based on velocity and curl
                float curl = texture(velocity, uv + vec2(1.0/resolution.x, 0.0)).x - 
                             texture(velocity, uv - vec2(1.0/resolution.x, 0.0)).x -
                             texture(velocity, uv + vec2(0.0, 1.0/resolution.y)).y + 
                             texture(velocity, uv - vec2(0.0, 1.0/resolution.y)).y;

                smokeIntensity = min(speed * 4.0 + abs(curl) * 2.0, 1.0);

                // Create smoke visuals - gray smoke on white background
                vec3 smokeColor = vec3(1.0) - vec3(smokeIntensity * 0.6);  // Whiter to grayer

                // Ball visualization
                if(d < 15.0) {
                    smokeColor = vec3(0.2, 0.2, 0.2);  // Dark gray ball
                }

                fragColor = vec4(smokeColor, 1.0);
            }
        """

        # Compile shaders with proper vertex shaders
        self.eulerian_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=eulerian_shader  # Your existing Eulerian shader
        )

        self.lbm_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=lbm_shader  # Your existing LBM shader
        )

        self.particle_program = self.ctx.program(
            vertex_shader=particle_vertex_shader,
            fragment_shader=particle_fragment_shader
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
        self.velocity_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Initialize particles
        self.reset_particles()

    def reset_particles(self):
        # Initialize particles with random positions
        self.particles = np.zeros((self.particle_count, 2), dtype=np.float32)

        if self.simulation_method == 'lbm':
            # For LBM, start particles in top-left corner to show smoke effect
            for i in range(self.particle_count):
                self.particles[i] = [
                    np.random.uniform(0.05, 0.2),  # x position - left side
                    np.random.uniform(0.7, 0.95)  # y position - top
                ]
        else:
            # For Eulerian, distribute particles randomly
            for i in range(self.particle_count):
                self.particles[i] = [
                    np.random.uniform(0.05, 0.95),
                    np.random.uniform(0.05, 0.95)
                ]

        self.particle_vbo = self.ctx.buffer(self.particles.tobytes())
        self.particle_vao = self.ctx.vertex_array(
            self.particle_program,
            [(self.particle_vbo, '2f', 'position')]
        )

        # Clear trail points
        self.trail_points = []

    def update(self):
        # Set background based on simulation method
        if self.simulation_method == 'lbm':
            self.ctx.clear(1.0, 1.0, 1.0, 1.0)  # White background for LBM
        else:
            self.ctx.clear(0.1, 0.1, 0.15, 1.0)  # Dark background for Eulerian

        # Update simulation based on selected method
        if self.simulation_method == 'eulerian':
            self.update_eulerian()
        elif self.simulation_method == 'lbm':
            self.update_lbm()

        # Update particles
        self.update_particles()

    def update_eulerian(self):
        # Create framebuffer for offscreen rendering
        fbo = self.ctx.framebuffer(color_attachments=[self.velocity_texture])
        fbo.use()

        # Run Eulerian simulation step
        self.velocity_texture.use(0)
        self.eulerian_program['velocity'].value = 0
        self.eulerian_program['resolution'].value = (self.width, self.height)
        self.eulerian_program['time'].value = pygame.time.get_ticks() / 1000.0
        self.eulerian_program['vorticity'].value = self.vorticity_strength
        self.eulerian_program['temperature'].value = self.temperature
        self.eulerian_program['mouseForce'].value = self.mouse_force
        mouse_pos = pygame.mouse.get_pos()
        self.eulerian_program['mousePosition'].value = (
            mouse_pos[0] / self.width,
            1.0 - mouse_pos[1] / self.height
        )
        self.eulerian_program['displayMode'].value = {
            'velocity': 0,
            'pressure': 1,
            'fluid': 2
        }[self.display_mode]

        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        # Switch back to default framebuffer to display results
        self.ctx.screen.use()
        self.velocity_texture.use(0)
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

    def update_lbm(self):
        # Create framebuffer for offscreen rendering
        fbo = self.ctx.framebuffer(color_attachments=[self.velocity_texture])
        fbo.use()

        # Run LBM simulation step
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
        self.lbm_program['displayMode'].value = 0  # Always fluid view for LBM

        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        # Update trail system
        current_time = pygame.time.get_ticks() / 1000.0
        if current_time - self.last_trail_time > 0.05:  # Add points every 50ms
            self.last_trail_time = current_time
            self.trail_points.append({
                'pos': mouse_pos,
                'time': current_time
            })
            # Limit trail length
            if len(self.trail_points) > self.max_trail_length:
                self.trail_points.pop(0)

        # Switch back to default framebuffer to display results
        self.ctx.screen.use()
        self.velocity_texture.use(0)
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

    def update_particles(self):
        # Sample velocity field for each particle
        for i in range(self.particle_count):
            x = int(self.particles[i, 0] * self.width)
            y = int(self.particles[i, 1] * self.height)

            # Clamp to bounds
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))

            # Read velocity data
            velx = self.velocity_field[y, x, 0]
            vely = self.velocity_field[y, x, 1]

            # Update position
            self.particles[i, 0] += velx * 0.01
            self.particles[i, 1] += vely * 0.01

            # Wrap around
            self.particles[i, 0] = self.particles[i, 0] % 1.0
            self.particles[i, 1] = self.particles[i, 1] % 1.0

        # Update particle buffer
        self.particle_vbo.write(self.particles)

        # Render particles
        glEnable(GL_POINT_SMOOTH)
        glPointSize(3.0)

        self.velocity_texture.use(0)
        self.particle_program['velocity'].value = 0
        self.particle_program['particleColor'].value = self.particle_color

        self.particle_vao.render(moderngl.POINTS)

        # Render trail for LBM
        if self.simulation_method == 'lbm':
            current_time = pygame.time.get_ticks() / 1000.0
            glBegin(GL_LINE_STRIP)
            for point in self.trail_points:
                age = current_time - point['time']
                if age < self.trail_duration:
                    alpha = 1.0 - (age / self.trail_duration)
                    glColor4f(0.2, 0.2, 0.2, alpha * 0.7)
                    x = point['pos'][0] / self.width * 2.0 - 1.0
                    y = -(point['pos'][1] / self.height * 2.0 - 1.0)
                    glVertex2f(x, y)
            glEnd()

    def render_ui(self):
        imgui.new_frame()

        if self.simulation_method is None:
            imgui.set_next_window_position(
                self.width // 2 - 150,
                self.height // 2 - 100
            )
            imgui.begin("Select Simulation Method", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
            imgui.text("Select a fluid simulation method:")
            if imgui.button("Eulerian Method", width=280):
                self.simulation_method = 'eulerian'
                self.reset()
            imgui.text("Interact with fluid using mouse")
            imgui.spacing()
            if imgui.button("Lattice Boltzmann Method", width=280):
                self.simulation_method = 'lbm'
                self.particle_color = (0.7, 0.7, 0.7)  # Gray particles for LBM
                self.reset()
            imgui.text("Shows smoke flowing around a moving ball")
            imgui.end()

        else:
            imgui.set_next_window_position(10, 10)
            imgui.begin("Simulation Controls", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

            if self.simulation_method == 'eulerian':
                changed, value = imgui.combo(
                    "Display Mode",
                    ["velocity", "pressure", "fluid"].index(self.display_mode),
                    ["Velocity", "Pressure", "Fluid"]
                )
                if changed:
                    self.display_mode = ["velocity", "pressure", "fluid"][value]

                changed, value = imgui.slider_float(
                    "Mouse Force", self.mouse_force, 0.0, 2.0
                )
                if changed:
                    self.mouse_force = value

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
                self.reset_particles()

            if imgui.button("Reset Simulation"):
                self.reset()

            imgui.text(f"Method: {self.simulation_method.capitalize()}")
            if self.simulation_method == 'lbm':
                imgui.text("Move mouse to control the ball")
            else:
                imgui.text("Move mouse to interact with fluid")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if imgui.button("Switch to Eulerian Method"):
                self.simulation_method = 'eulerian'
                self.reset()
            if imgui.button("Switch to Lattice Boltzmann Method"):
                self.simulation_method = 'lbm'
                self.reset()

            imgui.end()

        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

    def reset(self):
        # Reset velocity field
        self.velocity_field = np.zeros((self.height, self.width, 4), dtype=np.float32)
        self.velocity_texture.write(self.velocity_field)

        # Reset particles based on simulation method
        self.reset_particles()

        # Reset trail
        self.trail_points = []

        # Set display mode (velocity for Eulerian by default)
        if self.simulation_method == 'eulerian':
            self.display_mode = 'velocity'
            self.particle_color = (0.3, 0.6, 1.0)  # Blue particles for Eulerian
        else:
            self.display_mode = 'fluid'
            self.particle_color = (0.7, 0.7, 0.7)  # Gray particles for LBM

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                self.imgui_renderer.process_event(event)  # Process events for ImGui

            # Update ImGui IO
            imgui.get_io().display_size = self.width, self.height

            # Get the current velocity texture data
            if self.simulation_method:
                buffer = self.ctx.buffer(reserve=self.width * self.height * 16)
                self.velocity_texture.read_into(buffer)
                self.velocity_field = np.frombuffer(buffer.read(), dtype=np.float32).reshape(self.height,
                                                                                                 self.width, 4)

            self.update()
            self.render_ui()

            pygame.display.flip()
            clock.tick(60)

        # Cleanup
        self.imgui_renderer.shutdown()
        pygame.quit()

if __name__ == '__main__':
    simulation = FluidSimulation()
    simulation.run()