import numpy as np
import moderngl
import glfw
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import trimesh
from PIL import Image

class RenderMode(Enum):
    """Rendering modes"""
    SURFACE = "surface"
    VOLUME = "volume"
    STREAMLINE = "streamline"
    ISOSURFACE = "isosurface"
    POINT_CLOUD = "point_cloud"

@dataclass
class RenderConfig:
    """Rendering configuration"""
    mode: RenderMode = RenderMode.SURFACE
    resolution: Tuple[int, int] = (1920, 1080)
    samples: int = 4
    use_antialiasing: bool = True
    use_shadows: bool = True
    use_ambient_occlusion: bool = True
    use_global_illumination: bool = True
    use_volumetric_lighting: bool = True
    use_depth_of_field: bool = True
    use_motion_blur: bool = True

class Renderer:
    def __init__(self, config: Optional[RenderConfig] = None):
        """
        Initialize renderer
        
        Args:
            config: Rendering configuration
        """
        self.config = config or RenderConfig()
        self._setup_window()
        self._setup_context()
        self._setup_shaders()
        self._setup_buffers()
        
    def _setup_window(self):
        """Setup GLFW window"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
            
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, self.config.samples)
        
        self.window = glfw.create_window(
            *self.config.resolution,
            "NavierFlow Visualization",
            None,
            None
        )
        
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
            
        glfw.make_context_current(self.window)
        
    def _setup_context(self):
        """Setup ModernGL context"""
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        
        if self.config.use_antialiasing:
            self.ctx.enable(moderngl.MULTISAMPLE)
            
    def _setup_shaders(self):
        """Setup shader programs"""
        # Vertex shader
        vertex_shader = """
        #version 460
        in vec3 in_position;
        in vec3 in_normal;
        in vec2 in_texcoord;
        
        out vec3 v_position;
        out vec3 v_normal;
        out vec2 v_texcoord;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        void main() {
            v_position = (model * vec4(in_position, 1.0)).xyz;
            v_normal = mat3(transpose(inverse(model))) * in_normal;
            v_texcoord = in_texcoord;
            gl_Position = projection * view * model * vec4(in_position, 1.0);
        }
        """
        
        # Fragment shader
        fragment_shader = """
        #version 460
        in vec3 v_position;
        in vec3 v_normal;
        in vec2 v_texcoord;
        
        out vec4 f_color;
        
        uniform vec3 light_position;
        uniform vec3 light_color;
        uniform vec3 camera_position;
        uniform sampler2D texture0;
        
        void main() {
            // Ambient
            vec3 ambient = 0.1 * light_color;
            
            // Diffuse
            vec3 normal = normalize(v_normal);
            vec3 light_dir = normalize(light_position - v_position);
            float diff = max(dot(normal, light_dir), 0.0);
            vec3 diffuse = diff * light_color;
            
            // Specular
            vec3 view_dir = normalize(camera_position - v_position);
            vec3 reflect_dir = reflect(-light_dir, normal);
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
            vec3 specular = spec * light_color;
            
            // Combine
            vec3 result = (ambient + diffuse + specular) * texture(texture0, v_texcoord).rgb;
            f_color = vec4(result, 1.0);
        }
        """
        
        self.prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
    def _setup_buffers(self):
        """Setup vertex and index buffers"""
        # Create empty buffers
        self.vbo = self.ctx.buffer(reserve=1024 * 1024)  # 1MB
        self.ibo = self.ctx.buffer(reserve=1024 * 1024)  # 1MB
        
        # Create vertex array
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, "3f 3f 2f", "in_position", "in_normal", "in_texcoord")
            ],
            self.ibo
        )
        
    def render_mesh(self,
                   mesh: trimesh.Trimesh,
                   camera: Dict[str, np.ndarray],
                   light: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Render mesh
        
        Args:
            mesh: Input mesh
            camera: Camera parameters
            light: Light parameters
            
        Returns:
            Rendered image
        """
        # Update vertex buffer
        vertices = np.hstack((
            mesh.vertices,
            mesh.vertex_normals,
            mesh.visual.uv
        )).astype("f4")
        self.vbo.write(vertices.tobytes())
        
        # Update index buffer
        self.ibo.write(mesh.faces.tobytes())
        
        # Update uniforms
        self.prog["model"].write(camera["model"].astype("f4").tobytes())
        self.prog["view"].write(camera["view"].astype("f4").tobytes())
        self.prog["projection"].write(camera["projection"].astype("f4").tobytes())
        self.prog["light_position"].write(light["position"].astype("f4").tobytes())
        self.prog["light_color"].write(light["color"].astype("f4").tobytes())
        self.prog["camera_position"].write(camera["position"].astype("f4").tobytes())
        
        # Clear framebuffer
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # Render
        self.vao.render()
        
        # Read pixels
        pixels = self.ctx.screen.read()
        return np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.config.resolution[1],
            self.config.resolution[0],
            4
        )
        
    def render_volume(self,
                     volume: np.ndarray,
                     transfer_function: np.ndarray,
                     camera: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Render volume
        
        Args:
            volume: Volume data
            transfer_function: Transfer function
            camera: Camera parameters
            
        Returns:
            Rendered image
        """
        # Create 3D texture
        texture = self.ctx.texture3d(
            volume.shape,
            1,
            volume.astype("f4").tobytes()
        )
        
        # Create transfer function texture
        tf_texture = self.ctx.texture1d(
            transfer_function.shape[0],
            4,
            transfer_function.astype("f4").tobytes()
        )
        
        # Update uniforms
        self.prog["volume"].value = 0
        self.prog["transfer_function"].value = 1
        self.prog["model"].write(camera["model"].astype("f4").tobytes())
        self.prog["view"].write(camera["view"].astype("f4").tobytes())
        self.prog["projection"].write(camera["projection"].astype("f4").tobytes())
        
        # Clear framebuffer
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # Render
        self.vao.render()
        
        # Read pixels
        pixels = self.ctx.screen.read()
        return np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.config.resolution[1],
            self.config.resolution[0],
            4
        )
        
    def render_streamlines(self,
                         velocity_field: np.ndarray,
                         seeds: np.ndarray,
                         camera: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Render streamlines
        
        Args:
            velocity_field: Velocity field
            seeds: Seed points
            camera: Camera parameters
            
        Returns:
            Rendered image
        """
        # Create velocity field texture
        texture = self.ctx.texture3d(
            velocity_field.shape[:3],
            3,
            velocity_field.astype("f4").tobytes()
        )
        
        # Update vertex buffer with seed points
        self.vbo.write(seeds.astype("f4").tobytes())
        
        # Update uniforms
        self.prog["velocity_field"].value = 0
        self.prog["model"].write(camera["model"].astype("f4").tobytes())
        self.prog["view"].write(camera["view"].astype("f4").tobytes())
        self.prog["projection"].write(camera["projection"].astype("f4").tobytes())
        
        # Clear framebuffer
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # Render
        self.vao.render()
        
        # Read pixels
        pixels = self.ctx.screen.read()
        return np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.config.resolution[1],
            self.config.resolution[0],
            4
        )
        
    def render_isosurface(self,
                         scalar_field: np.ndarray,
                         isovalue: float,
                         camera: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Render isosurface
        
        Args:
            scalar_field: Scalar field
            isovalue: Isosurface value
            camera: Camera parameters
            
        Returns:
            Rendered image
        """
        # Create scalar field texture
        texture = self.ctx.texture3d(
            scalar_field.shape,
            1,
            scalar_field.astype("f4").tobytes()
        )
        
        # Update uniforms
        self.prog["scalar_field"].value = 0
        self.prog["isovalue"].value = isovalue
        self.prog["model"].write(camera["model"].astype("f4").tobytes())
        self.prog["view"].write(camera["view"].astype("f4").tobytes())
        self.prog["projection"].write(camera["projection"].astype("f4").tobytes())
        
        # Clear framebuffer
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # Render
        self.vao.render()
        
        # Read pixels
        pixels = self.ctx.screen.read()
        return np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.config.resolution[1],
            self.config.resolution[0],
            4
        )
        
    def render_point_cloud(self,
                          points: np.ndarray,
                          colors: np.ndarray,
                          camera: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Render point cloud
        
        Args:
            points: Point positions
            colors: Point colors
            camera: Camera parameters
            
        Returns:
            Rendered image
        """
        # Update vertex buffer
        vertices = np.hstack((points, colors)).astype("f4")
        self.vbo.write(vertices.tobytes())
        
        # Update uniforms
        self.prog["model"].write(camera["model"].astype("f4").tobytes())
        self.prog["view"].write(camera["view"].astype("f4").tobytes())
        self.prog["projection"].write(camera["projection"].astype("f4").tobytes())
        
        # Clear framebuffer
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # Render
        self.vao.render(moderngl.POINTS)
        
        # Read pixels
        pixels = self.ctx.screen.read()
        return np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.config.resolution[1],
            self.config.resolution[0],
            4
        )
        
    def save_image(self, image: np.ndarray, filename: str):
        """
        Save rendered image
        
        Args:
            image: Rendered image
            filename: Output filename
        """
        Image.fromarray(image).save(filename)
        
    def cleanup(self):
        """Cleanup resources"""
        self.vbo.release()
        self.ibo.release()
        self.vao.release()
        self.prog.release()
        glfw.destroy_window(self.window)
        glfw.terminate() 