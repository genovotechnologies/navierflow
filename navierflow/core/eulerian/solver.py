import taichi as ti
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from ..physics_core import MultiPhysicsSolver, PhysicsModel
from ..compute_engine import OptimizedComputeEngine, ComputeBackend

@ti.data_oriented
class CoreEulerianSolver:
    def __init__(self, width: int, height: int, config: Optional[Dict] = None):
        self.width = width
        self.height = height
        
        # Initialize physics solver with Navier-Stokes model
        self.physics_solver = MultiPhysicsSolver(
            width=width,
            height=height,
            models=[PhysicsModel.NAVIER_STOKES]
        )
        
        # Initialize compute engine for optimized operations
        self.compute_engine = OptimizedComputeEngine(backend=ComputeBackend.HYBRID)
        
        # Configuration
        self.config = {
            'dt': 0.05,
            'num_pressure_iterations': 50,
            'velocity_dissipation': 0.999,
            'density_dissipation': 0.995,
            'force_strength': 70.0,
            'force_radius': 20.0,
            'viscosity': 1.0e-6,
            'enable_turbulence': False
        }
        if config:
            self.config.update(config)
        
        # Core fields (managed by physics solver)
        self.velocity = self.physics_solver.velocity
        self.pressure = self.physics_solver.pressure
        self.density = self.physics_solver.density
        self.divergence = ti.field(dtype=ti.f32, shape=(width, height))
        
        # Mouse interaction
        self.prev_mouse_pos = ti.Vector([0.0, 0.0])
        
        # Initialize fields
        self.initialize_fields()

    @ti.kernel
    def initialize_fields(self):
        """Initialize simulation fields"""
        for i, j in self.velocity:
            self.velocity[i, j] = ti.Vector([0.0, 0.0])
            self.pressure[i, j] = 0.0
            self.density[i, j] = 0.0
            self.divergence[i, j] = 0.0

    @ti.kernel
    def add_force_and_density(self, pos_x: float, pos_y: float, vel_x: float, vel_y: float):
        """Add force and density at mouse position"""
        for i, j in ti.ndrange((-20, 20), (-20, 20)):
            x = int(pos_x + i)
            y = int(pos_y + j)

            if 0 <= x < self.width and 0 <= y < self.height:
                dx = float(i)
                dy = float(j)
                d2 = dx * dx + dy * dy
                factor = ti.exp(-d2 / self.config['force_radius'])

                # Add velocity
                self.velocity[x, y] += factor * self.config['force_strength'] * ti.Vector([vel_x, vel_y])

                # Add density with velocity-dependent intensity
                vel_magnitude = ti.sqrt(vel_x * vel_x + vel_y * vel_y)
                self.density[x, y] = ti.min(self.density[x, y] + factor * vel_magnitude * 0.1, 1.0)

    @ti.kernel
    def compute_divergence(self):
        """Compute velocity field divergence"""
        for i, j in self.divergence:
            vl = self.velocity[max(0, i - 1), j].x
            vr = self.velocity[min(self.width - 1, i + 1), j].x
            vb = self.velocity[i, max(0, j - 1)].y
            vt = self.velocity[i, min(self.height - 1, j + 1)].y
            self.divergence[i, j] = (vr - vl + vt - vb) * 0.5

    def solve_pressure(self):
        """Solve pressure Poisson equation using optimized compute engine"""
        try:
            # Compute divergence first
            self.compute_divergence()
            
            # Convert Taichi fields to PyTorch tensors
            pressure_tensor = torch.from_numpy(self.pressure.to_numpy()).cuda()
            divergence_tensor = torch.from_numpy(self.divergence.to_numpy()).cuda()
            
            # Solve using optimized compute engine
            pressure_tensor = self.compute_engine.solve_pressure_poisson(
                pressure_tensor,
                divergence_tensor,
                num_iterations=self.config['num_pressure_iterations']
            )
            
            # Update Taichi field
            self.pressure.from_torch(pressure_tensor.cpu().numpy())
        except Exception as e:
            print(f"Error in pressure solve: {str(e)}")
            # Fallback to basic pressure solve if optimization fails
            self._basic_pressure_solve()

    @ti.kernel
    def _basic_pressure_solve(self):
        """Basic pressure solver as fallback"""
        for _ in range(self.config['num_pressure_iterations']):
            for i, j in self.pressure:
                if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                    self.pressure[i, j] = (
                        self.pressure[i+1, j] + self.pressure[i-1, j] +
                        self.pressure[i, j+1] + self.pressure[i, j-1] -
                        self.divergence[i, j]
                    ) * 0.25

    @ti.kernel
    def apply_pressure_gradient(self):
        """Apply pressure gradient to velocity field"""
        for i, j in self.velocity:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                grad_p = ti.Vector([
                    self.pressure[i+1, j] - self.pressure[i-1, j],
                    self.pressure[i, j+1] - self.pressure[i, j-1]
                ]) * 0.5
                self.velocity[i, j] -= grad_p

    @ti.kernel
    def apply_viscosity(self):
        """Apply viscosity diffusion"""
        visc = self.config['viscosity']
        for i, j in self.velocity:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                laplacian = (
                    self.velocity[i+1, j] + self.velocity[i-1, j] +
                    self.velocity[i, j+1] + self.velocity[i, j-1] -
                    4.0 * self.velocity[i, j]
                )
                self.velocity[i, j] += visc * laplacian

    def step(self, mouse_pos=None, mouse_down=False):
        """Main simulation step"""
        # Handle mouse interaction
        if mouse_down and mouse_pos is not None:
            current_pos = ti.Vector([mouse_pos[0], mouse_pos[1]])
            velocity = (current_pos - self.prev_mouse_pos) * 0.5
            self.add_force_and_density(mouse_pos[0], mouse_pos[1], velocity[0], velocity[1])
            self.prev_mouse_pos = current_pos
        else:
            self.prev_mouse_pos = ti.Vector([mouse_pos[0], mouse_pos[1]]) if mouse_pos is not None else ti.Vector([0.0, 0.0])

        # Apply viscosity
        self.apply_viscosity()
        
        # Solve pressure
        self.solve_pressure()
        
        # Apply pressure gradient
        self.apply_pressure_gradient()
        
        # Physics solver step for additional effects
        if self.config['enable_turbulence']:
            self.physics_solver.step()

    def get_state(self) -> Dict:
        """Get current simulation state"""
        try:
            return {
                'velocity': self.velocity.to_numpy(),
                'pressure': self.pressure.to_numpy(),
                'density': self.density.to_numpy(),
                'divergence': self.divergence.to_numpy(),
                'metrics': {
                    'max_velocity': float(np.max(np.linalg.norm(self.velocity.to_numpy(), axis=2))),
                    'avg_pressure': float(np.mean(self.pressure.to_numpy())),
                    'max_density': float(np.max(self.density.to_numpy())),
                    'max_divergence': float(np.max(np.abs(self.divergence.to_numpy())))
                }
            }
        except Exception as e:
            print(f"Error in get_state: {str(e)}")
            return {}

    def update_config(self, config: Dict):
        """Update solver configuration"""
        self.config.update(config) 