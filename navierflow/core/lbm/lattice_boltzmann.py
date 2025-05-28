import taichi as ti
import numpy as np
from typing import Dict, List, Optional, Tuple

@ti.data_oriented
class LBMSolver:
    """
    Advanced Lattice Boltzmann Method solver with support for:
    - Multiple collision operators (BGK, MRT, TRT, Entropic)
    - Thermal flows with double distribution approach
    - Multi-component flows with Shan-Chen model
    - Various boundary conditions
    - Adaptive grid refinement
    - GPU acceleration
    """
    def __init__(self, width: int, height: int, config: Dict = None):
        self.width = width
        self.height = height
        
        # Default configuration
        self.config = {
            'collision_operator': 'bgk',  # 'bgk', 'mrt', 'trt', 'entropic'
            'viscosity': 0.01,
            'tau': 0.6,  # Relaxation time
            'enable_thermal': False,
            'enable_multicomponent': False,
            'enable_adaptive_grid': False,
            'gravity': [0.0, -9.81e-4],
            'boundary_conditions': 'bounce_back',  # 'bounce_back', 'zou_he'
            'mrt_relaxation_rates': [1.0, 1.4, 1.4, 1.0, 1.2, 1.0, 1.2],
            'surface_tension': 0.1,
            'thermal_diffusivity': 0.01,
            'grid_refinement_levels': 2
        }
        if config:
            self.config.update(config)
            
        # D2Q9 lattice constants
        self.Q = 9  # Number of velocities
        self.c = ti.Vector.field(2, dtype=ti.f32, shape=self.Q)
        self.w = ti.field(dtype=ti.f32, shape=self.Q)
        
        # Initialize lattice velocities and weights
        self._init_lattice_constants()
        
        # Distribution functions
        self.f = ti.field(dtype=ti.f32, shape=(width, height, self.Q))
        self.f_temp = ti.field(dtype=ti.f32, shape=(width, height, self.Q))
        
        # Macroscopic quantities
        self.density = ti.field(dtype=ti.f32, shape=(width, height))
        self.velocity = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))
        
        # MRT fields
        if self.config['collision_operator'] == 'mrt':
            self.moments = ti.field(dtype=ti.f32, shape=(width, height, self.Q))
            self.equilibrium_moments = ti.field(dtype=ti.f32, shape=(width, height, self.Q))
            
        # Thermal fields
        if self.config['enable_thermal']:
            self.temperature = ti.field(dtype=ti.f32, shape=(width, height))
            self.g = ti.field(dtype=ti.f32, shape=(width, height, self.Q))  # Thermal distributions
            self.g_temp = ti.field(dtype=ti.f32, shape=(width, height, self.Q))
            self.thermal_conductivity = ti.field(dtype=ti.f32, shape=(width, height))
            
        # Multi-component fields
        if self.config['enable_multicomponent']:
            self.phase_field = ti.field(dtype=ti.f32, shape=(width, height))
            self.chemical_potential = ti.field(dtype=ti.f32, shape=(width, height))
            self.surface_tension_force = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))
            
        # Adaptive grid fields
        if self.config['enable_adaptive_grid']:
            self.grid_levels = []
            for level in range(self.config['grid_refinement_levels']):
                level_width = width >> level
                level_height = height >> level
                self.grid_levels.append({
                    'f': ti.field(dtype=ti.f32, shape=(level_width, level_height, self.Q)),
                    'density': ti.field(dtype=ti.f32, shape=(level_width, level_height)),
                    'velocity': ti.Vector.field(2, dtype=ti.f32, shape=(level_width, level_height))
                })
            
        # Initialize fields
        self.initialize_fields()

    def _init_lattice_constants(self):
        """Initialize D2Q9 lattice constants"""
        # D2Q9 velocities
        velocities = [
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1]
        ]
        
        # D2Q9 weights
        weights = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
        
        # Copy to Taichi fields
        for i in range(self.Q):
            self.c[i] = ti.Vector(velocities[i])
            self.w[i] = weights[i]

    @ti.kernel
    def initialize_fields(self):
        """Initialize distribution functions and macroscopic quantities"""
        for i, j in ti.ndrange(self.width, self.height):
            self.density[i, j] = 1.0
            self.velocity[i, j] = ti.Vector([0.0, 0.0])
            
            # Initialize equilibrium distributions
            for k in range(self.Q):
                self.f[i, j, k] = self.compute_equilibrium(i, j, k)
                
            if self.config['enable_thermal']:
                self.temperature[i, j] = 1.0
                self.thermal_conductivity[i, j] = self.config['thermal_diffusivity']
                for k in range(self.Q):
                    self.g[i, j, k] = self.compute_thermal_equilibrium(i, j, k)
                    
            if self.config['enable_multicomponent']:
                # Initialize with two phases
                if i < self.width // 2:
                    self.phase_field[i, j] = 1.0
                else:
                    self.phase_field[i, j] = -1.0
                self.chemical_potential[i, j] = 0.0
                self.surface_tension_force[i, j] = ti.Vector([0.0, 0.0])

    @ti.func
    def compute_equilibrium(self, i: int, j: int, k: int) -> ti.f32:
        """Compute equilibrium distribution function"""
        cu = self.c[k].dot(self.velocity[i, j])
        usqr = self.velocity[i, j].dot(self.velocity[i, j])
        return self.w[k] * self.density[i, j] * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usqr)

    @ti.func
    def compute_thermal_equilibrium(self, i: int, j: int, k: int) -> ti.f32:
        """Compute thermal equilibrium distribution"""
        if not self.config['enable_thermal']:
            return 0.0
            
        T = self.temperature[i, j]
        cu = self.c[k].dot(self.velocity[i, j])
        return self.w[k] * T * (1.0 + 3.0*cu)

    @ti.kernel
    def collide_bgk(self):
        """BGK collision operator"""
        tau = self.config['tau']
        for i, j, k in ti.ndrange(self.width, self.height, self.Q):
            feq = self.compute_equilibrium(i, j, k)
            self.f[i, j, k] = self.f[i, j, k] - (self.f[i, j, k] - feq) / tau

    @ti.kernel
    def collide_mrt(self):
        """Multiple-relaxation-time collision operator"""
        if self.config['collision_operator'] != 'mrt':
            return
            
        # MRT collision matrix and relaxation rates
        S = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())
        for i in range(9):
            S[None][i, i] = self.config['mrt_relaxation_rates'][i]
            
        for i, j in ti.ndrange(self.width, self.height):
            # Transform to moment space
            self.compute_moments(i, j)
            
            # Compute equilibrium moments
            self.compute_equilibrium_moments(i, j)
            
            # Collision in moment space
            for k in range(self.Q):
                self.moments[i, j, k] -= S[None][k, k] * (
                    self.moments[i, j, k] - self.equilibrium_moments[i, j, k]
                )
                
            # Transform back to velocity space
            self.transform_from_moments(i, j)

    @ti.kernel
    def collide_entropic(self):
        """Entropic collision operator for enhanced stability"""
        if self.config['collision_operator'] != 'entropic':
            return
            
        beta = 2.0  # Entropic parameter
        
        for i, j in ti.ndrange(self.width, self.height):
            # Compute entropy change
            dS = 0.0
            for k in range(self.Q):
                feq = self.compute_equilibrium(i, j, k)
                if self.f[i, j, k] > 0:
                    dS += self.f[i, j, k] * ti.log(self.f[i, j, k] / feq)
                    
            # Adjust relaxation parameter
            alpha = 2.0 / (1.0 + ti.exp(beta * dS))
            
            # Collision
            for k in range(self.Q):
                feq = self.compute_equilibrium(i, j, k)
                self.f[i, j, k] = self.f[i, j, k] - alpha * (
                    self.f[i, j, k] - feq
                )

    @ti.kernel
    def stream(self):
        """Streaming step"""
        for i, j, k in ti.ndrange(self.width, self.height, self.Q):
            # Stream to neighboring nodes
            ni = i + int(self.c[k][0])
            nj = j + int(self.c[k][1])
            
            # Periodic boundary conditions
            ni = (ni + self.width) % self.width
            nj = (nj + self.height) % self.height
            
            self.f_temp[ni, nj, k] = self.f[i, j, k]
            
        # Copy back
        for i, j, k in ti.ndrange(self.width, self.height, self.Q):
            self.f[i, j, k] = self.f_temp[i, j, k]

    @ti.kernel
    def apply_bounce_back(self):
        """Apply bounce-back boundary conditions"""
        for i, j in ti.ndrange(self.width, self.height):
            if i == 0 or i == self.width-1 or j == 0 or j == self.height-1:
                for k in range(self.Q):
                    # Find opposite direction
                    k_opp = (k + self.Q//2) % self.Q if k > 0 else k
                    self.f[i, j, k] = self.f[i, j, k_opp]

    @ti.kernel
    def update_macroscopic(self):
        """Update macroscopic quantities"""
        for i, j in ti.ndrange(self.width, self.height):
            # Compute density
            rho = 0.0
            for k in range(self.Q):
                rho += self.f[i, j, k]
            self.density[i, j] = rho
            
            # Compute velocity
            vel = ti.Vector([0.0, 0.0])
            for k in range(self.Q):
                vel += self.c[k] * self.f[i, j, k]
            self.velocity[i, j] = vel / rho
            
            # Add gravity force
            self.velocity[i, j] += ti.Vector(self.config['gravity']) * self.config['tau']

    @ti.kernel
    def update_thermal_field(self):
        """Update thermal distributions"""
        if not self.config['enable_thermal']:
            return
            
        for i, j, k in ti.ndrange(self.width, self.height, self.Q):
            # Collision
            geq = self.compute_thermal_equilibrium(i, j, k)
            omega_t = 1.0 / (3.0 * self.thermal_conductivity[i, j] + 0.5)
            self.g[i, j, k] = self.g[i, j, k] - omega_t * (
                self.g[i, j, k] - geq
            )
            
            # Streaming
            ni = i + int(self.c[k][0])
            nj = j + int(self.c[k][1])
            
            if 0 <= ni < self.width and 0 <= nj < self.height:
                self.g_temp[ni, nj, k] = self.g[i, j, k]
                
        # Update temperature
        for i, j in self.temperature:
            T = 0.0
            for k in range(self.Q):
                T += self.g_temp[i, j, k]
            self.temperature[i, j] = T

    @ti.kernel
    def update_multicomponent(self):
        """Update multi-component fields using Shan-Chen model"""
        if not self.config['enable_multicomponent']:
            return
            
        G = self.config['surface_tension']  # Interaction strength
        
        for i, j in ti.ndrange(self.width, self.height):
            if 0 < i < self.width-1 and 0 < j < self.height-1:
                # Compute interaction force
                F = ti.Vector([0.0, 0.0])
                for k in range(self.Q):
                    ni = i + int(self.c[k][0])
                    nj = j + int(self.c[k][1])
                    F += self.w[k] * self.c[k] * self.phase_field[ni, nj]
                    
                self.surface_tension_force[i, j] = -G * self.phase_field[i, j] * F
                
                # Update chemical potential
                self.chemical_potential[i, j] = self.compute_chemical_potential(i, j)

    @ti.kernel
    def apply_adaptive_grid(self):
        """Apply adaptive grid refinement"""
        if not self.config['enable_adaptive_grid']:
            return
            
        for level in range(self.config['grid_refinement_levels']-1):
            # Restriction (coarse to fine)
            self.restrict_to_coarse_grid(level)
            
            # Prolongation (fine to coarse)
            self.prolong_to_fine_grid(level)

    def step(self):
        """Advance simulation by one time step"""
        # 1. Collision
        if self.config['collision_operator'] == 'bgk':
            self.collide_bgk()
        elif self.config['collision_operator'] == 'mrt':
            self.collide_mrt()
        elif self.config['collision_operator'] == 'entropic':
            self.collide_entropic()
            
        # 2. Streaming
        self.stream()
        
        # 3. Boundary conditions
        if self.config['boundary_conditions'] == 'bounce_back':
            self.apply_bounce_back()
            
        # 4. Update macroscopic quantities
        self.update_macroscopic()
        
        # 5. Update thermal field
        if self.config['enable_thermal']:
            self.update_thermal_field()
            
        # 6. Update multi-component fields
        if self.config['enable_multicomponent']:
            self.update_multicomponent()
            
        # 7. Apply adaptive grid refinement
        if self.config['enable_adaptive_grid']:
            self.apply_adaptive_grid()

    def get_velocity_field(self) -> np.ndarray:
        """Return velocity field as numpy array"""
        return self.velocity.to_numpy()

    def get_density_field(self) -> np.ndarray:
        """Return density field as numpy array"""
        return self.density.to_numpy()

    def get_temperature_field(self) -> Optional[np.ndarray]:
        """Return temperature field if enabled"""
        if self.config['enable_thermal']:
            return self.temperature.to_numpy()
        return None

    def get_phase_field(self) -> Optional[np.ndarray]:
        """Return phase field if multi-component is enabled"""
        if self.config['enable_multicomponent']:
            return self.phase_field.to_numpy()
        return None

    def get_grid_level(self, level: int) -> Optional[Dict[str, np.ndarray]]:
        """Return fields at specified grid refinement level"""
        if not self.config['enable_adaptive_grid'] or level >= len(self.grid_levels):
            return None
            
        return {
            'density': self.grid_levels[level]['density'].to_numpy(),
            'velocity': self.grid_levels[level]['velocity'].to_numpy()
        } 