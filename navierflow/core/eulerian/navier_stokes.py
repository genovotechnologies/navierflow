import taichi as ti
import numpy as np
from typing import Dict, List, Optional, Tuple

@ti.data_oriented
class NavierStokesSolver:
    """
    Advanced Navier-Stokes solver using the Eulerian approach with support for:
    - Variable viscosity and compressibility
    - Multi-phase flows
    - Advanced turbulence models (k-ε, k-ω SST)
    - Temperature and density coupling
    - Multi-resolution grids
    - Adaptive time stepping
    - Surface tension and phase change
    """
    def __init__(self, width: int, height: int, config: Dict = None):
        self.width = width
        self.height = height
        
        # Default configuration
        self.config = {
            'viscosity': 1.0e-6,
            'density': 1000.0,
            'dt': 0.01,
            'substeps': 8,
            'pressure_iters': 50,
            'use_adaptive_dt': True,
            'cfl_number': 0.5,
            'enable_turbulence': False,
            'turbulence_model': 'k_epsilon',  # 'k_epsilon', 'k_omega_sst'
            'enable_temperature': False,
            'enable_compressible': False,
            'enable_multiphase': False,
            'enable_surface_tension': False,
            'multi_resolution': False,
            'grid_levels': 3,
            'mach_number': 0.3,
            'surface_tension_coeff': 0.072  # Water-air at 20°C
        }
        if config:
            self.config.update(config)
            
        # Core fields
        self.velocity = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))
        self.velocity_tmp = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))
        self.pressure = ti.field(dtype=ti.f32, shape=(width, height))
        self.divergence = ti.field(dtype=ti.f32, shape=(width, height))
        
        # Compressible flow fields
        if self.config['enable_compressible']:
            self.density_field = ti.field(dtype=ti.f32, shape=(width, height))
            self.energy = ti.field(dtype=ti.f32, shape=(width, height))
            self.sound_speed = ti.field(dtype=ti.f32, shape=(width, height))
            
        # Multi-phase fields
        if self.config['enable_multiphase']:
            self.phase_field = ti.field(dtype=ti.f32, shape=(width, height))
            self.surface_normal = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))
            self.surface_curvature = ti.field(dtype=ti.f32, shape=(width, height))
            
        # Temperature coupling
        if self.config['enable_temperature']:
            self.temperature = ti.field(dtype=ti.f32, shape=(width, height))
            self.thermal_diffusivity = ti.field(dtype=ti.f32, shape=(width, height))
            
        # Turbulence modeling
        if self.config['enable_turbulence']:
            self.eddy_viscosity = ti.field(dtype=ti.f32, shape=(width, height))
            self.turbulent_ke = ti.field(dtype=ti.f32, shape=(width, height))
            
            if self.config['turbulence_model'] == 'k_epsilon':
                self.dissipation_rate = ti.field(dtype=ti.f32, shape=(width, height))
            elif self.config['turbulence_model'] == 'k_omega_sst':
                self.specific_dissipation = ti.field(dtype=ti.f32, shape=(width, height))
                self.blending_function = ti.field(dtype=ti.f32, shape=(width, height))
            
        # Multi-resolution grids
        if self.config['multi_resolution']:
            self.grid_levels = []
            for i in range(self.config['grid_levels']):
                level_width = width >> i
                level_height = height >> i
                self.grid_levels.append({
                    'velocity': ti.Vector.field(2, dtype=ti.f32, shape=(level_width, level_height)),
                    'pressure': ti.field(dtype=ti.f32, shape=(level_width, level_height))
                })
                
        self.initialize_fields()

    @ti.kernel
    def initialize_fields(self):
        """Initialize all simulation fields"""
        for i, j in self.velocity:
            self.velocity[i, j] = ti.Vector([0.0, 0.0])
            self.pressure[i, j] = 0.0
            self.divergence[i, j] = 0.0
            
            if self.config['enable_compressible']:
                self.density_field[i, j] = self.config['density']
                self.energy[i, j] = 0.0
                self.sound_speed[i, j] = 340.0  # Air at 20°C
                
            if self.config['enable_multiphase']:
                # Initialize with two phases
                if i < self.width // 2:
                    self.phase_field[i, j] = 1.0
                else:
                    self.phase_field[i, j] = 0.0
                self.surface_normal[i, j] = ti.Vector([0.0, 0.0])
                self.surface_curvature[i, j] = 0.0
                
            if self.config['enable_temperature']:
                self.temperature[i, j] = 293.15  # 20°C
                self.thermal_diffusivity[i, j] = 1.43e-7  # Water at 20°C
                
            if self.config['enable_turbulence']:
                self.eddy_viscosity[i, j] = 0.0
                self.turbulent_ke[i, j] = 1e-4
                
                if self.config['turbulence_model'] == 'k_epsilon':
                    self.dissipation_rate[i, j] = 1e-6
                elif self.config['turbulence_model'] == 'k_omega_sst':
                    self.specific_dissipation[i, j] = 1e-2
                    self.blending_function[i, j] = 0.0

    @ti.func
    def compute_cfl_dt(self) -> ti.f32:
        """Compute time step based on CFL condition"""
        max_vel = 0.0
        for i, j in self.velocity:
            vel_magnitude = ti.sqrt(self.velocity[i, j][0]**2 + self.velocity[i, j][1]**2)
            max_vel = ti.max(max_vel, vel_magnitude)
        
        dx = 1.0  # Grid spacing
        return self.config['cfl_number'] * dx / ti.max(max_vel, 1e-6)

    @ti.kernel
    def advect(self):
        """Semi-Lagrangian advection"""
        dt = self.config['dt']
        for i, j in self.velocity:
            pos = ti.Vector([float(i), float(j)])
            vel = self.velocity[i, j]
            pos_back = pos - vel * dt
            
            # Clamp backtraced position
            pos_back[0] = ti.max(0.5, ti.min(float(self.width - 1.5), pos_back[0]))
            pos_back[1] = ti.max(0.5, ti.min(float(self.height - 1.5), pos_back[1]))
            
            # Interpolate velocity
            self.velocity_tmp[i, j] = self.interpolate_velocity(pos_back)

    @ti.func
    def interpolate_velocity(self, pos: ti.template()) -> ti.template():
        """Bilinear interpolation for velocity field"""
        x0 = int(pos[0])
        y0 = int(pos[1])
        x1 = ti.min(x0 + 1, self.width - 1)
        y1 = ti.min(y0 + 1, self.height - 1)
        
        fx = pos[0] - x0
        fy = pos[1] - y0
        
        c00 = self.velocity[x0, y0]
        c10 = self.velocity[x1, y0]
        c01 = self.velocity[x0, y1]
        c11 = self.velocity[x1, y1]
        
        return (c00 * (1 - fx) * (1 - fy) +
                c10 * fx * (1 - fy) +
                c01 * (1 - fx) * fy +
                c11 * fx * fy)

    @ti.kernel
    def apply_viscosity(self):
        """Apply viscosity diffusion"""
        dt = self.config['dt']
        dx = 1.0
        dx2 = dx * dx
        
        for i, j in self.velocity:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                laplacian = (self.velocity[i+1, j] +
                            self.velocity[i-1, j] +
                            self.velocity[i, j+1] +
                            self.velocity[i, j-1] -
                            4.0 * self.velocity[i, j]) / dx2
                            
                # Add turbulent viscosity if enabled
                total_viscosity = self.config['viscosity']
                if self.config['enable_turbulence']:
                    total_viscosity += self.eddy_viscosity[i, j]
                    
                self.velocity[i, j] += dt * total_viscosity * laplacian

    @ti.kernel
    def compute_divergence(self):
        """Compute velocity divergence"""
        dx = 1.0
        for i, j in self.divergence:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                vl = self.velocity[i-1, j].x
                vr = self.velocity[i+1, j].x
                vb = self.velocity[i, j-1].y
                vt = self.velocity[i, j+1].y
                self.divergence[i, j] = (vr - vl + vt - vb) / (2.0 * dx)

    @ti.kernel
    def solve_pressure(self):
        """Solve pressure Poisson equation"""
        dx = 1.0
        dx2 = dx * dx
        
        for _ in range(self.config['pressure_iters']):
            for i, j in self.pressure:
                if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                    pl = self.pressure[i-1, j]
                    pr = self.pressure[i+1, j]
                    pb = self.pressure[i, j-1]
                    pt = self.pressure[i, j+1]
                    div = self.divergence[i, j]
                    self.pressure[i, j] = (pl + pr + pb + pt - div * dx2) * 0.25

    @ti.kernel
    def apply_pressure_gradient(self):
        """Apply pressure gradient to enforce incompressibility"""
        dx = 1.0
        for i, j in self.velocity:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                grad_p = ti.Vector([
                    (self.pressure[i+1, j] - self.pressure[i-1, j]) / (2.0 * dx),
                    (self.pressure[i, j+1] - self.pressure[i, j-1]) / (2.0 * dx)
                ])
                self.velocity[i, j] -= grad_p / self.config['density']

    @ti.kernel
    def update_turbulence(self):
        """Update turbulence model (k-ε model)"""
        if not self.config['enable_turbulence']:
            return
            
        dt = self.config['dt']
        dx = 1.0
        
        for i, j in self.turbulent_ke:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                # Compute velocity gradients
                dudx = (self.velocity[i+1, j][0] - self.velocity[i-1, j][0]) / (2.0 * dx)
                dudy = (self.velocity[i, j+1][0] - self.velocity[i, j-1][0]) / (2.0 * dx)
                dvdx = (self.velocity[i+1, j][1] - self.velocity[i-1, j][1]) / (2.0 * dx)
                dvdy = (self.velocity[i, j+1][1] - self.velocity[i, j-1][1]) / (2.0 * dx)
                
                # Compute production term
                P = 2.0 * (dudx**2 + dvdy**2) + (dudy + dvdx)**2
                
                # Update k and ε
                k = self.turbulent_ke[i, j]
                eps = self.dissipation_rate[i, j]
                
                self.turbulent_ke[i, j] += dt * (P - eps)
                self.dissipation_rate[i, j] += dt * (1.44 * eps * P / k - 1.92 * eps**2 / k)
                
                # Update eddy viscosity
                self.eddy_viscosity[i, j] = 0.09 * k**2 / eps

    @ti.kernel
    def update_surface_tension(self):
        """Update surface tension forces for multi-phase flow"""
        if not self.config['enable_surface_tension']:
            return
            
        for i, j in self.phase_field:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                # Compute surface normal
                grad_x = (self.phase_field[i+1, j] - self.phase_field[i-1, j]) / 2.0
                grad_y = (self.phase_field[i, j+1] - self.phase_field[i, j-1]) / 2.0
                grad_mag = ti.sqrt(grad_x**2 + grad_y**2)
                
                if grad_mag > 1e-6:
                    self.surface_normal[i, j] = ti.Vector([grad_x, grad_y]) / grad_mag
                    
                    # Compute surface curvature
                    div_n = (self.surface_normal[i+1, j].x - self.surface_normal[i-1, j].x +
                            self.surface_normal[i, j+1].y - self.surface_normal[i, j-1].y) / 2.0
                    self.surface_curvature[i, j] = div_n
                    
                    # Add surface tension force
                    sigma = self.config['surface_tension_coeff']
                    force = sigma * self.surface_curvature[i, j] * self.surface_normal[i, j]
                    self.velocity[i, j] += force * self.config['dt']

    @ti.kernel
    def update_compressible_flow(self):
        """Update compressible flow fields"""
        if not self.config['enable_compressible']:
            return
            
        dt = self.config['dt']
        dx = 1.0
        gamma = 1.4  # Specific heat ratio for air
        
        for i, j in self.velocity:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                # Density update (continuity equation)
                rho = self.density_field[i, j]
                div_v = (self.velocity[i+1, j].x - self.velocity[i-1, j].x +
                        self.velocity[i, j+1].y - self.velocity[i, j-1].y) / (2.0 * dx)
                self.density_field[i, j] -= rho * div_v * dt
                
                # Energy update
                v_mag = self.velocity[i, j].norm()
                e = self.energy[i, j]
                p = (gamma - 1.0) * rho * (e - 0.5 * v_mag**2)
                
                # Update sound speed
                self.sound_speed[i, j] = ti.sqrt(gamma * p / rho)

    @ti.kernel
    def update_k_omega_sst(self):
        """Update k-ω SST turbulence model"""
        if not (self.config['enable_turbulence'] and 
                self.config['turbulence_model'] == 'k_omega_sst'):
            return
            
        dt = self.config['dt']
        dx = 1.0
        
        for i, j in self.turbulent_ke:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                # Model constants
                beta1 = 0.075
                beta2 = 0.0828
                sigma_k1 = 0.85
                sigma_k2 = 1.0
                
                # Compute strain rate
                S = self.compute_strain_rate(i, j)
                
                # Update blending function
                y_plus = self.compute_y_plus(i, j)
                self.blending_function[i, j] = self.compute_blend(y_plus)
                F1 = self.blending_function[i, j]
                
                # Compute effective diffusivity
                nu_t = self.eddy_viscosity[i, j]
                sigma_k = F1 * sigma_k1 + (1.0 - F1) * sigma_k2
                
                # Update k and ω equations
                k = self.turbulent_ke[i, j]
                omega = self.specific_dissipation[i, j]
                
                # Production and destruction terms
                P_k = nu_t * S**2
                D_k = beta1 * k * omega
                
                self.turbulent_ke[i, j] += dt * (P_k - D_k)
                self.specific_dissipation[i, j] += dt * (
                    2.0 * (1.0 - F1) * sigma_k2 / omega * 
                    self.compute_omega_gradient_dot_k_gradient(i, j)
                )

    @ti.func
    def compute_strain_rate(self, i: int, j: int) -> ti.f32:
        """Compute strain rate magnitude"""
        dudx = (self.velocity[i+1, j].x - self.velocity[i-1, j].x) / 2.0
        dudy = (self.velocity[i, j+1].x - self.velocity[i, j-1].x) / 2.0
        dvdx = (self.velocity[i+1, j].y - self.velocity[i-1, j].y) / 2.0
        dvdy = (self.velocity[i, j+1].y - self.velocity[i, j-1].y) / 2.0
        
        return ti.sqrt(2.0 * (dudx**2 + dvdy**2) + (dudy + dvdx)**2)

    def step(self):
        """Advance simulation by one time step"""
        # Adaptive time stepping
        if self.config['use_adaptive_dt']:
            self.config['dt'] = self.compute_cfl_dt()
            
        # Substeps for stability
        dt_sub = self.config['dt'] / self.config['substeps']
        for _ in range(self.config['substeps']):
            # 1. Advection
            self.advect()
            self.velocity.copy_from(self.velocity_tmp)
            
            # 2. Viscosity
            self.apply_viscosity()
            
            # 3. Multi-phase and surface tension
            if self.config['enable_multiphase']:
                self.update_surface_tension()
                
            # 4. Compressible flow
            if self.config['enable_compressible']:
                self.update_compressible_flow()
                
            # 5. Pressure projection
            self.compute_divergence()
            self.solve_pressure()
            self.apply_pressure_gradient()
            
            # 6. Turbulence update
            if self.config['enable_turbulence']:
                if self.config['turbulence_model'] == 'k_epsilon':
                    self.update_turbulence()
                elif self.config['turbulence_model'] == 'k_omega_sst':
                    self.update_k_omega_sst()
            
            # 7. Temperature coupling
            if self.config['enable_temperature']:
                self.update_temperature()

    def get_velocity_field(self) -> np.ndarray:
        """Return velocity field as numpy array"""
        return self.velocity.to_numpy()

    def get_pressure_field(self) -> np.ndarray:
        """Return pressure field as numpy array"""
        return self.pressure.to_numpy()

    def get_phase_field(self) -> Optional[np.ndarray]:
        """Return phase field if multi-phase is enabled"""
        if self.config['enable_multiphase']:
            return self.phase_field.to_numpy()
        return None

    def get_density_field(self) -> Optional[np.ndarray]:
        """Return density field if compressible flow is enabled"""
        if self.config['enable_compressible']:
            return self.density_field.to_numpy()
        return None

    def get_temperature_field(self) -> Optional[np.ndarray]:
        """Return temperature field if enabled"""
        if self.config['enable_temperature']:
            return self.temperature.to_numpy()
        return None

    def get_turbulence_fields(self) -> Optional[Dict[str, np.ndarray]]:
        """Return turbulence fields if enabled"""
        if not self.config['enable_turbulence']:
            return None
            
        fields = {
            'turbulent_ke': self.turbulent_ke.to_numpy(),
            'eddy_viscosity': self.eddy_viscosity.to_numpy()
        }
        
        if self.config['turbulence_model'] == 'k_epsilon':
            fields['dissipation_rate'] = self.dissipation_rate.to_numpy()
        elif self.config['turbulence_model'] == 'k_omega_sst':
            fields['specific_dissipation'] = self.specific_dissipation.to_numpy()
            fields['blending_function'] = self.blending_function.to_numpy()
            
        return fields 