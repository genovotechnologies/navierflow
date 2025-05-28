import taichi as ti
import numpy as np
from typing import Dict, List, Optional, Tuple

@ti.data_oriented
class VortexMethods:
    """
    Advanced vortex methods for spectral solver enhancement:
    - Vortex particle method
    - Vortex filament tracking
    - Vortex sheet evolution
    - Vortex reconnection
    - Helicity dynamics
    """
    def __init__(self, width: int, height: int, config: Dict = None):
        self.width = width
        self.height = height
        
        # Default configuration
        self.config = {
            'max_vortex_particles': 10000,
            'particle_strength': 0.1,
            'core_size': 0.1,
            'enable_stretching': True,
            'enable_diffusion': True,
            'enable_reconnection': True,
            'reconnection_threshold': 0.1,
            'helicity_preservation': True
        }
        if config:
            self.config.update(config)
            
        # Vortex particle fields
        self.particle_positions = ti.Vector.field(2, dtype=ti.f32, 
                                               shape=self.config['max_vortex_particles'])
        self.particle_strengths = ti.Vector.field(2, dtype=ti.f32,
                                               shape=self.config['max_vortex_particles'])
        self.particle_active = ti.field(dtype=ti.i32,
                                     shape=self.config['max_vortex_particles'])
        
        # Vortex filament fields
        self.filament_points = ti.Vector.field(2, dtype=ti.f32,
                                            shape=(width, height))
        self.filament_strength = ti.field(dtype=ti.f32,
                                       shape=(width, height))
        
        # Vortex sheet fields
        self.sheet_position = ti.Vector.field(2, dtype=ti.f32,
                                           shape=(width, height))
        self.sheet_strength = ti.field(dtype=ti.f32,
                                    shape=(width, height))
        self.sheet_curvature = ti.field(dtype=ti.f32,
                                     shape=(width, height))
        
        # Helicity fields
        self.helicity_density = ti.field(dtype=ti.f32,
                                      shape=(width, height))
        self.relative_helicity = ti.field(dtype=ti.f32,
                                       shape=(width, height))
        
        self.initialize_fields()
        
    @ti.kernel
    def initialize_fields(self):
        """Initialize vortex fields"""
        # Initialize vortex particles
        for i in range(self.config['max_vortex_particles']):
            self.particle_active[i] = 0
            self.particle_positions[i] = ti.Vector([0.0, 0.0])
            self.particle_strengths[i] = ti.Vector([0.0, 0.0])
            
        # Initialize vortex filaments and sheets
        for i, j in self.filament_points:
            self.filament_points[i, j] = ti.Vector([float(i), float(j)])
            self.filament_strength[i, j] = 0.0
            self.sheet_position[i, j] = ti.Vector([float(i), float(j)])
            self.sheet_strength[i, j] = 0.0
            self.sheet_curvature[i, j] = 0.0
            self.helicity_density[i, j] = 0.0
            self.relative_helicity[i, j] = 0.0
            
    @ti.kernel
    def seed_vortex_particles(self, vorticity: ti.template()):
        """Seed vortex particles based on vorticity field"""
        threshold = 0.1  # Minimum vorticity magnitude
        particle_count = 0
        
        for i, j in vorticity:
            if particle_count < self.config['max_vortex_particles']:
                vort_mag = ti.abs(vorticity[i, j])
                if vort_mag > threshold:
                    idx = ti.atomic_add(particle_count, 1)
                    self.particle_active[idx] = 1
                    self.particle_positions[idx] = ti.Vector([float(i), float(j)])
                    self.particle_strengths[idx] = ti.Vector([0.0, vorticity[i, j]])
                    
    @ti.kernel
    def update_vortex_particles(self, velocity: ti.template(), dt: ti.f32):
        """Update vortex particle positions and strengths"""
        for i in range(self.config['max_vortex_particles']):
            if self.particle_active[i] == 1:
                pos = self.particle_positions[i]
                
                # Advection
                vel = self.interpolate_velocity(velocity, pos)
                self.particle_positions[i] += vel * dt
                
                # Stretching
                if self.config['enable_stretching']:
                    stretch = self.compute_stretching(velocity, pos)
                    self.particle_strengths[i] += stretch * dt
                    
                # Diffusion
                if self.config['enable_diffusion']:
                    self.particle_strengths[i] += self.compute_diffusion(i, dt)
                    
    @ti.func
    def interpolate_velocity(self, velocity: ti.template(),
                           pos: ti.template()) -> ti.Vector:
        """Bilinear interpolation of velocity field"""
        x = pos.x
        y = pos.y
        x0 = ti.floor(x)
        y0 = ti.floor(y)
        x1 = x0 + 1
        y1 = y0 + 1
        
        fx = x - x0
        fy = y - y0
        
        c00 = velocity[x0, y0]
        c10 = velocity[x1, y0]
        c01 = velocity[x0, y1]
        c11 = velocity[x1, y1]
        
        return (c00 * (1-fx) * (1-fy) +
                c10 * fx * (1-fy) +
                c01 * (1-fx) * fy +
                c11 * fx * fy)
                
    @ti.kernel
    def update_vortex_filaments(self, velocity: ti.template(), dt: ti.f32):
        """Update vortex filaments"""
        for i, j in self.filament_points:
            if self.filament_strength[i, j] > 0:
                # Advect filament points
                vel = velocity[i, j]
                self.filament_points[i, j] += vel * dt
                
                # Update strength
                stretch = self.compute_filament_stretching(i, j)
                self.filament_strength[i, j] *= (1.0 + stretch * dt)
                
                # Check for reconnection
                if self.config['enable_reconnection']:
                    self.check_filament_reconnection(i, j)
                    
    @ti.kernel
    def update_vortex_sheets(self, velocity: ti.template(), dt: ti.f32):
        """Update vortex sheets"""
        for i, j in self.sheet_position:
            if self.sheet_strength[i, j] > 0:
                # Compute sheet dynamics
                self.update_sheet_position(i, j, velocity, dt)
                self.update_sheet_strength(i, j, dt)
                self.update_sheet_curvature(i, j)
                
    @ti.kernel
    def compute_helicity(self, velocity: ti.template(),
                        vorticity: ti.template()):
        """Compute helicity density and relative helicity"""
        for i, j in self.helicity_density:
            # Helicity density = v · ω
            self.helicity_density[i, j] = velocity[i, j].dot(
                ti.Vector([0.0, vorticity[i, j]])
            )
            
            # Relative helicity = (v · ω) / (|v| |ω|)
            v_mag = velocity[i, j].norm()
            w_mag = ti.abs(vorticity[i, j])
            if v_mag > 1e-6 and w_mag > 1e-6:
                self.relative_helicity[i, j] = self.helicity_density[i, j] / (v_mag * w_mag)
                
    @ti.func
    def compute_stretching(self, velocity: ti.template(),
                         pos: ti.template()) -> ti.Vector:
        """Compute vortex stretching term"""
        # Compute velocity gradient tensor
        dx = 1.0
        dudx = (self.interpolate_velocity(velocity, pos + ti.Vector([dx, 0.0])) -
                self.interpolate_velocity(velocity, pos - ti.Vector([dx, 0.0]))) / (2.0 * dx)
        dudy = (self.interpolate_velocity(velocity, pos + ti.Vector([0.0, dx])) -
                self.interpolate_velocity(velocity, pos - ti.Vector([0.0, dx]))) / (2.0 * dx)
                
        return ti.Vector([dudx.x, dudy.y])
        
    @ti.func
    def compute_diffusion(self, idx: ti.i32, dt: ti.f32) -> ti.Vector:
        """Compute diffusion for vortex particle"""
        nu = 1e-6  # Kinematic viscosity
        sigma = ti.sqrt(2.0 * nu * dt)
        
        return ti.Vector([
            ti.random.gauss(0.0, sigma),
            ti.random.gauss(0.0, sigma)
        ])
        
    @ti.func
    def check_filament_reconnection(self, i: int, j: int):
        """Check and handle vortex filament reconnection"""
        threshold = self.config['reconnection_threshold']
        
        for di in ti.static(range(-1, 2)):
            for dj in ti.static(range(-1, 2)):
                if di != 0 or dj != 0:
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < self.width and 0 <= nj < self.height:
                        dist = (self.filament_points[i, j] - 
                               self.filament_points[ni, nj]).norm()
                        if dist < threshold:
                            self.reconnect_filaments(i, j, ni, nj)
                            
    @ti.func
    def reconnect_filaments(self, i1: int, j1: int, i2: int, j2: int):
        """Perform vortex filament reconnection"""
        # Average strength and position
        avg_strength = (self.filament_strength[i1, j1] + 
                       self.filament_strength[i2, j2]) * 0.5
        avg_pos = (self.filament_points[i1, j1] + 
                  self.filament_points[i2, j2]) * 0.5
                  
        # Update reconnected filaments
        self.filament_points[i1, j1] = avg_pos
        self.filament_points[i2, j2] = avg_pos
        self.filament_strength[i1, j1] = avg_strength
        self.filament_strength[i2, j2] = avg_strength
        
    def get_particle_positions(self) -> np.ndarray:
        """Return active particle positions"""
        return self.particle_positions.to_numpy()[
            self.particle_active.to_numpy() == 1
        ]
        
    def get_particle_strengths(self) -> np.ndarray:
        """Return active particle strengths"""
        return self.particle_strengths.to_numpy()[
            self.particle_active.to_numpy() == 1
        ]
        
    def get_helicity_field(self) -> np.ndarray:
        """Return helicity density field"""
        return self.helicity_density.to_numpy()
        
    def get_relative_helicity(self) -> np.ndarray:
        """Return relative helicity field"""
        return self.relative_helicity.to_numpy() 