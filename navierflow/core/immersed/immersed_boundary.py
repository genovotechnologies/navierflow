import taichi as ti
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial import cKDTree

@ti.data_oriented
class ImmersedBoundaryMethod:
    """
    Advanced immersed boundary method for complex geometries:
    - Direct forcing method
    - Distributed Lagrange multiplier method
    - Cut-cell method
    - Fluid-structure interaction
    - Adaptive mesh refinement near boundaries
    """
    def __init__(self, width: int, height: int, config: Dict = None):
        self.width = width
        self.height = height
        
        # Default configuration
        self.config = {
            'method': 'direct_forcing',  # 'direct_forcing', 'dlm', 'cut_cell'
            'interpolation_order': 2,
            'enable_fsi': True,
            'enable_adaptive_mesh': True,
            'max_boundary_points': 10000,
            'smoothing_width': 2.0,
            'rigidity_coefficient': 1.0,
            'damping_coefficient': 0.1,
            'refinement_levels': 3,
            'refinement_threshold': 0.1
        }
        if config:
            self.config.update(config)
            
        # Boundary representation
        self.boundary_points = ti.Vector.field(2, dtype=ti.f32,
                                           shape=self.config['max_boundary_points'])
        self.boundary_normals = ti.Vector.field(2, dtype=ti.f32,
                                            shape=self.config['max_boundary_points'])
        self.boundary_forces = ti.Vector.field(2, dtype=ti.f32,
                                           shape=self.config['max_boundary_points'])
        self.boundary_active = ti.field(dtype=ti.i32,
                                    shape=self.config['max_boundary_points'])
        
        # Fluid-structure interaction fields
        if self.config['enable_fsi']:
            self.displacement = ti.Vector.field(2, dtype=ti.f32,
                                            shape=self.config['max_boundary_points'])
            self.velocity = ti.Vector.field(2, dtype=ti.f32,
                                        shape=self.config['max_boundary_points'])
            self.acceleration = ti.Vector.field(2, dtype=ti.f32,
                                           shape=self.config['max_boundary_points'])
            
        # Cut-cell fields
        if self.config['method'] == 'cut_cell':
            self.cell_volume_fraction = ti.field(dtype=ti.f32, shape=(width, height))
            self.cell_center_distance = ti.field(dtype=ti.f32, shape=(width, height))
            self.cell_normal = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))
            
        # Adaptive mesh fields
        if self.config['enable_adaptive_mesh']:
            self.refinement_level = ti.field(dtype=ti.i32, shape=(width, height))
            self.refined_cells = []
            for level in range(self.config['refinement_levels']):
                level_width = width << level
                level_height = height << level
                self.refined_cells.append({
                    'velocity': ti.Vector.field(2, dtype=ti.f32,
                                           shape=(level_width, level_height)),
                    'pressure': ti.field(dtype=ti.f32,
                                     shape=(level_width, level_height))
                })
                
        self.initialize_fields()
        
    @ti.kernel
    def initialize_fields(self):
        """Initialize immersed boundary fields"""
        for i in range(self.config['max_boundary_points']):
            self.boundary_active[i] = 0
            self.boundary_points[i] = ti.Vector([0.0, 0.0])
            self.boundary_normals[i] = ti.Vector([0.0, 0.0])
            self.boundary_forces[i] = ti.Vector([0.0, 0.0])
            
            if self.config['enable_fsi']:
                self.displacement[i] = ti.Vector([0.0, 0.0])
                self.velocity[i] = ti.Vector([0.0, 0.0])
                self.acceleration[i] = ti.Vector([0.0, 0.0])
                
        if self.config['method'] == 'cut_cell':
            for i, j in self.cell_volume_fraction:
                self.cell_volume_fraction[i, j] = 1.0
                self.cell_center_distance[i, j] = 1e10
                self.cell_normal[i, j] = ti.Vector([0.0, 0.0])
                
        if self.config['enable_adaptive_mesh']:
            for i, j in self.refinement_level:
                self.refinement_level[i, j] = 0
                
    def add_boundary(self, points: np.ndarray, normals: np.ndarray):
        """Add boundary points and normals"""
        n_points = min(len(points), self.config['max_boundary_points'])
        
        # Convert to Taichi fields
        for i in range(n_points):
            self.boundary_points[i] = ti.Vector([points[i, 0], points[i, 1]])
            self.boundary_normals[i] = ti.Vector([normals[i, 0], normals[i, 1]])
            self.boundary_active[i] = 1
            
        # Build KD-tree for efficient neighbor searches
        self.boundary_tree = cKDTree(points[:n_points])
        
    @ti.kernel
    def compute_direct_forcing(self, fluid_velocity: ti.template()):
        """Compute direct forcing method"""
        if self.config['method'] != 'direct_forcing':
            return
            
        for i in range(self.config['max_boundary_points']):
            if self.boundary_active[i] == 1:
                pos = self.boundary_points[i]
                target_vel = ti.Vector([0.0, 0.0])
                if self.config['enable_fsi']:
                    target_vel = self.velocity[i]
                    
                # Interpolate fluid velocity
                fluid_vel = self.interpolate_velocity(fluid_velocity, pos)
                
                # Compute force
                self.boundary_forces[i] = (target_vel - fluid_vel) / self.config['dt']
                
    @ti.kernel
    def apply_dlm_method(self, fluid_velocity: ti.template(),
                        fluid_pressure: ti.template()):
        """Apply distributed Lagrange multiplier method"""
        if self.config['method'] != 'dlm':
            return
            
        for i in range(self.config['max_boundary_points']):
            if self.boundary_active[i] == 1:
                pos = self.boundary_points[i]
                
                # Compute Lagrange multipliers
                lambda_momentum = self.compute_momentum_multiplier(pos, fluid_velocity)
                lambda_incompress = self.compute_incompress_multiplier(pos, fluid_pressure)
                
                # Update forces
                self.boundary_forces[i] = lambda_momentum + lambda_incompress
                
    @ti.kernel
    def update_cut_cells(self):
        """Update cut-cell geometry information"""
        if self.config['method'] != 'cut_cell':
            return
            
        for i, j in self.cell_volume_fraction:
            min_dist = 1e10
            closest_normal = ti.Vector([0.0, 0.0])
            
            # Find closest boundary point
            cell_center = ti.Vector([float(i) + 0.5, float(j) + 0.5])
            for b in range(self.config['max_boundary_points']):
                if self.boundary_active[b] == 1:
                    dist = (cell_center - self.boundary_points[b]).norm()
                    if dist < min_dist:
                        min_dist = dist
                        closest_normal = self.boundary_normals[b]
                        
            # Update cell properties
            self.cell_center_distance[i, j] = min_dist
            self.cell_normal[i, j] = closest_normal
            self.cell_volume_fraction[i, j] = self.compute_volume_fraction(
                cell_center, min_dist, closest_normal
            )
            
    @ti.kernel
    def update_fsi(self, dt: ti.f32):
        """Update fluid-structure interaction"""
        if not self.config['enable_fsi']:
            return
            
        for i in range(self.config['max_boundary_points']):
            if self.boundary_active[i] == 1:
                # Update kinematics
                force = self.boundary_forces[i]
                mass = 1.0  # Mass per boundary point
                
                # Add elastic and damping forces
                k = self.config['rigidity_coefficient']
                c = self.config['damping_coefficient']
                elastic_force = -k * self.displacement[i]
                damping_force = -c * self.velocity[i]
                
                total_force = force + elastic_force + damping_force
                
                # Update motion
                self.acceleration[i] = total_force / mass
                self.velocity[i] += self.acceleration[i] * dt
                self.displacement[i] += self.velocity[i] * dt
                
                # Update boundary point position
                self.boundary_points[i] += self.velocity[i] * dt
                
    @ti.kernel
    def adapt_mesh(self):
        """Adapt mesh resolution near boundaries"""
        if not self.config['enable_adaptive_mesh']:
            return
            
        for i, j in self.refinement_level:
            min_dist = 1e10
            
            # Find distance to nearest boundary
            cell_center = ti.Vector([float(i) + 0.5, float(j) + 0.5])
            for b in range(self.config['max_boundary_points']):
                if self.boundary_active[b] == 1:
                    dist = (cell_center - self.boundary_points[b]).norm()
                    min_dist = ti.min(min_dist, dist)
                    
            # Determine refinement level
            threshold = self.config['refinement_threshold']
            if min_dist < threshold:
                self.refinement_level[i, j] = ti.min(
                    int(-ti.log(min_dist/threshold) / ti.log(2.0)),
                    self.config['refinement_levels'] - 1
                )
                
    @ti.func
    def interpolate_velocity(self, velocity: ti.template(),
                           pos: ti.template()) -> ti.Vector:
        """Interpolate velocity field at arbitrary position"""
        order = self.config['interpolation_order']
        
        if order == 1:
            return self.linear_interpolation(velocity, pos)
        else:
            return self.quadratic_interpolation(velocity, pos)
            
    @ti.func
    def compute_volume_fraction(self, cell_center: ti.template(),
                              distance: ti.f32,
                              normal: ti.template()) -> ti.f32:
        """Compute cut-cell volume fraction"""
        if distance >= 0.707:  # sqrt(2)/2
            return 1.0
        elif distance <= -0.707:
            return 0.0
        else:
            # Approximate volume fraction based on distance and normal
            return 0.5 * (1.0 + distance / 0.707)
            
    def spread_boundary_force(self, fluid_force: np.ndarray):
        """Spread boundary forces to fluid grid"""
        n_points = np.sum(self.boundary_active.to_numpy())
        points = self.boundary_points.to_numpy()[:n_points]
        forces = self.boundary_forces.to_numpy()[:n_points]
        
        # Use smoothed delta function
        width = self.config['smoothing_width']
        for i, (point, force) in enumerate(zip(points, forces)):
            # Find affected fluid cells
            x_min = max(0, int(point[0] - width))
            x_max = min(self.width, int(point[0] + width + 1))
            y_min = max(0, int(point[1] - width))
            y_max = min(self.height, int(point[1] + width + 1))
            
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    r = np.array([x + 0.5 - point[0], y + 0.5 - point[1]])
                    delta = self.smoothed_delta(r, width)
                    fluid_force[x, y] += force * delta
                    
    @staticmethod
    def smoothed_delta(r: np.ndarray, width: float) -> float:
        """Smoothed delta function for force spreading"""
        x, y = r
        if abs(x) > width or abs(y) > width:
            return 0.0
            
        phi_x = ImmersedBoundaryMethod.phi(x/width)
        phi_y = ImmersedBoundaryMethod.phi(y/width)
        return phi_x * phi_y / (width * width)
        
    @staticmethod
    def phi(r: float) -> float:
        """1D kernel function for smoothed delta function"""
        r = abs(r)
        if r > 2.0:
            return 0.0
        elif r > 1.0:
            return (2 - r) * (4 - 4*r + r*r) / 8
        else:
            return (3 - 2*r + sqrt(1 + 4*r - 4*r*r)) / 8
            
    def get_boundary_points(self) -> np.ndarray:
        """Return active boundary points"""
        return self.boundary_points.to_numpy()[
            self.boundary_active.to_numpy() == 1
        ]
        
    def get_boundary_forces(self) -> np.ndarray:
        """Return boundary forces"""
        return self.boundary_forces.to_numpy()[
            self.boundary_active.to_numpy() == 1
        ]
        
    def get_volume_fractions(self) -> Optional[np.ndarray]:
        """Return cut-cell volume fractions"""
        if self.config['method'] == 'cut_cell':
            return self.cell_volume_fraction.to_numpy()
        return None
        
    def get_refinement_levels(self) -> Optional[np.ndarray]:
        """Return mesh refinement levels"""
        if self.config['enable_adaptive_mesh']:
            return self.refinement_level.to_numpy()
        return None 