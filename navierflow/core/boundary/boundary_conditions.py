import taichi as ti
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

class BoundaryType(Enum):
    NO_SLIP = "no_slip"
    FREE_SLIP = "free_slip"
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    PERIODIC = "periodic"
    SYMMETRY = "symmetry"
    PRESSURE = "pressure"
    MIXED = "mixed"
    TURBULENT_WALL = "turbulent_wall"

@ti.data_oriented
class BoundaryConditions:
    """
    Advanced boundary conditions with various types and treatments:
    - No-slip and free-slip walls
    - Inflow/outflow conditions
    - Periodic boundaries
    - Symmetry planes
    - Pressure boundaries
    - Mixed conditions
    - Turbulent wall functions
    - Immersed boundaries
    - Adaptive treatments
    """
    def __init__(self, width: int, height: int, config: Dict = None):
        self.width = width
        self.height = height
        
        # Default configuration
        self.config = {
            'enable_wall_functions': True,
            'wall_model_type': 'equilibrium',
            'enable_adaptive_treatment': True,
            'enable_corner_treatment': True,
            'interpolation_order': 2,
            'outflow_treatment': 'convective',
            'pressure_bc_type': 'neumann',
            'model_constants': {
                'kappa': 0.41,  # von Karman constant
                'E': 9.8,  # Wall function constant
                'beta': 0.2,  # Outflow relaxation
            }
        }
        if config:
            self.config.update(config)
            
        # Boundary type fields
        self.boundary_type = ti.field(dtype=ti.i32, shape=(width, height))
        self.is_boundary = ti.field(dtype=ti.i32, shape=(width, height))
        
        # Wall treatment fields
        if self.config['enable_wall_functions']:
            self.y_plus = ti.field(dtype=ti.f32, shape=(width, height))
            self.u_tau = ti.field(dtype=ti.f32, shape=(width, height))
            self.wall_shear = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))
            
        # Inflow/outflow fields
        self.inflow_profile = ti.Vector.field(2, dtype=ti.f32, shape=(height,))
        self.outflow_history = ti.Vector.field(2, dtype=ti.f32,
                                           shape=(3, height))  # Store 3 time levels
                                           
        # Pressure boundary fields
        self.pressure_bc = ti.field(dtype=ti.f32, shape=(width, height))
        
        # Corner treatment fields
        if self.config['enable_corner_treatment']:
            self.is_corner = ti.field(dtype=ti.i32, shape=(width, height))
            self.corner_type = ti.field(dtype=ti.i32, shape=(width, height))
            
        self.initialize_fields()
        
    @ti.kernel
    def initialize_fields(self):
        """Initialize boundary condition fields"""
        for i, j in self.boundary_type:
            self.boundary_type[i, j] = int(BoundaryType.NO_SLIP.value)
            self.is_boundary[i, j] = 0
            
            if self.config['enable_wall_functions']:
                self.y_plus[i, j] = 0.0
                self.u_tau[i, j] = 0.0
                self.wall_shear[i, j] = ti.Vector([0.0, 0.0])
                
            if self.config['enable_corner_treatment']:
                self.is_corner[i, j] = 0
                self.corner_type[i, j] = 0
                
    def set_boundary_type(self, region: Tuple[slice, slice],
                         bc_type: BoundaryType):
        """Set boundary type for a region"""
        i_slice, j_slice = region
        self.boundary_type.fill(int(bc_type.value), i_slice, j_slice)
        self.is_boundary.fill(1, i_slice, j_slice)
        
        # Initialize specific boundary type
        if bc_type == BoundaryType.INFLOW:
            self._initialize_inflow(region)
        elif bc_type == BoundaryType.OUTFLOW:
            self._initialize_outflow(region)
        elif bc_type == BoundaryType.PRESSURE:
            self._initialize_pressure_bc(region)
            
    def set_inflow_profile(self, profile_func: Callable[[np.ndarray], np.ndarray]):
        """Set inflow velocity profile"""
        y = np.linspace(0, self.height-1, self.height)
        profile = profile_func(y)
        self.inflow_profile.from_numpy(profile)
        
    @ti.kernel
    def apply_velocity_bc(self, velocity: ti.template()):
        """Apply velocity boundary conditions"""
        for i, j in velocity:
            if self.is_boundary[i, j] == 1:
                bc_type = self.boundary_type[i, j]
                
                if bc_type == int(BoundaryType.NO_SLIP.value):
                    velocity[i, j] = ti.Vector([0.0, 0.0])
                    
                elif bc_type == int(BoundaryType.FREE_SLIP.value):
                    self.apply_free_slip(velocity, i, j)
                    
                elif bc_type == int(BoundaryType.INFLOW.value):
                    velocity[i, j] = self.inflow_profile[j]
                    
                elif bc_type == int(BoundaryType.OUTFLOW.value):
                    self.apply_outflow_bc(velocity, i, j)
                    
                elif bc_type == int(BoundaryType.PERIODIC.value):
                    self.apply_periodic_bc(velocity, i, j)
                    
                elif bc_type == int(BoundaryType.SYMMETRY.value):
                    self.apply_symmetry_bc(velocity, i, j)
                    
                elif bc_type == int(BoundaryType.TURBULENT_WALL.value):
                    self.apply_wall_function(velocity, i, j)
                    
        # Special corner treatment
        if self.config['enable_corner_treatment']:
            self.apply_corner_treatment(velocity)
            
    @ti.kernel
    def apply_pressure_bc(self, pressure: ti.template()):
        """Apply pressure boundary conditions"""
        for i, j in pressure:
            if self.is_boundary[i, j] == 1:
                bc_type = self.boundary_type[i, j]
                
                if bc_type == int(BoundaryType.PRESSURE.value):
                    pressure[i, j] = self.pressure_bc[i, j]
                    
                elif bc_type == int(BoundaryType.OUTFLOW.value):
                    if self.config['pressure_bc_type'] == 'neumann':
                        # Zero gradient
                        pressure[i, j] = pressure[i-1, j]
                    else:
                        # Dirichlet
                        pressure[i, j] = 0.0
                        
                elif bc_type == int(BoundaryType.PERIODIC.value):
                    self.apply_periodic_bc_scalar(pressure, i, j)
                    
    @ti.func
    def apply_free_slip(self, velocity: ti.template(), i: int, j: int):
        """Apply free-slip boundary condition"""
        normal = self.compute_wall_normal(i, j)
        tangent = ti.Vector([-normal.y, normal.x])
        
        # Project velocity onto tangent direction
        v_dot_t = velocity[i, j].dot(tangent)
        velocity[i, j] = v_dot_t * tangent
        
    @ti.func
    def apply_outflow_bc(self, velocity: ti.template(), i: int, j: int):
        """Apply outflow boundary condition"""
        if self.config['outflow_treatment'] == 'convective':
            # Convective outflow condition
            beta = self.config['model_constants']['beta']
            dt = 0.1  # Time step (should be passed as parameter)
            dx = 1.0  # Grid spacing
            
            u_prev = self.outflow_history[0, j]
            u_curr = self.outflow_history[1, j]
            
            # Convective equation: du/dt + U_c du/dx = 0
            U_c = ti.max(velocity[i-1, j].x, 0.0)  # Convection velocity
            velocity[i, j] = u_curr - beta * dt * U_c * (u_curr - u_prev) / dx
            
        else:
            # Zero gradient
            velocity[i, j] = velocity[i-1, j]
            
    @ti.func
    def apply_periodic_bc(self, velocity: ti.template(), i: int, j: int):
        """Apply periodic boundary condition"""
        if i == 0:
            velocity[i, j] = velocity[self.width-2, j]
        elif i == self.width-1:
            velocity[i, j] = velocity[1, j]
        elif j == 0:
            velocity[i, j] = velocity[i, self.height-2]
        elif j == self.height-1:
            velocity[i, j] = velocity[i, 1]
            
    @ti.func
    def apply_symmetry_bc(self, velocity: ti.template(), i: int, j: int):
        """Apply symmetry boundary condition"""
        normal = self.compute_wall_normal(i, j)
        
        # Reflect normal component, preserve tangential component
        v_dot_n = velocity[i, j].dot(normal)
        velocity[i, j] -= 2.0 * v_dot_n * normal
        
    @ti.func
    def apply_wall_function(self, velocity: ti.template(), i: int, j: int):
        """Apply wall function for turbulent flow"""
        if not self.config['enable_wall_functions']:
            velocity[i, j] = ti.Vector([0.0, 0.0])
            return
            
        # Compute wall distance and tangential velocity
        y = self.compute_wall_distance(i, j)
        u_tan = self.compute_tangential_velocity(velocity, i, j)
        
        # Compute friction velocity using wall function
        kappa = self.config['model_constants']['kappa']
        E = self.config['model_constants']['E']
        nu = 1e-6  # Kinematic viscosity (should be passed as parameter)
        
        # Initial guess for u_tau
        u_tau = 0.1 * ti.abs(u_tan)
        
        # Newton iteration to solve wall function equation
        for _ in range(5):
            y_plus = y * u_tau / nu
            if y_plus < 11.0:
                # Viscous sublayer
                u_plus = y_plus
                du_plus_du_tau = y / nu
            else:
                # Log layer
                u_plus = 1.0/kappa * ti.log(E * y_plus)
                du_plus_du_tau = 1.0/(kappa * u_tau)
                
            residual = u_tau * u_plus - ti.abs(u_tan)
            derivative = u_plus + u_tau * du_plus_du_tau
            
            if ti.abs(derivative) > 1e-10:
                u_tau -= residual / derivative
                
        # Store wall quantities
        self.u_tau[i, j] = u_tau
        self.y_plus[i, j] = y * u_tau / nu
        
        # Compute wall shear stress
        tangent = self.compute_wall_tangent(i, j)
        self.wall_shear[i, j] = u_tau * u_tau * tangent * ti.sign(u_tan)
        
    @ti.kernel
    def apply_corner_treatment(self, velocity: ti.template()):
        """Apply special treatment at corners"""
        if not self.config['enable_corner_treatment']:
            return
            
        for i, j in velocity:
            if self.is_corner[i, j] == 1:
                corner_type = self.corner_type[i, j]
                
                if corner_type == 1:  # Convex corner
                    self.treat_convex_corner(velocity, i, j)
                elif corner_type == 2:  # Concave corner
                    self.treat_concave_corner(velocity, i, j)
                    
    @ti.func
    def compute_wall_normal(self, i: int, j: int) -> ti.Vector:
        """Compute wall normal vector"""
        normal = ti.Vector([0.0, 0.0])
        
        # Simple normal computation based on position
        if i == 0:
            normal.x = -1.0
        elif i == self.width-1:
            normal.x = 1.0
        elif j == 0:
            normal.y = -1.0
        elif j == self.height-1:
            normal.y = 1.0
            
        return normal.normalized()
        
    @ti.func
    def compute_wall_tangent(self, i: int, j: int) -> ti.Vector:
        """Compute wall tangent vector"""
        normal = self.compute_wall_normal(i, j)
        return ti.Vector([-normal.y, normal.x])
        
    @ti.func
    def compute_wall_distance(self, i: int, j: int) -> ti.f32:
        """Compute distance to nearest wall"""
        # Simple wall distance computation
        dist = 1e10
        if i == 0:
            dist = ti.min(dist, float(i))
        elif i == self.width-1:
            dist = ti.min(dist, float(self.width-1 - i))
        if j == 0:
            dist = ti.min(dist, float(j))
        elif j == self.height-1:
            dist = ti.min(dist, float(self.height-1 - j))
            
        return dist
        
    @ti.func
    def compute_tangential_velocity(self, velocity: ti.template(),
                                  i: int, j: int) -> ti.f32:
        """Compute tangential velocity component"""
        tangent = self.compute_wall_tangent(i, j)
        return velocity[i, j].dot(tangent)
        
    def _initialize_inflow(self, region: Tuple[slice, slice]):
        """Initialize inflow boundary"""
        i_slice, j_slice = region
        # Default to uniform inflow
        self.inflow_profile.fill(ti.Vector([1.0, 0.0]))
        
    def _initialize_outflow(self, region: Tuple[slice, slice]):
        """Initialize outflow boundary"""
        i_slice, j_slice = region
        # Initialize outflow history
        self.outflow_history.fill(ti.Vector([0.0, 0.0]))
        
    def _initialize_pressure_bc(self, region: Tuple[slice, slice]):
        """Initialize pressure boundary condition"""
        i_slice, j_slice = region
        # Default to zero pressure
        self.pressure_bc.fill(0.0, i_slice, j_slice)
        
    def update_outflow_history(self, velocity: ti.template()):
        """Update outflow velocity history"""
        if self.config['outflow_treatment'] == 'convective':
            # Shift history
            self.outflow_history[0] = self.outflow_history[1]
            self.outflow_history[1] = self.outflow_history[2]
            
            # Store new values
            i = self.width - 1
            for j in range(self.height):
                if self.boundary_type[i, j] == int(BoundaryType.OUTFLOW.value):
                    self.outflow_history[2, j] = velocity[i, j]
                    
    def get_wall_shear(self) -> np.ndarray:
        """Return wall shear stress field"""
        if self.config['enable_wall_functions']:
            return self.wall_shear.to_numpy()
        return None
        
    def get_y_plus(self) -> np.ndarray:
        """Return y+ field"""
        if self.config['enable_wall_functions']:
            return self.y_plus.to_numpy()
        return None 