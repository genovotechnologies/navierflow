import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class MaterialProperties:
    """Properties of a material"""
    permittivity: float
    permeability: float
    conductivity: float

class Electromagnetic:
    def __init__(self,
                 materials: List[MaterialProperties],
                 courant_number: float = 0.5):
        """
        Initialize electromagnetic solver
        
        Args:
            materials: List of material properties
            courant_number: Courant number for stability
        """
        self.materials = materials
        self.courant_number = courant_number
        
    def compute_electric_field(self,
                             E: np.ndarray,
                             H: np.ndarray,
                             J: np.ndarray,
                             dt: float,
                             dx: float) -> np.ndarray:
        """
        Update electric field using Maxwell's equations
        
        Args:
            E: Electric field
            H: Magnetic field
            J: Current density
            dt: Time step
            dx: Grid spacing
            
        Returns:
            Updated electric field
        """
        # Compute curl of magnetic field
        curl_H = self._compute_curl(H, dx)
        
        # Update electric field
        dE = (curl_H - J) * dt / self.materials[0].permittivity
        return E + dE
        
    def compute_magnetic_field(self,
                             H: np.ndarray,
                             E: np.ndarray,
                             dt: float,
                             dx: float) -> np.ndarray:
        """
        Update magnetic field using Maxwell's equations
        
        Args:
            H: Magnetic field
            E: Electric field
            J: Current density
            dt: Time step
            dx: Grid spacing
            
        Returns:
            Updated magnetic field
        """
        # Compute curl of electric field
        curl_E = self._compute_curl(E, dx)
        
        # Update magnetic field
        dH = -curl_E * dt / self.materials[0].permeability
        return H + dH
        
    def _compute_curl(self, field: np.ndarray, dx: float) -> np.ndarray:
        """Compute curl of vector field"""
        if field.shape[-1] == 3:  # 3D
            curl = np.zeros_like(field)
            curl[..., 0] = (np.gradient(field[..., 2], dx)[1] -
                           np.gradient(field[..., 1], dx)[2])
            curl[..., 1] = (np.gradient(field[..., 0], dx)[2] -
                           np.gradient(field[..., 2], dx)[0])
            curl[..., 2] = (np.gradient(field[..., 1], dx)[0] -
                           np.gradient(field[..., 0], dx)[1])
        else:  # 2D
            curl = np.zeros_like(field)
            curl[..., 0] = np.gradient(field[..., 1], dx)[0]
            curl[..., 1] = -np.gradient(field[..., 0], dx)[1]
            
        return curl
        
    def compute_current_density(self,
                              E: np.ndarray,
                              velocity: Optional[np.ndarray] = None,
                              B: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute current density
        
        Args:
            E: Electric field
            velocity: Fluid velocity (for MHD)
            B: Magnetic field (for MHD)
            
        Returns:
            Current density
        """
        # Ohmic current
        J = self.materials[0].conductivity * E
        
        # Add MHD effects if velocity and magnetic field are provided
        if velocity is not None and B is not None:
            # Compute Lorentz force
            v_cross_B = np.cross(velocity, B)
            J += self.materials[0].conductivity * v_cross_B
            
        return J
        
    def compute_lorentz_force(self,
                            J: np.ndarray,
                            B: np.ndarray) -> np.ndarray:
        """
        Compute Lorentz force
        
        Args:
            J: Current density
            B: Magnetic field
            
        Returns:
            Lorentz force
        """
        return np.cross(J, B)
        
    def compute_energy(self,
                      E: np.ndarray,
                      H: np.ndarray) -> float:
        """
        Compute electromagnetic energy
        
        Args:
            E: Electric field
            H: Magnetic field
            
        Returns:
            Total electromagnetic energy
        """
        # Electric energy
        W_e = 0.5 * self.materials[0].permittivity * np.sum(E**2)
        
        # Magnetic energy
        W_m = 0.5 * self.materials[0].permeability * np.sum(H**2)
        
        return W_e + W_m
        
    def solve_step(self,
                  E: np.ndarray,
                  H: np.ndarray,
                  velocity: Optional[np.ndarray] = None,
                  dt: float = 0.0,
                  dx: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one time step of electromagnetic simulation
        
        Args:
            E: Electric field
            H: Magnetic field
            velocity: Fluid velocity (for MHD)
            dt: Time step
            dx: Grid spacing
            
        Returns:
            Updated electric and magnetic fields
        """
        # Compute current density
        J = self.compute_current_density(E, velocity, H)
        
        # Update electric field
        E_new = self.compute_electric_field(E, H, J, dt, dx)
        
        # Update magnetic field
        H_new = self.compute_magnetic_field(H, E_new, dt, dx)
        
        return E_new, H_new
        
    def compute_boundary_conditions(self,
                                  field: np.ndarray,
                                  boundary_type: str) -> np.ndarray:
        """
        Apply boundary conditions
        
        Args:
            field: Field to apply boundary conditions to
            boundary_type: Type of boundary condition
            
        Returns:
            Field with boundary conditions applied
        """
        if boundary_type == "periodic":
            # Periodic boundary conditions
            field[0] = field[-2]
            field[-1] = field[1]
        elif boundary_type == "perfect_conductor":
            # Perfect conductor boundary conditions
            field[0] = 0
            field[-1] = 0
        elif boundary_type == "absorbing":
            # Absorbing boundary conditions (PML)
            # This is a simplified version
            alpha = 0.1
            field[0] *= np.exp(-alpha)
            field[-1] *= np.exp(-alpha)
            
        return field 