import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

class RheologyModel(Enum):
    """Types of rheological models"""
    POWER_LAW = "power_law"
    BINGHAM = "bingham"
    HERSHEL_BULKLEY = "hershel_bulkley"
    CROSS = "cross"
    CARREAU = "carreau"

@dataclass
class RheologyParameters:
    """Parameters for rheological models"""
    # Power law parameters
    k: float = 1.0  # Consistency index
    n: float = 1.0  # Power law index
    
    # Bingham parameters
    yield_stress: float = 0.0  # Yield stress
    plastic_viscosity: float = 1.0  # Plastic viscosity
    
    # Cross model parameters
    eta_0: float = 1.0  # Zero-shear viscosity
    eta_inf: float = 0.0  # Infinite-shear viscosity
    lambda_cross: float = 1.0  # Time constant
    m: float = 1.0  # Cross model index
    
    # Carreau model parameters
    lambda_carreau: float = 1.0  # Time constant
    a: float = 2.0  # Carreau model index

class NonNewtonianFlow:
    def __init__(self,
                 model: RheologyModel,
                 params: Optional[RheologyParameters] = None):
        """
        Initialize non-Newtonian flow solver
        
        Args:
            model: Rheological model
            params: Model parameters
        """
        self.model = model
        self.params = params or RheologyParameters()
        
    def compute_viscosity(self,
                         strain_rate: np.ndarray) -> np.ndarray:
        """
        Compute effective viscosity based on strain rate
        
        Args:
            strain_rate: Strain rate tensor
            
        Returns:
            Effective viscosity
        """
        # Compute second invariant of strain rate
        gamma_dot = self._compute_strain_rate_magnitude(strain_rate)
        
        if self.model == RheologyModel.POWER_LAW:
            return self._power_law_viscosity(gamma_dot)
        elif self.model == RheologyModel.BINGHAM:
            return self._bingham_viscosity(gamma_dot)
        elif self.model == RheologyModel.HERSHEL_BULKLEY:
            return self._hershel_bulkley_viscosity(gamma_dot)
        elif self.model == RheologyModel.CROSS:
            return self._cross_viscosity(gamma_dot)
        elif self.model == RheologyModel.CARREAU:
            return self._carreau_viscosity(gamma_dot)
        else:
            raise ValueError(f"Unknown rheological model: {self.model}")
            
    def _compute_strain_rate_magnitude(self, strain_rate: np.ndarray) -> np.ndarray:
        """Compute magnitude of strain rate tensor"""
        if strain_rate.shape[-1] == 3:  # 3D
            return np.sqrt(0.5 * np.sum(strain_rate**2, axis=(-2, -1)))
        else:  # 2D
            return np.sqrt(0.5 * np.sum(strain_rate**2, axis=-1))
            
    def _power_law_viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Compute power law viscosity"""
        return self.params.k * gamma_dot**(self.params.n - 1)
        
    def _bingham_viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Compute Bingham plastic viscosity"""
        # Add small number to avoid division by zero
        gamma_dot = np.maximum(gamma_dot, 1e-10)
        return self.params.plastic_viscosity + self.params.yield_stress / gamma_dot
        
    def _hershel_bulkley_viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Compute Hershel-Bulkley viscosity"""
        # Add small number to avoid division by zero
        gamma_dot = np.maximum(gamma_dot, 1e-10)
        return self.params.k * gamma_dot**(self.params.n - 1) + self.params.yield_stress / gamma_dot
        
    def _cross_viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Compute Cross model viscosity"""
        eta = (self.params.eta_0 - self.params.eta_inf) / (
            1 + (self.params.lambda_cross * gamma_dot)**self.params.m
        ) + self.params.eta_inf
        return eta
        
    def _carreau_viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Compute Carreau model viscosity"""
        eta = (self.params.eta_0 - self.params.eta_inf) * (
            1 + (self.params.lambda_carreau * gamma_dot)**2
        )**((self.params.a - 1) / 2) + self.params.eta_inf
        return eta
        
    def compute_stress(self,
                      strain_rate: np.ndarray) -> np.ndarray:
        """
        Compute stress tensor
        
        Args:
            strain_rate: Strain rate tensor
            
        Returns:
            Stress tensor
        """
        # Compute effective viscosity
        viscosity = self.compute_viscosity(strain_rate)
        
        # Compute stress tensor
        if strain_rate.shape[-1] == 3:  # 3D
            stress = np.zeros_like(strain_rate)
            for i in range(3):
                for j in range(3):
                    stress[..., i, j] = 2 * viscosity * strain_rate[..., i, j]
        else:  # 2D
            stress = 2 * viscosity[..., np.newaxis, np.newaxis] * strain_rate
            
        return stress
        
    def compute_yield_criterion(self,
                              stress: np.ndarray) -> np.ndarray:
        """
        Compute yield criterion
        
        Args:
            stress: Stress tensor
            
        Returns:
            Yield criterion (1 if yielded, 0 if not)
        """
        if self.model in [RheologyModel.BINGHAM, RheologyModel.HERSHEL_BULKLEY]:
            # Compute von Mises stress
            if stress.shape[-1] == 3:  # 3D
                von_mises = np.sqrt(0.5 * np.sum(stress**2, axis=(-2, -1)))
            else:  # 2D
                von_mises = np.sqrt(0.5 * np.sum(stress**2, axis=-1))
                
            # Check if material has yielded
            return (von_mises > self.params.yield_stress).astype(float)
        else:
            # No yield stress for other models
            return np.ones_like(stress[..., 0])
            
    def compute_power_dissipation(self,
                                stress: np.ndarray,
                                strain_rate: np.ndarray) -> np.ndarray:
        """
        Compute power dissipation
        
        Args:
            stress: Stress tensor
            strain_rate: Strain rate tensor
            
        Returns:
            Power dissipation
        """
        if stress.shape[-1] == 3:  # 3D
            return np.sum(stress * strain_rate, axis=(-2, -1))
        else:  # 2D
            return np.sum(stress * strain_rate, axis=-1)
            
    def compute_apparent_viscosity(self,
                                 velocity_grad: np.ndarray) -> np.ndarray:
        """
        Compute apparent viscosity from velocity gradient
        
        Args:
            velocity_grad: Velocity gradient tensor
            
        Returns:
            Apparent viscosity
        """
        # Compute strain rate tensor
        strain_rate = 0.5 * (velocity_grad + velocity_grad.transpose(0, 2, 1))
        
        # Compute effective viscosity
        return self.compute_viscosity(strain_rate) 