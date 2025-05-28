import taichi as ti
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum

class TurbulenceModelType(Enum):
    RANS_SPALART_ALLMARAS = "spalart_allmaras"
    RANS_K_EPSILON = "k_epsilon"
    RANS_K_OMEGA = "k_omega"
    RANS_K_OMEGA_SST = "k_omega_sst"
    LES_SMAGORINSKY = "smagorinsky"
    LES_DYNAMIC_SMAGORINSKY = "dynamic_smagorinsky"
    LES_WALE = "wale"
    LES_SIGMA = "sigma"

@ti.data_oriented
class TurbulenceModel:
    """
    Advanced turbulence modeling with multiple RANS and LES models:
    - RANS models: Spalart-Allmaras, k-ε, k-ω, k-ω SST
    - LES models: Smagorinsky, Dynamic Smagorinsky, WALE, Sigma
    - Wall functions
    - Dynamic procedure for model coefficients
    - Hybrid RANS-LES capabilities
    """
    def __init__(self, width: int, height: int, config: Dict = None):
        self.width = width
        self.height = height
        
        # Default configuration
        self.config = {
            'model_type': TurbulenceModelType.LES_DYNAMIC_SMAGORINSKY,
            'enable_wall_functions': True,
            'enable_dynamic_procedure': True,
            'enable_hybrid_rans_les': False,
            'wall_model_type': 'equilibrium',
            'filter_type': 'box',
            'test_filter_ratio': 2.0,
            'model_constants': {
                'Cs': 0.17,  # Smagorinsky constant
                'Cw': 0.325,  # WALE constant
                'kappa': 0.41,  # von Karman constant
                'sigma_k': 1.0,  # k-ε/ω turbulent Schmidt number
                'C_mu': 0.09,  # k-ε model constant
                'C_epsilon1': 1.44,  # k-ε model constant
                'C_epsilon2': 1.92,  # k-ε model constant
                'beta_star': 0.09,  # k-ω model constant
                'alpha': 5.0/9.0,  # k-ω model constant
                'beta': 0.075,  # k-ω model constant
            }
        }
        if config:
            self.config.update(config)
            
        # RANS model fields
        if self._is_rans_model():
            # Spalart-Allmaras
            if self.config['model_type'] == TurbulenceModelType.RANS_SPALART_ALLMARAS:
                self.nu_tilde = ti.field(dtype=ti.f32, shape=(width, height))
                
            # k-ε model
            elif self.config['model_type'] == TurbulenceModelType.RANS_K_EPSILON:
                self.k = ti.field(dtype=ti.f32, shape=(width, height))
                self.epsilon = ti.field(dtype=ti.f32, shape=(width, height))
                
            # k-ω and k-ω SST models
            elif self.config['model_type'] in [TurbulenceModelType.RANS_K_OMEGA,
                                           TurbulenceModelType.RANS_K_OMEGA_SST]:
                self.k = ti.field(dtype=ti.f32, shape=(width, height))
                self.omega = ti.field(dtype=ti.f32, shape=(width, height))
                if self.config['model_type'] == TurbulenceModelType.RANS_K_OMEGA_SST:
                    self.f1 = ti.field(dtype=ti.f32, shape=(width, height))
                    self.f2 = ti.field(dtype=ti.f32, shape=(width, height))
                    
        # LES model fields
        else:
            self.nu_sgs = ti.field(dtype=ti.f32, shape=(width, height))
            if self.config['model_type'] == TurbulenceModelType.LES_DYNAMIC_SMAGORINSKY:
                self.Cs_field = ti.field(dtype=ti.f32, shape=(width, height))
                
            # Test-filtered quantities for dynamic procedure
            if self.config['enable_dynamic_procedure']:
                self.u_test = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))
                self.S_test = ti.Matrix.field(2, 2, dtype=ti.f32, shape=(width, height))
                
        # Common fields
        self.S = ti.Matrix.field(2, 2, dtype=ti.f32, shape=(width, height))
        self.nu_t = ti.field(dtype=ti.f32, shape=(width, height))
        
        # Wall model fields
        if self.config['enable_wall_functions']:
            self.y_plus = ti.field(dtype=ti.f32, shape=(width, height))
            self.u_tau = ti.field(dtype=ti.f32, shape=(width, height))
            
        self.initialize_fields()
        
    def _is_rans_model(self) -> bool:
        """Check if current model is RANS type"""
        return self.config['model_type'].value.startswith('rans')
        
    @ti.kernel
    def initialize_fields(self):
        """Initialize turbulence model fields"""
        for i, j in self.nu_t:
            self.nu_t[i, j] = 0.0
            
            if self._is_rans_model():
                if hasattr(self, 'nu_tilde'):
                    self.nu_tilde[i, j] = 1e-4
                if hasattr(self, 'k'):
                    self.k[i, j] = 1e-4
                if hasattr(self, 'epsilon'):
                    self.epsilon[i, j] = 1e-4
                if hasattr(self, 'omega'):
                    self.omega[i, j] = 1e-4
            else:
                self.nu_sgs[i, j] = 0.0
                if hasattr(self, 'Cs_field'):
                    self.Cs_field[i, j] = self.config['model_constants']['Cs']
                    
    @ti.kernel
    def compute_strain_rate(self, velocity: ti.template()):
        """Compute strain rate tensor"""
        for i, j in velocity:
            # Compute velocity gradients
            dudx = self.central_diff_x(velocity, i, j, 0)
            dudy = self.central_diff_y(velocity, i, j, 0)
            dvdx = self.central_diff_x(velocity, i, j, 1)
            dvdy = self.central_diff_y(velocity, i, j, 1)
            
            # Strain rate components
            self.S[i, j][0, 0] = dudx
            self.S[i, j][0, 1] = 0.5 * (dudy + dvdx)
            self.S[i, j][1, 0] = self.S[i, j][0, 1]
            self.S[i, j][1, 1] = dvdy
            
    @ti.kernel
    def update_eddy_viscosity(self):
        """Update turbulent/eddy viscosity based on model type"""
        if self._is_rans_model():
            self.update_rans_eddy_viscosity()
        else:
            self.update_les_eddy_viscosity()
            
    @ti.kernel
    def update_rans_eddy_viscosity(self):
        """Update RANS model eddy viscosity"""
        for i, j in self.nu_t:
            if self.config['model_type'] == TurbulenceModelType.RANS_SPALART_ALLMARAS:
                # Spalart-Allmaras model
                chi = self.nu_tilde[i, j] / self.config['nu']
                fv1 = chi**3 / (chi**3 + 7.1**3)
                self.nu_t[i, j] = self.nu_tilde[i, j] * fv1
                
            elif self.config['model_type'] == TurbulenceModelType.RANS_K_EPSILON:
                # k-ε model
                C_mu = self.config['model_constants']['C_mu']
                self.nu_t[i, j] = C_mu * self.k[i, j]**2 / self.epsilon[i, j]
                
            elif self.config['model_type'] in [TurbulenceModelType.RANS_K_OMEGA,
                                           TurbulenceModelType.RANS_K_OMEGA_SST]:
                # k-ω and k-ω SST models
                if self.config['model_type'] == TurbulenceModelType.RANS_K_OMEGA:
                    self.nu_t[i, j] = self.k[i, j] / self.omega[i, j]
                else:
                    # k-ω SST blending
                    a1 = 0.31
                    F2 = self.f2[i, j]
                    S = ti.sqrt(2.0 * self.contract_tensor(self.S[i, j], self.S[i, j]))
                    self.nu_t[i, j] = (a1 * self.k[i, j]) / \
                                   ti.max(a1 * self.omega[i, j], F2 * S)
                                   
    @ti.kernel
    def update_les_eddy_viscosity(self):
        """Update LES model eddy viscosity"""
        delta = 1.0  # Grid spacing (assumed uniform for simplicity)
        
        for i, j in self.nu_t:
            if self.config['model_type'] == TurbulenceModelType.LES_SMAGORINSKY:
                # Standard Smagorinsky model
                Cs = self.config['model_constants']['Cs']
                S_mag = ti.sqrt(2.0 * self.contract_tensor(self.S[i, j], self.S[i, j]))
                self.nu_sgs[i, j] = (Cs * delta)**2 * S_mag
                
            elif self.config['model_type'] == TurbulenceModelType.LES_DYNAMIC_SMAGORINSKY:
                # Dynamic Smagorinsky model
                Cs = self.Cs_field[i, j]
                S_mag = ti.sqrt(2.0 * self.contract_tensor(self.S[i, j], self.S[i, j]))
                self.nu_sgs[i, j] = (Cs * delta)**2 * S_mag
                
            elif self.config['model_type'] == TurbulenceModelType.LES_WALE:
                # WALE model
                Cw = self.config['model_constants']['Cw']
                S_mag = ti.sqrt(2.0 * self.contract_tensor(self.S[i, j], self.S[i, j]))
                Sd = self.compute_wale_tensor(self.S[i, j])
                Sd_mag = ti.sqrt(self.contract_tensor(Sd, Sd))
                self.nu_sgs[i, j] = (Cw * delta)**2 * \
                                 Sd_mag**3 / (S_mag**5 + Sd_mag**2.5)
                                 
            elif self.config['model_type'] == TurbulenceModelType.LES_SIGMA:
                # Sigma model
                D = self.compute_velocity_gradient_tensor(i, j)
                sigma = self.compute_sigma_model_coefficient(D)
                self.nu_sgs[i, j] = (0.165 * delta)**2 * sigma
                
            # Apply wall damping if enabled
            if self.config['enable_wall_functions']:
                self.nu_sgs[i, j] *= self.compute_van_driest_damping(i, j)
                
            self.nu_t[i, j] = self.nu_sgs[i, j]
            
    @ti.kernel
    def update_dynamic_coefficient(self, velocity: ti.template()):
        """Update dynamic model coefficient using Germano identity"""
        if not self.config['enable_dynamic_procedure']:
            return
            
        # Apply test filter to velocity and compute test-filtered quantities
        self.apply_test_filter(velocity)
        self.compute_test_filtered_quantities()
        
        # Compute Germano identity terms
        alpha = self.config['test_filter_ratio']
        delta = 1.0
        
        for i, j in self.Cs_field:
            L_ij = self.compute_leonard_tensor(i, j)
            M_ij = -2.0 * delta**2 * (alpha**2 * self.contract_tensor(
                self.S_test[i, j], self.S_test[i, j]
            )**0.5 * self.S_test[i, j] - \
            self.apply_test_filter_to_tensor(
                self.contract_tensor(self.S[i, j], self.S[i, j])**0.5 * self.S[i, j]
            ))
            
            # Compute Cs using least squares
            LM = self.contract_tensor(L_ij, M_ij)
            MM = self.contract_tensor(M_ij, M_ij)
            
            if MM > 1e-10:
                self.Cs_field[i, j] = 0.5 * LM / MM
            else:
                self.Cs_field[i, j] = 0.0
                
            # Clip to avoid negative values
            self.Cs_field[i, j] = ti.max(self.Cs_field[i, j], 0.0)
            
    @ti.func
    def compute_wale_tensor(self, S: ti.template()) -> ti.Matrix:
        """Compute WALE model tensor"""
        g = ti.Matrix.zero(ti.f32, 2, 2)
        S2 = S @ S
        
        # Compute velocity gradient tensor
        g[0, 0] = S2[0, 0]
        g[0, 1] = S2[0, 1]
        g[1, 0] = S2[1, 0]
        g[1, 1] = S2[1, 1]
        
        # Compute traceless symmetric part
        trace = g[0, 0] + g[1, 1]
        g[0, 0] -= trace / 3.0
        g[1, 1] -= trace / 3.0
        
        return g
        
    @ti.func
    def compute_sigma_model_coefficient(self, D: ti.template()) -> ti.f32:
        """Compute sigma model coefficient from velocity gradient tensor"""
        # Compute eigenvalues using characteristic equation
        trace = D[0, 0] + D[1, 1]
        det = D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0]
        
        # Solve quadratic equation
        p = -trace
        q = det
        
        # Get eigenvalues
        disc = p * p - 4.0 * q
        if disc < 0:
            disc = 0.0
            
        sqrt_disc = ti.sqrt(disc)
        lambda1 = (-p + sqrt_disc) / 2.0
        lambda2 = (-p - sqrt_disc) / 2.0
        
        # Sort eigenvalues
        if lambda2 > lambda1:
            temp = lambda1
            lambda1 = lambda2
            lambda2 = temp
            
        # Compute sigma coefficient
        if abs(lambda1) < 1e-10:
            return 0.0
            
        return lambda2 * (lambda1 - lambda2) * (lambda1 + lambda2) / lambda1**3
        
    @ti.func
    def compute_van_driest_damping(self, i: int, j: int) -> ti.f32:
        """Compute van Driest damping function"""
        if not self.config['enable_wall_functions']:
            return 1.0
            
        y_plus = self.y_plus[i, j]
        A_plus = 26.0
        
        return (1.0 - ti.exp(-y_plus / A_plus))**2
        
    @ti.func
    def contract_tensor(self, A: ti.template(), B: ti.template()) -> ti.f32:
        """Compute tensor contraction A:B"""
        return A[0, 0] * B[0, 0] + A[0, 1] * B[0, 1] + \
               A[1, 0] * B[1, 0] + A[1, 1] * B[1, 1]
               
    @ti.func
    def central_diff_x(self, field: ti.template(), i: int, j: int,
                      component: int) -> ti.f32:
        """Central difference in x-direction"""
        if i == 0:
            return (field[i+1, j][component] - field[i, j][component])
        elif i == self.width - 1:
            return (field[i, j][component] - field[i-1, j][component])
        else:
            return 0.5 * (field[i+1, j][component] - field[i-1, j][component])
            
    @ti.func
    def central_diff_y(self, field: ti.template(), i: int, j: int,
                      component: int) -> ti.f32:
        """Central difference in y-direction"""
        if j == 0:
            return (field[i, j+1][component] - field[i, j][component])
        elif j == self.height - 1:
            return (field[i, j][component] - field[i, j-1][component])
        else:
            return 0.5 * (field[i, j+1][component] - field[i, j-1][component])
            
    def get_eddy_viscosity(self) -> np.ndarray:
        """Return eddy viscosity field"""
        return self.nu_t.to_numpy()
        
    def get_turbulent_quantities(self) -> Dict[str, np.ndarray]:
        """Return relevant turbulent quantities based on model type"""
        result = {'nu_t': self.nu_t.to_numpy()}
        
        if self._is_rans_model():
            if hasattr(self, 'k'):
                result['k'] = self.k.to_numpy()
            if hasattr(self, 'epsilon'):
                result['epsilon'] = self.epsilon.to_numpy()
            if hasattr(self, 'omega'):
                result['omega'] = self.omega.to_numpy()
        else:
            result['nu_sgs'] = self.nu_sgs.to_numpy()
            if hasattr(self, 'Cs_field'):
                result['Cs'] = self.Cs_field.to_numpy()
                
        return result 