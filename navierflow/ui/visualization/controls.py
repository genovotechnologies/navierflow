import streamlit as st
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class BoundaryType(Enum):
    """Types of boundary conditions"""
    NO_SLIP = "No-Slip Wall"
    FREE_SLIP = "Free-Slip Wall"
    INFLOW = "Inflow"
    OUTFLOW = "Outflow"
    PERIODIC = "Periodic"
    SYMMETRY = "Symmetry"

@dataclass
class MaterialProperties:
    """Material properties for fluid simulation"""
    density: float = 1.0  # kg/m³
    viscosity: float = 1.0e-3  # Pa·s
    thermal_conductivity: float = 0.6  # W/(m·K)
    specific_heat: float = 4186.0  # J/(kg·K)
    thermal_expansion: float = 2.0e-4  # 1/K

class ControlPanel:
    """Control panel for simulation parameters and settings"""
    
    def __init__(self):
        self.initialize_state()

    def initialize_state(self):
        """Initialize control panel state"""
        if 'control_panel' not in st.session_state:
            st.session_state.control_panel = {
                'boundaries': {
                    'left': BoundaryType.NO_SLIP,
                    'right': BoundaryType.NO_SLIP,
                    'top': BoundaryType.NO_SLIP,
                    'bottom': BoundaryType.NO_SLIP
                },
                'inflow_velocity': 1.0,
                'inflow_temperature': 293.15,
                'material': MaterialProperties(),
                'solver': {
                    'dt': 0.01,
                    'cfl': 0.5,
                    'pressure_iterations': 50,
                    'tolerance': 1e-6
                },
                'physics': {
                    'gravity': True,
                    'buoyancy': False,
                    'surface_tension': False,
                    'turbulence_model': None
                }
            }

    def render_boundary_conditions(self):
        """Render boundary condition controls"""
        st.markdown("### Boundary Conditions")
        
        # Create columns for each boundary
        col1, col2 = st.columns(2)
        
        with col1:
            # Left boundary
            st.selectbox(
                "Left Boundary",
                [bc.value for bc in BoundaryType],
                index=list(BoundaryType).index(st.session_state.control_panel['boundaries']['left']),
                key='boundary_left',
                on_change=self._update_boundary_condition,
                args=('left',)
            )
            
            # Right boundary
            st.selectbox(
                "Right Boundary",
                [bc.value for bc in BoundaryType],
                index=list(BoundaryType).index(st.session_state.control_panel['boundaries']['right']),
                key='boundary_right',
                on_change=self._update_boundary_condition,
                args=('right',)
            )
        
        with col2:
            # Top boundary
            st.selectbox(
                "Top Boundary",
                [bc.value for bc in BoundaryType],
                index=list(BoundaryType).index(st.session_state.control_panel['boundaries']['top']),
                key='boundary_top',
                on_change=self._update_boundary_condition,
                args=('top',)
            )
            
            # Bottom boundary
            st.selectbox(
                "Bottom Boundary",
                [bc.value for bc in BoundaryType],
                index=list(BoundaryType).index(st.session_state.control_panel['boundaries']['bottom']),
                key='boundary_bottom',
                on_change=self._update_boundary_condition,
                args=('bottom',)
            )
        
        # Inflow conditions (if any boundary is INFLOW)
        if any(bc == BoundaryType.INFLOW for bc in st.session_state.control_panel['boundaries'].values()):
            st.markdown("#### Inflow Conditions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input(
                    "Inflow Velocity (m/s)",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.control_panel['inflow_velocity'],
                    step=0.1,
                    key='inflow_velocity',
                    on_change=self._update_inflow_conditions
                )
            
            with col2:
                st.number_input(
                    "Inflow Temperature (K)",
                    min_value=273.15,
                    max_value=373.15,
                    value=st.session_state.control_panel['inflow_temperature'],
                    step=0.1,
                    key='inflow_temperature',
                    on_change=self._update_inflow_conditions
                )

    def render_material_properties(self):
        """Render material property controls"""
        st.markdown("### Material Properties")
        
        material = st.session_state.control_panel['material']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input(
                "Density (kg/m³)",
                min_value=0.1,
                max_value=2000.0,
                value=material.density,
                step=0.1,
                key='material_density',
                on_change=self._update_material_properties
            )
            
            st.number_input(
                "Viscosity (Pa·s)",
                min_value=1e-6,
                max_value=1.0,
                value=material.viscosity,
                format="%.2e",
                key='material_viscosity',
                on_change=self._update_material_properties
            )
            
            st.number_input(
                "Thermal Conductivity (W/(m·K))",
                min_value=0.1,
                max_value=10.0,
                value=material.thermal_conductivity,
                step=0.1,
                key='material_thermal_conductivity',
                on_change=self._update_material_properties
            )
        
        with col2:
            st.number_input(
                "Specific Heat (J/(kg·K))",
                min_value=100.0,
                max_value=10000.0,
                value=material.specific_heat,
                step=100.0,
                key='material_specific_heat',
                on_change=self._update_material_properties
            )
            
            st.number_input(
                "Thermal Expansion (1/K)",
                min_value=0.0,
                max_value=1e-3,
                value=material.thermal_expansion,
                format="%.2e",
                key='material_thermal_expansion',
                on_change=self._update_material_properties
            )

    def render_solver_settings(self):
        """Render solver settings controls"""
        st.markdown("### Solver Settings")
        
        solver = st.session_state.control_panel['solver']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input(
                "Time Step (dt)",
                min_value=0.001,
                max_value=0.1,
                value=solver['dt'],
                format="%.3f",
                key='solver_dt',
                on_change=self._update_solver_settings
            )
            
            st.number_input(
                "CFL Number",
                min_value=0.1,
                max_value=1.0,
                value=solver['cfl'],
                step=0.1,
                key='solver_cfl',
                on_change=self._update_solver_settings
            )
        
        with col2:
            st.number_input(
                "Pressure Iterations",
                min_value=10,
                max_value=200,
                value=solver['pressure_iterations'],
                step=10,
                key='solver_pressure_iterations',
                on_change=self._update_solver_settings
            )
            
            st.number_input(
                "Convergence Tolerance",
                min_value=1e-8,
                max_value=1e-4,
                value=solver['tolerance'],
                format="%.2e",
                key='solver_tolerance',
                on_change=self._update_solver_settings
            )

    def render_physics_settings(self):
        """Render physics model settings"""
        st.markdown("### Physics Settings")
        
        physics = st.session_state.control_panel['physics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox(
                "Include Gravity",
                value=physics['gravity'],
                key='physics_gravity',
                on_change=self._update_physics_settings
            )
            
            st.checkbox(
                "Include Buoyancy",
                value=physics['buoyancy'],
                key='physics_buoyancy',
                on_change=self._update_physics_settings
            )
        
        with col2:
            st.checkbox(
                "Include Surface Tension",
                value=physics['surface_tension'],
                key='physics_surface_tension',
                on_change=self._update_physics_settings
            )
            
            st.selectbox(
                "Turbulence Model",
                ["None", "k-ε", "k-ω", "Smagorinsky"],
                index=0 if physics['turbulence_model'] is None else 
                     ["None", "k-ε", "k-ω", "Smagorinsky"].index(physics['turbulence_model']),
                key='physics_turbulence_model',
                on_change=self._update_physics_settings
            )

    def _update_boundary_condition(self, boundary: str):
        """Update boundary condition state"""
        st.session_state.control_panel['boundaries'][boundary] = BoundaryType(
            getattr(st.session_state, f'boundary_{boundary}')
        )

    def _update_inflow_conditions(self):
        """Update inflow condition state"""
        st.session_state.control_panel['inflow_velocity'] = st.session_state.inflow_velocity
        st.session_state.control_panel['inflow_temperature'] = st.session_state.inflow_temperature

    def _update_material_properties(self):
        """Update material properties state"""
        st.session_state.control_panel['material'] = MaterialProperties(
            density=st.session_state.material_density,
            viscosity=st.session_state.material_viscosity,
            thermal_conductivity=st.session_state.material_thermal_conductivity,
            specific_heat=st.session_state.material_specific_heat,
            thermal_expansion=st.session_state.material_thermal_expansion
        )

    def _update_solver_settings(self):
        """Update solver settings state"""
        st.session_state.control_panel['solver'] = {
            'dt': st.session_state.solver_dt,
            'cfl': st.session_state.solver_cfl,
            'pressure_iterations': st.session_state.solver_pressure_iterations,
            'tolerance': st.session_state.solver_tolerance
        }

    def _update_physics_settings(self):
        """Update physics settings state"""
        st.session_state.control_panel['physics'] = {
            'gravity': st.session_state.physics_gravity,
            'buoyancy': st.session_state.physics_buoyancy,
            'surface_tension': st.session_state.physics_surface_tension,
            'turbulence_model': None if st.session_state.physics_turbulence_model == "None"
                               else st.session_state.physics_turbulence_model
        }

    def get_state(self) -> Dict:
        """Get current control panel state"""
        return st.session_state.control_panel

    def apply_preset(self, preset: Dict):
        """Apply a preset configuration"""
        st.session_state.control_panel.update(preset)
        
        # Update all UI elements
        for boundary, value in preset['boundaries'].items():
            setattr(st.session_state, f'boundary_{boundary}', value.value)
        
        st.session_state.inflow_velocity = preset['inflow_velocity']
        st.session_state.inflow_temperature = preset['inflow_temperature']
        
        # Update material properties
        material = preset['material']
        st.session_state.material_density = material.density
        st.session_state.material_viscosity = material.viscosity
        st.session_state.material_thermal_conductivity = material.thermal_conductivity
        st.session_state.material_specific_heat = material.specific_heat
        st.session_state.material_thermal_expansion = material.thermal_expansion
        
        # Update solver settings
        solver = preset['solver']
        st.session_state.solver_dt = solver['dt']
        st.session_state.solver_cfl = solver['cfl']
        st.session_state.solver_pressure_iterations = solver['pressure_iterations']
        st.session_state.solver_tolerance = solver['tolerance']
        
        # Update physics settings
        physics = preset['physics']
        st.session_state.physics_gravity = physics['gravity']
        st.session_state.physics_buoyancy = physics['buoyancy']
        st.session_state.physics_surface_tension = physics['surface_tension']
        st.session_state.physics_turbulence_model = "None" if physics['turbulence_model'] is None \
                                                   else physics['turbulence_model'] 