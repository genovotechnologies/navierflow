"""
Simulation presets for NavierFlow
"""
from dataclasses import dataclass
from typing import Dict, Any, List
from enum import Enum
import json
from pathlib import Path


class PresetCategory(Enum):
    """Preset categories"""
    EDUCATIONAL = "educational"
    RESEARCH = "research"
    INDUSTRIAL = "industrial"
    BENCHMARK = "benchmark"


@dataclass
class SimulationPreset:
    """Simulation preset configuration"""
    name: str
    description: str
    category: PresetCategory
    physics_model: str
    numerical_method: str
    parameters: Dict[str, Any]
    boundary_conditions: Dict[str, Any]
    initial_conditions: Dict[str, Any]
    visualization: Dict[str, Any]
    tags: List[str]


class PresetManager:
    """Manage simulation presets"""
    
    def __init__(self):
        self.presets: Dict[str, SimulationPreset] = {}
        self._initialize_presets()
    
    def _initialize_presets(self):
        """Initialize built-in presets"""
        
        # Educational presets
        self.add_preset(SimulationPreset(
            name="Lid-Driven Cavity",
            description="Classic CFD benchmark problem. Flow in a square cavity with moving top wall.",
            category=PresetCategory.EDUCATIONAL,
            physics_model="incompressible",
            numerical_method="finite_volume",
            parameters={
                "viscosity": 0.01,
                "density": 1.0,
                "time_step": 0.001,
                "max_steps": 10000,
                "lid_velocity": 1.0,
                "grid_resolution": (64, 64, 1)
            },
            boundary_conditions={
                "top": {"type": "velocity", "value": (1.0, 0.0, 0.0)},
                "bottom": {"type": "no_slip", "value": (0.0, 0.0, 0.0)},
                "left": {"type": "no_slip", "value": (0.0, 0.0, 0.0)},
                "right": {"type": "no_slip", "value": (0.0, 0.0, 0.0)}
            },
            initial_conditions={
                "velocity": (0.0, 0.0, 0.0),
                "pressure": 0.0
            },
            visualization={
                "mode": "streamline",
                "colormap": "viridis",
                "show_vectors": True
            },
            tags=["2d", "educational", "benchmark", "laminar"]
        ))
        
        self.add_preset(SimulationPreset(
            name="Flow Around Cylinder",
            description="Flow around a circular cylinder demonstrating vortex shedding.",
            category=PresetCategory.EDUCATIONAL,
            physics_model="incompressible",
            numerical_method="lattice_boltzmann",
            parameters={
                "viscosity": 0.01,
                "density": 1.0,
                "time_step": 0.001,
                "inlet_velocity": 1.0,
                "cylinder_radius": 0.1,
                "reynolds_number": 100,
                "grid_resolution": (128, 64, 1)
            },
            boundary_conditions={
                "inlet": {"type": "velocity", "value": (1.0, 0.0, 0.0)},
                "outlet": {"type": "outflow", "value": None},
                "top": {"type": "free_slip", "value": None},
                "bottom": {"type": "free_slip", "value": None},
                "cylinder": {"type": "no_slip", "value": (0.0, 0.0, 0.0)}
            },
            initial_conditions={
                "velocity": (1.0, 0.0, 0.0),
                "pressure": 0.0
            },
            visualization={
                "mode": "surface",
                "colormap": "coolwarm",
                "show_vectors": False
            },
            tags=["2d", "educational", "vortex", "unsteady"]
        ))
        
        self.add_preset(SimulationPreset(
            name="Thermal Convection",
            description="Natural convection in a heated cavity. Demonstrates buoyancy-driven flow.",
            category=PresetCategory.EDUCATIONAL,
            physics_model="heat_transfer",
            numerical_method="finite_volume",
            parameters={
                "viscosity": 0.01,
                "density": 1.0,
                "thermal_diffusivity": 0.01,
                "thermal_expansion": 0.001,
                "gravity": (0.0, -9.81, 0.0),
                "time_step": 0.001,
                "hot_temp": 100.0,
                "cold_temp": 0.0,
                "rayleigh_number": 10000,
                "grid_resolution": (64, 64, 1)
            },
            boundary_conditions={
                "left": {"type": "temperature", "value": 100.0},
                "right": {"type": "temperature", "value": 0.0},
                "top": {"type": "adiabatic", "value": None},
                "bottom": {"type": "adiabatic", "value": None}
            },
            initial_conditions={
                "velocity": (0.0, 0.0, 0.0),
                "pressure": 0.0,
                "temperature": 50.0
            },
            visualization={
                "mode": "surface",
                "colormap": "hot",
                "show_vectors": True
            },
            tags=["2d", "educational", "thermal", "natural_convection"]
        ))
        
        # Research presets
        self.add_preset(SimulationPreset(
            name="Turbulent Channel Flow",
            description="High-Reynolds-number turbulent flow in a channel.",
            category=PresetCategory.RESEARCH,
            physics_model="turbulence",
            numerical_method="finite_volume",
            parameters={
                "viscosity": 0.001,
                "density": 1.0,
                "time_step": 0.0001,
                "max_steps": 100000,
                "reynolds_number": 5000,
                "turbulence_model": "k_epsilon",
                "grid_resolution": (128, 128, 64)
            },
            boundary_conditions={
                "inlet": {"type": "velocity", "value": (1.0, 0.0, 0.0)},
                "outlet": {"type": "outflow", "value": None},
                "top": {"type": "no_slip", "value": (0.0, 0.0, 0.0)},
                "bottom": {"type": "no_slip", "value": (0.0, 0.0, 0.0)}
            },
            initial_conditions={
                "velocity": (1.0, 0.0, 0.0),
                "pressure": 0.0,
                "turbulent_kinetic_energy": 0.001,
                "turbulent_dissipation": 0.0001
            },
            visualization={
                "mode": "volume",
                "colormap": "viridis",
                "show_vectors": False
            },
            tags=["3d", "research", "turbulence", "high_re"]
        ))
        
        self.add_preset(SimulationPreset(
            name="Multiphase Flow",
            description="Two-phase flow simulation with phase interface tracking.",
            category=PresetCategory.RESEARCH,
            physics_model="multiphase",
            numerical_method="lattice_boltzmann",
            parameters={
                "viscosity_phase1": 0.01,
                "viscosity_phase2": 0.001,
                "density_phase1": 1.0,
                "density_phase2": 0.1,
                "surface_tension": 0.1,
                "time_step": 0.001,
                "grid_resolution": (128, 128, 128)
            },
            boundary_conditions={
                "all": {"type": "periodic", "value": None}
            },
            initial_conditions={
                "velocity": (0.0, 0.0, 0.0),
                "pressure": 0.0,
                "phase_fraction": 0.5
            },
            visualization={
                "mode": "isosurface",
                "colormap": "coolwarm",
                "show_interface": True
            },
            tags=["3d", "research", "multiphase", "interface"]
        ))
        
        self.add_preset(SimulationPreset(
            name="Non-Newtonian Flow",
            description="Shear-thinning fluid flow in a pipe.",
            category=PresetCategory.RESEARCH,
            physics_model="non_newtonian",
            numerical_method="finite_element",
            parameters={
                "power_law_index": 0.5,
                "consistency_index": 0.1,
                "density": 1.0,
                "time_step": 0.001,
                "inlet_velocity": 1.0,
                "grid_resolution": (128, 32, 32)
            },
            boundary_conditions={
                "inlet": {"type": "velocity", "value": (1.0, 0.0, 0.0)},
                "outlet": {"type": "outflow", "value": None},
                "wall": {"type": "no_slip", "value": (0.0, 0.0, 0.0)}
            },
            initial_conditions={
                "velocity": (0.5, 0.0, 0.0),
                "pressure": 0.0
            },
            visualization={
                "mode": "streamline",
                "colormap": "plasma",
                "show_vectors": True
            },
            tags=["3d", "research", "non_newtonian", "rheology"]
        ))
        
        # Industrial presets
        self.add_preset(SimulationPreset(
            name="Pipe Flow",
            description="Fully developed flow in a circular pipe.",
            category=PresetCategory.INDUSTRIAL,
            physics_model="incompressible",
            numerical_method="finite_volume",
            parameters={
                "viscosity": 0.001,
                "density": 1000.0,
                "time_step": 0.001,
                "inlet_velocity": 2.0,
                "pipe_diameter": 0.1,
                "pipe_length": 1.0,
                "grid_resolution": (256, 32, 32)
            },
            boundary_conditions={
                "inlet": {"type": "velocity", "value": (2.0, 0.0, 0.0)},
                "outlet": {"type": "outflow", "value": None},
                "wall": {"type": "no_slip", "value": (0.0, 0.0, 0.0)}
            },
            initial_conditions={
                "velocity": (1.0, 0.0, 0.0),
                "pressure": 0.0
            },
            visualization={
                "mode": "surface",
                "colormap": "jet",
                "show_vectors": False
            },
            tags=["3d", "industrial", "pipe", "internal_flow"]
        ))
        
        self.add_preset(SimulationPreset(
            name="Heat Exchanger",
            description="Counter-flow heat exchanger simulation.",
            category=PresetCategory.INDUSTRIAL,
            physics_model="heat_transfer",
            numerical_method="finite_volume",
            parameters={
                "viscosity": 0.001,
                "density": 1000.0,
                "thermal_diffusivity": 0.0001,
                "time_step": 0.001,
                "inlet_temp_hot": 100.0,
                "inlet_temp_cold": 20.0,
                "grid_resolution": (128, 64, 64)
            },
            boundary_conditions={
                "hot_inlet": {"type": "velocity_temperature", "value": (1.0, 100.0)},
                "cold_inlet": {"type": "velocity_temperature", "value": (1.0, 20.0)},
                "hot_outlet": {"type": "outflow", "value": None},
                "cold_outlet": {"type": "outflow", "value": None},
                "walls": {"type": "coupled_heat_transfer", "value": None}
            },
            initial_conditions={
                "velocity": (0.5, 0.0, 0.0),
                "pressure": 0.0,
                "temperature": 60.0
            },
            visualization={
                "mode": "volume",
                "colormap": "hot",
                "show_vectors": True
            },
            tags=["3d", "industrial", "heat_transfer", "heat_exchanger"]
        ))
        
        # Benchmark presets
        self.add_preset(SimulationPreset(
            name="Taylor-Green Vortex",
            description="Standard benchmark for temporal accuracy.",
            category=PresetCategory.BENCHMARK,
            physics_model="incompressible",
            numerical_method="finite_difference",
            parameters={
                "viscosity": 0.01,
                "density": 1.0,
                "time_step": 0.001,
                "max_steps": 10000,
                "grid_resolution": (128, 128, 1)
            },
            boundary_conditions={
                "all": {"type": "periodic", "value": None}
            },
            initial_conditions={
                "velocity": "taylor_green",  # Special initialization
                "pressure": "taylor_green"
            },
            visualization={
                "mode": "streamline",
                "colormap": "viridis",
                "show_vectors": True
            },
            tags=["2d", "benchmark", "temporal_accuracy", "periodic"]
        ))
        
        self.add_preset(SimulationPreset(
            name="Shock Tube",
            description="Sod's shock tube problem for compressible flow.",
            category=PresetCategory.BENCHMARK,
            physics_model="compressible",
            numerical_method="finite_volume",
            parameters={
                "gamma": 1.4,
                "time_step": 0.0001,
                "max_steps": 5000,
                "left_density": 1.0,
                "left_pressure": 1.0,
                "right_density": 0.125,
                "right_pressure": 0.1,
                "grid_resolution": (512, 1, 1)
            },
            boundary_conditions={
                "left": {"type": "reflective", "value": None},
                "right": {"type": "reflective", "value": None}
            },
            initial_conditions={
                "density": "shock_tube",
                "velocity": (0.0, 0.0, 0.0),
                "pressure": "shock_tube"
            },
            visualization={
                "mode": "surface",
                "colormap": "plasma",
                "show_shock": True
            },
            tags=["1d", "benchmark", "compressible", "shock"]
        ))
    
    def add_preset(self, preset: SimulationPreset):
        """Add a preset"""
        self.presets[preset.name] = preset
    
    def get_preset(self, name: str) -> SimulationPreset:
        """Get a preset by name"""
        return self.presets.get(name)
    
    def get_presets_by_category(self, category: PresetCategory) -> List[SimulationPreset]:
        """Get all presets in a category"""
        return [p for p in self.presets.values() if p.category == category]
    
    def get_presets_by_tags(self, tags: List[str]) -> List[SimulationPreset]:
        """Get presets matching any of the given tags"""
        return [p for p in self.presets.values() if any(tag in p.tags for tag in tags)]
    
    def get_all_presets(self) -> List[SimulationPreset]:
        """Get all presets"""
        return list(self.presets.values())
    
    def search_presets(self, query: str) -> List[SimulationPreset]:
        """Search presets by name or description"""
        query = query.lower()
        return [
            p for p in self.presets.values()
            if query in p.name.lower() or query in p.description.lower()
        ]
    
    def export_preset(self, name: str, filepath: str):
        """Export a preset to JSON file"""
        preset = self.get_preset(name)
        if preset:
            data = {
                'name': preset.name,
                'description': preset.description,
                'category': preset.category.value,
                'physics_model': preset.physics_model,
                'numerical_method': preset.numerical_method,
                'parameters': preset.parameters,
                'boundary_conditions': preset.boundary_conditions,
                'initial_conditions': preset.initial_conditions,
                'visualization': preset.visualization,
                'tags': preset.tags
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
    
    def import_preset(self, filepath: str) -> SimulationPreset:
        """Import a preset from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        preset = SimulationPreset(
            name=data['name'],
            description=data['description'],
            category=PresetCategory(data['category']),
            physics_model=data['physics_model'],
            numerical_method=data['numerical_method'],
            parameters=data['parameters'],
            boundary_conditions=data['boundary_conditions'],
            initial_conditions=data['initial_conditions'],
            visualization=data['visualization'],
            tags=data['tags']
        )
        
        self.add_preset(preset)
        return preset
