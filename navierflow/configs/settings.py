import yaml
import json
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import os

class ConfigFormat(Enum):
    """Configuration file formats"""
    YAML = "yaml"
    JSON = "json"

@dataclass
class SimulationConfig:
    """Simulation configuration"""
    # Physics parameters
    reynolds_number: float = 1000.0
    mach_number: float = 0.1
    prandtl_number: float = 0.71
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    
    # Numerical parameters
    time_step: float = 0.001
    max_steps: int = 1000
    cfl_number: float = 0.5
    tolerance: float = 1e-6
    
    # Mesh parameters
    mesh_type: str = "unstructured"
    mesh_resolution: float = 0.1
    refinement_level: int = 0
    
    # Boundary conditions
    boundary_conditions: Dict[str, str] = None
    
    # Output parameters
    output_dir: str = "results"
    output_frequency: int = 100
    output_format: str = "vtk"
    
    # Parallel computing
    num_processes: int = 1
    num_threads: int = 4
    use_gpu: bool = True
    
    # Visualization
    visualization_type: str = "surface"
    visualization_frequency: int = 10
    
    def __post_init__(self):
        """Initialize default values"""
        if self.boundary_conditions is None:
            self.boundary_conditions = {
                "x_min": "wall",
                "x_max": "outlet",
                "y_min": "wall",
                "y_max": "inlet",
                "z_min": "wall",
                "z_max": "wall"
            }

class ConfigManager:
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize configuration manager
        
        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        
    def save(self,
             filename: str,
             format: ConfigFormat = ConfigFormat.YAML):
        """
        Save configuration to file
        
        Args:
            filename: Output filename
            format: File format
        """
        # Convert configuration to dictionary
        config_dict = self._config_to_dict()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save configuration
        if format == ConfigFormat.YAML:
            with open(filename, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif format == ConfigFormat.JSON:
            with open(filename, "w") as f:
                json.dump(config_dict, f, indent=4)
        else:
            raise ValueError(f"Unknown format: {format}")
            
    def load(self,
             filename: str,
             format: ConfigFormat = ConfigFormat.YAML):
        """
        Load configuration from file
        
        Args:
            filename: Input filename
            format: File format
        """
        # Load configuration
        if format == ConfigFormat.YAML:
            with open(filename, "r") as f:
                config_dict = yaml.safe_load(f)
        elif format == ConfigFormat.JSON:
            with open(filename, "r") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unknown format: {format}")
            
        # Update configuration
        self._update_config(config_dict)
        
    def _config_to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        config_dict = {}
        for field in self.config.__dataclass_fields__:
            value = getattr(self.config, field)
            if isinstance(value, tuple):
                value = list(value)
            config_dict[field] = value
        return config_dict
        
    def _update_config(self, config_dict: Dict):
        """Update configuration from dictionary"""
        for field, value in config_dict.items():
            if field in self.config.__dataclass_fields__:
                if isinstance(getattr(self.config, field), tuple):
                    value = tuple(value)
                setattr(self.config, field, value)
                
    def validate(self) -> List[str]:
        """
        Validate configuration
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check physics parameters
        if self.config.reynolds_number <= 0:
            errors.append("Reynolds number must be positive")
        if self.config.mach_number <= 0:
            errors.append("Mach number must be positive")
        if self.config.prandtl_number <= 0:
            errors.append("Prandtl number must be positive")
            
        # Check numerical parameters
        if self.config.time_step <= 0:
            errors.append("Time step must be positive")
        if self.config.max_steps <= 0:
            errors.append("Maximum steps must be positive")
        if self.config.cfl_number <= 0:
            errors.append("CFL number must be positive")
        if self.config.tolerance <= 0:
            errors.append("Tolerance must be positive")
            
        # Check mesh parameters
        if self.config.mesh_resolution <= 0:
            errors.append("Mesh resolution must be positive")
        if self.config.refinement_level < 0:
            errors.append("Refinement level must be non-negative")
            
        # Check output parameters
        if self.config.output_frequency <= 0:
            errors.append("Output frequency must be positive")
            
        # Check parallel computing parameters
        if self.config.num_processes <= 0:
            errors.append("Number of processes must be positive")
        if self.config.num_threads <= 0:
            errors.append("Number of threads must be positive")
            
        # Check visualization parameters
        if self.config.visualization_frequency <= 0:
            errors.append("Visualization frequency must be positive")
            
        return errors
        
    def get_boundary_condition(self, boundary: str) -> str:
        """
        Get boundary condition
        
        Args:
            boundary: Boundary name
            
        Returns:
            Boundary condition
        """
        return self.config.boundary_conditions.get(boundary, "wall")
        
    def set_boundary_condition(self, boundary: str, condition: str):
        """
        Set boundary condition
        
        Args:
            boundary: Boundary name
            condition: Boundary condition
        """
        self.config.boundary_conditions[boundary] = condition
        
    def get_output_path(self, step: int) -> str:
        """
        Get output path for step
        
        Args:
            step: Simulation step
            
        Returns:
            Output path
        """
        return os.path.join(
            self.config.output_dir,
            f"step_{step:06d}.{self.config.output_format}"
        )
        
    def get_visualization_path(self, step: int) -> str:
        """
        Get visualization path for step
        
        Args:
            step: Simulation step
            
        Returns:
            Visualization path
        """
        return os.path.join(
            self.config.output_dir,
            "visualization",
            f"step_{step:06d}.png"
        )
        
    def get_checkpoint_path(self, step: int) -> str:
        """
        Get checkpoint path for step
        
        Args:
            step: Simulation step
            
        Returns:
            Checkpoint path
        """
        return os.path.join(
            self.config.output_dir,
            "checkpoints",
            f"step_{step:06d}.h5"
        )
        
    def get_log_path(self) -> str:
        """
        Get log path
        
        Returns:
            Log path
        """
        return os.path.join(self.config.output_dir, "simulation.log")
        
    def get_config_path(self) -> str:
        """
        Get configuration path
        
        Returns:
            Configuration path
        """
        return os.path.join(self.config.output_dir, "config.yaml")
        
    def create_output_dirs(self):
        """Create output directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "visualization"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "checkpoints"), exist_ok=True)
        
    def cleanup(self):
        """Cleanup output directories"""
        import shutil
        shutil.rmtree(self.config.output_dir, ignore_errors=True) 