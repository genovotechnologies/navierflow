import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class SimulationPreset:
    name: str
    description: str
    method: str
    parameters: Dict[str, Any]
    visualization: Dict[str, Any]
    tags: List[str]
    created: str
    last_modified: str
    author: str
    category: str
    version: str

class PresetManager:
    def __init__(self, preset_dir: str = "presets"):
        self.preset_dir = preset_dir
        self.default_presets = {
            "educational": {
                "basic_fluid": {
                    "name": "Basic Fluid Dynamics",
                    "description": "Simple fluid simulation for learning basics",
                    "method": "eulerian",
                    "parameters": {
                        "viscosity": 0.1,
                        "time_step": 0.05,
                        "iterations": 50
                    },
                    "visualization": {
                        "color_scheme": "blue",
                        "show_vectors": True,
                        "show_pressure": False
                    },
                    "tags": ["educational", "beginner", "basic"],
                    "category": "educational"
                },
                "vortex_study": {
                    "name": "Vortex Formation Study",
                    "description": "Demonstration of vortex formation and behavior",
                    "method": "eulerian",
                    "parameters": {
                        "viscosity": 0.05,
                        "time_step": 0.03,
                        "iterations": 75
                    },
                    "visualization": {
                        "color_scheme": "rainbow",
                        "show_vectors": True,
                        "show_pressure": True
                    },
                    "tags": ["educational", "intermediate", "vortex"],
                    "category": "educational"
                }
            },
            "research": {
                "high_accuracy": {
                    "name": "High Accuracy Simulation",
                    "description": "Research-grade simulation with high accuracy",
                    "method": "lbm",
                    "parameters": {
                        "viscosity": 0.01,
                        "time_step": 0.01,
                        "iterations": 200,
                        "grid_resolution": 512
                    },
                    "visualization": {
                        "color_scheme": "grayscale",
                        "show_vectors": True,
                        "show_pressure": True,
                        "show_analytics": True
                    },
                    "tags": ["research", "high-accuracy", "advanced"],
                    "category": "research"
                },
                "turbulence_study": {
                    "name": "Turbulence Analysis",
                    "description": "Focused study of turbulent flow patterns",
                    "method": "lbm",
                    "parameters": {
                        "viscosity": 0.001,
                        "time_step": 0.005,
                        "iterations": 300,
                        "grid_resolution": 1024
                    },
                    "visualization": {
                        "color_scheme": "fire",
                        "show_vectors": True,
                        "show_pressure": True,
                        "show_analytics": True
                    },
                    "tags": ["research", "turbulence", "advanced"],
                    "category": "research"
                }
            }
        }
        self._ensure_preset_directory()
        self._initialize_default_presets()

    def _ensure_preset_directory(self):
        """Create preset directory if it doesn't exist"""
        if not os.path.exists(self.preset_dir):
            os.makedirs(self.preset_dir)
            os.makedirs(os.path.join(self.preset_dir, "educational"))
            os.makedirs(os.path.join(self.preset_dir, "research"))
            os.makedirs(os.path.join(self.preset_dir, "custom"))

    def _initialize_default_presets(self):
        """Initialize default presets if they don't exist"""
        for category, presets in self.default_presets.items():
            for preset_name, preset_data in presets.items():
                preset_data.update({
                    "created": datetime.now().isoformat(),
                    "last_modified": datetime.now().isoformat(),
                    "author": "NavierFlow Team",
                    "version": "1.0.0"
                })
                self.save_preset(SimulationPreset(**preset_data))

    def load_preset(self, name: str, category: str = "educational") -> Optional[SimulationPreset]:
        """Load a preset by name and category"""
        try:
            file_path = os.path.join(self.preset_dir, category, f"{name}.json")
            with open(file_path, 'r') as f:
                data = json.load(f)
                return SimulationPreset(**data)
        except FileNotFoundError:
            print(f"Preset {name} not found in category {category}")
            return None
        except json.JSONDecodeError:
            print(f"Error reading preset {name}")
            return None

    def save_preset(self, preset: SimulationPreset) -> bool:
        """Save a preset to file"""
        try:
            file_path = os.path.join(self.preset_dir, preset.category, f"{preset.name}.json")
            with open(file_path, 'w') as f:
                json.dump(preset.__dict__, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving preset: {e}")
            return False

    def list_presets(self, category: Optional[str] = None) -> List[str]:
        """List all available presets, optionally filtered by category"""
        presets = []
        categories = [category] if category else ["educational", "research", "custom"]
        
        for cat in categories:
            cat_dir = os.path.join(self.preset_dir, cat)
            if os.path.exists(cat_dir):
                for file in os.listdir(cat_dir):
                    if file.endswith('.json'):
                        presets.append(f"{cat}/{file[:-5]}")
        return presets

    def create_preset(self, 
                     name: str,
                     description: str,
                     method: str,
                     parameters: Dict[str, Any],
                     visualization: Dict[str, Any],
                     tags: List[str],
                     category: str = "custom",
                     author: str = "User") -> SimulationPreset:
        """Create a new preset"""
        preset = SimulationPreset(
            name=name,
            description=description,
            method=method,
            parameters=parameters,
            visualization=visualization,
            tags=tags,
            created=datetime.now().isoformat(),
            last_modified=datetime.now().isoformat(),
            author=author,
            category=category,
            version="1.0.0"
        )
        self.save_preset(preset)
        return preset

    def update_preset(self, preset: SimulationPreset) -> bool:
        """Update an existing preset"""
        preset.last_modified = datetime.now().isoformat()
        return self.save_preset(preset)

    def delete_preset(self, name: str, category: str) -> bool:
        """Delete a preset"""
        try:
            file_path = os.path.join(self.preset_dir, category, f"{name}.json")
            os.remove(file_path)
            return True
        except FileNotFoundError:
            print(f"Preset {name} not found in category {category}")
            return False

    def export_preset(self, preset: SimulationPreset, export_path: str) -> bool:
        """Export a preset to a specific location"""
        try:
            with open(export_path, 'w') as f:
                json.dump(preset.__dict__, f, indent=4)
            return True
        except Exception as e:
            print(f"Error exporting preset: {e}")
            return False

    def import_preset(self, import_path: str) -> Optional[SimulationPreset]:
        """Import a preset from a file"""
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)
                preset = SimulationPreset(**data)
                self.save_preset(preset)
                return preset
        except Exception as e:
            print(f"Error importing preset: {e}")
            return None 