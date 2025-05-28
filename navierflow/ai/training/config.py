from typing import Dict, List, Optional, Union
from pathlib import Path
import yaml
import logging
from dataclasses import dataclass, field
import json

@dataclass
class ModelConfig:
    """Base configuration for models"""
    model_type: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    
    # Training settings
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    
    # Optimizer settings
    optimizer: str = "adam"
    optimizer_params: Dict = field(default_factory=lambda: {
        "betas": (0.9, 0.999),
        "eps": 1e-8
    })
    
    # Learning rate scheduler
    scheduler: Optional[str] = None
    scheduler_params: Dict = field(default_factory=dict)
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    save_best: bool = True
    checkpoint_frequency: int = 10
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class PINNConfig(ModelConfig):
    """Configuration for Physics-Informed Neural Networks"""
    model_type: str = "pinn"
    
    # Architecture
    hidden_layers: List[int] = field(default_factory=lambda: [64, 128, 128, 64])
    activation: str = "tanh"
    
    # PDE settings
    num_pde_points: int = 10000
    num_bc_points: int = 1000
    
    # Loss weights
    pde_weight: float = 1.0
    bc_weight: float = 1.0
    data_weight: float = 1.0

@dataclass
class MeshOptimizerConfig(ModelConfig):
    """Configuration for mesh optimization"""
    model_type: str = "mesh_optimizer"
    
    # Mesh settings
    base_resolution: tuple = (64, 64)
    max_refinement_level: int = 3
    feature_threshold: float = 0.1
    
    # Feature detection network
    feature_channels: List[int] = field(default_factory=lambda: [32, 64, 32])
    pooling_type: str = "max"
    
    # Loss weights
    feature_weight: float = 1.0
    smoothness_weight: float = 0.1
    efficiency_weight: float = 0.1

@dataclass
class AnomalyDetectorConfig(ModelConfig):
    """Configuration for anomaly detection"""
    model_type: str = "anomaly_detector"
    
    # Autoencoder architecture
    input_channels: int = 4
    latent_dim: int = 32
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 32])
    
    # Detection settings
    threshold_percentile: float = 95.0
    min_anomaly_size: int = 10
    
    # Loss weights
    reconstruction_weight: float = 1.0
    sparsity_weight: float = 0.1

@dataclass
class DataConfig:
    """Configuration for data handling"""
    # Data paths
    data_dir: Path
    output_dir: Path
    
    # Data splits
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Preprocessing
    normalize: bool = True
    stats_file: Optional[Path] = None
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_factor: int = 2
    
    # Transforms
    crop_size: Optional[tuple] = None
    use_flips: bool = True
    
    # Cache settings
    cache_size: int = 100
    prefetch_factor: int = 2

@dataclass
class TrainingConfig:
    """Complete training configuration"""
    # Basic info
    experiment_name: str
    timestamp: str
    seed: int = 42
    
    # Components
    model: ModelConfig
    data: DataConfig
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "navierflow"
    log_frequency: int = 10
    
    # Output
    output_dir: Path = Path("outputs")
    save_predictions: bool = True
    export_format: str = "h5"

class ConfigManager:
    """Manager for handling configurations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: Union[str, Path]) -> TrainingConfig:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load YAML
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        
        # Create model config based on type
        model_type = config_dict['model']['model_type']
        if model_type == "pinn":
            model_config = PINNConfig(**config_dict['model'])
        elif model_type == "mesh_optimizer":
            model_config = MeshOptimizerConfig(**config_dict['model'])
        elif model_type == "anomaly_detector":
            model_config = AnomalyDetectorConfig(**config_dict['model'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create data config
        data_config = DataConfig(**config_dict['data'])
        
        # Create complete config
        config = TrainingConfig(
            model=model_config,
            data=data_config,
            **{k: v for k, v in config_dict.items()
               if k not in ['model', 'data']}
        )
        
        return config

    def save_config(
        self,
        config: TrainingConfig,
        output_path: Union[str, Path]
    ):
        """Save configuration to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = {
            'experiment_name': config.experiment_name,
            'timestamp': config.timestamp,
            'seed': config.seed,
            'model': self._dataclass_to_dict(config.model),
            'data': self._dataclass_to_dict(config.data),
            'use_wandb': config.use_wandb,
            'wandb_project': config.wandb_project,
            'log_frequency': config.log_frequency,
            'output_dir': str(config.output_dir),
            'save_predictions': config.save_predictions,
            'export_format': config.export_format
        }
        
        # Save as YAML
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        self.logger.info(f"Saved configuration to {output_path}")

    def _dataclass_to_dict(self, obj) -> Dict:
        """Convert dataclass instance to dictionary"""
        result = {}
        for field_name, field_value in obj.__dict__.items():
            if isinstance(field_value, Path):
                result[field_name] = str(field_value)
            elif isinstance(field_value, (list, dict, str, int, float, bool)) or field_value is None:
                result[field_name] = field_value
            else:
                result[field_name] = str(field_value)
        return result

    def create_experiment_dir(
        self,
        config: TrainingConfig
    ) -> Path:
        """Create experiment directory with config"""
        # Create experiment directory
        experiment_dir = config.output_dir / config.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = experiment_dir / "config.yaml"
        self.save_config(config, config_path)
        
        # Create subdirectories
        (experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (experiment_dir / "predictions").mkdir(exist_ok=True)
        (experiment_dir / "logs").mkdir(exist_ok=True)
        
        return experiment_dir

def load_default_config(model_type: str) -> Dict:
    """Load default configuration for model type"""
    if model_type == "pinn":
        return {
            "model": PINNConfig().__dict__,
            "data": DataConfig(
                data_dir=Path("data/raw"),
                output_dir=Path("data/processed")
            ).__dict__
        }
    elif model_type == "mesh_optimizer":
        return {
            "model": MeshOptimizerConfig().__dict__,
            "data": DataConfig(
                data_dir=Path("data/raw"),
                output_dir=Path("data/processed")
            ).__dict__
        }
    elif model_type == "anomaly_detector":
        return {
            "model": AnomalyDetectorConfig().__dict__,
            "data": DataConfig(
                data_dir=Path("data/raw"),
                output_dir=Path("data/processed")
            ).__dict__
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_example_config(
    model_type: str,
    output_path: Union[str, Path]
):
    """Create example configuration file"""
    config = load_default_config(model_type)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created example config at {output_path}") 