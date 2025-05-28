import pytest
import numpy as np
import torch
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    import taichi as ti
    import numpy as np
    import torch
    import wandb
    import h5py
    import yaml
    
    assert ti.__version__ >= "1.6.0"
    assert torch.__version__ >= "2.0.0"

def test_project_structure():
    """Test that the project structure is correct"""
    required_dirs = [
        "navierflow/core/eulerian",
        "navierflow/core/lbm",
        "navierflow/gui",
        "navierflow/utils",
        "configs",
        "data/raw",
        "data/processed",
        "outputs",
        "tests"
    ]
    
    for directory in required_dirs:
        assert Path(directory).exists(), f"Directory {directory} does not exist"

def test_config_loading():
    """Test that the default config can be loaded"""
    import yaml
    
    config_path = Path("configs/default.yaml")
    assert config_path.exists(), "Default config file not found"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    assert "model" in config
    assert "data" in config
    assert config["model"]["model_type"] in ["pinn", "mesh_optimizer", "anomaly_detector"]

def test_cuda_availability():
    """Test CUDA availability if using GPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.randn(100, 100, device=device)
        assert x.device.type == "cuda"

def test_basic_simulation():
    """Test basic fluid simulation setup"""
    import taichi as ti
    
    ti.init(arch=ti.cpu)
    
    # Create a simple grid
    n = 32
    velocity = ti.Vector.field(2, dtype=ti.f32, shape=(n, n))
    pressure = ti.field(dtype=ti.f32, shape=(n, n))
    
    @ti.kernel
    def initialize():
        for i, j in velocity:
            velocity[i, j] = ti.Vector([0.0, 0.0])
            pressure[i, j] = 0.0
    
    initialize()
    
    assert np.all(velocity.to_numpy() == 0)
    assert np.all(pressure.to_numpy() == 0) 