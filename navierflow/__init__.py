"""
NavierFlow - Advanced Fluid Dynamics Simulation
"""

from navierflow.core import (
    CoreEulerianSolver,
    CoreLBMSolver,
    MultiPhysicsSolver,
    PhysicsModel,
    OptimizedComputeEngine,
    ComputeBackend
)

from navierflow.ui.visualization.simulation_interface import (
    GUIManager,
    SimulationManager
)

from navierflow.ui.dashboard.main import Dashboard
from navierflow.ai.ai_integration import SimulationEnhancer
from navierflow.utils import setup_logging, create_logger

__version__ = "2.4.0"
__author__ = "tafolabi009"
__created__ = "2025-02-22"

__all__ = [
    # Core components
    'CoreEulerianSolver',
    'CoreLBMSolver',
    'MultiPhysicsSolver',
    'PhysicsModel',
    'OptimizedComputeEngine',
    'ComputeBackend',
    
    # UI components
    'GUIManager',
    'SimulationManager',
    'Dashboard',
    
    # AI components
    'SimulationEnhancer',
    
    # Utilities
    'setup_logging',
    'create_logger'
]
