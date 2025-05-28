import logging
from typing import Optional, Any, Dict
import traceback
import sys
import numpy as np

class SimulationError(Exception):
    """Base class for simulation-related errors"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}

class SolverError(SimulationError):
    """Error in numerical solver"""
    pass

class BoundaryError(SimulationError):
    """Error in boundary conditions"""
    pass

class PhysicsError(SimulationError):
    """Error in physics calculations"""
    pass

class ConfigurationError(SimulationError):
    """Error in configuration"""
    pass

def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('simulation.log')
        ]
    )

def handle_simulation_error(e: Exception, logger: logging.Logger) -> None:
    """Handle simulation errors with appropriate logging"""
    if isinstance(e, SimulationError):
        logger.error(f"{e.__class__.__name__}: {str(e)}")
        if e.details:
            logger.debug(f"Error details: {e.details}")
    else:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")

def safe_operation(logger: logging.Logger):
    """Decorator for safe operation execution with error handling"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_simulation_error(e, logger)
                return None
        return wrapper
    return decorator

def validate_config(config: Dict[str, Any], required_fields: Dict[str, type]) -> None:
    """Validate configuration dictionary"""
    for field, field_type in required_fields.items():
        if field not in config:
            raise ConfigurationError(f"Missing required field: {field}")
        if not isinstance(config[field], field_type):
            raise ConfigurationError(
                f"Invalid type for field {field}. Expected {field_type}, got {type(config[field])}"
            )

def check_numerical_stability(value: float, field_name: str) -> None:
    """Check numerical stability of a value"""
    if not np.isfinite(value):
        raise PhysicsError(f"Non-finite value detected in {field_name}")
    if np.abs(value) > 1e6:
        raise PhysicsError(f"Value too large in {field_name}: {value}")

def check_array_bounds(array: np.ndarray, field_name: str) -> None:
    """Check array bounds and values"""
    if not np.all(np.isfinite(array)):
        raise PhysicsError(f"Non-finite values detected in {field_name}")
    if np.any(np.abs(array) > 1e6):
        raise PhysicsError(f"Values too large in {field_name}")

def log_performance_metrics(metrics: Dict[str, float], logger: logging.Logger) -> None:
    """Log performance metrics"""
    logger.info("Performance metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.6f}")

def create_logger(name: str) -> logging.Logger:
    """Create a logger with standard configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Add console handler if not already present
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)
    
    return logger 