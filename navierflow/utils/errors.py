from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import traceback
import sys

class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

@dataclass
class SimulationError:
    """Simulation error information"""
    message: str
    severity: ErrorSeverity
    context: Optional[Dict[str, Any]] = None
    traceback: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation"""
        error_str = f"{self.severity.name}: {self.message}"
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            error_str += f" ({context_str})"
            
        if self.traceback:
            error_str += f"\n{self.traceback}"
            
        return error_str

class ErrorHandler:
    def __init__(self):
        """Initialize error handler"""
        self.errors: List[SimulationError] = []
        self.max_errors = 1000  # Maximum number of errors to store
        
    def handle_error(self,
                    error: Exception,
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    context: Optional[Dict[str, Any]] = None) -> SimulationError:
        """
        Handle error
        
        Args:
            error: Exception to handle
            severity: Error severity
            context: Optional error context
            
        Returns:
            Simulation error
        """
        # Create simulation error
        sim_error = SimulationError(
            message=str(error),
            severity=severity,
            context=context,
            traceback="".join(traceback.format_exception(type(error), error, error.__traceback__))
        )
        
        # Add error to list
        self.errors.append(sim_error)
        
        # Trim error list if necessary
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
            
        return sim_error
        
    def get_errors(self,
                  min_severity: ErrorSeverity = ErrorSeverity.DEBUG,
                  max_count: Optional[int] = None) -> List[SimulationError]:
        """
        Get errors
        
        Args:
            min_severity: Minimum error severity
            max_count: Maximum number of errors to return
            
        Returns:
            List of errors
        """
        # Filter errors by severity
        filtered_errors = [
            error for error in self.errors
            if error.severity.value >= min_severity.value
        ]
        
        # Limit number of errors
        if max_count is not None:
            filtered_errors = filtered_errors[-max_count:]
            
        return filtered_errors
        
    def clear_errors(self):
        """Clear errors"""
        self.errors.clear()
        
    def has_errors(self,
                   min_severity: ErrorSeverity = ErrorSeverity.ERROR) -> bool:
        """
        Check if there are errors
        
        Args:
            min_severity: Minimum error severity
            
        Returns:
            Whether there are errors
        """
        return any(
            error.severity.value >= min_severity.value
            for error in self.errors
        )
        
    def get_error_summary(self) -> str:
        """
        Get error summary
        
        Returns:
            Error summary
        """
        if not self.errors:
            return "No errors"
            
        # Count errors by severity
        severity_counts = {
            severity: len([
                error for error in self.errors
                if error.severity == severity
            ])
            for severity in ErrorSeverity
        }
        
        # Create summary
        summary = "Error Summary:\n"
        for severity in ErrorSeverity:
            count = severity_counts[severity]
            if count > 0:
                summary += f"{severity.name}: {count}\n"
                
        return summary
        
class SimulationException(Exception):
    """Base class for simulation exceptions"""
    def __init__(self,
                 message: str,
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize exception
        
        Args:
            message: Error message
            severity: Error severity
            context: Optional error context
        """
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        
class ConfigurationError(SimulationException):
    """Configuration error"""
    pass
    
class MeshError(SimulationException):
    """Mesh error"""
    pass
    
class SolverError(SimulationException):
    """Solver error"""
    pass
    
class BoundaryError(SimulationException):
    """Boundary condition error"""
    pass
    
class PhysicsError(SimulationException):
    """Physics error"""
    pass
    
class ValidationError(SimulationException):
    """Validation error"""
    pass
    
class ConvergenceError(SimulationException):
    """Convergence error"""
    pass
    
class ResourceError(SimulationException):
    """Resource error"""
    pass
    
def handle_exception(error: Exception,
                    handler: ErrorHandler,
                    context: Optional[Dict[str, Any]] = None) -> SimulationError:
    """
    Handle exception
    
    Args:
        error: Exception to handle
        handler: Error handler
        context: Optional error context
        
    Returns:
        Simulation error
    """
    # Determine error severity
    if isinstance(error, SimulationException):
        severity = error.severity
        if error.context:
            context = {**(context or {}), **error.context}
    else:
        severity = ErrorSeverity.ERROR
        
    # Handle error
    return handler.handle_error(error, severity, context)
    
def setup_exception_hook(handler: ErrorHandler):
    """
    Setup exception hook
    
    Args:
        handler: Error handler
    """
    def exception_hook(exc_type, exc_value, exc_traceback):
        """Exception hook"""
        # Handle exception
        handle_exception(exc_value, handler)
        
        # Call original hook
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
    # Set exception hook
    sys.excepthook = exception_hook 