from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ValidationType(Enum):
    """Validation types"""
    PARAMETER = "parameter"
    RESULT = "result"
    CONSERVATION = "conservation"
    STABILITY = "stability"
    BOUNDARY = "boundary"

@dataclass
class ValidationResult:
    """Validation result"""
    type: ValidationType
    name: str
    passed: bool
    message: str
    value: Optional[Any] = None
    expected: Optional[Any] = None
    tolerance: Optional[float] = None
    
    def __str__(self) -> str:
        """String representation"""
        result_str = f"{self.type.value.upper()} - {self.name}: "
        result_str += "PASSED" if self.passed else "FAILED"
        
        if self.message:
            result_str += f" - {self.message}"
            
        if self.value is not None:
            result_str += f"\nValue: {self.value}"
            
        if self.expected is not None:
            result_str += f"\nExpected: {self.expected}"
            
        if self.tolerance is not None:
            result_str += f"\nTolerance: {self.tolerance}"
            
        return result_str

class Validator:
    def __init__(self):
        """Initialize validator"""
        self.results: List[ValidationResult] = []
        
    def validate_parameter(self,
                          name: str,
                          value: Any,
                          expected: Any,
                          tolerance: Optional[float] = None) -> ValidationResult:
        """
        Validate parameter
        
        Args:
            name: Parameter name
            value: Parameter value
            expected: Expected value
            tolerance: Optional tolerance
            
        Returns:
            Validation result
        """
        # Check if values are numeric
        if isinstance(value, (int, float)) and isinstance(expected, (int, float)):
            if tolerance is not None:
                passed = abs(value - expected) <= tolerance
                message = f"Value {value} is within tolerance {tolerance} of expected {expected}"
            else:
                passed = value == expected
                message = f"Value {value} matches expected {expected}"
        else:
            passed = value == expected
            message = f"Value {value} matches expected {expected}"
            
        # Create result
        result = ValidationResult(
            type=ValidationType.PARAMETER,
            name=name,
            passed=passed,
            message=message,
            value=value,
            expected=expected,
            tolerance=tolerance
        )
        
        # Add result
        self.results.append(result)
        
        return result
        
    def validate_result(self,
                       name: str,
                       value: np.ndarray,
                       expected: np.ndarray,
                       tolerance: float = 1e-6) -> ValidationResult:
        """
        Validate result
        
        Args:
            name: Result name
            value: Result value
            expected: Expected value
            tolerance: Tolerance
            
        Returns:
            Validation result
        """
        # Check shapes
        if value.shape != expected.shape:
            passed = False
            message = f"Shape mismatch: {value.shape} != {expected.shape}"
        else:
            # Compute error
            error = np.abs(value - expected).max()
            passed = error <= tolerance
            message = f"Maximum error: {error} (tolerance: {tolerance})"
            
        # Create result
        result = ValidationResult(
            type=ValidationType.RESULT,
            name=name,
            passed=passed,
            message=message,
            value=value,
            expected=expected,
            tolerance=tolerance
        )
        
        # Add result
        self.results.append(result)
        
        return result
        
    def validate_conservation(self,
                            name: str,
                            initial: float,
                            final: float,
                            tolerance: float = 1e-6) -> ValidationResult:
        """
        Validate conservation
        
        Args:
            name: Conservation name
            initial: Initial value
            final: Final value
            tolerance: Tolerance
            
        Returns:
            Validation result
        """
        # Compute relative error
        error = abs(final - initial) / (abs(initial) + 1e-10)
        passed = error <= tolerance
        message = f"Relative error: {error} (tolerance: {tolerance})"
        
        # Create result
        result = ValidationResult(
            type=ValidationType.CONSERVATION,
            name=name,
            passed=passed,
            message=message,
            value=final,
            expected=initial,
            tolerance=tolerance
        )
        
        # Add result
        self.results.append(result)
        
        return result
        
    def validate_stability(self,
                          name: str,
                          value: np.ndarray,
                          threshold: float) -> ValidationResult:
        """
        Validate stability
        
        Args:
            name: Stability name
            value: Value to check
            threshold: Stability threshold
            
        Returns:
            Validation result
        """
        # Check if value exceeds threshold
        max_value = np.abs(value).max()
        passed = max_value <= threshold
        message = f"Maximum value: {max_value} (threshold: {threshold})"
        
        # Create result
        result = ValidationResult(
            type=ValidationType.STABILITY,
            name=name,
            passed=passed,
            message=message,
            value=max_value,
            expected=threshold
        )
        
        # Add result
        self.results.append(result)
        
        return result
        
    def validate_boundary(self,
                         name: str,
                         value: np.ndarray,
                         boundary: str,
                         expected: Optional[Any] = None) -> ValidationResult:
        """
        Validate boundary condition
        
        Args:
            name: Boundary name
            value: Boundary value
            boundary: Boundary type
            expected: Expected value
            
        Returns:
            Validation result
        """
        # Check boundary condition
        if boundary == "dirichlet":
            if expected is not None:
                error = np.abs(value - expected).max()
                passed = error <= 1e-6
                message = f"Maximum error: {error}"
            else:
                passed = True
                message = "Dirichlet boundary condition satisfied"
        elif boundary == "neumann":
            gradient = np.gradient(value)
            error = np.abs(gradient).max()
            passed = error <= 1e-6
            message = f"Maximum gradient: {error}"
        elif boundary == "periodic":
            start = value[0]
            end = value[-1]
            error = np.abs(start - end).max()
            passed = error <= 1e-6
            message = f"Periodicity error: {error}"
        else:
            passed = False
            message = f"Unknown boundary type: {boundary}"
            
        # Create result
        result = ValidationResult(
            type=ValidationType.BOUNDARY,
            name=name,
            passed=passed,
            message=message,
            value=value,
            expected=expected
        )
        
        # Add result
        self.results.append(result)
        
        return result
        
    def get_results(self,
                   type: Optional[ValidationType] = None,
                   passed: Optional[bool] = None) -> List[ValidationResult]:
        """
        Get validation results
        
        Args:
            type: Optional result type
            passed: Optional pass/fail filter
            
        Returns:
            List of results
        """
        # Filter results
        filtered_results = self.results
        
        if type is not None:
            filtered_results = [
                result for result in filtered_results
                if result.type == type
            ]
            
        if passed is not None:
            filtered_results = [
                result for result in filtered_results
                if result.passed == passed
            ]
            
        return filtered_results
        
    def clear_results(self):
        """Clear results"""
        self.results.clear()
        
    def get_summary(self) -> str:
        """
        Get validation summary
        
        Returns:
            Validation summary
        """
        if not self.results:
            return "No validation results"
            
        # Count results by type and status
        counts = {
            type: {
                "passed": 0,
                "failed": 0
            }
            for type in ValidationType
        }
        
        for result in self.results:
            status = "passed" if result.passed else "failed"
            counts[result.type][status] += 1
            
        # Create summary
        summary = "Validation Summary:\n"
        for type in ValidationType:
            passed = counts[type]["passed"]
            failed = counts[type]["failed"]
            total = passed + failed
            
            if total > 0:
                summary += f"\n{type.value.upper()}:\n"
                summary += f"  Passed: {passed}/{total}\n"
                summary += f"  Failed: {failed}/{total}\n"
                
        return summary 