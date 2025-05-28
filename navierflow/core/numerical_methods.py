import numpy as np
import torch
from typing import Tuple, Optional

def poisson_solver(rhs: np.ndarray, dx: float, dy: float,
                  boundary_conditions: dict,
                  max_iter: int = 1000,
                  tolerance: float = 1e-6) -> np.ndarray:
    """
    Solve the Poisson equation using successive over-relaxation (SOR).
    
    Args:
        rhs: Right-hand side of the Poisson equation
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction
        boundary_conditions: Dictionary specifying boundary conditions
        max_iter: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Solution of the Poisson equation
    """
    nx, ny = rhs.shape
    solution = np.zeros_like(rhs)
    
    # SOR parameter
    omega = 1.5
    
    # Apply boundary conditions
    for boundary, value in boundary_conditions.items():
        if boundary == 'left':
            solution[0, :] = value
        elif boundary == 'right':
            solution[-1, :] = value
        elif boundary == 'bottom':
            solution[:, 0] = value
        elif boundary == 'top':
            solution[:, -1] = value
    
    dx2 = dx * dx
    dy2 = dy * dy
    
    # Iteration loop
    for it in range(max_iter):
        old_solution = solution.copy()
        
        # Update interior points
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                solution[i, j] = ((1.0 - omega) * solution[i, j] +
                                omega * 0.25 * (solution[i+1, j] + solution[i-1, j] +
                                              solution[i, j+1] + solution[i, j-1] -
                                              dx2 * dy2 * rhs[i, j] / (dx2 + dy2)))
        
        # Check convergence
        error = np.max(np.abs(solution - old_solution))
        if error < tolerance:
            break
            
    return solution

def advection_diffusion(field: np.ndarray,
                       velocity_x: np.ndarray,
                       velocity_y: np.ndarray,
                       diffusion_coeff: float,
                       dt: float,
                       dx: float,
                       dy: float) -> np.ndarray:
    """
    Solve the advection-diffusion equation using an upwind scheme for advection
    and central differences for diffusion.
    
    Args:
        field: Scalar field to be advected and diffused
        velocity_x: x-component of velocity field
        velocity_y: y-component of velocity field
        diffusion_coeff: Diffusion coefficient
        dt: Time step
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction
        
    Returns:
        Updated scalar field
    """
    nx, ny = field.shape
    new_field = field.copy()
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Advection terms (upwind)
            if velocity_x[i, j] > 0:
                adv_x = velocity_x[i, j] * (field[i, j] - field[i-1, j]) / dx
            else:
                adv_x = velocity_x[i, j] * (field[i+1, j] - field[i, j]) / dx
                
            if velocity_y[i, j] > 0:
                adv_y = velocity_y[i, j] * (field[i, j] - field[i, j-1]) / dy
            else:
                adv_y = velocity_y[i, j] * (field[i, j+1] - field[i, j]) / dy
            
            # Diffusion terms (central)
            diff_x = diffusion_coeff * (field[i+1, j] - 2*field[i, j] + field[i-1, j]) / (dx*dx)
            diff_y = diffusion_coeff * (field[i, j+1] - 2*field[i, j] + field[i, j-1]) / (dy*dy)
            
            # Update field
            new_field[i, j] = field[i, j] + dt * (-(adv_x + adv_y) + diff_x + diff_y)
            
    return new_field

def vorticity_stream(velocity_x: np.ndarray,
                    velocity_y: np.ndarray,
                    dx: float,
                    dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute vorticity and stream function from velocity field.
    
    Args:
        velocity_x: x-component of velocity field
        velocity_y: y-component of velocity field
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction
        
    Returns:
        Tuple of (vorticity, stream function)
    """
    # Compute vorticity
    dvydx = np.gradient(velocity_y, dx, axis=0)
    dvxdy = np.gradient(velocity_x, dy, axis=1)
    vorticity = dvydx - dvxdy
    
    # Solve for stream function
    boundary_conditions = {
        'left': 0.0,
        'right': 0.0,
        'top': 0.0,
        'bottom': 0.0
    }
    
    stream_function = poisson_solver(-vorticity, dx, dy, boundary_conditions)
    
    return vorticity, stream_function

def pressure_projection(velocity_x: np.ndarray,
                       velocity_y: np.ndarray,
                       density: float,
                       dx: float,
                       dy: float,
                       dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform pressure projection to enforce incompressibility.
    
    Args:
        velocity_x: x-component of velocity field
        velocity_y: y-component of velocity field
        density: Fluid density
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction
        dt: Time step
        
    Returns:
        Tuple of (corrected velocity_x, corrected velocity_y, pressure)
    """
    # Compute divergence
    dudx = np.gradient(velocity_x, dx, axis=0)
    dvdy = np.gradient(velocity_y, dy, axis=1)
    divergence = dudx + dvdy
    
    # Solve pressure Poisson equation
    boundary_conditions = {
        'left': 0.0,
        'right': 0.0,
        'top': 0.0,
        'bottom': 0.0
    }
    
    pressure = poisson_solver(divergence/dt, dx, dy, boundary_conditions)
    
    # Correct velocities
    pressure_grad_x = np.gradient(pressure, dx, axis=0)
    pressure_grad_y = np.gradient(pressure, dy, axis=1)
    
    velocity_x_corrected = velocity_x - dt * pressure_grad_x / density
    velocity_y_corrected = velocity_y - dt * pressure_grad_y / density
    
    return velocity_x_corrected, velocity_y_corrected, pressure 