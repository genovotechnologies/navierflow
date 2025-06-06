import numpy as np
import yaml
from pathlib import Path
import sys

# Add parent directory to path to import navierflow modules
sys.path.append(str(Path(__file__).parent.parent))

from core.physics.navier_stokes import NavierStokes
from core.numerics.mesh import Mesh
from visualization.plotter import SimulationPlotter

def create_mesh(nx: int, ny: int) -> Mesh:
    """Create structured mesh for lid-driven cavity"""
    # Create vertices
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    vertices = np.column_stack((X.flatten(), Y.flatten()))
    
    # Create cells (triangles)
    cells = []
    for i in range(ny-1):
        for j in range(nx-1):
            # First triangle
            v1 = i * nx + j
            v2 = i * nx + j + 1
            v3 = (i + 1) * nx + j
            cells.append([v1, v2, v3])
            
            # Second triangle
            v1 = i * nx + j + 1
            v2 = (i + 1) * nx + j + 1
            v3 = (i + 1) * nx + j
            cells.append([v1, v2, v3])
            
    return Mesh(vertices, cells)

def main():
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Create mesh
    nx, ny = config["simulation"]["mesh"]["resolution"]
    mesh = create_mesh(nx, ny)
    
    # Initialize Navier-Stokes solver
    viscosity = config["simulation"]["physics"]["fluid"]["viscosity"]
    density = config["simulation"]["physics"]["fluid"]["density"]
    solver = NavierStokes(viscosity, density)
    
    # Initialize fields
    n_vertices = len(mesh.vertices)
    velocity = np.zeros((n_vertices, 2))
    pressure = np.zeros(n_vertices)
    
    # Set lid velocity (top boundary)
    lid_velocity = 1.0
    top_boundary = mesh.vertices[:, 1] == 1.0
    velocity[top_boundary, 0] = lid_velocity
    
    # Time stepping
    dt = config["simulation"]["time_step"]
    max_time = config["simulation"]["max_time"]
    save_interval = config["simulation"]["save_interval"]
    
    # Initialize plotter
    plotter = SimulationPlotter(mesh)
    
    # Time loop
    time = 0.0
    while time < max_time:
        # Update fields
        velocity, pressure = solver.solve_step(velocity, pressure, dt)
        
        # Save results
        if time % save_interval < dt:
            # Plot velocity field
            plotter.plot_vector_field(
                velocity,
                title=f"Velocity field at t = {time:.2f}",
                scale=0.1,
                show_mesh=True
            )
            
            # Plot streamlines
            plotter.plot_streamlines(
                velocity,
                title=f"Streamlines at t = {time:.2f}",
                density=2.0,
                show_mesh=True
            )
            
            # Plot pressure field
            plotter.plot_scalar_field(
                pressure,
                title=f"Pressure field at t = {time:.2f}",
                cmap="coolwarm",
                show_mesh=True
            )
            
        time += dt
        
if __name__ == "__main__":
    main() 