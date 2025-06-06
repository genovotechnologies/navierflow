import numpy as np
import matplotlib.pyplot as plt
from navierflow.core.physics.electromagnetic import ElectromagneticField
from navierflow.core.numerics.solver import Solver
from navierflow.core.mesh.generation import MeshGenerator
from navierflow.visualization.renderer import Renderer, VisualizationConfig
from navierflow.utils.logging import SimulationLogger
from navierflow.utils.errors import ErrorHandler
from navierflow.utils.validation import Validator
from navierflow.utils.performance import PerformanceMonitor
from navierflow.configs.settings import ConfigManager, SimulationConfig

def main():
    # Initialize logger
    logger = SimulationLogger(
        log_file="electromagnetic_wave.log",
        level="INFO",
        stream=True
    )
    
    # Initialize error handler
    handler = ErrorHandler()
    
    # Initialize validator
    validator = Validator()
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    # Initialize config manager
    config = SimulationConfig(
        physics={
            "permittivity": 8.85e-12,
            "permeability": 1.26e-6,
            "conductivity": 1.0
        },
        numerical={
            "method": "explicit",
            "time_step": 0.001,
            "max_steps": 1000,
            "tolerance": 1e-6
        },
        mesh={
            "type": "structured",
            "dimension": 3,
            "resolution": (50, 50, 50)
        },
        boundary={
            "type": "periodic",
            "value": 0.0
        },
        output={
            "path": "output",
            "format": "vtk",
            "frequency": 100
        },
        parallel={
            "backend": "cpu",
            "n_processes": 4,
            "n_threads": 2,
            "use_gpu": False
        },
        visualization={
            "type": "surface",
            "colormap": "viridis",
            "background_color": "white",
            "show_axes": True,
            "show_grid": False,
            "show_legend": True,
            "show_colorbar": True,
            "window_size": (800, 600),
            "dpi": 100,
            "animation_fps": 30,
            "animation_duration": 10.0
        }
    )
    
    manager = ConfigManager(config)
    
    try:
        # Start simulation
        logger.start_simulation("Electromagnetic wave simulation started")
        
        # Create output directory
        manager.create_output_dir()
        
        # Initialize mesh generator
        with monitor.measure_time("mesh_generation"):
            mesh = MeshGenerator(
                mesh_type="structured",
                dimension=3,
                resolution=(50, 50, 50)
            )
            
            # Generate mesh
            mesh = mesh.generate_structured_mesh(
                bounds=((0, 0, 0), (1, 1, 1)),
                periodic=(True, True, True)
            )
            
        # Initialize electromagnetic field
        em = ElectromagneticField(
            permittivity=8.85e-12,
            permeability=1.26e-6,
            conductivity=1.0
        )
        
        # Initialize solver
        solver = Solver(
            method="explicit",
            time_step=0.001,
            max_steps=1000,
            tolerance=1e-6
        )
        
        # Initialize renderer
        renderer = Renderer(VisualizationConfig(
            type="surface",
            colormap="viridis",
            background_color="white",
            show_axes=True,
            show_grid=False,
            show_legend=True,
            show_colorbar=True,
            window_size=(800, 600),
            dpi=100,
            animation_fps=30,
            animation_duration=10.0
        ))
        
        # Initial condition
        charge_density = np.zeros((50, 50, 50))
        charge_density[25, 25, 25] = 1.0
        
        current_density = np.zeros((50, 50, 50, 3))
        current_density[..., 0] = 1.0
        
        # Solve
        with monitor.measure_time("solving"):
            solution = solver.solve(charge_density)
            
        # Compute electric field
        with monitor.measure_time("electric_field_computation"):
            electric_field = em.compute_electric_field(solution)
            
        # Compute magnetic field
        with monitor.measure_time("magnetic_field_computation"):
            magnetic_field = em.compute_magnetic_field(current_density)
            
        # Compute Lorentz force
        with monitor.measure_time("lorentz_force_computation"):
            lorentz_force = em.compute_lorentz_force(
                electric_field,
                magnetic_field
            )
            
        # Compute boundary conditions
        with monitor.measure_time("boundary_condition_computation"):
            em.compute_boundary_conditions(
                electric_field,
                magnetic_field,
                boundary_type="periodic"
            )
            
        # Validate results
        with monitor.measure_time("validation"):
            # Validate electric field
            validator.validate_parameter(
                name="electric_field",
                value=electric_field.mean(),
                expected=0.0,
                tolerance=1e-6
            )
            
            # Validate magnetic field
            validator.validate_parameter(
                name="magnetic_field",
                value=magnetic_field.mean(),
                expected=0.0,
                tolerance=1e-6
            )
            
            # Validate Lorentz force
            validator.validate_parameter(
                name="lorentz_force",
                value=lorentz_force.mean(),
                expected=0.0,
                tolerance=1e-6
            )
            
        # Render results
        with monitor.measure_time("visualization"):
            # Render electric field
            renderer.render_surface(mesh, electric_field[..., 0], "Electric Field X")
            renderer.save_plot("electric_field_x.png")
            
            # Render magnetic field
            renderer.render_surface(mesh, magnetic_field[..., 0], "Magnetic Field X")
            renderer.save_plot("magnetic_field_x.png")
            
            # Render Lorentz force
            renderer.render_surface(mesh, lorentz_force[..., 0], "Lorentz Force X")
            renderer.save_plot("lorentz_force_x.png")
            
        # Generate summary
        with monitor.measure_time("summary_generation"):
            # Log summary
            logger.log_message("Simulation completed successfully")
            logger.log_message(f"Best electric field: {electric_field.min():.6f}")
            logger.log_message(f"Best magnetic field: {magnetic_field.min():.6f}")
            logger.log_message(f"Best lorentz force: {lorentz_force.min():.6f}")
            
            # Generate validation summary
            validation_summary = validator.generate_summary()
            logger.log_message(f"Validation summary: {validation_summary}")
            
            # Generate performance summary
            performance_summary = monitor.generate_summary()
            logger.log_message(f"Performance summary: {performance_summary}")
            
        # End simulation
        logger.end_simulation(True, "Simulation completed successfully")
        
    except Exception as e:
        # Handle error
        handler.handle_error(
            str(e),
            severity="ERROR",
            context={"simulation": "electromagnetic_wave"}
        )
        
        # Log error
        logger.log_message(f"Error: {str(e)}", level="ERROR")
        
        # End simulation
        logger.end_simulation(False, "Simulation failed")
        
    finally:
        # Cleanup
        renderer.cleanup()
        manager.cleanup_output_dir()
        
if __name__ == "__main__":
    main() 