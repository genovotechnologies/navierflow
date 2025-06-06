import numpy as np
import matplotlib.pyplot as plt
from navierflow.core.physics.compressible import CompressibleFlow
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
        log_file="compressible_flow.log",
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
            "model_type": "euler",
            "mach_number": 0.8,
            "gamma": 1.4
        },
        numerical={
            "method": "explicit",
            "time_step": 0.001,
            "max_steps": 1000,
            "tolerance": 1e-6
        },
        mesh={
            "type": "structured",
            "dimension": 2,
            "resolution": (100, 100)
        },
        boundary={
            "type": "supersonic",
            "value": 1.0
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
        logger.start_simulation("Compressible flow simulation started")
        
        # Create output directory
        manager.create_output_dir()
        
        # Initialize mesh generator
        with monitor.measure_time("mesh_generation"):
            mesh = MeshGenerator(
                mesh_type="structured",
                dimension=2,
                resolution=(100, 100)
            )
            
            # Generate mesh
            mesh = mesh.generate_structured_mesh(
                bounds=((0, 0), (1, 1)),
                periodic=(False, False)
            )
            
        # Initialize compressible flow
        compressible = CompressibleFlow(
            model_type="euler",
            mach_number=0.8,
            gamma=1.4
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
        density = np.ones((100, 100))
        velocity = np.zeros((100, 100, 2))
        velocity[..., 0] = 1.0  # Uniform flow in x-direction
        pressure = np.ones((100, 100))
        
        # Solve
        with monitor.measure_time("solving"):
            solution = solver.solve(
                density,
                velocity,
                pressure
            )
            
        # Compute density
        with monitor.measure_time("density_computation"):
            density = compressible.compute_density(solution)
            
        # Compute velocity
        with monitor.measure_time("velocity_computation"):
            velocity = compressible.compute_velocity(solution)
            
        # Compute pressure
        with monitor.measure_time("pressure_computation"):
            pressure = compressible.compute_pressure(solution)
            
        # Compute temperature
        with monitor.measure_time("temperature_computation"):
            temperature = compressible.compute_temperature(
                density,
                pressure
            )
            
        # Compute mach number
        with monitor.measure_time("mach_number_computation"):
            mach_number = compressible.compute_mach_number(
                velocity,
                temperature
            )
            
        # Validate results
        with monitor.measure_time("validation"):
            # Validate density
            validator.validate_parameter(
                name="density",
                value=density.mean(),
                expected=1.0,
                tolerance=1e-6
            )
            
            # Validate velocity
            validator.validate_parameter(
                name="velocity",
                value=velocity.mean(),
                expected=1.0,
                tolerance=1e-6
            )
            
            # Validate pressure
            validator.validate_parameter(
                name="pressure",
                value=pressure.mean(),
                expected=1.0,
                tolerance=1e-6
            )
            
            # Validate temperature
            validator.validate_parameter(
                name="temperature",
                value=temperature.mean(),
                expected=1.0,
                tolerance=1e-6
            )
            
            # Validate mach number
            validator.validate_parameter(
                name="mach_number",
                value=mach_number.mean(),
                expected=0.8,
                tolerance=1e-6
            )
            
        # Render results
        with monitor.measure_time("visualization"):
            # Render density
            renderer.render_surface(mesh, density, "Density")
            renderer.save_plot("density.png")
            
            # Render velocity
            renderer.render_surface(mesh, velocity[..., 0], "Velocity X")
            renderer.save_plot("velocity_x.png")
            
            # Render pressure
            renderer.render_surface(mesh, pressure, "Pressure")
            renderer.save_plot("pressure.png")
            
            # Render temperature
            renderer.render_surface(mesh, temperature, "Temperature")
            renderer.save_plot("temperature.png")
            
            # Render mach number
            renderer.render_surface(mesh, mach_number, "Mach Number")
            renderer.save_plot("mach_number.png")
            
        # Generate summary
        with monitor.measure_time("summary_generation"):
            # Log summary
            logger.log_message("Simulation completed successfully")
            logger.log_message(f"Best density: {density.max():.6f}")
            logger.log_message(f"Best velocity: {velocity.max():.6f}")
            logger.log_message(f"Best pressure: {pressure.max():.6f}")
            logger.log_message(f"Best temperature: {temperature.max():.6f}")
            logger.log_message(f"Best mach number: {mach_number.max():.6f}")
            
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
            context={"simulation": "compressible_flow"}
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