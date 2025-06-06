import numpy as np
import matplotlib.pyplot as plt
from navierflow.core.physics.heat import HeatTransfer
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
        log_file="heat_transfer.log",
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
            "model_type": "conduction",
            "thermal_conductivity": 1.0,
            "heat_capacity": 1.0,
            "density": 1.0
        },
        numerical={
            "method": "implicit",
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
            "type": "dirichlet",
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
        logger.start_simulation("Heat transfer simulation started")
        
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
            
        # Initialize heat transfer
        heat = HeatTransfer(
            model_type="conduction",
            thermal_conductivity=1.0,
            heat_capacity=1.0,
            density=1.0
        )
        
        # Initialize solver
        solver = Solver(
            method="implicit",
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
        temperature = np.zeros((100, 100))
        temperature[0, :] = 1.0  # Hot wall at x=0
        
        # Solve
        with monitor.measure_time("solving"):
            solution = solver.solve(temperature)
            
        # Compute temperature
        with monitor.measure_time("temperature_computation"):
            temperature = heat.compute_temperature(solution)
            
        # Compute heat flux
        with monitor.measure_time("heat_flux_computation"):
            heat_flux = heat.compute_heat_flux(temperature)
            
        # Compute thermal gradient
        with monitor.measure_time("thermal_gradient_computation"):
            thermal_gradient = heat.compute_thermal_gradient(temperature)
            
        # Compute thermal energy
        with monitor.measure_time("thermal_energy_computation"):
            thermal_energy = heat.compute_thermal_energy(
                temperature,
                heat_flux
            )
            
        # Validate results
        with monitor.measure_time("validation"):
            # Validate temperature
            validator.validate_parameter(
                name="temperature",
                value=temperature.mean(),
                expected=0.5,
                tolerance=1e-6
            )
            
            # Validate heat flux
            validator.validate_parameter(
                name="heat_flux",
                value=heat_flux.mean(),
                expected=1.0,
                tolerance=1e-6
            )
            
            # Validate thermal gradient
            validator.validate_parameter(
                name="thermal_gradient",
                value=thermal_gradient.mean(),
                expected=1.0,
                tolerance=1e-6
            )
            
            # Validate thermal energy
            validator.validate_parameter(
                name="thermal_energy",
                value=thermal_energy.mean(),
                expected=1.0,
                tolerance=1e-6
            )
            
        # Render results
        with monitor.measure_time("visualization"):
            # Render temperature
            renderer.render_surface(mesh, temperature, "Temperature")
            renderer.save_plot("temperature.png")
            
            # Render heat flux
            renderer.render_surface(mesh, heat_flux[..., 0], "Heat Flux X")
            renderer.save_plot("heat_flux_x.png")
            
            # Render thermal gradient
            renderer.render_surface(mesh, thermal_gradient[..., 0], "Thermal Gradient X")
            renderer.save_plot("thermal_gradient_x.png")
            
            # Render thermal energy
            renderer.render_surface(mesh, thermal_energy, "Thermal Energy")
            renderer.save_plot("thermal_energy.png")
            
        # Generate summary
        with monitor.measure_time("summary_generation"):
            # Log summary
            logger.log_message("Simulation completed successfully")
            logger.log_message(f"Best temperature: {temperature.max():.6f}")
            logger.log_message(f"Best heat flux: {heat_flux.max():.6f}")
            logger.log_message(f"Best thermal gradient: {thermal_gradient.max():.6f}")
            logger.log_message(f"Best thermal energy: {thermal_energy.max():.6f}")
            
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
            context={"simulation": "heat_transfer"}
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