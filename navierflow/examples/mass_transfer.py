import numpy as np
import matplotlib.pyplot as plt
from navierflow.core.physics.mass import MassTransfer
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
        log_file="mass_transfer.log",
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
            "model_type": "diffusion",
            "diffusion_coefficient": 1.0,
            "reaction_rate": 0.1,
            "source_term": 0.0
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
        logger.start_simulation("Mass transfer simulation started")
        
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
            
        # Initialize mass transfer
        mass = MassTransfer(
            model_type="diffusion",
            diffusion_coefficient=1.0,
            reaction_rate=0.1,
            source_term=0.0
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
        concentration = np.zeros((100, 100))
        concentration[0, :] = 1.0  # Source at x=0
        
        # Solve
        with monitor.measure_time("solving"):
            solution = solver.solve(concentration)
            
        # Compute concentration
        with monitor.measure_time("concentration_computation"):
            concentration = mass.compute_concentration(solution)
            
        # Compute mass flux
        with monitor.measure_time("mass_flux_computation"):
            mass_flux = mass.compute_mass_flux(concentration)
            
        # Compute concentration gradient
        with monitor.measure_time("concentration_gradient_computation"):
            concentration_gradient = mass.compute_concentration_gradient(concentration)
            
        # Compute reaction rate
        with monitor.measure_time("reaction_rate_computation"):
            reaction_rate = mass.compute_reaction_rate(concentration)
            
        # Validate results
        with monitor.measure_time("validation"):
            # Validate concentration
            validator.validate_parameter(
                name="concentration",
                value=concentration.mean(),
                expected=0.5,
                tolerance=1e-6
            )
            
            # Validate mass flux
            validator.validate_parameter(
                name="mass_flux",
                value=mass_flux.mean(),
                expected=1.0,
                tolerance=1e-6
            )
            
            # Validate concentration gradient
            validator.validate_parameter(
                name="concentration_gradient",
                value=concentration_gradient.mean(),
                expected=1.0,
                tolerance=1e-6
            )
            
            # Validate reaction rate
            validator.validate_parameter(
                name="reaction_rate",
                value=reaction_rate.mean(),
                expected=0.1,
                tolerance=1e-6
            )
            
        # Render results
        with monitor.measure_time("visualization"):
            # Render concentration
            renderer.render_surface(mesh, concentration, "Concentration")
            renderer.save_plot("concentration.png")
            
            # Render mass flux
            renderer.render_surface(mesh, mass_flux[..., 0], "Mass Flux X")
            renderer.save_plot("mass_flux_x.png")
            
            # Render concentration gradient
            renderer.render_surface(mesh, concentration_gradient[..., 0], "Concentration Gradient X")
            renderer.save_plot("concentration_gradient_x.png")
            
            # Render reaction rate
            renderer.render_surface(mesh, reaction_rate, "Reaction Rate")
            renderer.save_plot("reaction_rate.png")
            
        # Generate summary
        with monitor.measure_time("summary_generation"):
            # Log summary
            logger.log_message("Simulation completed successfully")
            logger.log_message(f"Best concentration: {concentration.max():.6f}")
            logger.log_message(f"Best mass flux: {mass_flux.max():.6f}")
            logger.log_message(f"Best concentration gradient: {concentration_gradient.max():.6f}")
            logger.log_message(f"Best reaction rate: {reaction_rate.max():.6f}")
            
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
            context={"simulation": "mass_transfer"}
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