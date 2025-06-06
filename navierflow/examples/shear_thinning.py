import numpy as np
import matplotlib.pyplot as plt
from navierflow.core.physics.non_newtonian import NonNewtonianFlow
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
        log_file="shear_thinning.log",
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
            "model_type": "power_law",
            "k": 1.0,
            "n": 0.5
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
            "type": "no_slip",
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
        logger.start_simulation("Shear-thinning fluid simulation started")
        
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
            
        # Initialize non-Newtonian flow
        non_newtonian = NonNewtonianFlow(
            model_type="power_law",
            k=1.0,
            n=0.5
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
        strain_rate = np.zeros((100, 100, 2, 2))
        strain_rate[..., 0, 1] = 1.0  # Simple shear
        
        # Solve
        with monitor.measure_time("solving"):
            solution = solver.solve(strain_rate)
            
        # Compute viscosity
        with monitor.measure_time("viscosity_computation"):
            viscosity = non_newtonian.compute_viscosity(solution)
            
        # Compute stress
        with monitor.measure_time("stress_computation"):
            stress = non_newtonian.compute_stress(solution)
            
        # Compute yield criterion
        with monitor.measure_time("yield_criterion_computation"):
            yield_criterion = non_newtonian.compute_yield_criterion(stress)
            
        # Compute power dissipation
        with monitor.measure_time("power_dissipation_computation"):
            power_dissipation = non_newtonian.compute_power_dissipation(
                solution,
                stress
            )
            
        # Validate results
        with monitor.measure_time("validation"):
            # Validate viscosity
            validator.validate_parameter(
                name="viscosity",
                value=viscosity.mean(),
                expected=1.0,
                tolerance=1e-6
            )
            
            # Validate stress
            validator.validate_parameter(
                name="stress",
                value=stress.mean(),
                expected=1.0,
                tolerance=1e-6
            )
            
            # Validate yield criterion
            validator.validate_parameter(
                name="yield_criterion",
                value=yield_criterion.mean(),
                expected=0.0,
                tolerance=1e-6
            )
            
            # Validate power dissipation
            validator.validate_parameter(
                name="power_dissipation",
                value=power_dissipation.mean(),
                expected=1.0,
                tolerance=1e-6
            )
            
        # Render results
        with monitor.measure_time("visualization"):
            # Render viscosity
            renderer.render_surface(mesh, viscosity, "Viscosity")
            renderer.save_plot("viscosity.png")
            
            # Render stress
            renderer.render_surface(mesh, stress[..., 0, 0], "Stress XX")
            renderer.save_plot("stress_xx.png")
            
            # Render yield criterion
            renderer.render_surface(mesh, yield_criterion, "Yield Criterion")
            renderer.save_plot("yield_criterion.png")
            
            # Render power dissipation
            renderer.render_surface(mesh, power_dissipation, "Power Dissipation")
            renderer.save_plot("power_dissipation.png")
            
        # Generate summary
        with monitor.measure_time("summary_generation"):
            # Log summary
            logger.log_message("Simulation completed successfully")
            logger.log_message(f"Best viscosity: {viscosity.min():.6f}")
            logger.log_message(f"Best stress: {stress.min():.6f}")
            logger.log_message(f"Best yield criterion: {yield_criterion.min():.6f}")
            logger.log_message(f"Best power dissipation: {power_dissipation.min():.6f}")
            
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
            context={"simulation": "shear_thinning"}
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