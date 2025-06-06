import numpy as np
import matplotlib.pyplot as plt
from navierflow.core.physics.multiphase import MultiphaseFlow
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
        log_file="rising_bubble.log",
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
            "phases": ["water", "air"],
            "densities": [1000.0, 1.0],
            "viscosities": [0.001, 0.00001],
            "surface_tension": 0.072
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
            "resolution": (100, 200)
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
        logger.start_simulation("Rising bubble simulation started")
        
        # Create output directory
        manager.create_output_dir()
        
        # Initialize mesh generator
        with monitor.measure_time("mesh_generation"):
            mesh = MeshGenerator(
                mesh_type="structured",
                dimension=2,
                resolution=(100, 200)
            )
            
            # Generate mesh
            mesh = mesh.generate_structured_mesh(
                bounds=((0, 0), (1, 2)),
                periodic=(False, False)
            )
            
        # Initialize multiphase flow
        multiphase = MultiphaseFlow(
            phases=["water", "air"],
            densities=[1000.0, 1.0],
            viscosities=[0.001, 0.00001],
            surface_tension=0.072
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
        level_set = np.zeros((100, 200))
        x, y = np.meshgrid(
            np.linspace(0, 1, 100),
            np.linspace(0, 2, 200)
        )
        level_set = np.sqrt((x - 0.5)**2 + (y - 0.5)**2) - 0.2
        
        # Solve
        with monitor.measure_time("solving"):
            solution = solver.solve(level_set)
            
        # Compute volume fraction
        with monitor.measure_time("volume_fraction_computation"):
            volume_fraction = multiphase.compute_volume_fraction(solution)
            
        # Compute interface normal
        with monitor.measure_time("interface_normal_computation"):
            normal = multiphase.compute_interface_normal(solution)
            
        # Compute curvature
        with monitor.measure_time("curvature_computation"):
            curvature = multiphase.compute_curvature(solution)
            
        # Compute surface tension
        with monitor.measure_time("surface_tension_computation"):
            surface_tension = multiphase.compute_surface_tension(solution)
            
        # Validate results
        with monitor.measure_time("validation"):
            # Validate volume fraction
            validator.validate_parameter(
                name="volume_fraction",
                value=volume_fraction.mean(),
                expected=0.5,
                tolerance=1e-6
            )
            
            # Validate interface normal
            validator.validate_parameter(
                name="interface_normal",
                value=normal.mean(),
                expected=0.0,
                tolerance=1e-6
            )
            
            # Validate curvature
            validator.validate_parameter(
                name="curvature",
                value=curvature.mean(),
                expected=5.0,
                tolerance=1e-6
            )
            
            # Validate surface tension
            validator.validate_parameter(
                name="surface_tension",
                value=surface_tension.mean(),
                expected=0.072,
                tolerance=1e-6
            )
            
        # Render results
        with monitor.measure_time("visualization"):
            # Render volume fraction
            renderer.render_surface(mesh, volume_fraction, "Volume Fraction")
            renderer.save_plot("volume_fraction.png")
            
            # Render interface normal
            renderer.render_surface(mesh, normal[..., 0], "Interface Normal X")
            renderer.save_plot("interface_normal_x.png")
            
            # Render curvature
            renderer.render_surface(mesh, curvature, "Curvature")
            renderer.save_plot("curvature.png")
            
            # Render surface tension
            renderer.render_surface(mesh, surface_tension[..., 0], "Surface Tension X")
            renderer.save_plot("surface_tension_x.png")
            
        # Generate summary
        with monitor.measure_time("summary_generation"):
            # Log summary
            logger.log_message("Simulation completed successfully")
            logger.log_message(f"Best volume fraction: {volume_fraction.min():.6f}")
            logger.log_message(f"Best interface normal: {normal.min():.6f}")
            logger.log_message(f"Best curvature: {curvature.min():.6f}")
            logger.log_message(f"Best surface tension: {surface_tension.min():.6f}")
            
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
            context={"simulation": "rising_bubble"}
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