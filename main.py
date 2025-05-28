import argparse
import taichi as ti
from pathlib import Path
from navierflow.ui.visualization.simulation_interface import GUIManager, SimulationManager
from navierflow.ui.dashboard.main import Dashboard
from navierflow.ai.ai_integration import SimulationEnhancer
from navierflow.utils import setup_logging, create_logger, handle_simulation_error

def run_visualization_mode(width: int = 800, height: int = 800):
    """Run the application in visualization mode with interactive GUI"""
    logger = create_logger("visualization")
    try:
        # Initialize Taichi
        ti.init(arch=ti.vulkan, default_fp=ti.f32)
        
        gui_manager = GUIManager(width, height)
        start_screen = gui_manager.start_screen
        sim_manager = None

        while gui_manager.window.running:
            if not start_screen.start_simulation:
                start_screen.render()
                gui_manager.window.show()
            else:
                if sim_manager is None:
                    sim_manager = SimulationManager(width, height, start_screen.brush_size)
                    sim_manager.method = start_screen.selected_method
                    logger.info(f"Initialized simulation with method: {start_screen.selected_method}")

                mouse_pos = gui_manager.window.get_cursor_pos()
                mouse_pos = (mouse_pos[0] * width, mouse_pos[1] * height)
                mouse_down = gui_manager.window.is_pressed(ti.ui.LMB)

                sim_manager.update(mouse_pos, mouse_down)
                gui_manager.render(sim_manager)
    except Exception as e:
        handle_simulation_error(e, logger)
        raise

def run_dashboard_mode():
    """Run the application in dashboard mode with analytics"""
    logger = create_logger("dashboard")
    try:
        dashboard = Dashboard()
        dashboard.run()
    except Exception as e:
        handle_simulation_error(e, logger)
        raise

def main():
    parser = argparse.ArgumentParser(description="NavierFlow - Advanced Fluid Dynamics Simulation")
    parser.add_argument(
        "--mode",
        choices=["visualization", "dashboard"],
        default="visualization",
        help="Application mode: visualization (interactive GUI) or dashboard (analytics)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Window width for visualization mode"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Window height for visualization mode"
    )
    parser.add_argument(
        "--enable-ai",
        action="store_true",
        help="Enable AI-powered simulation enhancement"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )

    args = parser.parse_args()
    setup_logging(getattr(logging, args.log_level))
    logger = create_logger(__name__)

    try:
        if args.enable_ai:
            # Initialize AI enhancement if enabled
            enhancer = SimulationEnhancer()
            logger.info("AI-powered simulation enhancement enabled")

        if args.mode == "visualization":
            logger.info("Starting NavierFlow in visualization mode")
            run_visualization_mode(args.width, args.height)
        else:
            logger.info("Starting NavierFlow in dashboard mode")
            run_dashboard_mode()

    except Exception as e:
        handle_simulation_error(e, logger)
        raise SystemExit(1)

if __name__ == "__main__":
    main() 