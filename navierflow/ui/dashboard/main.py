import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Optional
import time
from datetime import datetime
import json
import os

from ...core.physics.navier_stokes import NavierStokesSolver, SolverMode, PhysicsModel
from ..visualization.plots import create_velocity_plot, create_pressure_plot, create_temperature_plot
from ..visualization.metrics import MetricsPanel
from ..visualization.controls import ControlPanel
from ..visualization.export import ExportManager

class Dashboard:
    """Modern dashboard interface for NavierFlow"""
    
    def __init__(self):
        self.setup_page()
        self.initialize_session_state()
        self.metrics_panel = MetricsPanel()
        self.control_panel = ControlPanel()
        self.export_manager = ExportManager()
        
        # Theme and styling
        self.dark_theme = {
            'bg_color': '#1E1E1E',
            'text_color': '#FFFFFF',
            'accent_color': '#00A0DC',
            'secondary_color': '#404040'
        }
        
        self.light_theme = {
            'bg_color': '#FFFFFF',
            'text_color': '#000000',
            'accent_color': '#2196F3',
            'secondary_color': '#E0E0E0'
        }
        
        # Default to dark theme
        self.current_theme = self.dark_theme

    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="NavierFlow Dashboard",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for Material Design styling
        st.markdown("""
            <style>
                .stButton>button {
                    background-color: #2196F3;
                    color: white;
                    border-radius: 4px;
                    padding: 0.5rem 1rem;
                    border: none;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    transition: all 0.3s ease;
                }
                .stButton>button:hover {
                    background-color: #1976D2;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                }
                .stProgress > div > div > div {
                    background-color: #2196F3;
                }
                .stSelectbox > div {
                    background-color: #FFFFFF;
                    border-radius: 4px;
                }
                .reportview-container {
                    background: #1E1E1E;
                    color: #FFFFFF;
                }
                .sidebar .sidebar-content {
                    background: #262626;
                }
            </style>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'solver' not in st.session_state:
            st.session_state.solver = NavierStokesSolver(
                width=256,
                height=256,
                mode=SolverMode.HYBRID,
                physics_models=[PhysicsModel.LAMINAR]
            )
        
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
            
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0
            
        if 'history' not in st.session_state:
            st.session_state.history = {
                'time': [],
                'max_velocity': [],
                'avg_pressure': [],
                'max_temperature': []
            }
            
        if 'settings' not in st.session_state:
            st.session_state.settings = {
                'theme': 'dark',
                'visualization_mode': 'velocity',
                'colormap': 'viridis',
                'show_streamlines': True,
                'show_vectors': False,
                'auto_range': True
            }

    def render_header(self):
        """Render dashboard header"""
        col1, col2, col3 = st.columns([3, 6, 3])
        
        with col1:
            st.image("logo.png", width=100)
            
        with col2:
            st.title("NavierFlow Dashboard")
            st.markdown("Advanced Fluid Dynamics Simulation")
            
        with col3:
            if st.button("⚙️ Settings"):
                self.show_settings_modal()

    def render_sidebar(self):
        """Render control sidebar"""
        with st.sidebar:
            st.subheader("Simulation Controls")
            
            # Physics models
            st.markdown("### Physics Models")
            physics_models = st.multiselect(
                "Select Physics Models",
                ["Laminar", "Turbulent", "Heat Transfer", "Multiphase"],
                default=["Laminar"]
            )
            
            # Solver mode
            st.markdown("### Solver Mode")
            solver_mode = st.selectbox(
                "Select Solver Mode",
                ["Real-time", "High Accuracy", "Hybrid"],
                index=2
            )
            
            # Numerical parameters
            st.markdown("### Numerical Parameters")
            dt = st.slider("Time Step (dt)", 0.001, 0.1, 0.01)
            iterations = st.slider("Pressure Iterations", 10, 200, 50)
            
            # Boundary conditions
            st.markdown("### Boundary Conditions")
            self.control_panel.render_boundary_conditions()
            
            # Material properties
            st.markdown("### Material Properties")
            self.control_panel.render_material_properties()
            
            # Export options
            st.markdown("### Export Options")
            self.export_manager.render_export_controls()

    def render_main_view(self):
        """Render main visualization area"""
        col1, col2 = st.columns([7, 3])
        
        with col1:
            # Visualization tabs
            tabs = st.tabs(["Velocity", "Pressure", "Temperature", "Vorticity"])
            
            with tabs[0]:
                fig = create_velocity_plot(
                    st.session_state.solver.velocity.to_numpy(),
                    colormap=st.session_state.settings['colormap'],
                    show_streamlines=st.session_state.settings['show_streamlines'],
                    show_vectors=st.session_state.settings['show_vectors']
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tabs[1]:
                fig = create_pressure_plot(
                    st.session_state.solver.pressure.to_numpy(),
                    colormap=st.session_state.settings['colormap']
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tabs[2]:
                fig = create_temperature_plot(
                    st.session_state.solver.temperature.to_numpy(),
                    colormap=st.session_state.settings['colormap']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Simulation controls
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("▶️ Start" if not st.session_state.simulation_running else "⏸️ Pause"):
                    st.session_state.simulation_running = not st.session_state.simulation_running
            with col2:
                if st.button("⏹️ Reset"):
                    self.reset_simulation()
            with col3:
                if st.button("💾 Save State"):
                    self.save_simulation_state()
        
        with col2:
            # Real-time metrics
            st.markdown("### Performance Metrics")
            self.metrics_panel.render_metrics(st.session_state.solver.get_state())
            
            # Time series plots
            st.markdown("### Time Series")
            self.render_time_series()

    def render_time_series(self):
        """Render time series plots of key metrics"""
        history = st.session_state.history
        
        # Velocity plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history['time'],
            y=history['max_velocity'],
            mode='lines',
            name='Max Velocity'
        ))
        fig.update_layout(
            title="Maximum Velocity vs Time",
            xaxis_title="Time (s)",
            yaxis_title="Velocity (m/s)",
            template="plotly_dark" if st.session_state.settings['theme'] == 'dark' else "plotly"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Pressure plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history['time'],
            y=history['avg_pressure'],
            mode='lines',
            name='Average Pressure'
        ))
        fig.update_layout(
            title="Average Pressure vs Time",
            xaxis_title="Time (s)",
            yaxis_title="Pressure (Pa)",
            template="plotly_dark" if st.session_state.settings['theme'] == 'dark' else "plotly"
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_settings_modal(self):
        """Show settings modal dialog"""
        with st.expander("Settings", expanded=True):
            # Theme selection
            theme = st.selectbox(
                "Theme",
                ["Dark", "Light"],
                index=0 if st.session_state.settings['theme'] == 'dark' else 1
            )
            st.session_state.settings['theme'] = theme.lower()
            
            # Visualization settings
            st.markdown("### Visualization Settings")
            st.session_state.settings['colormap'] = st.selectbox(
                "Color Map",
                ["viridis", "plasma", "inferno", "magma", "cividis"]
            )
            st.session_state.settings['show_streamlines'] = st.checkbox(
                "Show Streamlines",
                value=st.session_state.settings['show_streamlines']
            )
            st.session_state.settings['show_vectors'] = st.checkbox(
                "Show Velocity Vectors",
                value=st.session_state.settings['show_vectors']
            )
            st.session_state.settings['auto_range'] = st.checkbox(
                "Auto Range",
                value=st.session_state.settings['auto_range']
            )
            
            # Performance settings
            st.markdown("### Performance Settings")
            st.slider("Maximum FPS", 1, 60, 30)
            st.checkbox("Enable GPU Acceleration", value=True)
            
            # Export settings
            st.markdown("### Export Settings")
            st.selectbox("Export Format", ["MP4", "GIF", "PNG Sequence"])
            st.text_input("Export Directory", "exports/")

    def reset_simulation(self):
        """Reset simulation to initial state"""
        st.session_state.solver.initialize_fields()
        st.session_state.current_step = 0
        st.session_state.history = {
            'time': [],
            'max_velocity': [],
            'avg_pressure': [],
            'max_temperature': []
        }
        st.session_state.simulation_running = False

    def save_simulation_state(self):
        """Save current simulation state"""
        state = st.session_state.solver.get_state()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create exports directory if it doesn't exist
        os.makedirs("exports", exist_ok=True)
        
        # Save state as NPZ file
        np.savez(
            f"exports/state_{timestamp}.npz",
            velocity=state['velocity'],
            pressure=state['pressure'],
            temperature=state['temperature']
        )
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'step': st.session_state.current_step,
            'settings': st.session_state.settings,
            'metrics': state['metrics']
        }
        
        with open(f"exports/metadata_{timestamp}.json", 'w') as f:
            json.dump(metadata, f, indent=4)
            
        st.success(f"Simulation state saved to exports/state_{timestamp}.npz")

    def update_simulation(self):
        """Update simulation state"""
        if st.session_state.simulation_running:
            state = st.session_state.solver.step()
            
            # Update history
            st.session_state.history['time'].append(st.session_state.current_step * 0.01)
            st.session_state.history['max_velocity'].append(
                np.max(np.linalg.norm(state['velocity'], axis=2))
            )
            st.session_state.history['avg_pressure'].append(
                np.mean(state['pressure'])
            )
            st.session_state.history['max_temperature'].append(
                np.max(state['temperature'])
            )
            
            st.session_state.current_step += 1

    def run(self):
        """Main dashboard loop"""
        self.render_header()
        self.render_sidebar()
        self.render_main_view()
        
        # Update simulation if running
        self.update_simulation()
        
        # Rerun the app
        time.sleep(0.01)
        st.experimental_rerun()

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run() 