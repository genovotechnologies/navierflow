import streamlit as st
import numpy as np
from typing import Dict, List, Optional
import plotly.graph_objects as go
from datetime import datetime, timedelta

class MetricsPanel:
    """Panel for displaying simulation metrics and performance statistics"""
    
    def __init__(self):
        self.history_length = 100  # Number of historical data points to keep
        self.initialize_metrics()

    def initialize_metrics(self):
        """Initialize metrics storage"""
        if 'metrics_history' not in st.session_state:
            st.session_state.metrics_history = {
                'time': [],
                'performance': {
                    'solve_time': [],
                    'mesh_update_time': [],
                    'memory_usage': []
                },
                'physics': {
                    'max_velocity': [],
                    'avg_pressure': [],
                    'max_temperature': [],
                    'total_energy': [],
                    'enstrophy': []
                },
                'numerics': {
                    'residual': [],
                    'iterations': [],
                    'cfl_number': []
                }
            }

    def update_metrics(self, state: Dict):
        """Update metrics with new simulation state"""
        metrics_history = st.session_state.metrics_history
        
        # Update time
        current_time = datetime.now()
        metrics_history['time'].append(current_time)
        
        # Update performance metrics
        metrics_history['performance']['solve_time'].append(state['metrics']['solve_time'])
        metrics_history['performance']['mesh_update_time'].append(state['metrics']['mesh_update_time'])
        metrics_history['performance']['memory_usage'].append(state['metrics']['memory_usage'])
        
        # Update physics metrics
        velocity_mag = np.sqrt(np.sum(state['velocity']**2, axis=2))
        metrics_history['physics']['max_velocity'].append(np.max(velocity_mag))
        metrics_history['physics']['avg_pressure'].append(np.mean(state['pressure']))
        metrics_history['physics']['max_temperature'].append(np.max(state['temperature']))
        
        # Compute total kinetic energy
        energy = 0.5 * np.sum(velocity_mag**2)
        metrics_history['physics']['total_energy'].append(energy)
        
        # Compute enstrophy (if vorticity is available)
        if 'vorticity' in state:
            enstrophy = 0.5 * np.sum(state['vorticity']**2)
            metrics_history['physics']['enstrophy'].append(enstrophy)
        else:
            metrics_history['physics']['enstrophy'].append(0.0)
        
        # Trim history if too long
        if len(metrics_history['time']) > self.history_length:
            for category in metrics_history.values():
                if isinstance(category, dict):
                    for metric in category.values():
                        metric.pop(0)
                else:
                    category.pop(0)

    def render_metrics(self, state: Dict):
        """Render metrics panel"""
        # Update metrics
        self.update_metrics(state)
        
        # Performance metrics
        st.markdown("#### Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_performance_metrics()
        
        with col2:
            self.render_memory_usage()
        
        # Physics metrics
        st.markdown("#### Physics Metrics")
        self.render_physics_metrics()
        
        # Numerical metrics
        st.markdown("#### Numerical Metrics")
        self.render_numerical_metrics()
        
        # Time series plots
        st.markdown("#### Time Series")
        self.render_time_series()

    def render_performance_metrics(self):
        """Render performance metrics section"""
        metrics = st.session_state.metrics_history['performance']
        
        # Compute statistics
        avg_solve_time = np.mean(metrics['solve_time'][-10:])  # Average of last 10 steps
        avg_mesh_time = np.mean(metrics['mesh_update_time'][-10:])
        fps = 1.0 / (avg_solve_time + avg_mesh_time) if (avg_solve_time + avg_mesh_time) > 0 else 0
        
        # Display metrics
        st.metric(
            label="Simulation FPS",
            value=f"{fps:.1f}",
            delta=f"{fps - 1.0/np.mean(metrics['solve_time'][-20:-10]):.1f}"
        )
        
        st.metric(
            label="Solve Time (ms)",
            value=f"{avg_solve_time*1000:.1f}",
            delta=f"{(avg_solve_time - np.mean(metrics['solve_time'][-20:-10]))*1000:.1f}"
        )

    def render_memory_usage(self):
        """Render memory usage metrics"""
        metrics = st.session_state.metrics_history['performance']
        current_memory = metrics['memory_usage'][-1]
        
        st.metric(
            label="Memory Usage (GB)",
            value=f"{current_memory:.2f}",
            delta=f"{current_memory - metrics['memory_usage'][-2]:.2f}"
        )
        
        # Memory usage gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_memory,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 16]},  # Assume 16GB max
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 4], 'color': "lightgray"},
                    {'range': [4, 8], 'color': "gray"},
                    {'range': [8, 16], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 14
                }
            }
        ))
        
        fig.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    def render_physics_metrics(self):
        """Render physics metrics section"""
        metrics = st.session_state.metrics_history['physics']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Max Velocity (m/s)",
                value=f"{metrics['max_velocity'][-1]:.2f}",
                delta=f"{metrics['max_velocity'][-1] - metrics['max_velocity'][-2]:.2f}"
            )
            
        with col2:
            st.metric(
                label="Avg Pressure (Pa)",
                value=f"{metrics['avg_pressure'][-1]:.2f}",
                delta=f"{metrics['avg_pressure'][-1] - metrics['avg_pressure'][-2]:.2f}"
            )
            
        with col3:
            st.metric(
                label="Max Temperature (K)",
                value=f"{metrics['max_temperature'][-1]:.1f}",
                delta=f"{metrics['max_temperature'][-1] - metrics['max_temperature'][-2]:.1f}"
            )
        
        # Energy conservation plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=metrics['total_energy'],
            mode='lines',
            name='Total Energy'
        ))
        
        fig.update_layout(
            title="Energy Conservation",
            xaxis_title="Time Step",
            yaxis_title="Total Energy",
            height=200,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_numerical_metrics(self):
        """Render numerical metrics section"""
        metrics = st.session_state.metrics_history['numerics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residual convergence
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=metrics['residual'],
                mode='lines',
                name='Residual'
            ))
            
            fig.update_layout(
                title="Residual Convergence",
                yaxis_type="log",
                height=200,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # CFL number distribution
            if len(metrics['cfl_number']) > 0:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=metrics['cfl_number'][-1],
                    nbinsx=30,
                    name='CFL Distribution'
                ))
                
                fig.update_layout(
                    title="CFL Number Distribution",
                    height=200,
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)

    def render_time_series(self):
        """Render time series plots"""
        metrics = st.session_state.metrics_history
        
        # Create time series figure
        fig = go.Figure()
        
        # Add traces for different metrics
        fig.add_trace(go.Scatter(
            x=metrics['time'],
            y=metrics['physics']['max_velocity'],
            mode='lines',
            name='Max Velocity'
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics['time'],
            y=metrics['physics']['avg_pressure'],
            mode='lines',
            name='Avg Pressure'
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics['time'],
            y=metrics['physics']['enstrophy'],
            mode='lines',
            name='Enstrophy'
        ))
        
        # Update layout
        fig.update_layout(
            title="Time Evolution of Key Metrics",
            xaxis_title="Time",
            yaxis_title="Value",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def export_metrics(self, filename: str):
        """Export metrics to file"""
        metrics = st.session_state.metrics_history
        
        # Convert to numpy arrays for saving
        export_data = {
            'time': np.array([t.timestamp() for t in metrics['time']]),
            'performance': {
                key: np.array(value) for key, value in metrics['performance'].items()
            },
            'physics': {
                key: np.array(value) for key, value in metrics['physics'].items()
            },
            'numerics': {
                key: np.array(value) for key, value in metrics['numerics'].items()
            }
        }
        
        np.savez(filename, **export_data) 