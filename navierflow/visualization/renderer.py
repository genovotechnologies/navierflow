from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class VisualizationType(Enum):
    """Visualization types"""
    SURFACE = "surface"
    VOLUME = "volume"
    STREAMLINE = "streamline"
    ISOSURFACE = "isosurface"
    POINT_CLOUD = "point_cloud"

@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    type: VisualizationType
    colormap: str = "viridis"
    background_color: str = "white"
    show_axes: bool = True
    show_grid: bool = False
    show_legend: bool = True
    show_colorbar: bool = True
    window_size: Tuple[int, int] = (800, 600)
    dpi: int = 100
    animation_fps: int = 30
    animation_duration: float = 10.0
    
    def __post_init__(self):
        """Initialize configuration"""
        if self.window_size[0] <= 0 or self.window_size[1] <= 0:
            raise ValueError("Window size must be positive")
            
        if self.dpi <= 0:
            raise ValueError("DPI must be positive")
            
        if self.animation_fps <= 0:
            raise ValueError("Animation FPS must be positive")
            
        if self.animation_duration <= 0:
            raise ValueError("Animation duration must be positive")

class Renderer:
    def __init__(self, config: VisualizationConfig):
        """
        Initialize renderer
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.plotter = None
        self.fig = None
        
    def render_surface(self,
                      mesh: pv.PolyData,
                      field: Optional[np.ndarray] = None,
                      field_name: Optional[str] = None):
        """
        Render surface
        
        Args:
            mesh: Mesh
            field: Optional field to visualize
            field_name: Optional field name
        """
        # Create plotter
        self.plotter = pv.Plotter(
            window_size=self.config.window_size,
            background=self.config.background_color
        )
        
        # Add mesh
        if field is not None:
            self.plotter.add_mesh(
                mesh,
                scalars=field,
                cmap=self.config.colormap,
                show_edges=True,
                show_scalar_bar=self.config.show_colorbar,
                scalar_bar_args={"title": field_name}
            )
        else:
            self.plotter.add_mesh(
                mesh,
                show_edges=True
            )
            
        # Add axes and grid
        if self.config.show_axes:
            self.plotter.add_axes()
        if self.config.show_grid:
            self.plotter.show_grid()
            
        # Show plot
        self.plotter.show()
        
    def render_volume(self,
                     mesh: pv.UnstructuredGrid,
                     field: np.ndarray,
                     field_name: str,
                     opacity: Optional[float] = None):
        """
        Render volume
        
        Args:
            mesh: Mesh
            field: Field to visualize
            field_name: Field name
            opacity: Optional opacity
        """
        # Create plotter
        self.plotter = pv.Plotter(
            window_size=self.config.window_size,
            background=self.config.background_color
        )
        
        # Add volume
        self.plotter.add_mesh(
            mesh,
            scalars=field,
            cmap=self.config.colormap,
            opacity=opacity,
            show_scalar_bar=self.config.show_colorbar,
            scalar_bar_args={"title": field_name}
        )
        
        # Add axes and grid
        if self.config.show_axes:
            self.plotter.add_axes()
        if self.config.show_grid:
            self.plotter.show_grid()
            
        # Show plot
        self.plotter.show()
        
    def render_streamlines(self,
                          mesh: pv.PolyData,
                          velocity: np.ndarray,
                          start_points: Optional[np.ndarray] = None,
                          max_steps: int = 1000):
        """
        Render streamlines
        
        Args:
            mesh: Mesh
            velocity: Velocity field
            start_points: Optional start points
            max_steps: Maximum number of steps
        """
        # Create plotter
        self.plotter = pv.Plotter(
            window_size=self.config.window_size,
            background=self.config.background_color
        )
        
        # Add mesh
        self.plotter.add_mesh(
            mesh,
            show_edges=True
        )
        
        # Add streamlines
        if start_points is None:
            start_points = mesh.points[::len(mesh.points)//10]
            
        self.plotter.add_mesh(
            mesh.streamlines(
                vectors=velocity,
                start_points=start_points,
                max_steps=max_steps
            ),
            color="black",
            line_width=2
        )
        
        # Add axes and grid
        if self.config.show_axes:
            self.plotter.add_axes()
        if self.config.show_grid:
            self.plotter.show_grid()
            
        # Show plot
        self.plotter.show()
        
    def render_isosurface(self,
                         mesh: pv.UnstructuredGrid,
                         field: np.ndarray,
                         field_name: str,
                         isovalues: Optional[List[float]] = None):
        """
        Render isosurface
        
        Args:
            mesh: Mesh
            field: Field to visualize
            field_name: Field name
            isovalues: Optional isovalues
        """
        # Create plotter
        self.plotter = pv.Plotter(
            window_size=self.config.window_size,
            background=self.config.background_color
        )
        
        # Compute isovalues if not provided
        if isovalues is None:
            isovalues = np.linspace(field.min(), field.max(), 5)
            
        # Add isosurfaces
        for i, isovalue in enumerate(isovalues):
            isosurface = mesh.contour(
                isosurfaces=[isovalue],
                scalars=field
            )
            
            self.plotter.add_mesh(
                isosurface,
                cmap=self.config.colormap,
                show_edges=True,
                show_scalar_bar=self.config.show_colorbar,
                scalar_bar_args={"title": field_name}
            )
            
        # Add axes and grid
        if self.config.show_axes:
            self.plotter.add_axes()
        if self.config.show_grid:
            self.plotter.show_grid()
            
        # Show plot
        self.plotter.show()
        
    def render_point_cloud(self,
                          points: np.ndarray,
                          values: Optional[np.ndarray] = None,
                          value_name: Optional[str] = None):
        """
        Render point cloud
        
        Args:
            points: Points
            values: Optional values
            value_name: Optional value name
        """
        # Create plotter
        self.plotter = pv.Plotter(
            window_size=self.config.window_size,
            background=self.config.background_color
        )
        
        # Create point cloud
        cloud = pv.PolyData(points)
        
        # Add points
        if values is not None:
            self.plotter.add_mesh(
                cloud,
                scalars=values,
                cmap=self.config.colormap,
                point_size=5,
                show_scalar_bar=self.config.show_colorbar,
                scalar_bar_args={"title": value_name}
            )
        else:
            self.plotter.add_mesh(
                cloud,
                point_size=5
            )
            
        # Add axes and grid
        if self.config.show_axes:
            self.plotter.add_axes()
        if self.config.show_grid:
            self.plotter.show_grid()
            
        # Show plot
        self.plotter.show()
        
    def create_animation(self,
                        frames: List[Tuple[pv.PolyData, Optional[np.ndarray]]],
                        output_file: str):
        """
        Create animation
        
        Args:
            frames: List of (mesh, field) tuples
            output_file: Output file
        """
        # Create plotter
        self.plotter = pv.Plotter(
            window_size=self.config.window_size,
            background=self.config.background_color,
            off_screen=True
        )
        
        # Add axes and grid
        if self.config.show_axes:
            self.plotter.add_axes()
        if self.config.show_grid:
            self.plotter.show_grid()
            
        # Open movie file
        self.plotter.open_movie(
            output_file,
            framerate=self.config.animation_fps
        )
        
        # Write frames
        for mesh, field in frames:
            self.plotter.clear()
            
            if field is not None:
                self.plotter.add_mesh(
                    mesh,
                    scalars=field,
                    cmap=self.config.colormap,
                    show_edges=True,
                    show_scalar_bar=self.config.show_colorbar
                )
            else:
                self.plotter.add_mesh(
                    mesh,
                    show_edges=True
                )
                
            self.plotter.write_frame()
            
        # Close movie file
        self.plotter.close()
        
    def create_plot(self,
                   x: np.ndarray,
                   y: np.ndarray,
                   title: Optional[str] = None,
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None):
        """
        Create plot
        
        Args:
            x: X values
            y: Y values
            title: Optional title
            xlabel: Optional x-axis label
            ylabel: Optional y-axis label
        """
        # Create figure
        self.fig, ax = plt.subplots(figsize=self.config.window_size)
        
        # Plot data
        ax.plot(x, y)
        
        # Add labels
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
            
        # Add grid
        if self.config.show_grid:
            ax.grid(True)
            
        # Show plot
        plt.show()
        
    def create_interactive_plot(self,
                              x: np.ndarray,
                              y: np.ndarray,
                              title: Optional[str] = None,
                              xlabel: Optional[str] = None,
                              ylabel: Optional[str] = None):
        """
        Create interactive plot
        
        Args:
            x: X values
            y: Y values
            title: Optional title
            xlabel: Optional x-axis label
            ylabel: Optional y-axis label
        """
        # Create figure
        fig = go.Figure()
        
        # Add trace
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Data"
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            showlegend=self.config.show_legend,
            template="plotly_white"
        )
        
        # Show plot
        fig.show()
        
    def create_multi_plot(self,
                         data: List[Tuple[np.ndarray, np.ndarray, str]],
                         title: Optional[str] = None,
                         xlabel: Optional[str] = None,
                         ylabel: Optional[str] = None):
        """
        Create multi-plot
        
        Args:
            data: List of (x, y, name) tuples
            title: Optional title
            xlabel: Optional x-axis label
            ylabel: Optional y-axis label
        """
        # Create figure
        self.fig, ax = plt.subplots(figsize=self.config.window_size)
        
        # Plot data
        for x, y, name in data:
            ax.plot(x, y, label=name)
            
        # Add labels
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
            
        # Add grid and legend
        if self.config.show_grid:
            ax.grid(True)
        if self.config.show_legend:
            ax.legend()
            
        # Show plot
        plt.show()
        
    def create_interactive_multi_plot(self,
                                    data: List[Tuple[np.ndarray, np.ndarray, str]],
                                    title: Optional[str] = None,
                                    xlabel: Optional[str] = None,
                                    ylabel: Optional[str] = None):
        """
        Create interactive multi-plot
        
        Args:
            data: List of (x, y, name) tuples
            title: Optional title
            xlabel: Optional x-axis label
            ylabel: Optional y-axis label
        """
        # Create figure
        fig = go.Figure()
        
        # Add traces
        for x, y, name in data:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=name
                )
            )
            
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            showlegend=self.config.show_legend,
            template="plotly_white"
        )
        
        # Show plot
        fig.show()
        
    def save_plot(self, filename: str):
        """
        Save plot
        
        Args:
            filename: Output filename
        """
        if self.fig is not None:
            self.fig.savefig(
                filename,
                dpi=self.config.dpi,
                bbox_inches="tight"
            )
            
    def cleanup(self):
        """Cleanup resources"""
        if self.plotter is not None:
            self.plotter.close()
        if self.fig is not None:
            plt.close(self.fig) 