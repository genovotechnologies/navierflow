import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import matplotlib.colors as colors

class SimulationPlotter:
    def __init__(self, mesh):
        """
        Initialize simulation plotter
        
        Args:
            mesh: Computational mesh object
        """
        self.mesh = mesh
        
    def plot_scalar_field(self,
                         field: np.ndarray,
                         title: str = "",
                         cmap: str = "viridis",
                         show_mesh: bool = False,
                         save_path: Optional[str] = None):
        """
        Plot scalar field on mesh
        
        Args:
            field: Scalar field values
            title: Plot title
            cmap: Colormap name
            show_mesh: Whether to show mesh lines
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(10, 8))
        
        # Create triangulation
        if self.mesh.vertices.shape[1] == 2:
            plt.tripcolor(self.mesh.vertices[:, 0],
                         self.mesh.vertices[:, 1],
                         self.mesh.cells,
                         field,
                         cmap=cmap,
                         shading='flat')
            
            if show_mesh:
                plt.triplot(self.mesh.vertices[:, 0],
                           self.mesh.vertices[:, 1],
                           self.mesh.cells,
                           'k-',
                           linewidth=0.5)
                
        plt.colorbar(label=title)
        plt.title(title)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_vector_field(self,
                         field: np.ndarray,
                         title: str = "",
                         scale: float = 1.0,
                         show_mesh: bool = False,
                         save_path: Optional[str] = None):
        """
        Plot vector field on mesh
        
        Args:
            field: Vector field values
            title: Plot title
            scale: Arrow scale factor
            show_mesh: Whether to show mesh lines
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(10, 8))
        
        if self.mesh.vertices.shape[1] == 2:
            # Plot vector field
            plt.quiver(self.mesh.vertices[:, 0],
                      self.mesh.vertices[:, 1],
                      field[:, 0],
                      field[:, 1],
                      scale=scale)
            
            if show_mesh:
                plt.triplot(self.mesh.vertices[:, 0],
                           self.mesh.vertices[:, 1],
                           self.mesh.cells,
                           'k-',
                           linewidth=0.5)
                
        plt.title(title)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_streamlines(self,
                        velocity: np.ndarray,
                        title: str = "",
                        density: float = 1.0,
                        show_mesh: bool = False,
                        save_path: Optional[str] = None):
        """
        Plot streamlines of velocity field
        
        Args:
            velocity: Velocity field values
            title: Plot title
            density: Streamline density
            show_mesh: Whether to show mesh lines
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(10, 8))
        
        if self.mesh.vertices.shape[1] == 2:
            # Create regular grid for streamlines
            x = np.linspace(self.mesh.vertices[:, 0].min(),
                          self.mesh.vertices[:, 0].max(),
                          100)
            y = np.linspace(self.mesh.vertices[:, 1].min(),
                          self.mesh.vertices[:, 1].max(),
                          100)
            X, Y = np.meshgrid(x, y)
            
            # Interpolate velocity to grid
            U = self.mesh.interpolate_to_vertices(velocity[:, 0])
            V = self.mesh.interpolate_to_vertices(velocity[:, 1])
            
            # Plot streamlines
            plt.streamplot(X, Y, U, V,
                         density=density,
                         color='b',
                         linewidth=1)
            
            if show_mesh:
                plt.triplot(self.mesh.vertices[:, 0],
                           self.mesh.vertices[:, 1],
                           self.mesh.cells,
                           'k-',
                           linewidth=0.5)
                
        plt.title(title)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_time_series(self,
                        time: np.ndarray,
                        values: np.ndarray,
                        labels: List[str],
                        title: str = "",
                        save_path: Optional[str] = None):
        """
        Plot time series data
        
        Args:
            time: Time points
            values: Array of values at each time point
            labels: Labels for each value series
            title: Plot title
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(10, 6))
        
        for i, label in enumerate(labels):
            plt.plot(time, values[:, i], label=label)
            
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() 