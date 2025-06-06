from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import trimesh
import pyvista as pv
from scipy.spatial import Delaunay

class MeshType(Enum):
    """Mesh types"""
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    ADAPTIVE = "adaptive"
    CURVED = "curved"

@dataclass
class MeshConfig:
    """Mesh configuration"""
    type: MeshType
    dimension: int = 3
    resolution: float = 0.1
    refinement_level: int = 0
    quality_threshold: float = 0.3
    boundary_layers: int = 1
    growth_rate: float = 1.2
    
    def __post_init__(self):
        """Initialize configuration"""
        if self.dimension not in [2, 3]:
            raise ValueError("Dimension must be 2 or 3")
            
        if self.resolution <= 0:
            raise ValueError("Resolution must be positive")
            
        if self.refinement_level < 0:
            raise ValueError("Refinement level must be non-negative")
            
        if self.quality_threshold <= 0 or self.quality_threshold > 1:
            raise ValueError("Quality threshold must be in (0, 1]")
            
        if self.boundary_layers < 0:
            raise ValueError("Number of boundary layers must be non-negative")
            
        if self.growth_rate <= 1:
            raise ValueError("Growth rate must be greater than 1")

class MeshGenerator:
    def __init__(self, config: MeshConfig):
        """
        Initialize mesh generator
        
        Args:
            config: Mesh configuration
        """
        self.config = config
        self.mesh = None
        
    def generate_structured_mesh(self,
                               bounds: Tuple[Tuple[float, float], ...],
                               periodic: Optional[List[bool]] = None):
        """
        Generate structured mesh
        
        Args:
            bounds: Domain bounds
            periodic: Optional periodic flags
        """
        # Check dimensions
        if len(bounds) != self.config.dimension:
            raise ValueError("Number of bounds must match dimension")
            
        # Initialize periodic flags
        if periodic is None:
            periodic = [False] * self.config.dimension
            
        # Create grid points
        points = []
        for i, (min_val, max_val) in enumerate(bounds):
            num_points = int((max_val - min_val) / self.config.resolution) + 1
            points.append(np.linspace(min_val, max_val, num_points))
            
        # Create mesh
        if self.config.dimension == 2:
            x, y = np.meshgrid(*points)
            self.mesh = pv.StructuredGrid(x, y)
        else:
            x, y, z = np.meshgrid(*points)
            self.mesh = pv.StructuredGrid(x, y, z)
            
        # Set periodic boundaries
        self.mesh.set_periodic_boundaries(periodic)
        
    def generate_unstructured_mesh(self,
                                 points: np.ndarray,
                                 boundary: Optional[np.ndarray] = None):
        """
        Generate unstructured mesh
        
        Args:
            points: Mesh points
            boundary: Optional boundary points
        """
        # Check dimensions
        if points.shape[1] != self.config.dimension:
            raise ValueError("Point dimension must match mesh dimension")
            
        # Create mesh
        if self.config.dimension == 2:
            # Create Delaunay triangulation
            tri = Delaunay(points)
            self.mesh = pv.PolyData(points, tri.simplices)
        else:
            # Create tetrahedral mesh
            self.mesh = pv.PolyData(points)
            self.mesh.delaunay_3d()
            
        # Add boundary if provided
        if boundary is not None:
            self.mesh.set_boundary(boundary)
            
    def generate_adaptive_mesh(self,
                             points: np.ndarray,
                             metric: Optional[np.ndarray] = None):
        """
        Generate adaptive mesh
        
        Args:
            points: Initial mesh points
            metric: Optional metric field
        """
        # Generate initial mesh
        self.generate_unstructured_mesh(points)
        
        # Refine mesh
        for _ in range(self.config.refinement_level):
            # Compute error indicators
            if metric is not None:
                error = self._compute_error_indicators(metric)
            else:
                error = self._compute_geometric_error()
                
            # Mark cells for refinement
            marked_cells = self._mark_cells_for_refinement(error)
            
            # Refine marked cells
            self._refine_cells(marked_cells)
            
    def generate_curved_mesh(self,
                           points: np.ndarray,
                           boundary: np.ndarray,
                           curvature: Optional[np.ndarray] = None):
        """
        Generate curved mesh
        
        Args:
            points: Mesh points
            boundary: Boundary points
            curvature: Optional curvature field
        """
        # Generate initial mesh
        self.generate_unstructured_mesh(points, boundary)
        
        # Add boundary layers
        for _ in range(self.config.boundary_layers):
            self._add_boundary_layer()
            
        # Apply curvature if provided
        if curvature is not None:
            self._apply_curvature(curvature)
            
    def _compute_error_indicators(self, metric: np.ndarray) -> np.ndarray:
        """
        Compute error indicators
        
        Args:
            metric: Metric field
            
        Returns:
            Error indicators
        """
        # Compute gradient of metric
        gradient = np.gradient(metric)
        
        # Compute error indicators
        error = np.zeros(self.mesh.n_cells)
        for i in range(self.config.dimension):
            error += np.abs(gradient[i]).mean(axis=1)
            
        return error
        
    def _compute_geometric_error(self) -> np.ndarray:
        """
        Compute geometric error
        
        Returns:
            Geometric error
        """
        # Compute cell quality
        quality = self.mesh.compute_cell_quality()
        
        # Compute error
        error = 1 - quality
        
        return error
        
    def _mark_cells_for_refinement(self, error: np.ndarray) -> np.ndarray:
        """
        Mark cells for refinement
        
        Args:
            error: Error indicators
            
        Returns:
            Marked cells
        """
        # Compute threshold
        threshold = np.percentile(error, 90)
        
        # Mark cells
        marked_cells = error > threshold
        
        return marked_cells
        
    def _refine_cells(self, marked_cells: np.ndarray):
        """
        Refine cells
        
        Args:
            marked_cells: Marked cells
        """
        # Get cell centers
        centers = self.mesh.cell_centers()
        
        # Add new points
        new_points = centers[marked_cells]
        self.mesh.points = np.vstack([self.mesh.points, new_points])
        
        # Update connectivity
        self.mesh._rebuild()
        
    def _add_boundary_layer(self):
        """Add boundary layer"""
        # Get boundary cells
        boundary_cells = self.mesh.extract_surface().cell_centers()
        
        # Compute normal vectors
        normals = self.mesh.compute_normals()
        
        # Add new points
        new_points = boundary_cells + normals * self.config.resolution
        self.mesh.points = np.vstack([self.mesh.points, new_points])
        
        # Update connectivity
        self.mesh._rebuild()
        
    def _apply_curvature(self, curvature: np.ndarray):
        """
        Apply curvature
        
        Args:
            curvature: Curvature field
        """
        # Get boundary points
        boundary_points = self.mesh.extract_surface().points
        
        # Compute normal vectors
        normals = self.mesh.compute_normals()
        
        # Apply curvature
        displacement = normals * curvature[:, np.newaxis]
        self.mesh.points += displacement
        
    def get_mesh(self) -> pv.PolyData:
        """
        Get mesh
        
        Returns:
            Mesh
        """
        return self.mesh
        
    def save_mesh(self, filename: str):
        """
        Save mesh
        
        Args:
            filename: Output filename
        """
        self.mesh.save(filename)
        
    def load_mesh(self, filename: str):
        """
        Load mesh
        
        Args:
            filename: Input filename
        """
        self.mesh = pv.read(filename)
        
    def get_boundary(self) -> pv.PolyData:
        """
        Get boundary mesh
        
        Returns:
            Boundary mesh
        """
        return self.mesh.extract_surface()
        
    def get_cell_quality(self) -> np.ndarray:
        """
        Get cell quality
        
        Returns:
            Cell quality
        """
        return self.mesh.compute_cell_quality()
        
    def get_cell_centers(self) -> np.ndarray:
        """
        Get cell centers
        
        Returns:
            Cell centers
        """
        return self.mesh.cell_centers()
        
    def get_cell_volumes(self) -> np.ndarray:
        """
        Get cell volumes
        
        Returns:
            Cell volumes
        """
        return self.mesh.compute_cell_sizes()
        
    def get_cell_normals(self) -> np.ndarray:
        """
        Get cell normals
        
        Returns:
            Cell normals
        """
        return self.mesh.compute_normals()
        
    def get_cell_gradients(self, field: np.ndarray) -> np.ndarray:
        """
        Get cell gradients
        
        Args:
            field: Field to compute gradients for
            
        Returns:
            Cell gradients
        """
        return self.mesh.compute_gradient(field)
        
    def get_cell_laplacians(self, field: np.ndarray) -> np.ndarray:
        """
        Get cell Laplacians
        
        Args:
            field: Field to compute Laplacians for
            
        Returns:
            Cell Laplacians
        """
        return self.mesh.compute_laplacian(field) 