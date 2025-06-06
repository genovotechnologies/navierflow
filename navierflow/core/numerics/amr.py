import numpy as np
from typing import List, Tuple, Optional, Callable
from .mesh import Mesh, Cell

class AMR:
    def __init__(self,
                 mesh: Mesh,
                 max_level: int = 3,
                 refine_threshold: float = 0.1,
                 coarsen_threshold: float = 0.01):
        """
        Initialize adaptive mesh refinement
        
        Args:
            mesh: Initial mesh
            max_level: Maximum refinement level
            refine_threshold: Threshold for refinement
            coarsen_threshold: Threshold for coarsening
        """
        self.mesh = mesh
        self.max_level = max_level
        self.refine_threshold = refine_threshold
        self.coarsen_threshold = coarsen_threshold
        self.cell_levels = np.zeros(len(mesh.cells), dtype=int)
        
    def estimate_error(self,
                      field: np.ndarray,
                      method: str = "gradient") -> np.ndarray:
        """
        Estimate error in field
        
        Args:
            field: Field to estimate error for
            method: Error estimation method
            
        Returns:
            Error estimate for each cell
        """
        if method == "gradient":
            return self._gradient_based_error(field)
        elif method == "recovery":
            return self._recovery_based_error(field)
        else:
            raise ValueError(f"Unknown error estimation method: {method}")
            
    def _gradient_based_error(self, field: np.ndarray) -> np.ndarray:
        """Compute gradient-based error estimate"""
        errors = np.zeros(len(self.mesh.cells))
        
        for i, cell in enumerate(self.mesh.cells):
            # Get field values at cell vertices
            vertex_values = field[cell.vertices]
            
            # Compute gradient
            grad = np.gradient(vertex_values)
            
            # Error is maximum gradient magnitude
            errors[i] = np.max(np.abs(grad))
            
        return errors
        
    def _recovery_based_error(self, field: np.ndarray) -> np.ndarray:
        """Compute recovery-based error estimate"""
        errors = np.zeros(len(self.mesh.cells))
        
        for i, cell in enumerate(self.mesh.cells):
            # Get field values at cell vertices
            vertex_values = field[cell.vertices]
            
            # Compute recovered gradient using superconvergent patch recovery
            recovered_grad = self._recover_gradient(cell, vertex_values)
            
            # Compute actual gradient
            actual_grad = np.gradient(vertex_values)
            
            # Error is difference between recovered and actual gradient
            errors[i] = np.max(np.abs(recovered_grad - actual_grad))
            
        return errors
        
    def _recover_gradient(self, cell: Cell, values: np.ndarray) -> np.ndarray:
        """Recover gradient using superconvergent patch recovery"""
        # Get neighboring cells
        neighbors = [self.mesh.cells[j] for j in cell.neighbors]
        
        # Collect all vertices in patch
        patch_vertices = set(cell.vertices)
        for neighbor in neighbors:
            patch_vertices.update(neighbor.vertices)
            
        # Compute least squares fit for gradient
        A = np.zeros((len(patch_vertices), 3))
        b = np.zeros(len(patch_vertices))
        
        for i, vertex in enumerate(patch_vertices):
            A[i, 0] = 1
            A[i, 1:] = self.mesh.vertices[vertex]
            b[i] = values[vertex]
            
        # Solve least squares problem
        grad = np.linalg.lstsq(A, b, rcond=None)[0][1:]
        
        return grad
        
    def adapt_mesh(self,
                  field: np.ndarray,
                  error_method: str = "gradient") -> Mesh:
        """
        Adapt mesh based on error estimate
        
        Args:
            field: Field to base adaptation on
            error_method: Error estimation method
            
        Returns:
            New adapted mesh
        """
        # Estimate error
        errors = self.estimate_error(field, error_method)
        
        # Normalize errors
        errors = errors / np.max(errors)
        
        # Mark cells for refinement/coarsening
        refine_cells = errors > self.refine_threshold
        coarsen_cells = errors < self.coarsen_threshold
        
        # Create new mesh
        new_vertices = self.mesh.vertices.copy()
        new_cells = []
        new_levels = []
        
        # Process each cell
        for i, cell in enumerate(self.mesh.cells):
            if refine_cells[i] and self.cell_levels[i] < self.max_level:
                # Refine cell
                refined_vertices, refined_cells = self._refine_cell(cell, new_vertices)
                new_vertices = np.vstack((new_vertices, refined_vertices))
                new_cells.extend(refined_cells)
                new_levels.extend([self.cell_levels[i] + 1] * len(refined_cells))
            elif coarsen_cells[i] and self.cell_levels[i] > 0:
                # Coarsen cell
                coarsened_cells = self._coarsen_cell(cell)
                new_cells.extend(coarsened_cells)
                new_levels.extend([self.cell_levels[i] - 1] * len(coarsened_cells))
            else:
                # Keep cell as is
                new_cells.append(cell.vertices)
                new_levels.append(self.cell_levels[i])
                
        # Create new mesh
        new_mesh = Mesh(new_vertices, new_cells)
        self.mesh = new_mesh
        self.cell_levels = np.array(new_levels)
        
        return new_mesh
        
    def _refine_cell(self, cell: Cell, vertices: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
        """Refine a single cell"""
        # Create new vertices at cell edges
        new_vertices = []
        for i in range(len(cell.vertices)):
            v1 = vertices[cell.vertices[i]]
            v2 = vertices[cell.vertices[(i + 1) % len(cell.vertices)]]
            new_vertex = 0.5 * (v1 + v2)
            new_vertices.append(new_vertex)
            
        # Add new vertices to vertex list
        new_vertex_indices = list(range(len(vertices), len(vertices) + len(new_vertices)))
        vertices = np.vstack((vertices, new_vertices))
        
        # Create refined cells
        refined_cells = []
        center_vertex = np.mean(vertices[cell.vertices], axis=0)
        center_index = len(vertices)
        vertices = np.vstack((vertices, [center_vertex]))
        
        for i in range(len(cell.vertices)):
            refined_cells.append([
                cell.vertices[i],
                new_vertex_indices[i],
                center_index,
                new_vertex_indices[(i - 1) % len(cell.vertices)]
            ])
            
        return vertices, refined_cells
        
    def _coarsen_cell(self, cell: Cell) -> List[List[int]]:
        """Coarsen a single cell"""
        # For now, just return the original cell
        # In practice, this would merge with neighboring cells
        return [cell.vertices]
        
    def interpolate_field(self,
                         field: np.ndarray,
                         new_mesh: Mesh) -> np.ndarray:
        """
        Interpolate field to new mesh
        
        Args:
            field: Field to interpolate
            new_mesh: New mesh
            
        Returns:
            Interpolated field
        """
        new_field = np.zeros(len(new_mesh.vertices))
        
        for i, vertex in enumerate(new_mesh.vertices):
            # Find containing cell in old mesh
            cell = self._find_containing_cell(vertex)
            
            if cell is not None:
                # Interpolate using barycentric coordinates
                new_field[i] = self._interpolate_vertex(vertex, cell, field)
                
        return new_field
        
    def _find_containing_cell(self, vertex: np.ndarray) -> Optional[Cell]:
        """Find cell containing given vertex"""
        for cell in self.mesh.cells:
            if self._is_point_in_cell(vertex, cell):
                return cell
        return None
        
    def _is_point_in_cell(self, point: np.ndarray, cell: Cell) -> bool:
        """Check if point is inside cell"""
        # Simple check using cross products
        vertices = self.mesh.vertices[cell.vertices]
        n = len(vertices)
        
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            if np.cross(v2 - v1, point - v1) < 0:
                return False
                
        return True
        
    def _interpolate_vertex(self,
                           vertex: np.ndarray,
                           cell: Cell,
                           field: np.ndarray) -> float:
        """Interpolate field value at vertex using barycentric coordinates"""
        vertices = self.mesh.vertices[cell.vertices]
        values = field[cell.vertices]
        
        # Compute barycentric coordinates
        coords = self._compute_barycentric_coords(vertex, vertices)
        
        # Interpolate
        return np.sum(coords * values)
        
    def _compute_barycentric_coords(self,
                                  point: np.ndarray,
                                  vertices: np.ndarray) -> np.ndarray:
        """Compute barycentric coordinates of point in cell"""
        # For triangles
        if len(vertices) == 3:
            v0 = vertices[1] - vertices[0]
            v1 = vertices[2] - vertices[0]
            v2 = point - vertices[0]
            
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)
            
            denom = d00 * d11 - d01 * d01
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            
            return np.array([u, v, w])
        else:
            raise NotImplementedError("Barycentric coordinates only implemented for triangles") 