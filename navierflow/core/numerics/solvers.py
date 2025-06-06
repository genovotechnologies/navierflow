import numpy as np
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass
from enum import Enum
from scipy.sparse import spmatrix, csr_matrix
from scipy.sparse.linalg import spsolve, gmres, bicgstab, cg

class SolverType(Enum):
    """Types of numerical solvers"""
    DIRECT = "direct"
    GMRES = "gmres"
    BICGSTAB = "bicgstab"
    CG = "cg"
    SOR = "sor"
    JACOBI = "jacobi"
    GAUSS_SEIDEL = "gauss_seidel"

@dataclass
class SolverParameters:
    """Parameters for numerical solvers"""
    max_iterations: int = 1000
    tolerance: float = 1e-6
    preconditioner: Optional[str] = None
    omega: float = 1.0  # Relaxation parameter for SOR
    verbose: bool = False

class LinearSolver:
    def __init__(self,
                 solver_type: SolverType,
                 params: Optional[SolverParameters] = None):
        """
        Initialize linear solver
        
        Args:
            solver_type: Type of solver
            params: Solver parameters
        """
        self.solver_type = solver_type
        self.params = params or SolverParameters()
        
    def solve(self,
              A: spmatrix,
              b: np.ndarray,
              x0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve linear system Ax = b
        
        Args:
            A: System matrix
            b: Right-hand side vector
            x0: Initial guess
            
        Returns:
            Solution vector
        """
        if self.solver_type == SolverType.DIRECT:
            return self._solve_direct(A, b)
        elif self.solver_type == SolverType.GMRES:
            return self._solve_gmres(A, b, x0)
        elif self.solver_type == SolverType.BICGSTAB:
            return self._solve_bicgstab(A, b, x0)
        elif self.solver_type == SolverType.CG:
            return self._solve_cg(A, b, x0)
        elif self.solver_type == SolverType.SOR:
            return self._solve_sor(A, b, x0)
        elif self.solver_type == SolverType.JACOBI:
            return self._solve_jacobi(A, b, x0)
        elif self.solver_type == SolverType.GAUSS_SEIDEL:
            return self._solve_gauss_seidel(A, b, x0)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
            
    def _solve_direct(self, A: spmatrix, b: np.ndarray) -> np.ndarray:
        """Solve using direct method"""
        return spsolve(A, b)
        
    def _solve_gmres(self,
                     A: spmatrix,
                     b: np.ndarray,
                     x0: Optional[np.ndarray]) -> np.ndarray:
        """Solve using GMRES"""
        x, info = gmres(A, b,
                       x0=x0,
                       maxiter=self.params.max_iterations,
                       tol=self.params.tolerance)
        if info > 0:
            raise RuntimeError(f"GMRES failed to converge: {info}")
        return x
        
    def _solve_bicgstab(self,
                       A: spmatrix,
                       b: np.ndarray,
                       x0: Optional[np.ndarray]) -> np.ndarray:
        """Solve using BiCGSTAB"""
        x, info = bicgstab(A, b,
                          x0=x0,
                          maxiter=self.params.max_iterations,
                          tol=self.params.tolerance)
        if info > 0:
            raise RuntimeError(f"BiCGSTAB failed to converge: {info}")
        return x
        
    def _solve_cg(self,
                 A: spmatrix,
                 b: np.ndarray,
                 x0: Optional[np.ndarray]) -> np.ndarray:
        """Solve using Conjugate Gradient"""
        x, info = cg(A, b,
                    x0=x0,
                    maxiter=self.params.max_iterations,
                    tol=self.params.tolerance)
        if info > 0:
            raise RuntimeError(f"CG failed to converge: {info}")
        return x
        
    def _solve_sor(self,
                  A: spmatrix,
                  b: np.ndarray,
                  x0: Optional[np.ndarray]) -> np.ndarray:
        """Solve using Successive Over-Relaxation"""
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
            
        D = A.diagonal()
        L = -A.tril(k=-1)
        U = -A.triu(k=1)
        
        for _ in range(self.params.max_iterations):
            x_new = x.copy()
            for i in range(len(b)):
                x_new[i] = (1 - self.params.omega) * x[i] + \
                          self.params.omega / D[i] * (
                              b[i] - L[i].dot(x_new) - U[i].dot(x)
                          )
            if np.linalg.norm(x_new - x) < self.params.tolerance:
                break
            x = x_new
            
        return x
        
    def _solve_jacobi(self,
                     A: spmatrix,
                     b: np.ndarray,
                     x0: Optional[np.ndarray]) -> np.ndarray:
        """Solve using Jacobi iteration"""
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
            
        D = A.diagonal()
        R = A - csr_matrix(np.diag(D))
        
        for _ in range(self.params.max_iterations):
            x_new = (b - R.dot(x)) / D
            if np.linalg.norm(x_new - x) < self.params.tolerance:
                break
            x = x_new
            
        return x
        
    def _solve_gauss_seidel(self,
                          A: spmatrix,
                          b: np.ndarray,
                          x0: Optional[np.ndarray]) -> np.ndarray:
        """Solve using Gauss-Seidel iteration"""
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
            
        D = A.diagonal()
        L = -A.tril(k=-1)
        U = -A.triu(k=1)
        
        for _ in range(self.params.max_iterations):
            x_new = x.copy()
            for i in range(len(b)):
                x_new[i] = (b[i] - L[i].dot(x_new) - U[i].dot(x)) / D[i]
            if np.linalg.norm(x_new - x) < self.params.tolerance:
                break
            x = x_new
            
        return x

class NonlinearSolver:
    def __init__(self,
                 linear_solver: LinearSolver,
                 params: Optional[SolverParameters] = None):
        """
        Initialize nonlinear solver
        
        Args:
            linear_solver: Linear solver for Jacobian system
            params: Solver parameters
        """
        self.linear_solver = linear_solver
        self.params = params or SolverParameters()
        
    def solve(self,
              residual: Callable[[np.ndarray], np.ndarray],
              jacobian: Callable[[np.ndarray], spmatrix],
              x0: np.ndarray) -> np.ndarray:
        """
        Solve nonlinear system F(x) = 0
        
        Args:
            residual: Residual function F(x)
            jacobian: Jacobian function J(x)
            x0: Initial guess
            
        Returns:
            Solution vector
        """
        x = x0.copy()
        
        for i in range(self.params.max_iterations):
            # Compute residual
            F = residual(x)
            
            # Check convergence
            if np.linalg.norm(F) < self.params.tolerance:
                break
                
            # Compute Jacobian
            J = jacobian(x)
            
            # Solve linear system
            dx = self.linear_solver.solve(J, -F)
            
            # Update solution
            x += dx
            
            if self.params.verbose:
                print(f"Iteration {i+1}: residual = {np.linalg.norm(F)}")
                
        return x
        
    def solve_with_line_search(self,
                              residual: Callable[[np.ndarray], np.ndarray],
                              jacobian: Callable[[np.ndarray], spmatrix],
                              x0: np.ndarray) -> np.ndarray:
        """
        Solve nonlinear system with line search
        
        Args:
            residual: Residual function F(x)
            jacobian: Jacobian function J(x)
            x0: Initial guess
            
        Returns:
            Solution vector
        """
        x = x0.copy()
        
        for i in range(self.params.max_iterations):
            # Compute residual
            F = residual(x)
            
            # Check convergence
            if np.linalg.norm(F) < self.params.tolerance:
                break
                
            # Compute Jacobian
            J = jacobian(x)
            
            # Solve linear system
            dx = self.linear_solver.solve(J, -F)
            
            # Line search
            alpha = 1.0
            while True:
                x_new = x + alpha * dx
                F_new = residual(x_new)
                if np.linalg.norm(F_new) < np.linalg.norm(F):
                    break
                alpha *= 0.5
                if alpha < 1e-10:
                    break
                    
            # Update solution
            x = x_new
            
            if self.params.verbose:
                print(f"Iteration {i+1}: residual = {np.linalg.norm(F)}")
                
        return x 