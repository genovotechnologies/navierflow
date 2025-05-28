import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class FluidPINN(nn.Module):
    """Physics-Informed Neural Network for fluid dynamics"""
    
    def __init__(
        self,
        hidden_layers: List[int] = [64, 128, 128, 64],
        activation: nn.Module = nn.Tanh(),
        learning_rate: float = 1e-4
    ):
        super().__init__()
        
        self.input_dim = 3  # (x, y, t)
        self.output_dim = 3  # (u, v, p)
        
        # Build network layers
        layers = []
        prev_dim = self.input_dim
        
        for dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, dim),
                activation,
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Loss weights
        self.lambda_data = 1.0
        self.lambda_pde = 1.0
        self.lambda_bc = 1.0
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)

    def compute_pde_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual loss (Navier-Stokes equations)"""
        x.requires_grad_(True)
        
        # Get predictions
        y = self.forward(x)
        u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        
        # Compute gradients
        du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 0:1]
        du_dy = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 1:2]
        du_dt = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 2:3]
        
        dv_dx = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 0:1]
        dv_dy = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 1:2]
        dv_dt = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 2:3]
        
        dp_dx = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:, 0:1]
        dp_dy = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:, 1:2]
        
        # Second derivatives
        d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0][:, 0:1]
        d2u_dy2 = torch.autograd.grad(du_dy.sum(), x, create_graph=True)[0][:, 1:2]
        
        d2v_dx2 = torch.autograd.grad(dv_dx.sum(), x, create_graph=True)[0][:, 0:1]
        d2v_dy2 = torch.autograd.grad(dv_dy.sum(), x, create_graph=True)[0][:, 1:2]
        
        # Physical parameters
        rho = 1.0  # Density
        nu = 0.01  # Kinematic viscosity
        
        # Navier-Stokes residuals
        continuity = du_dx + dv_dy
        
        momentum_x = (
            du_dt + u * du_dx + v * du_dy +
            (1/rho) * dp_dx - nu * (d2u_dx2 + d2u_dy2)
        )
        
        momentum_y = (
            dv_dt + u * dv_dx + v * dv_dy +
            (1/rho) * dp_dy - nu * (d2v_dx2 + d2v_dy2)
        )
        
        return torch.mean(
            continuity**2 +
            momentum_x**2 +
            momentum_y**2
        )

    def compute_bc_loss(self, x_bc: torch.Tensor, bc_values: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss"""
        y_pred = self.forward(x_bc)
        return torch.mean((y_pred - bc_values)**2)

    def compute_data_loss(self, x_data: torch.Tensor, y_data: torch.Tensor) -> torch.Tensor:
        """Compute data fitting loss"""
        y_pred = self.forward(x_data)
        return torch.mean((y_pred - y_data)**2)

    def training_step(
        self,
        x_pde: torch.Tensor,
        x_bc: torch.Tensor,
        bc_values: torch.Tensor,
        x_data: Optional[torch.Tensor] = None,
        y_data: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Execute one training step"""
        self.optimizer.zero_grad()
        
        # Compute PDE loss
        pde_loss = self.compute_pde_loss(x_pde)
        
        # Compute BC loss
        bc_loss = self.compute_bc_loss(x_bc, bc_values)
        
        # Compute data loss if available
        data_loss = (
            self.compute_data_loss(x_data, y_data)
            if x_data is not None and y_data is not None
            else torch.tensor(0.0)
        )
        
        # Total loss
        total_loss = (
            self.lambda_pde * pde_loss +
            self.lambda_bc * bc_loss +
            self.lambda_data * data_loss
        )
        
        # Backpropagation
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'pde_loss': pde_loss.item(),
            'bc_loss': bc_loss.item(),
            'data_loss': data_loss.item()
        }

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Make prediction at given points"""
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(x)
        return y_pred.numpy()

    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Model loaded from {path}") 