import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List, Tuple, Dict, Callable
from dataclasses import dataclass
from enum import Enum

class NetworkType(Enum):
    """Types of neural networks"""
    FNN = "fnn"  # Fully Connected Neural Network
    CNN = "cnn"  # Convolutional Neural Network
    RNN = "rnn"  # Recurrent Neural Network
    GNN = "gnn"  # Graph Neural Network
    PINN = "pinn"  # Physics-Informed Neural Network

@dataclass
class NetworkConfig:
    """Neural network configuration"""
    network_type: NetworkType = NetworkType.PINN
    hidden_layers: List[int] = (64, 128, 64)
    activation: str = "tanh"
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 1000
    use_gpu: bool = True
    use_amp: bool = True  # Automatic Mixed Precision

class PhysicsInformedNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 config: Optional[NetworkConfig] = None):
        """
        Initialize physics-informed neural network
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            config: Network configuration
        """
        super().__init__()
        self.config = config or NetworkConfig()
        self._build_network(input_dim, output_dim)
        self._setup_optimizer()
        
    def _build_network(self, input_dim: int, output_dim: int):
        """Build neural network architecture"""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, self.config.hidden_layers[0]))
        layers.append(self._get_activation())
        
        # Hidden layers
        for i in range(len(self.config.hidden_layers) - 1):
            layers.append(nn.Linear(
                self.config.hidden_layers[i],
                self.config.hidden_layers[i + 1]
            ))
            layers.append(self._get_activation())
            
        # Output layer
        layers.append(nn.Linear(self.config.hidden_layers[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def _get_activation(self) -> nn.Module:
        """Get activation function"""
        if self.config.activation == "tanh":
            return nn.Tanh()
        elif self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}")
            
    def _setup_optimizer(self):
        """Setup optimizer"""
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate
        )
        
        if self.config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.network(x)
        
    def compute_derivatives(self,
                          x: torch.Tensor,
                          order: int = 1) -> List[torch.Tensor]:
        """
        Compute derivatives using automatic differentiation
        
        Args:
            x: Input tensor
            order: Derivative order
            
        Returns:
            List of derivatives
        """
        derivatives = []
        y = self.forward(x)
        
        for i in range(order):
            if i == 0:
                grad = torch.autograd.grad(
                    y.sum(),
                    x,
                    create_graph=True,
                    retain_graph=True
                )[0]
            else:
                grad = torch.autograd.grad(
                    grad.sum(),
                    x,
                    create_graph=True,
                    retain_graph=True
                )[0]
            derivatives.append(grad)
            
        return derivatives
        
    def compute_physics_loss(self,
                           x: torch.Tensor,
                           physics_equations: List[Callable]) -> torch.Tensor:
        """
        Compute physics-informed loss
        
        Args:
            x: Input tensor
            physics_equations: List of physics equations
            
        Returns:
            Physics loss
        """
        y = self.forward(x)
        derivatives = self.compute_derivatives(x)
        
        loss = 0.0
        for equation in physics_equations:
            loss += equation(x, y, derivatives)
            
        return loss
        
    def train_step(self,
                  x: torch.Tensor,
                  y: torch.Tensor,
                  physics_equations: List[Callable]) -> Dict[str, float]:
        """
        Perform one training step
        
        Args:
            x: Input tensor
            y: Target tensor
            physics_equations: List of physics equations
            
        Returns:
            Dictionary of losses
        """
        self.optimizer.zero_grad()
        
        if self.config.use_amp:
            with torch.cuda.amp.autocast():
                # Forward pass
                y_pred = self.forward(x)
                
                # Compute losses
                data_loss = nn.MSELoss()(y_pred, y)
                physics_loss = self.compute_physics_loss(x, physics_equations)
                total_loss = data_loss + physics_loss
                
            # Backward pass with gradient scaling
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Forward pass
            y_pred = self.forward(x)
            
            # Compute losses
            data_loss = nn.MSELoss()(y_pred, y)
            physics_loss = self.compute_physics_loss(x, physics_equations)
            total_loss = data_loss + physics_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
        return {
            "total_loss": total_loss.item(),
            "data_loss": data_loss.item(),
            "physics_loss": physics_loss.item()
        }
        
    def train(self,
             x: torch.Tensor,
             y: torch.Tensor,
             physics_equations: List[Callable]) -> Dict[str, List[float]]:
        """
        Train network
        
        Args:
            x: Input tensor
            y: Target tensor
            physics_equations: List of physics equations
            
        Returns:
            Dictionary of training history
        """
        history = {
            "total_loss": [],
            "data_loss": [],
            "physics_loss": []
        }
        
        for epoch in range(self.config.epochs):
            # Create batches
            indices = torch.randperm(len(x))
            for i in range(0, len(x), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                x_batch = x[batch_indices]
                y_batch = y[batch_indices]
                
                # Training step
                losses = self.train_step(x_batch, y_batch, physics_equations)
                
                # Update history
                for key, value in losses.items():
                    history[key].append(value)
                    
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.config.epochs}")
                print(f"Total Loss: {losses['total_loss']:.6f}")
                print(f"Data Loss: {losses['data_loss']:.6f}")
                print(f"Physics Loss: {losses['physics_loss']:.6f}")
                
        return history
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions
        
        Args:
            x: Input tensor
            
        Returns:
            Predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
            
    def save(self, path: str):
        """
        Save model
        
        Args:
            path: Save path
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }, path)
        
    def load(self, path: str):
        """
        Load model
        
        Args:
            path: Load path
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.config = checkpoint["config"] 