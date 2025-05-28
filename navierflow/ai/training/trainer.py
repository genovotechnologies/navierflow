import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import wandb
from datetime import datetime
from pathlib import Path
import json

from ..models.pinn import FluidPINN
from ..models.mesh_optimizer import MeshOptimizer
from ..models.anomaly_detector import AnomalyDetector

class ModelTrainer:
    """Trainer for NavierFlow AI models"""
    
    def __init__(
        self,
        model_type: str,
        config: Dict,
        experiment_name: str,
        use_wandb: bool = True
    ):
        self.model_type = model_type
        self.config = config
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = self._create_model()
        
        # Training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'epochs_trained': 0
        }
        
        # Initialize wandb if enabled
        if self.use_wandb:
            wandb.init(
                project="navierflow",
                name=experiment_name,
                config=config
            )

    def _create_model(self) -> Union[FluidPINN, MeshOptimizer, AnomalyDetector]:
        """Create model instance based on type"""
        if self.model_type == "pinn":
            return FluidPINN(
                hidden_layers=self.config.get('hidden_layers', [64, 128, 128, 64]),
                learning_rate=self.config.get('learning_rate', 1e-4)
            )
        elif self.model_type == "mesh_optimizer":
            return MeshOptimizer(
                base_resolution=self.config['base_resolution'],
                max_refinement_level=self.config.get('max_refinement_level', 3)
            )
        elif self.model_type == "anomaly_detector":
            return AnomalyDetector(
                input_shape=self.config['input_shape'],
                threshold_percentile=self.config.get('threshold_percentile', 95.0)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        train_data: List[Dict[str, np.ndarray]],
        val_data: Optional[List[Dict[str, np.ndarray]]] = None,
        num_epochs: int = 100,
        batch_size: int = 16,
        checkpoint_dir: str = "checkpoints",
        save_best: bool = True
    ):
        """Train the model"""
        # Create checkpoint directory
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        for epoch in range(num_epochs):
            # Training step
            train_metrics = self._train_epoch(train_data, batch_size)
            
            # Validation step
            if val_data is not None:
                val_metrics = self._validate(val_data, batch_size)
            else:
                val_metrics = {}
            
            # Update metrics
            self.metrics['train_loss'].append(train_metrics['loss'])
            if val_metrics:
                self.metrics['val_loss'].append(val_metrics['loss'])
                
                # Save best model
                if save_best and val_metrics['loss'] < self.metrics['best_val_loss']:
                    self.metrics['best_val_loss'] = val_metrics['loss']
                    self.save_checkpoint(checkpoint_dir / "best_model.pt")
            
            # Log metrics
            self._log_metrics(epoch + 1, train_metrics, val_metrics)
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                )
        
        self.metrics['epochs_trained'] += num_epochs
        
        # Save final model
        self.save_checkpoint(checkpoint_dir / "final_model.pt")
        
        # Save training history
        self.save_history(checkpoint_dir / "training_history.json")

    def _train_epoch(
        self,
        train_data: List[Dict[str, np.ndarray]],
        batch_size: int
    ) -> Dict[str, float]:
        """Execute one training epoch"""
        if self.model_type == "pinn":
            return self._train_pinn_epoch(train_data, batch_size)
        elif self.model_type == "mesh_optimizer":
            return self._train_mesh_optimizer_epoch(train_data, batch_size)
        else:  # anomaly_detector
            return self._train_anomaly_detector_epoch(train_data, batch_size)

    def _train_pinn_epoch(
        self,
        train_data: List[Dict[str, np.ndarray]],
        batch_size: int
    ) -> Dict[str, float]:
        """Train PINN for one epoch"""
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Prepare batch data
            x_pde = torch.cat([sample['x_pde'] for sample in batch])
            x_bc = torch.cat([sample['x_bc'] for sample in batch])
            bc_values = torch.cat([sample['bc_values'] for sample in batch])
            
            # Training step
            metrics = self.model.training_step(x_pde, x_bc, bc_values)
            total_loss += metrics['total_loss']
            num_batches += 1
        
        return {'loss': total_loss / num_batches}

    def _train_mesh_optimizer_epoch(
        self,
        train_data: List[Dict[str, np.ndarray]],
        batch_size: int
    ) -> Dict[str, float]:
        """Train mesh optimizer for one epoch"""
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Training step
            feature_scores = []
            for sample in batch:
                scores = self.model.compute_feature_scores(
                    sample['velocity'],
                    sample['pressure'],
                    sample['vorticity']
                )
                feature_scores.append(scores)
            
            # Compute loss against target refinement
            target_refinement = np.stack([
                sample['refinement_mask'] for sample in batch
            ])
            loss = np.mean((np.stack(feature_scores) - target_refinement) ** 2)
            
            total_loss += loss
            num_batches += 1
        
        return {'loss': total_loss / num_batches}

    def _train_anomaly_detector_epoch(
        self,
        train_data: List[Dict[str, np.ndarray]],
        batch_size: int
    ) -> Dict[str, float]:
        """Train anomaly detector for one epoch"""
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Prepare batch data
            x_batch = []
            for sample in batch:
                x = self.model.preprocess_data(
                    sample['velocity'],
                    sample['pressure'],
                    sample['vorticity']
                )
                x_batch.append(x)
            
            x_batch = torch.cat(x_batch)
            
            # Forward pass
            reconstructed, _ = self.model.autoencoder(x_batch)
            
            # Compute loss
            loss = torch.nn.MSELoss()(reconstructed, x_batch)
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'loss': total_loss / num_batches}

    def _validate(
        self,
        val_data: List[Dict[str, np.ndarray]],
        batch_size: int
    ) -> Dict[str, float]:
        """Validate model performance"""
        with torch.no_grad():
            return self._train_epoch(val_data, batch_size)

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log training metrics"""
        # Console logging
        log_str = f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}"
        if val_metrics:
            log_str += f", val_loss={val_metrics['loss']:.4f}"
        self.logger.info(log_str)
        
        # WandB logging
        if self.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics['loss']
            }
            if val_metrics:
                log_dict['val_loss'] = val_metrics['loss']
            wandb.log(log_dict)

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_type': self.model_type,
            'model_state': self.model.state_dict(),
            'config': self.config,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        
        # Verify model type
        if checkpoint['model_type'] != self.model_type:
            raise ValueError(
                f"Checkpoint model type ({checkpoint['model_type']}) "
                f"does not match current model type ({self.model_type})"
            )
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state'])
        self.metrics = checkpoint['metrics']
        self.logger.info(f"Checkpoint loaded from {path}")

    def save_history(self, path: str):
        """Save training history"""
        history = {
            'model_type': self.model_type,
            'config': self.config,
            'metrics': {
                k: v if isinstance(v, (int, float)) else v.tolist()
                for k, v in self.metrics.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(history, f, indent=4)
        self.logger.info(f"Training history saved to {path}")

    def get_metrics(self) -> Dict:
        """Get current training metrics"""
        return self.metrics.copy() 