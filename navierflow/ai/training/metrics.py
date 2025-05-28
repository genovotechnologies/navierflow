import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from sklearn.metrics import mean_squared_error, r2_score
import logging

class TrainingMetrics:
    """Metrics for evaluating AI model performance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': [],
            'relative_error': [],
            'max_error': []
        }

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ):
        """Update metrics with new predictions"""
        # Mean squared error
        mse = mean_squared_error(
            y_true, y_pred,
            sample_weight=sample_weight
        )
        self.metrics['mse'].append(mse)
        
        # Root mean squared error
        rmse = np.sqrt(mse)
        self.metrics['rmse'].append(rmse)
        
        # Mean absolute error
        mae = np.mean(np.abs(y_true - y_pred))
        self.metrics['mae'].append(mae)
        
        # R² score
        r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)
        self.metrics['r2'].append(r2)
        
        # Relative error
        relative_error = np.mean(
            np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-10)
        )
        self.metrics['relative_error'].append(relative_error)
        
        # Maximum error
        max_error = np.max(np.abs(y_true - y_pred))
        self.metrics['max_error'].append(max_error)

    def get_current(self) -> Dict[str, float]:
        """Get current metric values"""
        return {
            name: values[-1] if values else 0.0
            for name, values in self.metrics.items()
        }

    def get_average(self) -> Dict[str, float]:
        """Get average metric values"""
        return {
            name: np.mean(values) if values else 0.0
            for name, values in self.metrics.items()
        }

    def get_history(self) -> Dict[str, List[float]]:
        """Get metric history"""
        return self.metrics.copy()

class PINNMetrics(TrainingMetrics):
    """Specific metrics for Physics-Informed Neural Networks"""
    
    def __init__(self):
        super().__init__()
        self.pde_metrics = {
            'continuity_residual': [],
            'momentum_residual': [],
            'energy_residual': [],
            'boundary_error': []
        }

    def reset(self):
        """Reset all metrics"""
        super().reset()
        for key in self.pde_metrics:
            self.pde_metrics[key] = []

    def update_pde_metrics(
        self,
        continuity_residual: float,
        momentum_residual: float,
        energy_residual: float,
        boundary_error: float
    ):
        """Update PDE-specific metrics"""
        self.pde_metrics['continuity_residual'].append(continuity_residual)
        self.pde_metrics['momentum_residual'].append(momentum_residual)
        self.pde_metrics['energy_residual'].append(energy_residual)
        self.pde_metrics['boundary_error'].append(boundary_error)

    def get_current(self) -> Dict[str, float]:
        """Get current metric values including PDE metrics"""
        metrics = super().get_current()
        metrics.update({
            name: values[-1] if values else 0.0
            for name, values in self.pde_metrics.items()
        })
        return metrics

class MeshMetrics(TrainingMetrics):
    """Specific metrics for mesh optimization"""
    
    def __init__(self):
        super().__init__()
        self.mesh_metrics = {
            'num_cells': [],
            'refinement_levels': [],
            'aspect_ratio': [],
            'smoothness': []
        }

    def reset(self):
        """Reset all metrics"""
        super().reset()
        for key in self.mesh_metrics:
            self.mesh_metrics[key] = []

    def update_mesh_metrics(
        self,
        num_cells: int,
        refinement_levels: np.ndarray,
        cell_sizes: np.ndarray
    ):
        """Update mesh-specific metrics"""
        self.mesh_metrics['num_cells'].append(num_cells)
        
        # Distribution of refinement levels
        self.mesh_metrics['refinement_levels'].append(
            np.bincount(refinement_levels.flatten())
        )
        
        # Compute aspect ratio statistics
        if cell_sizes.ndim == 2:
            aspect_ratios = cell_sizes[:, 0] / cell_sizes[:, 1]
            self.mesh_metrics['aspect_ratio'].append(
                np.mean(np.maximum(aspect_ratios, 1/aspect_ratios))
            )
        
        # Compute mesh smoothness
        if cell_sizes.ndim == 2:
            size_gradients = np.gradient(cell_sizes)
            smoothness = np.mean(np.abs(size_gradients))
            self.mesh_metrics['smoothness'].append(smoothness)

    def get_current(self) -> Dict[str, float]:
        """Get current metric values including mesh metrics"""
        metrics = super().get_current()
        metrics.update({
            name: values[-1] if values else 0.0
            for name, values in self.mesh_metrics.items()
            if name != 'refinement_levels'  # Skip array-valued metric
        })
        return metrics

class AnomalyMetrics(TrainingMetrics):
    """Specific metrics for anomaly detection"""
    
    def __init__(self):
        super().__init__()
        self.anomaly_metrics = {
            'num_anomalies': [],
            'anomaly_scores': [],
            'false_positive_rate': [],
            'detection_rate': [],
            'precision': [],
            'recall': []
        }

    def reset(self):
        """Reset all metrics"""
        super().reset()
        for key in self.anomaly_metrics:
            self.anomaly_metrics[key] = []

    def update_anomaly_metrics(
        self,
        anomaly_mask: np.ndarray,
        true_anomalies: Optional[np.ndarray] = None,
        anomaly_scores: Optional[np.ndarray] = None
    ):
        """Update anomaly detection metrics"""
        self.anomaly_metrics['num_anomalies'].append(
            np.sum(anomaly_mask)
        )
        
        if anomaly_scores is not None:
            self.anomaly_metrics['anomaly_scores'].append(
                np.mean(anomaly_scores)
            )
        
        if true_anomalies is not None:
            # Compute detection performance metrics
            tp = np.sum(anomaly_mask & true_anomalies)
            fp = np.sum(anomaly_mask & ~true_anomalies)
            fn = np.sum(~anomaly_mask & true_anomalies)
            
            # False positive rate
            total_negatives = np.sum(~true_anomalies)
            fpr = fp / total_negatives if total_negatives > 0 else 0
            self.anomaly_metrics['false_positive_rate'].append(fpr)
            
            # Detection rate (true positive rate)
            total_positives = np.sum(true_anomalies)
            tpr = tp / total_positives if total_positives > 0 else 0
            self.anomaly_metrics['detection_rate'].append(tpr)
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            self.anomaly_metrics['precision'].append(precision)
            
            # Recall (same as detection rate)
            self.anomaly_metrics['recall'].append(tpr)

    def get_current(self) -> Dict[str, float]:
        """Get current metric values including anomaly metrics"""
        metrics = super().get_current()
        metrics.update({
            name: values[-1] if values else 0.0
            for name, values in self.anomaly_metrics.items()
        })
        return metrics 