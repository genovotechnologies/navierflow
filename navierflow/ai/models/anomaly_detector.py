import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler

class FlowAutoencoder(nn.Module):
    """Autoencoder for flow field anomaly detection"""
    
    def __init__(self, input_channels: int = 4):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder"""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class AnomalyDetector:
    """Anomaly detector for fluid flow fields"""
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        threshold_percentile: float = 95.0
    ):
        self.input_shape = input_shape
        self.threshold_percentile = threshold_percentile
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.autoencoder = FlowAutoencoder()
        self.optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=1e-4
        )
        
        # Initialize scalers
        self.scalers = {
            'velocity': StandardScaler(),
            'pressure': StandardScaler(),
            'vorticity': StandardScaler()
        }
        
        # Statistics
        self.reconstruction_errors = []
        self.anomaly_threshold = None
        self.stats = {
            'num_anomalies': 0,
            'anomaly_scores': [],
            'detection_history': []
        }

    def preprocess_data(
        self,
        velocity: np.ndarray,
        pressure: np.ndarray,
        vorticity: np.ndarray,
        fit_scalers: bool = False
    ) -> torch.Tensor:
        """Preprocess input data"""
        # Reshape and scale velocity components
        u = velocity[:,:,0].reshape(-1, 1)
        v = velocity[:,:,1].reshape(-1, 1)
        p = pressure.reshape(-1, 1)
        w = vorticity.reshape(-1, 1)
        
        if fit_scalers:
            self.scalers['velocity'].fit(np.hstack([u, v]))
            self.scalers['pressure'].fit(p)
            self.scalers['vorticity'].fit(w)
        
        # Apply scaling
        uv_scaled = self.scalers['velocity'].transform(np.hstack([u, v]))
        p_scaled = self.scalers['pressure'].transform(p)
        w_scaled = self.scalers['vorticity'].transform(w)
        
        # Reshape back to original dimensions
        u_scaled = uv_scaled[:,0].reshape(self.input_shape)
        v_scaled = uv_scaled[:,1].reshape(self.input_shape)
        p_scaled = p_scaled.reshape(self.input_shape)
        w_scaled = w_scaled.reshape(self.input_shape)
        
        # Stack channels
        x = np.stack([u_scaled, v_scaled, p_scaled, w_scaled], axis=0)
        return torch.FloatTensor(x).unsqueeze(0)

    def train(
        self,
        training_data: List[Dict[str, np.ndarray]],
        num_epochs: int = 100,
        batch_size: int = 16
    ):
        """Train the anomaly detector"""
        self.autoencoder.train()
        reconstruction_errors = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Shuffle training data
            np.random.shuffle(training_data)
            
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                
                # Prepare batch data
                x_batch = []
                for sample in batch:
                    x = self.preprocess_data(
                        sample['velocity'],
                        sample['pressure'],
                        sample['vorticity'],
                        fit_scalers=(epoch == 0 and i == 0)
                    )
                    x_batch.append(x)
                
                x_batch = torch.cat(x_batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                reconstructed, _ = self.autoencoder(x_batch)
                
                # Compute reconstruction loss
                loss = nn.MSELoss()(reconstructed, x_batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                reconstruction_errors.append(loss.item())
            
            # Log progress
            avg_loss = epoch_loss / (len(training_data) / batch_size)
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Set anomaly threshold
        self.anomaly_threshold = np.percentile(
            reconstruction_errors,
            self.threshold_percentile
        )
        self.logger.info(f"Anomaly threshold set to: {self.anomaly_threshold:.4f}")

    def detect_anomalies(
        self,
        velocity: np.ndarray,
        pressure: np.ndarray,
        vorticity: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Detect anomalies in flow field"""
        self.autoencoder.eval()
        
        # Preprocess input
        x = self.preprocess_data(velocity, pressure, vorticity)
        
        # Forward pass
        with torch.no_grad():
            reconstructed, latent = self.autoencoder(x)
        
        # Compute reconstruction error
        error = nn.MSELoss(reduction='none')(reconstructed, x).numpy()
        error = error.mean(axis=1)  # Average over channels
        
        # Compute anomaly score
        anomaly_score = np.mean(error)
        
        # Generate anomaly mask
        anomaly_mask = error > self.anomaly_threshold
        
        # Update statistics
        self.stats['num_anomalies'] += np.sum(anomaly_mask)
        self.stats['anomaly_scores'].append(anomaly_score)
        self.stats['detection_history'].append(anomaly_score > self.anomaly_threshold)
        
        return anomaly_mask[0], anomaly_score

    def analyze_anomaly(
        self,
        velocity: np.ndarray,
        pressure: np.ndarray,
        vorticity: np.ndarray,
        anomaly_mask: np.ndarray
    ) -> Dict:
        """Analyze detected anomalies"""
        analysis = {
            'location': {
                'x': np.where(anomaly_mask)[1],
                'y': np.where(anomaly_mask)[0]
            },
            'severity': {
                'velocity': np.mean(np.abs(velocity[anomaly_mask])),
                'pressure': np.mean(np.abs(pressure[anomaly_mask])),
                'vorticity': np.mean(np.abs(vorticity[anomaly_mask]))
            },
            'extent': np.sum(anomaly_mask) / anomaly_mask.size
        }
        
        # Classify anomaly type
        if analysis['severity']['velocity'] > 2.0 * np.mean(np.abs(velocity)):
            analysis['type'] = 'velocity_instability'
        elif analysis['severity']['pressure'] > 2.0 * np.mean(np.abs(pressure)):
            analysis['type'] = 'pressure_instability'
        elif analysis['severity']['vorticity'] > 2.0 * np.mean(np.abs(vorticity)):
            analysis['type'] = 'vorticity_anomaly'
        else:
            analysis['type'] = 'unknown'
        
        return analysis

    def get_latent_representation(
        self,
        velocity: np.ndarray,
        pressure: np.ndarray,
        vorticity: np.ndarray
    ) -> np.ndarray:
        """Get latent space representation of flow field"""
        self.autoencoder.eval()
        
        # Preprocess input
        x = self.preprocess_data(velocity, pressure, vorticity)
        
        # Forward pass
        with torch.no_grad():
            _, latent = self.autoencoder(x)
        
        return latent.numpy()

    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.autoencoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'anomaly_threshold': self.anomaly_threshold,
            'scalers': self.scalers
        }, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.anomaly_threshold = checkpoint['anomaly_threshold']
        self.scalers = checkpoint['scalers']
        self.logger.info(f"Model loaded from {path}")

    def get_stats(self) -> Dict:
        """Get detection statistics"""
        return self.stats.copy() 