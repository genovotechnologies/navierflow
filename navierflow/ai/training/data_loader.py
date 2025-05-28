import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
import h5py
import logging
from pathlib import Path
import json

class FluidDataset(Dataset):
    """Base dataset class for fluid simulation data"""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        transform: Optional[callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        
        # Load data index
        self.index = self._load_index()
        
        # Initialize cache
        self.cache = {}
        self.cache_size = 100  # Number of samples to keep in memory

    def _load_index(self) -> List[Dict]:
        """Load data index from JSON file"""
        index_path = self.data_dir / f"{self.split}_index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Data index not found at {index_path}"
            )
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        self.logger.info(
            f"Loaded {len(index)} samples for {self.split} split"
        )
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a data sample"""
        if idx in self.cache:
            return self.cache[idx]
        
        # Load data from file
        sample_info = self.index[idx]
        data_path = self.data_dir / sample_info['file']
        
        with h5py.File(data_path, 'r') as f:
            sample = {
                'velocity': f['velocity'][:],
                'pressure': f['pressure'][:],
                'vorticity': f['vorticity'][:]
            }
        
        # Apply transform if specified
        if self.transform is not None:
            sample = self.transform(sample)
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[idx] = sample
        return sample

class PINNDataset(FluidDataset):
    """Dataset for Physics-Informed Neural Networks"""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        num_pde_points: int = 10000,
        num_bc_points: int = 1000,
        transform: Optional[callable] = None
    ):
        super().__init__(data_dir, split, transform)
        self.num_pde_points = num_pde_points
        self.num_bc_points = num_bc_points

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a data sample with PDE points and boundary conditions"""
        sample = super().__getitem__(idx)
        
        # Generate PDE collocation points
        x_pde = self._generate_pde_points()
        
        # Generate boundary points
        x_bc, bc_values = self._generate_boundary_points(sample)
        
        return {
            'x_pde': torch.FloatTensor(x_pde),
            'x_bc': torch.FloatTensor(x_bc),
            'bc_values': torch.FloatTensor(bc_values),
            'velocity': torch.FloatTensor(sample['velocity']),
            'pressure': torch.FloatTensor(sample['pressure'])
        }

    def _generate_pde_points(self) -> np.ndarray:
        """Generate random points for PDE residual computation"""
        # Generate points in [0,1]^3 space (x, y, t)
        x_pde = np.random.rand(self.num_pde_points, 3)
        return x_pde

    def _generate_boundary_points(
        self,
        sample: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate boundary condition points"""
        # Initialize arrays
        x_bc = np.zeros((self.num_bc_points, 3))
        bc_values = np.zeros((self.num_bc_points, 3))  # u, v, p
        
        # Randomly sample boundary points
        for i in range(self.num_bc_points):
            # Randomly choose boundary type
            boundary_type = np.random.choice(['wall', 'inlet', 'outlet'])
            
            if boundary_type == 'wall':
                # No-slip boundary condition
                y = np.random.rand()
                t = np.random.rand()
                x_bc[i] = [0.0, y, t]  # Wall at x=0
                bc_values[i] = [0.0, 0.0, 0.0]  # Zero velocity
                
            elif boundary_type == 'inlet':
                # Inlet flow condition
                y = np.random.rand()
                t = np.random.rand()
                x_bc[i] = [0.0, y, t]
                bc_values[i] = [1.0, 0.0, 0.0]  # Unit x-velocity
                
            else:  # outlet
                # Zero pressure gradient
                y = np.random.rand()
                t = np.random.rand()
                x_bc[i] = [1.0, y, t]
                bc_values[i] = [0.0, 0.0, 0.0]  # To be handled in loss function
        
        return x_bc, bc_values

class MeshDataset(FluidDataset):
    """Dataset for mesh optimization"""
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a data sample with mesh refinement information"""
        sample = super().__getitem__(idx)
        
        # Load mesh refinement mask if available
        sample_info = self.index[idx]
        data_path = self.data_dir / sample_info['file']
        
        with h5py.File(data_path, 'r') as f:
            refinement_mask = f['refinement_mask'][:] if 'refinement_mask' in f else None
            cell_sizes = f['cell_sizes'][:] if 'cell_sizes' in f else None
        
        return {
            'velocity': torch.FloatTensor(sample['velocity']),
            'pressure': torch.FloatTensor(sample['pressure']),
            'vorticity': torch.FloatTensor(sample['vorticity']),
            'refinement_mask': torch.FloatTensor(refinement_mask) if refinement_mask is not None else None,
            'cell_sizes': torch.FloatTensor(cell_sizes) if cell_sizes is not None else None
        }

class AnomalyDataset(FluidDataset):
    """Dataset for anomaly detection"""
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a data sample with anomaly labels if available"""
        sample = super().__getitem__(idx)
        
        # Load anomaly labels if available
        sample_info = self.index[idx]
        data_path = self.data_dir / sample_info['file']
        
        with h5py.File(data_path, 'r') as f:
            anomaly_mask = f['anomaly_mask'][:] if 'anomaly_mask' in f else None
        
        return {
            'velocity': torch.FloatTensor(sample['velocity']),
            'pressure': torch.FloatTensor(sample['pressure']),
            'vorticity': torch.FloatTensor(sample['vorticity']),
            'anomaly_mask': torch.FloatTensor(anomaly_mask) if anomaly_mask is not None else None
        }

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader for the dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

class DataTransform:
    """Base class for data transformations"""
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply transformation to sample"""
        return sample

class Normalize(DataTransform):
    """Normalize data to zero mean and unit variance"""
    
    def __init__(self, stats_file: Union[str, Path]):
        self.stats_file = Path(stats_file)
        self.stats = self._load_stats()

    def _load_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Load normalization statistics"""
        with np.load(self.stats_file) as f:
            return {
                'velocity': {
                    'mean': f['velocity_mean'],
                    'std': f['velocity_std']
                },
                'pressure': {
                    'mean': f['pressure_mean'],
                    'std': f['pressure_std']
                },
                'vorticity': {
                    'mean': f['vorticity_mean'],
                    'std': f['vorticity_std']
                }
            }

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize sample"""
        normalized = {}
        for key in ['velocity', 'pressure', 'vorticity']:
            if key in sample:
                normalized[key] = (
                    sample[key] - self.stats[key]['mean']
                ) / self.stats[key]['std']
        return normalized

class RandomCrop(DataTransform):
    """Random crop of the spatial domain"""
    
    def __init__(self, crop_size: Tuple[int, int]):
        self.crop_size = crop_size

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply random crop to sample"""
        h, w = sample['velocity'].shape[:2]
        new_h, new_w = self.crop_size
        
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        
        cropped = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 2:
                    cropped[key] = value[top:top+new_h, left:left+new_w]
                elif value.ndim == 3:
                    cropped[key] = value[top:top+new_h, left:left+new_w, :]
        
        return cropped

class RandomFlip(DataTransform):
    """Random horizontal and vertical flips"""
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply random flips to sample"""
        # Horizontal flip
        if np.random.rand() > 0.5:
            flipped = {}
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    flipped[key] = np.flip(value, axis=1)
                    if key == 'velocity' and value.ndim == 3:
                        # Flip x-component of velocity
                        flipped[key][..., 0] *= -1
            sample = flipped
        
        # Vertical flip
        if np.random.rand() > 0.5:
            flipped = {}
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    flipped[key] = np.flip(value, axis=0)
                    if key == 'velocity' and value.ndim == 3:
                        # Flip y-component of velocity
                        flipped[key][..., 1] *= -1
            sample = flipped
        
        return sample

class ComposeTransforms:
    """Compose multiple transforms"""
    
    def __init__(self, transforms: List[DataTransform]):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply all transforms in sequence"""
        for transform in self.transforms:
            sample = transform(sample)
        return sample