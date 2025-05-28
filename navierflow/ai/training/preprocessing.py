import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import h5py
from pathlib import Path
import json
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DataPreprocessor:
    """Preprocessor for fluid simulation data"""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize statistics
        self.stats = {
            'velocity': {'mean': None, 'std': None},
            'pressure': {'mean': None, 'std': None},
            'vorticity': {'mean': None, 'std': None}
        }

    def process_dataset(self):
        """Process the entire dataset"""
        # Find all data files
        data_files = list(self.data_dir.glob('*.h5'))
        self.logger.info(f"Found {len(data_files)} data files")
        
        # Split data into train/val/test
        train_files, test_files = train_test_split(
            data_files,
            test_size=self.test_split,
            random_state=self.seed
        )
        
        train_files, val_files = train_test_split(
            train_files,
            test_size=self.val_split / (1 - self.test_split),
            random_state=self.seed
        )
        
        # Process each split
        self.logger.info("Processing training data...")
        train_index = self._process_split(train_files, 'train')
        
        self.logger.info("Processing validation data...")
        val_index = self._process_split(val_files, 'val')
        
        self.logger.info("Processing test data...")
        test_index = self._process_split(test_files, 'test')
        
        # Save data splits
        self._save_splits(train_index, val_index, test_index)
        
        # Save statistics
        self._save_statistics()

    def _process_split(
        self,
        files: List[Path],
        split: str
    ) -> List[Dict]:
        """Process files for a data split"""
        index = []
        
        for file in tqdm(files):
            # Load and process file
            processed = self._process_file(file)
            
            if processed is not None:
                # Save processed data
                output_file = self.output_dir / f"{split}_{file.stem}.h5"
                self._save_processed_data(processed, output_file)
                
                # Update index
                index.append({
                    'file': output_file.name,
                    'original_file': str(file),
                    'shape': {
                        key: value.shape
                        for key, value in processed.items()
                        if isinstance(value, np.ndarray)
                    }
                })
        
        return index

    def _process_file(self, file: Path) -> Optional[Dict[str, np.ndarray]]:
        """Process a single data file"""
        try:
            with h5py.File(file, 'r') as f:
                data = {
                    'velocity': f['velocity'][:],
                    'pressure': f['pressure'][:],
                    'vorticity': f['vorticity'][:]
                }
                
                # Load optional fields if available
                for field in ['refinement_mask', 'cell_sizes', 'anomaly_mask']:
                    if field in f:
                        data[field] = f[field][:]
                
                # Update statistics
                self._update_statistics(data)
                
                return data
                
        except Exception as e:
            self.logger.warning(f"Error processing {file}: {str(e)}")
            return None

    def _update_statistics(self, data: Dict[str, np.ndarray]):
        """Update running statistics"""
        for key in ['velocity', 'pressure', 'vorticity']:
            if key in data:
                values = data[key].reshape(-1, data[key].shape[-1] if data[key].ndim > 2 else 1)
                
                # Update mean
                if self.stats[key]['mean'] is None:
                    self.stats[key]['mean'] = values.mean(axis=0)
                    self.stats[key]['std'] = values.std(axis=0)
                else:
                    # Online mean and variance updates
                    n = len(self.stats[key]['mean'])
                    m = len(values)
                    
                    # Update mean
                    delta = values.mean(axis=0) - self.stats[key]['mean']
                    self.stats[key]['mean'] += delta * m / (n + m)
                    
                    # Update std
                    old_ssd = self.stats[key]['std'] ** 2 * n
                    new_ssd = values.std(axis=0) ** 2 * m
                    combined_ssd = old_ssd + new_ssd + \
                        (delta ** 2) * n * m / (n + m)
                    self.stats[key]['std'] = np.sqrt(combined_ssd / (n + m))

    def _save_processed_data(
        self,
        data: Dict[str, np.ndarray],
        output_file: Path
    ):
        """Save processed data to file"""
        with h5py.File(output_file, 'w') as f:
            for key, value in data.items():
                f.create_dataset(
                    key,
                    data=value,
                    compression='gzip',
                    compression_opts=9
                )

    def _save_splits(
        self,
        train_index: List[Dict],
        val_index: List[Dict],
        test_index: List[Dict]
    ):
        """Save data split indices"""
        splits = {
            'train': train_index,
            'val': val_index,
            'test': test_index
        }
        
        for split, index in splits.items():
            index_file = self.output_dir / f"{split}_index.json"
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=4)
            
            self.logger.info(
                f"Saved {split} index with {len(index)} samples"
            )

    def _save_statistics(self):
        """Save dataset statistics"""
        stats_file = self.output_dir / 'statistics.npz'
        
        save_dict = {}
        for key in self.stats:
            save_dict[f'{key}_mean'] = self.stats[key]['mean']
            save_dict[f'{key}_std'] = self.stats[key]['std']
        
        np.savez(stats_file, **save_dict)
        self.logger.info(f"Saved dataset statistics to {stats_file}")

class DataAugmentor:
    """Data augmentation for fluid simulation data"""
    
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        augmentation_factor: int = 2
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.augmentation_factor = augmentation_factor
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def augment_dataset(self):
        """Augment the entire dataset"""
        # Process each split
        for split in ['train', 'val', 'test']:
            self.logger.info(f"Augmenting {split} split...")
            self._augment_split(split)

    def _augment_split(self, split: str):
        """Augment data for a specific split"""
        # Load index
        index_file = self.input_dir / f"{split}_index.json"
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        new_index = []
        
        for sample_info in tqdm(index):
            input_file = self.input_dir / sample_info['file']
            
            # Load original data
            with h5py.File(input_file, 'r') as f:
                original_data = {
                    key: f[key][:]
                    for key in f.keys()
                }
            
            # Add original data to new index
            new_index.append(sample_info)
            
            # Generate augmented samples
            for i in range(self.augmentation_factor - 1):
                augmented_data = self._augment_sample(original_data)
                
                # Save augmented data
                output_file = self.output_dir / f"{split}_aug_{i}_{sample_info['file']}"
                self._save_augmented_data(augmented_data, output_file)
                
                # Update index
                aug_info = sample_info.copy()
                aug_info['file'] = output_file.name
                aug_info['augmented'] = True
                aug_info['original_file'] = sample_info['file']
                new_index.append(aug_info)
        
        # Save new index
        output_index_file = self.output_dir / f"{split}_index.json"
        with open(output_index_file, 'w') as f:
            json.dump(new_index, f, indent=4)
        
        self.logger.info(
            f"Augmented {split} split: {len(index)} -> {len(new_index)} samples"
        )

    def _augment_sample(
        self,
        data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply augmentation to a single sample"""
        augmented = {}
        
        # Random rotation angle
        angle = np.random.uniform(0, 360)
        
        # Random scaling factor
        scale = np.random.uniform(0.8, 1.2)
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Apply transformations based on data type
                if key == 'velocity':
                    # Rotate velocity field
                    augmented[key] = self._rotate_vector_field(
                        value, angle
                    )
                    # Scale velocities
                    augmented[key] *= scale
                    
                elif key in ['pressure', 'vorticity']:
                    # Rotate scalar fields
                    augmented[key] = self._rotate_scalar_field(
                        value, angle
                    )
                    
                elif key in ['refinement_mask', 'anomaly_mask']:
                    # Rotate binary masks
                    augmented[key] = self._rotate_scalar_field(
                        value, angle
                    ) > 0.5
                    
                else:
                    # Other fields are copied as is
                    augmented[key] = value.copy()
        
        return augmented

    def _rotate_vector_field(
        self,
        field: np.ndarray,
        angle: float
    ) -> np.ndarray:
        """Rotate a vector field"""
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # Reshape field for rotation
        shape = field.shape
        vectors = field.reshape(-1, 2)
        
        # Apply rotation
        rotated = np.dot(vectors, R.T)
        
        return rotated.reshape(shape)

    def _rotate_scalar_field(
        self,
        field: np.ndarray,
        angle: float
    ) -> np.ndarray:
        """Rotate a scalar field"""
        from scipy.ndimage import rotate
        return rotate(field, angle, reshape=False)

    def _save_augmented_data(
        self,
        data: Dict[str, np.ndarray],
        output_file: Path
    ):
        """Save augmented data to file"""
        with h5py.File(output_file, 'w') as f:
            for key, value in data.items():
                f.create_dataset(
                    key,
                    data=value,
                    compression='gzip',
                    compression_opts=9
                ) 