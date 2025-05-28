import argparse
from datetime import datetime
import logging
from pathlib import Path
import torch
import wandb
from typing import Dict, Optional

from .config import ConfigManager, TrainingConfig
from .data_loader import (
    PINNDataset, MeshDataset, AnomalyDataset,
    create_dataloader, Normalize, RandomCrop, RandomFlip,
    ComposeTransforms
)
from .preprocessing import DataPreprocessor, DataAugmentor
from .metrics import PINNMetrics, MeshMetrics, AnomalyMetrics
from .trainer import ModelTrainer

def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

def prepare_data(config: TrainingConfig) -> Dict[str, torch.utils.data.DataLoader]:
    """Prepare data loaders"""
    logger = logging.getLogger(__name__)
    
    # Preprocess data if needed
    if not (config.data.output_dir / 'statistics.npz').exists():
        logger.info("Preprocessing dataset...")
        preprocessor = DataPreprocessor(
            data_dir=config.data.data_dir,
            output_dir=config.data.output_dir,
            val_split=config.data.val_split,
            test_split=config.data.test_split
        )
        preprocessor.process_dataset()
    
    # Augment data if enabled
    if config.data.use_augmentation:
        logger.info("Augmenting dataset...")
        augmentor = DataAugmentor(
            input_dir=config.data.output_dir,
            output_dir=config.data.output_dir,
            augmentation_factor=config.data.augmentation_factor
        )
        augmentor.augment_dataset()
    
    # Create transforms
    transforms = []
    if config.data.normalize:
        transforms.append(
            Normalize(config.data.output_dir / 'statistics.npz')
        )
    if config.data.crop_size:
        transforms.append(RandomCrop(config.data.crop_size))
    if config.data.use_flips:
        transforms.append(RandomFlip())
    
    transform = ComposeTransforms(transforms) if transforms else None
    
    # Create datasets based on model type
    if config.model.model_type == "pinn":
        dataset_class = PINNDataset
        dataset_params = {
            'num_pde_points': config.model.num_pde_points,
            'num_bc_points': config.model.num_bc_points
        }
    elif config.model.model_type == "mesh_optimizer":
        dataset_class = MeshDataset
        dataset_params = {}
    else:  # anomaly_detector
        dataset_class = AnomalyDataset
        dataset_params = {}
    
    # Create data loaders
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        dataset = dataset_class(
            data_dir=config.data.output_dir,
            split=split,
            transform=transform,
            **dataset_params
        )
        
        dataloaders[split] = create_dataloader(
            dataset,
            batch_size=config.model.batch_size,
            shuffle=(split == 'train'),
            num_workers=config.model.num_workers
        )
    
    return dataloaders

def train_model(
    config: TrainingConfig,
    dataloaders: Dict[str, torch.utils.data.DataLoader]
) -> ModelTrainer:
    """Train the model"""
    logger = logging.getLogger(__name__)
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_type=config.model.model_type,
        config=config.model.__dict__,
        experiment_name=config.experiment_name,
        use_wandb=config.use_wandb
    )
    
    # Initialize metrics
    if config.model.model_type == "pinn":
        metrics = PINNMetrics()
    elif config.model.model_type == "mesh_optimizer":
        metrics = MeshMetrics()
    else:  # anomaly_detector
        metrics = AnomalyMetrics()
    
    # Training loop
    logger.info("Starting training...")
    trainer.train(
        train_data=dataloaders['train'],
        val_data=dataloaders['val'],
        num_epochs=config.model.num_epochs,
        checkpoint_dir=config.output_dir / config.experiment_name / "checkpoints",
        save_best=config.model.save_best
    )
    
    return trainer

def evaluate_model(
    trainer: ModelTrainer,
    test_loader: torch.utils.data.DataLoader,
    config: TrainingConfig
):
    """Evaluate model on test set"""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model on test set...")
    
    # Load best model if available
    best_model_path = (
        config.output_dir / config.experiment_name /
        "checkpoints/best_model.pt"
    )
    if best_model_path.exists():
        trainer.load_checkpoint(best_model_path)
    
    # Evaluate and save predictions
    predictions_dir = config.output_dir / config.experiment_name / "predictions"
    trainer.evaluate(
        test_loader,
        save_predictions=config.save_predictions,
        output_dir=predictions_dir,
        export_format=config.export_format
    )

def main(config_path: Optional[str] = None):
    """Main training function"""
    # Load configuration
    config_manager = ConfigManager()
    if config_path:
        config = config_manager.load_config(config_path)
    else:
        # Use default configuration
        config = config_manager.load_config("configs/default.yaml")
    
    # Set timestamp
    config.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment directory
    experiment_dir = config_manager.create_experiment_dir(config)
    
    # Setup logging
    setup_logging(experiment_dir / "logs")
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {config.experiment_name}")
    
    # Set random seed
    torch.manual_seed(config.seed)
    
    try:
        # Prepare data
        dataloaders = prepare_data(config)
        
        # Train model
        trainer = train_model(config, dataloaders)
        
        # Evaluate model
        evaluate_model(trainer, dataloaders['test'], config)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Clean up
        if config.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NavierFlow AI models")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    main(args.config) 