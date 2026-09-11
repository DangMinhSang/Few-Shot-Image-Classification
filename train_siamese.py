#!/usr/bin/env python3
"""
Level 2: Train Siamese Network

Usage:
    python train_siamese.py --config config.yaml
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets import FlowersDataset, SiameseDataset
from src.models import SiameseNetwork
from src.trainers import SiameseTrainer
from src.utils import set_seed, load_config, save_results
from src.utils.visualization import plot_learning_curves


def main(args):
    config = load_config(args.config)
    siamese_cfg = config.get('siamese', {})
    
    if args.epochs:
        siamese_cfg['num_epochs'] = args.epochs
    
    set_seed(config['project'].get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = config['dataset'].get('data_dir', 'data/')
    image_size = config['dataset'].get('image_size', 224)
    num_workers = config['dataset'].get('num_workers', 4)
    
    # Load base datasets to get class info
    print("Loading datasets...")
    train_base = FlowersDataset(root=data_dir, split='train', image_size=image_size)
    val_base = FlowersDataset(root=data_dir, split='val', image_size=image_size)
    
    # Create Siamese datasets
    train_dataset = SiameseDataset(
        class_to_samples=train_base.class_to_samples,
        n_pairs=15000,
        positive_ratio=0.5,
        image_size=image_size,
    )
    val_dataset = SiameseDataset(
        class_to_samples=val_base.class_to_samples,
        n_pairs=3000,
        positive_ratio=0.5,
        image_size=image_size,
    )
    
    print(f"Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}")
    
    batch_size = siamese_cfg.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Model
    model = SiameseNetwork(
        backbone_name=siamese_cfg.get('backbone', 'resnet18'),
        pretrained=siamese_cfg.get('pretrained', True),
        embedding_dim=siamese_cfg.get('embedding_dim', 256),
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # W&B
    use_wandb = config.get('logging', {}).get('use_wandb', False)
    if use_wandb:
        import wandb
        wandb.init(project=config['project']['name'], name='siamese', config=siamese_cfg)
    
    # Trainer
    trainer = SiameseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=siamese_cfg,
        device=device,
        use_wandb=use_wandb,
    )
    
    history = trainer.train()
    
    # Save
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    save_results({
        'model': 'siamese',
        'config': siamese_cfg,
        'history': history,
    }, str(results_dir / 'siamese_history.json'))
    
    fig = plot_learning_curves(history, title='Siamese Network',
                               save_path=str(results_dir / 'siamese_learning_curves.png'))
    
    print(f"\nSiamese Training Complete! Best Val Loss: {trainer.best_val_loss:.4f}")
    
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    main(args)
