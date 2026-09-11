#!/usr/bin/env python3
"""
Level 1: Train Baseline ResNet18 Classifier

Usage:
    python train_baseline.py --config config.yaml
    python train_baseline.py --config config.yaml --epochs 50 --lr 0.001
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets import FlowersDataset
from src.models import BaselineClassifier
from src.trainers import BaselineTrainer
from src.utils import set_seed, load_config, save_results
from src.utils.visualization import plot_learning_curves


def main(args):
    # Load config
    config = load_config(args.config)
    baseline_cfg = config.get('baseline', {})
    
    # Override with CLI args
    if args.epochs:
        baseline_cfg['num_epochs'] = args.epochs
    if args.lr:
        baseline_cfg['lr'] = args.lr
    if args.batch_size:
        baseline_cfg['batch_size'] = args.batch_size
    
    # Setup
    set_seed(config['project'].get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = config['dataset'].get('data_dir', 'data/')
    image_size = config['dataset'].get('image_size', 224)
    batch_size = baseline_cfg.get('batch_size', 32)
    num_workers = config['dataset'].get('num_workers', 4)
    
    # Datasets
    print("Loading datasets...")
    train_dataset = FlowersDataset(
        root=data_dir,
        split='train',
        image_size=image_size,
    )
    val_dataset = FlowersDataset(
        root=data_dir,
        split='val',
        image_size=image_size,
    )
    
    print(f"Train: {len(train_dataset)} images | {train_dataset.num_classes} classes")
    print(f"Val:   {len(val_dataset)} images")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    
    # Model
    model = BaselineClassifier(
        num_classes=train_dataset.num_classes,
        backbone_name=baseline_cfg.get('backbone', 'resnet18'),
        pretrained=baseline_cfg.get('pretrained', True),
        dropout=0.3,
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # W&B
    use_wandb = config.get('logging', {}).get('use_wandb', False)
    if use_wandb:
        import wandb
        wandb.init(
            project=config['project']['name'],
            name='baseline',
            config=baseline_cfg,
        )
    
    # Trainer
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=baseline_cfg,
        device=device,
        use_wandb=use_wandb,
    )
    
    # Train
    history = trainer.train()
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    save_results({
        'model': 'baseline',
        'config': baseline_cfg,
        'history': history,
        'best_val_acc': trainer.best_val_acc,
    }, str(results_dir / 'baseline_history.json'))
    
    # Plot
    plot_learning_curves(
        history,
        title='Baseline ResNet18',
        save_path=str(results_dir / 'baseline_learning_curves.png'),
    )
    
    print(f"\n{'='*50}")
    print(f"Baseline Training Complete")
    print(f"Best Val Accuracy: {trainer.best_val_acc:.4f} ({trainer.best_val_acc*100:.2f}%)")
    print(f"Results saved to {results_dir}/")
    
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Baseline Classifier')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()
    main(args)
