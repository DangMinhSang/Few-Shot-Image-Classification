"""
Level 1: Baseline Trainer

Standard supervised classification trainer.
"""
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import numpy as np


class BaselineTrainer:
    """
    Trainer for standard supervised classification.
    
    Features:
    - Cosine / Step LR scheduling
    - Best model checkpoint saving
    - Training history logging
    - W&B integration (optional)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
        )
        
        # LR Scheduler
        num_epochs = config.get('num_epochs', 50)
        scheduler_type = config.get('lr_scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        else:
            self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)
        
        # Output
        self.output_dir = Path(config.get('output_dir', 'checkpoints/baseline/'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_val_acc = 0.0
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Train', leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += images.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return {
            'loss': total_loss / total,
            'acc': correct / total,
        }
    
    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        loader = loader or self.val_loader
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item() * images.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += images.size(0)
        
        return {
            'loss': total_loss / total,
            'acc': correct / total,
        }
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        save_every: int = 10,
    ) -> Dict[str, list]:
        """Full training loop."""
        num_epochs = num_epochs or self.config.get('num_epochs', 50)
        
        print(f"Training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()
            self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            
            # Log
            if epoch % 5 == 0 or epoch == 1:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['acc']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['acc']:.4f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )
            
            # Save best
            if val_metrics['acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc']
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
            
            # Periodic save
            if epoch % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth', epoch, val_metrics)
            
            # W&B logging
            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/acc': train_metrics['acc'],
                    'val/loss': val_metrics['loss'],
                    'val/acc': val_metrics['acc'],
                    'lr': self.scheduler.get_last_lr()[0],
                })
        
        print(f"\nTraining complete! Best Val Acc: {self.best_val_acc:.4f}")
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }
        torch.save(ckpt, self.output_dir / filename)
    
    def load_checkpoint(self, filename: str) -> Dict:
        """Load model checkpoint."""
        ckpt = torch.load(self.output_dir / filename, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt
