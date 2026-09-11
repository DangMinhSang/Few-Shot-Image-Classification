"""
Level 2: Siamese Network Trainer
"""
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.losses import ContrastiveLoss, TripletLoss


class SiameseTrainer:
    """
    Trainer for Siamese Network with Contrastive or Triplet Loss.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
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
        
        # Loss
        loss_type = config.get('loss', 'contrastive')
        if loss_type == 'contrastive':
            self.criterion = ContrastiveLoss(margin=config.get('margin', 1.0))
        else:
            self.criterion = TripletLoss(margin=config.get('margin', 0.3))
        self.loss_type = loss_type
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=1e-5,
        )
        
        num_epochs = config.get('num_epochs', 50)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        
        self.output_dir = Path(config.get('output_dir', 'checkpoints/siamese/'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {'train_loss': [], 'val_loss': [], 'pos_dist': [], 'neg_dist': []}
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_pos_dist = 0.0
        total_neg_dist = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Siamese Train', leave=False)
        for batch in pbar:
            if len(batch) == 3:
                img1, img2, labels = batch
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                emb1, emb2, distance = self.model(img1, img2)
                loss = self.criterion(emb1, emb2, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                metrics = self.criterion.get_metrics(emb1, emb2, labels) if hasattr(self.criterion, 'get_metrics') else {}
                total_pos_dist += metrics.get('pos_dist', 0)
                total_neg_dist += metrics.get('neg_dist', 0)
                n_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return {
            'loss': total_loss / n_batches,
            'pos_dist': total_pos_dist / n_batches,
            'neg_dist': total_neg_dist / n_batches,
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {'loss': 0.0}
        
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for img1, img2, labels in self.val_loader:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            emb1, emb2, _ = self.model(img1, img2)
            loss = self.criterion(emb1, emb2, labels)
            total_loss += loss.item()
            n_batches += 1
        
        return {'loss': total_loss / max(n_batches, 1)}
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, list]:
        num_epochs = num_epochs or self.config.get('num_epochs', 50)
        print(f"Training Siamese Network for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()
            self.scheduler.step()
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['pos_dist'].append(train_metrics.get('pos_dist', 0))
            self.history['neg_dist'].append(train_metrics.get('neg_dist', 0))
            
            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Pos Dist: {train_metrics.get('pos_dist', 0):.4f} | "
                    f"Neg Dist: {train_metrics.get('neg_dist', 0):.4f}"
                )
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'metrics': val_metrics,
                }, self.output_dir / 'best_model.pth')
            
            if self.use_wandb:
                import wandb
                wandb.log({'epoch': epoch, **{f'train/{k}': v for k, v in train_metrics.items()}, 'val/loss': val_metrics['loss']})
        
        print(f"\nSiamese training complete! Best Val Loss: {self.best_val_loss:.4f}")
        return self.history
