"""
Level 3: Prototypical Network Trainer

Episodic training for few-shot learning.
"""
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from src.datasets import EpisodeSampler, FewShotEpisode
from src.losses import PrototypicalLoss


class ProtoNetTrainer:
    """
    Episodic trainer for Prototypical Network.
    
    Each "episode" = one few-shot classification task.
    Model is optimized over many episodes per epoch.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_sampler: EpisodeSampler,
        val_sampler: Optional[EpisodeSampler],
        config: Dict[str, Any],
        device: torch.device,
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        self.criterion = PrototypicalLoss()
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=1e-5,
        )
        
        scheduler_type = config.get('lr_scheduler', 'step')
        if scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=config.get('lr_step_size', 20),
                gamma=config.get('lr_gamma', 0.5),
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('num_epochs', 100),
                eta_min=1e-6,
            )
        
        self.output_dir = Path(config.get('output_dir', 'checkpoints/protonet/'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_val_acc = 0.0
    
    def _run_episode(self, episode: FewShotEpisode) -> Dict[str, float]:
        """Run a single few-shot episode."""
        support_images = episode.support_images.to(self.device)
        support_labels = episode.support_labels.to(self.device)
        query_images = episode.query_images.to(self.device)
        query_labels = episode.query_labels.to(self.device)
        
        log_probs, prototypes = self.model(support_images, support_labels, query_images)
        loss = self.criterion(log_probs, query_labels)
        acc = self.criterion.accuracy(log_probs, query_labels)
        
        return {'loss': loss, 'acc': acc, 'prototypes': prototypes}
    
    def train_epoch(self, episodes_per_epoch: int) -> Dict[str, float]:
        """Run training for one epoch (N episodes)."""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        
        pbar = tqdm(range(episodes_per_epoch), desc='ProtoNet Train', leave=False)
        for i in pbar:
            episode = self.train_sampler[i % len(self.train_sampler)]
            
            self.optimizer.zero_grad()
            result = self._run_episode(episode)
            result['loss'].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += result['loss'].item()
            total_acc += result['acc']
            
            pbar.set_postfix({
                'loss': f"{result['loss'].item():.4f}",
                'acc': f"{result['acc']:.4f}"
            })
        
        return {
            'loss': total_loss / episodes_per_epoch,
            'acc': total_acc / episodes_per_epoch,
        }
    
    @torch.no_grad()
    def evaluate(self, n_episodes: int = 200) -> Dict[str, float]:
        """Evaluate on validation episodes."""
        if self.val_sampler is None:
            return {'loss': 0.0, 'acc': 0.0}
        
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        accs = []
        
        for i in range(n_episodes):
            episode = self.val_sampler[i % len(self.val_sampler)]
            result = self._run_episode(episode)
            total_loss += result['loss'].item()
            accs.append(result['acc'])
        
        # Confidence interval
        accs_array = np.array(accs)
        mean_acc = accs_array.mean()
        std_acc = accs_array.std()
        ci95 = 1.96 * std_acc / np.sqrt(n_episodes)
        
        return {
            'loss': total_loss / n_episodes,
            'acc': mean_acc,
            'acc_std': std_acc,
            'ci95': ci95,
        }
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        episodes_per_epoch: Optional[int] = None,
        val_episodes: int = 200,
    ) -> Dict[str, list]:
        """Full episodic training."""
        num_epochs = num_epochs or self.config.get('num_epochs', 100)
        episodes_per_epoch = episodes_per_epoch or self.config.get('episodes_per_epoch', 100)
        
        print(f"ProtoNet Training: {num_epochs} epochs x {episodes_per_epoch} episodes")
        start = time.time()
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(episodes_per_epoch)
            val_metrics = self.evaluate(val_episodes)
            self.scheduler.step()
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            
            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.time() - start
                print(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['acc']:.4f} | "
                    f"Val Acc: {val_metrics['acc']:.4f} ± {val_metrics.get('ci95', 0):.4f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f} | "
                    f"Time: {elapsed:.0f}s"
                )
            
            if val_metrics['acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics,
                    'config': self.config,
                }, self.output_dir / 'best_model.pth')
                print(f"  >> New best! Val Acc: {self.best_val_acc:.4f}")
            
            if self.use_wandb:
                import wandb
                wandb.log({'epoch': epoch, **{f'train/{k}': v for k, v in train_metrics.items()}, **{f'val/{k}': v for k, v in val_metrics.items()}})
        
        print(f"\nProtoNet training complete! Best Val Acc: {self.best_val_acc:.4f}")
        return self.history
