#!/usr/bin/env python3
"""
Level 3: Train Prototypical Network

Usage:
    python train_protonet.py --config config.yaml
    python train_protonet.py --config config.yaml --n_way 5 --k_shot 5
    python train_protonet.py --config config.yaml --augment_support
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets import FlowersDataset, EpisodeSampler
from src.models import PrototypicalNetwork
from src.trainers import ProtoNetTrainer
from src.utils import set_seed, load_config, save_results, compute_few_shot_accuracy
from src.utils.visualization import plot_learning_curves


def main(args):
    config = load_config(args.config)
    proto_cfg = config.get('protonet', {})
    
    # CLI overrides
    if args.n_way:
        proto_cfg['n_way'] = args.n_way
    if args.k_shot:
        proto_cfg['k_shot'] = args.k_shot
    if args.epochs:
        proto_cfg['num_epochs'] = args.epochs
    if args.distance:
        proto_cfg['distance'] = args.distance
    
    set_seed(config['project'].get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = config['dataset'].get('data_dir', 'data/')
    image_size = config['dataset'].get('image_size', 224)
    n_way = proto_cfg.get('n_way', 5)
    k_shot = proto_cfg.get('k_shot', 5)
    n_query = proto_cfg.get('n_query', 15)
    
    print(f"Episode setup: {n_way}-way {k_shot}-shot, {n_query} queries/class")
    
    # Datasets
    print("Loading datasets...")
    train_base = FlowersDataset(root=data_dir, split='train', image_size=image_size)
    val_base = FlowersDataset(root=data_dir, split='val', image_size=image_size)
    
    # Episode samplers
    train_sampler = EpisodeSampler(
        class_to_samples=train_base.class_to_samples,
        idx_to_class=train_base.idx_to_class,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        n_episodes=proto_cfg.get('episodes_per_epoch', 100),
        augment_support=args.augment_support,
        image_size=image_size,
    )
    val_sampler = EpisodeSampler(
        class_to_samples=val_base.class_to_samples,
        idx_to_class=val_base.idx_to_class,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        n_episodes=200,
        image_size=image_size,
    )
    
    print(f"Train classes: {len(train_sampler.valid_classes)} | Val classes: {len(val_sampler.valid_classes)}")
    
    # Model
    model = PrototypicalNetwork(
        backbone_name=proto_cfg.get('backbone', 'resnet18'),
        pretrained=proto_cfg.get('pretrained', True),
        embedding_dim=proto_cfg.get('embedding_dim', 256),
        distance=proto_cfg.get('distance', 'euclidean'),
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Suffix for saving (with/without augmentation)
    suffix = '_aug' if args.augment_support else ''
    proto_cfg['output_dir'] = proto_cfg.get('output_dir', 'checkpoints/protonet/').rstrip('/') + suffix + '/'
    
    # W&B
    use_wandb = config.get('logging', {}).get('use_wandb', False)
    if use_wandb:
        import wandb
        run_name = f'protonet_{n_way}way_{k_shot}shot{suffix}'
        wandb.init(project=config['project']['name'], name=run_name, config=proto_cfg)
    
    # Trainer
    trainer = ProtoNetTrainer(
        model=model,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        config=proto_cfg,
        device=device,
        use_wandb=use_wandb,
    )
    
    history = trainer.train(
        num_epochs=proto_cfg.get('num_epochs', 100),
        episodes_per_epoch=proto_cfg.get('episodes_per_epoch', 100),
    )
    
    # Final evaluation
    print("\nRunning final evaluation (600 episodes)...")
    eval_cfg = config.get('evaluation', {})
    shots = eval_cfg.get('shots', [1, 5, 10])
    
    final_results = {'model': f'protonet{suffix}', 'config': proto_cfg}
    for shot in shots:
        eval_sampler = EpisodeSampler(
            class_to_samples=val_base.class_to_samples,
            idx_to_class=val_base.idx_to_class,
            n_way=n_way,
            k_shot=shot,
            n_query=n_query,
            n_episodes=600,
            image_size=image_size,
        )
        acc, ci = compute_few_shot_accuracy(model, eval_sampler, n_episodes=600, device=device)
        print(f"  {shot}-shot: {acc*100:.2f}% ± {ci*100:.2f}%")
        final_results[f'{shot}-shot'] = {'acc': acc, 'ci95': ci}
    
    # Save
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    save_results(final_results, str(results_dir / f'protonet{suffix}_results.json'))
    plot_learning_curves(history, title=f'ProtoNet{suffix}',
                        save_path=str(results_dir / f'protonet{suffix}_curves.png'))
    
    print(f"\nProtoNet{suffix} Training Complete!")
    print(f"Best Val Acc: {trainer.best_val_acc:.4f}")
    
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--n_way', type=int, default=None)
    parser.add_argument('--k_shot', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--distance', type=str, choices=['euclidean', 'cosine'], default=None)
    parser.add_argument('--augment_support', action='store_true', help='Augment support images')
    args = parser.parse_args()
    main(args)
