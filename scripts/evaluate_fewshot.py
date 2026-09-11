#!/usr/bin/env python3
"""
Evaluate a trained model in few-shot setting.

Usage:
    python scripts/evaluate_fewshot.py --model protonet --checkpoint checkpoints/protonet/best_model.pth
    python scripts/evaluate_fewshot.py --model baseline --checkpoint checkpoints/baseline/best_model.pth
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import FlowersDataset, EpisodeSampler
from src.models import BaselineClassifier, SiameseNetwork, PrototypicalNetwork
from src.utils import set_seed, load_config, compute_few_shot_accuracy


def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    data_dir = config['dataset'].get('data_dir', 'data/')
    image_size = config['dataset'].get('image_size', 224)
    n_way = args.n_way
    n_query = args.n_query
    shots = [int(s) for s in args.shots.split(',')]
    
    # Load val dataset
    val_dataset = FlowersDataset(root=data_dir, split=args.split, image_size=image_size)
    print(f"Evaluating on {args.split}: {val_dataset.num_classes} classes")
    
    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    if args.model == 'protonet':
        model = PrototypicalNetwork(backbone_name='resnet18', pretrained=False,
                                    embedding_dim=256, distance=args.distance)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"\nPrototypical Network ({args.distance} distance):")
        print(f"{'Shot':<10} {'Accuracy':>12} {'95% CI':>12}")
        print("-" * 35)
        
        for shot in shots:
            sampler = EpisodeSampler(
                class_to_samples=val_dataset.class_to_samples,
                idx_to_class=val_dataset.idx_to_class,
                n_way=n_way, k_shot=shot, n_query=n_query,
                n_episodes=args.n_episodes, image_size=image_size,
            )
            acc, ci = compute_few_shot_accuracy(model, sampler, args.n_episodes, device)
            print(f"{shot}-shot{'':<4} {acc*100:>10.2f}% {ci*100:>10.2f}%")
    
    elif args.model == 'baseline':
        model = BaselineClassifier(num_classes=val_dataset.num_classes,
                                   backbone_name='resnet18', pretrained=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"\nBaseline (ResNet18, nearest-prototype):")
        print(f"{'Shot':<10} {'Accuracy':>12} {'95% CI':>12}")
        print("-" * 35)
        
        for shot in shots:
            sampler = EpisodeSampler(
                class_to_samples=val_dataset.class_to_samples,
                idx_to_class=val_dataset.idx_to_class,
                n_way=n_way, k_shot=shot, n_query=n_query,
                n_episodes=args.n_episodes, image_size=image_size,
            )
            accs = []
            with torch.no_grad():
                for i in range(args.n_episodes):
                    episode = sampler[i % len(sampler)]
                    n_way_ep, k_shot_ep = episode.support_images.shape[:2]
                    support_flat = episode.support_images.view(n_way_ep * k_shot_ep, *episode.support_images.shape[2:]).to(device)
                    support_emb = model.extract_features(support_flat).view(n_way_ep, k_shot_ep, -1).mean(1)
                    query_emb = model.extract_features(episode.query_images.to(device))
                    dists = ((query_emb.unsqueeze(1) - support_emb.unsqueeze(0)) ** 2).sum(-1)
                    preds = dists.argmin(-1)
                    accs.append((preds == episode.query_labels.to(device)).float().mean().item())
            acc = np.mean(accs)
            ci = 1.96 * np.std(accs) / np.sqrt(args.n_episodes)
            print(f"{shot}-shot{'':<4} {acc*100:>10.2f}% {ci*100:>10.2f}%")
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['baseline', 'siamese', 'protonet'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_query', type=int, default=15)
    parser.add_argument('--shots', type=str, default='1,5,10')
    parser.add_argument('--n_episodes', type=int, default=600)
    parser.add_argument('--distance', type=str, default='euclidean')
    args = parser.parse_args()
    main(args)
