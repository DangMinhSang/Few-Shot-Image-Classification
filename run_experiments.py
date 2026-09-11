#!/usr/bin/env python3
"""
Level 4: Full Experiment Runner

Runs all models, collects results, and generates comprehensive comparison plots.

Usage:
    python run_experiments.py --config config.yaml
    python run_experiments.py --config config.yaml --skip_training
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets import FlowersDataset, EpisodeSampler
from src.models import BaselineClassifier, SiameseNetwork, PrototypicalNetwork
from src.utils import set_seed, load_config, save_results, compute_few_shot_accuracy
from src.utils.visualization import (
    plot_learning_curves,
    plot_experiment_results,
    plot_confusion_matrix,
    plot_tsne,
)


def evaluate_baseline_fewshot(
    model_path: str,
    val_dataset: FlowersDataset,
    n_way: int,
    shots: List[int],
    n_query: int,
    n_episodes: int,
    device: torch.device,
    image_size: int = 224,
) -> Dict:
    """Evaluate baseline in few-shot regime using nearest-prototype."""
    ckpt = torch.load(model_path, map_location=device)
    num_classes = val_dataset.num_classes
    model = BaselineClassifier(num_classes=num_classes, backbone_name='resnet18', pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    results = {}

    for shot in shots:
        accs = []
        sampler = EpisodeSampler(
            class_to_samples=val_dataset.class_to_samples,
            idx_to_class=val_dataset.idx_to_class,
            n_way=n_way, k_shot=shot, n_query=n_query,
            n_episodes=n_episodes, image_size=image_size,
        )

        with torch.no_grad():
            for i in range(n_episodes):
                episode = sampler[i % len(sampler)]
                n_way_ep, k_shot_ep = episode.support_images.shape[:2]
                support_flat = episode.support_images.view(n_way_ep * k_shot_ep, *episode.support_images.shape[2:]).to(device)
                support_emb = model.extract_features(support_flat).view(n_way_ep, k_shot_ep, -1).mean(dim=1)
                query_emb = model.extract_features(episode.query_images.to(device))
                diff = query_emb.unsqueeze(1) - support_emb.unsqueeze(0)
                dists = (diff ** 2).sum(dim=-1)
                preds = dists.argmin(dim=-1)
                query_labels = episode.query_labels.to(device)
                acc = (preds == query_labels).float().mean().item()
                accs.append(acc)

        mean_acc = np.mean(accs)
        ci95 = 1.96 * np.std(accs) / np.sqrt(n_episodes)
        results[f'{shot}-shot'] = {'acc': mean_acc, 'ci95': ci95}
        print(f"  Baseline {shot}-shot: {mean_acc*100:.2f}% ± {ci95*100:.2f}%")

    return results


def evaluate_siamese_fewshot(
    model_path: str,
    val_dataset: FlowersDataset,
    n_way: int,
    shots: List[int],
    n_query: int,
    n_episodes: int,
    device: torch.device,
    image_size: int = 224,
) -> Dict:
    """Evaluate Siamese Network in few-shot regime."""
    ckpt = torch.load(model_path, map_location=device)
    model = SiameseNetwork(backbone_name='resnet18', pretrained=False, embedding_dim=256)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    results = {}

    for shot in shots:
        accs = []
        sampler = EpisodeSampler(
            class_to_samples=val_dataset.class_to_samples,
            idx_to_class=val_dataset.idx_to_class,
            n_way=n_way, k_shot=shot, n_query=n_query,
            n_episodes=n_episodes, image_size=image_size,
        )

        with torch.no_grad():
            for i in range(n_episodes):
                episode = sampler[i % len(sampler)]
                n_way_ep, k_shot_ep = episode.support_images.shape[:2]
                support_flat = episode.support_images.view(n_way_ep * k_shot_ep, *episode.support_images.shape[2:]).to(device)
                support_emb = model.encode(support_flat).view(n_way_ep, k_shot_ep, -1).mean(dim=1)
                query_emb = model.encode(episode.query_images.to(device))
                diff = query_emb.unsqueeze(1) - support_emb.unsqueeze(0)
                dists = (diff ** 2).sum(dim=-1)
                preds = dists.argmin(dim=-1)
                query_labels = episode.query_labels.to(device)
                acc = (preds == query_labels).float().mean().item()
                accs.append(acc)

        mean_acc = np.mean(accs)
        ci95 = 1.96 * np.std(accs) / np.sqrt(n_episodes)
        results[f'{shot}-shot'] = {'acc': mean_acc, 'ci95': ci95}
        print(f"  Siamese {shot}-shot: {mean_acc*100:.2f}% ± {ci95*100:.2f}%")

    return results


def evaluate_protonet(
    model_path: str,
    val_dataset: FlowersDataset,
    n_way: int,
    shots: List[int],
    n_query: int,
    n_episodes: int,
    device: torch.device,
    image_size: int = 224,
    distance: str = 'euclidean',
) -> Dict:
    """Evaluate ProtoNet in few-shot regime."""
    ckpt = torch.load(model_path, map_location=device)
    model = PrototypicalNetwork(backbone_name='resnet18', pretrained=False,
                                embedding_dim=256, distance=distance)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    results = {}

    for shot in shots:
        sampler = EpisodeSampler(
            class_to_samples=val_dataset.class_to_samples,
            idx_to_class=val_dataset.idx_to_class,
            n_way=n_way, k_shot=shot, n_query=n_query,
            n_episodes=n_episodes, image_size=image_size,
        )
        acc, ci = compute_few_shot_accuracy(model, sampler, n_episodes=n_episodes, device=device)
        results[f'{shot}-shot'] = {'acc': acc, 'ci95': ci}
        print(f"  ProtoNet {shot}-shot: {acc*100:.2f}% ± {ci*100:.2f}%")

    return results


def generate_comparison_table(all_results: Dict) -> pd.DataFrame:
    """Generate comparison DataFrame."""
    rows = []
    shots = ['1-shot', '5-shot', '10-shot']
    for model_name, model_results in all_results.items():
        row = {'Model': model_name}
        for shot in shots:
            if shot in model_results:
                acc = model_results[shot]['acc'] * 100
                ci = model_results[shot].get('ci95', 0) * 100
                row[shot] = f"{acc:.2f}% ± {ci:.2f}%"
            else:
                row[shot] = '-'
        rows.append(row)
    return pd.DataFrame(rows)


def main(args):
    config = load_config(args.config)
    set_seed(config['project'].get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = config['dataset'].get('data_dir', 'data/')
    image_size = config['dataset'].get('image_size', 224)
    eval_cfg = config.get('evaluation', {})
    n_way = eval_cfg.get('n_way', 5)
    shots = eval_cfg.get('shots', [1, 5, 10])
    n_query = eval_cfg.get('n_query', 15)
    n_episodes = eval_cfg.get('n_episodes', 600)

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    if not args.skip_training:
        print("\n" + "="*60)
        print("STEP 1: Training Baseline...")
        print("="*60)
        subprocess.run([sys.executable, 'train_baseline.py', '--config', args.config], check=True)

        print("\n" + "="*60)
        print("STEP 2: Training Siamese Network...")
        print("="*60)
        subprocess.run([sys.executable, 'train_siamese.py', '--config', args.config], check=True)

        print("\n" + "="*60)
        print("STEP 3: Training ProtoNet (no aug)...")
        print("="*60)
        subprocess.run([sys.executable, 'train_protonet.py', '--config', args.config], check=True)

        print("\n" + "="*60)
        print("STEP 4: Training ProtoNet (with aug)...")
        print("="*60)
        subprocess.run([sys.executable, 'train_protonet.py', '--config', args.config, '--augment_support'], check=True)

    # Evaluate all
    print("\n" + "="*60)
    print("EVALUATING ALL MODELS")
    print("="*60)

    val_dataset = FlowersDataset(root=data_dir, split='val', image_size=image_size)
    all_results = {}

    baseline_ckpt = 'checkpoints/baseline/best_model.pth'
    if Path(baseline_ckpt).exists():
        print("\nBaseline (ResNet18):")
        all_results['ResNet18'] = evaluate_baseline_fewshot(
            baseline_ckpt, val_dataset, n_way, shots, n_query, n_episodes, device, image_size)

    siamese_ckpt = 'checkpoints/siamese/best_model.pth'
    if Path(siamese_ckpt).exists():
        print("\nSiamese Network:")
        all_results['Siamese'] = evaluate_siamese_fewshot(
            siamese_ckpt, val_dataset, n_way, shots, n_query, n_episodes, device, image_size)

    protonet_ckpt = 'checkpoints/protonet/best_model.pth'
    if Path(protonet_ckpt).exists():
        print("\nPrototypical Network:")
        all_results['ProtoNet'] = evaluate_protonet(
            protonet_ckpt, val_dataset, n_way, shots, n_query, n_episodes, device, image_size)

    protonet_aug_ckpt = 'checkpoints/protonet_aug/best_model.pth'
    if Path(protonet_aug_ckpt).exists():
        print("\nProtoNet + Augmentation:")
        all_results['ProtoNet+Aug'] = evaluate_protonet(
            protonet_aug_ckpt, val_dataset, n_way, shots, n_query, n_episodes, device, image_size)

    if not all_results:
        print("No trained models found. Run with --skip_training=False first.")
        return

    save_results(all_results, str(results_dir / 'all_results.json'))

    df = generate_comparison_table(all_results)
    print("\n" + "="*60)
    print("RESULTS COMPARISON TABLE")
    print("="*60)
    print(df.to_string(index=False))
    df.to_csv(str(results_dir / 'comparison_table.csv'), index=False)

    plot_experiment_results(
        all_results, shots=shots,
        save_path=str(results_dir / 'experiment_comparison.png'),
    )

    print(f"\nResults saved to {results_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Full Experiments')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--skip_training', action='store_true')
    args = parser.parse_args()
    main(args)
