#!/usr/bin/env python3
"""
Visualize experiment results and generate publication-ready figures.

Usage:
    python scripts/visualize_results.py
    python scripts/visualize_results.py --results_dir results/
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_all(results_dir: str = 'results/'):
    results_path = Path(results_dir)
    
    # Collect available histories
    histories = {}
    for name, filename in [
        ('Baseline', 'baseline_history.json'),
        ('Siamese', 'siamese_history.json'),
        ('ProtoNet', 'protonet_results.json'),
        ('ProtoNet+Aug', 'protonet_aug_results.json'),
    ]:
        filepath = results_path / filename
        if filepath.exists():
            data = load_json(str(filepath))
            if 'history' in data:
                histories[name] = data['history']
    
    # Plot all learning curves
    if histories:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, (name, history) in enumerate(histories.items()):
            ax = axes[i]
            epochs = range(1, len(history.get('train_loss', [])) + 1)
            
            if 'train_loss' in history:
                ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
            if 'val_loss' in history:
                ax.plot(epochs, history['val_loss'], 'r--', label='Val', linewidth=2)
            
            ax.set_title(name, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Training Curves Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(results_path / 'all_learning_curves.png'), dpi=150, bbox_inches='tight')
        print("Saved: all_learning_curves.png")
    
    # Comparison bar chart
    all_results_file = results_path / 'all_results.json'
    if all_results_file.exists():
        all_results = load_json(str(all_results_file))
        shots = [1, 5, 10]
        models = list(all_results.keys())
        
        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(shots))
        width = 0.8 / len(models)
        colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
        
        for i, model in enumerate(models):
            accs = [all_results[model].get(f'{s}-shot', {}).get('acc', 0) * 100 for s in shots]
            cis = [all_results[model].get(f'{s}-shot', {}).get('ci95', 0) * 100 for s in shots]
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, accs, width*0.9, label=model, color=colors[i % 4], alpha=0.85)
            ax.errorbar(x + offset, accs, yerr=cis, fmt='none', color='black', capsize=4)
            
            for bar, acc in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{acc:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s}-shot' for s in shots], fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Few-Shot Classification Accuracy (5-way, Oxford Flowers 102)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(str(results_path / 'final_comparison.png'), dpi=150, bbox_inches='tight')
        print("Saved: final_comparison.png")
        
        # Print table
        print("\n" + "="*65)
        print(f"{'Model':<20} {'1-shot':^15} {'5-shot':^15} {'10-shot':^15}")
        print("="*65)
        for model, results in all_results.items():
            row = f"{model:<20}"
            for shot in [1, 5, 10]:
                key = f'{shot}-shot'
                if key in results:
                    acc = results[key]['acc'] * 100
                    ci = results[key].get('ci95', 0) * 100
                    row += f" {acc:.2f}±{ci:.2f}%  "
                else:
                    row += f" {'N/A':^13} "
            print(row)
        print("="*65)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/')
    args = parser.parse_args()
    plot_all(args.results_dir)
