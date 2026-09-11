#!/usr/bin/env python3
"""
Analysis Script: Level 4 Experiments

This script performs comprehensive analysis:
1. Load all trained models
2. Evaluate on few-shot episodes
3. Plot learning curves, confusion matrix, t-SNE
4. Print comparison table

Can be run directly or converted to Jupyter notebook with:
    jupytext --to notebook notebooks/analysis.py
"""
# %% [markdown]
# # Few-Shot Image Classification Analysis
# ## Oxford Flowers 102 - Full Experiment Results

# %%
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path().absolute().parent))

from src.datasets import FlowersDataset, EpisodeSampler
from src.models import BaselineClassifier, SiameseNetwork, PrototypicalNetwork
from src.utils import set_seed, compute_few_shot_accuracy
from src.utils.visualization import (
    plot_learning_curves,
    plot_confusion_matrix,
    plot_tsne,
    plot_episode,
    plot_experiment_results,
)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# %% [markdown]
# ## 1. Load Dataset

# %%
DATA_DIR = '../data/'
IMAGE_SIZE = 224

train_dataset = FlowersDataset(root=DATA_DIR, split='train', image_size=IMAGE_SIZE)
val_dataset = FlowersDataset(root=DATA_DIR, split='val', image_size=IMAGE_SIZE)
test_dataset = FlowersDataset(root=DATA_DIR, split='test', image_size=IMAGE_SIZE)

print(f'Train: {len(train_dataset)} images, {train_dataset.num_classes} classes')
print(f'Val:   {len(val_dataset)} images')
print(f'Test:  {len(test_dataset)} images')

# %% [markdown]
# ## 2. Visualize Sample Episode

# %%
N_WAY = 5
K_SHOT = 5
N_QUERY = 15

sampler = EpisodeSampler(
    class_to_samples=val_dataset.class_to_samples,
    idx_to_class=val_dataset.idx_to_class,
    n_way=N_WAY, k_shot=K_SHOT, n_query=N_QUERY,
    n_episodes=100, image_size=IMAGE_SIZE,
)

episode = sampler[0]
print(f'Episode classes: {episode.class_names}')
print(f'Support: {episode.support_images.shape}')
print(f'Query:   {episode.query_images.shape}')

fig = plot_episode(episode)
plt.show()

# %% [markdown]
# ## 3. Load Training Histories

# %%
RESULTS_DIR = Path('../results/')

def load_json(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None

baseline_history = load_json(RESULTS_DIR / 'baseline_history.json')
siamese_history = load_json(RESULTS_DIR / 'siamese_history.json')
protonet_results = load_json(RESULTS_DIR / 'protonet_results.json')
protonet_aug_results = load_json(RESULTS_DIR / 'protonet_aug_results.json')

# Plot learning curves
for name, data in [('Baseline', baseline_history), ('Siamese', siamese_history)]:
    if data and 'history' in data:
        plot_learning_curves(data['history'], title=name)
        plt.show()

# %% [markdown]
# ## 4. Few-Shot Accuracy Comparison

# %%
all_results = load_json(RESULTS_DIR / 'all_results.json')
if all_results:
    fig = plot_experiment_results(all_results, shots=[1, 5, 10])
    plt.show()
    
    # Print table
    print(f"\n{'='*70}")
    print(f"{'Model':<20} {'1-shot':^17} {'5-shot':^17} {'10-shot':^17}")
    print('='*70)
    for model, results in all_results.items():
        row = f'{model:<20}'
        for shot in [1, 5, 10]:
            key = f'{shot}-shot'
            if key in results:
                acc = results[key]['acc'] * 100
                ci = results[key].get('ci95', 0) * 100
                row += f'  {acc:.2f}±{ci:.2f}%   '
            else:
                row += f"  {'N/A':^13}  "
        print(row)
    print('='*70)

# %% [markdown]
# ## 5. t-SNE Embedding Visualization

# %%
# Load ProtoNet and extract embeddings
protonet_ckpt = Path('../checkpoints/protonet/best_model.pth')

if protonet_ckpt.exists():
    ckpt = torch.load(str(protonet_ckpt), map_location=device)
    model = PrototypicalNetwork(
        backbone_name='resnet18', pretrained=False,
        embedding_dim=256, distance='euclidean'
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    
    from torch.utils.data import DataLoader
    from src.datasets import FlowersDataset
    import torchvision.transforms as T
    
    # Use test dataset with clean transforms
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = FlowersDataset(root=DATA_DIR, split='test', transform=transform)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    embeddings = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            emb = model.encode(imgs.to(device))
            embeddings.append(emb.cpu().numpy())
            labels.append(lbls.numpy())
    
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    # Show only top 20 classes for clarity
    unique_classes = np.unique(labels)[:20]
    mask = np.isin(labels, unique_classes)
    
    fig = plot_tsne(
        embeddings[mask],
        labels[mask],
        class_names=test_ds.FLOWER_NAMES,
        title='t-SNE: ProtoNet Embeddings (Top 20 Classes)',
    )
    plt.show()
else:
    print('ProtoNet checkpoint not found. Train the model first.')

# %% [markdown]
# ## 6. Error Analysis

# %%
print('Run train_protonet.py first, then re-run this analysis.')
print('The confusion matrix will appear here after evaluation.')
