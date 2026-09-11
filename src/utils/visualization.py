"""
Visualization utilities.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import torch

# Set style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def plot_learning_curves(
    history: Dict[str, List[float]],
    title: str = 'Learning Curves',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation loss/accuracy curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss
    if 'train_loss' in history:
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r--', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    
    # Accuracy
    if 'train_acc' in history:
        axes[1].plot(epochs, [a * 100 for a in history['train_acc']], 'b-', label='Train Acc', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(epochs, [a * 100 for a in history['val_acc']], 'r--', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
    top_n: int = 20,
) -> plt.Figure:
    """
    Plot normalized confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # If too many classes, show only top_n most confused
    if cm.shape[0] > top_n:
        # Find classes with most errors
        errors = cm.sum(axis=1) - cm.diagonal()
        top_classes = errors.argsort()[-top_n:][::-1]
        cm = cm[np.ix_(top_classes, top_classes)]
        if class_names:
            class_names = [class_names[i] for i in top_classes]
    
    # Normalize
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(max(10, len(cm) * 0.5), max(8, len(cm) * 0.4)))
    
    sns.heatmap(
        cm_norm,
        annot=len(cm) <= 20,
        fmt='.2f',
        cmap='Blues',
        ax=ax,
        xticklabels=class_names or 'auto',
        yticklabels=class_names or 'auto',
        linewidths=0.5,
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = 't-SNE Embedding Visualization',
    save_path: Optional[str] = None,
    perplexity: int = 30,
) -> plt.Figure:
    """
    Plot t-SNE visualization of embeddings.
    """
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    emb_2d = tsne.fit_transform(embeddings)
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_classes, 20)))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names and label < len(class_names) else f'Class {label}'
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=[colors[i % 20]],
            label=name,
            alpha=0.7,
            s=30,
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    if n_classes <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_episode(
    episode,
    class_names: Optional[List[str]] = None,
    predictions: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a few-shot episode with support and query images.
    """
    n_way = episode.support_images.shape[0]
    k_shot = episode.support_images.shape[1]
    n_query = min(episode.query_images.shape[0] // n_way, 3)
    
    fig = plt.figure(figsize=(2 * (k_shot + n_query + 1), 2 * n_way))
    gs = gridspec.GridSpec(n_way, k_shot + n_query + 1)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    def denormalize(img_tensor):
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        return np.clip(img, 0, 1)
    
    for cls_idx in range(n_way):
        class_name = episode.class_names[cls_idx] if episode.class_names else f'Class {cls_idx}'
        
        # Label column
        ax = fig.add_subplot(gs[cls_idx, 0])
        ax.text(0.5, 0.5, class_name, ha='center', va='center', fontsize=9, wrap=True)
        ax.axis('off')
        
        # Support images
        for shot_idx in range(k_shot):
            ax = fig.add_subplot(gs[cls_idx, shot_idx + 1])
            img = denormalize(episode.support_images[cls_idx, shot_idx])
            ax.imshow(img)
            ax.axis('off')
            if cls_idx == 0:
                ax.set_title(f'S{shot_idx+1}', fontsize=8)
        
        # Query images
        for q_idx in range(n_query):
            ax = fig.add_subplot(gs[cls_idx, k_shot + 1 + q_idx])
            query_global_idx = cls_idx * (episode.query_images.shape[0] // n_way) + q_idx
            if query_global_idx < episode.query_images.shape[0]:
                img = denormalize(episode.query_images[query_global_idx])
                ax.imshow(img)
                
                if predictions is not None:
                    pred_label = predictions[query_global_idx].item()
                    true_label = episode.query_labels[query_global_idx].item()
                    color = 'green' if pred_label == true_label else 'red'
                    ax.set_title(f'Pred:{pred_label}', color=color, fontsize=7)
            ax.axis('off')
            if cls_idx == 0:
                ax.set_title(f'Q{q_idx+1}', fontsize=8)
    
    plt.suptitle('Few-Shot Episode Visualization', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_experiment_results(
    results: Dict[str, Dict],
    shots: List[int] = [1, 5, 10],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot comparison table of model performances across shots.
    """
    models = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(shots))
    width = 0.8 / len(models)
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    
    for i, model_name in enumerate(models):
        accs = [results[model_name].get(f'{s}-shot', {}).get('acc', 0) * 100 for s in shots]
        cis = [results[model_name].get(f'{s}-shot', {}).get('ci95', 0) * 100 for s in shots]
        
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, accs, width * 0.9,
            label=model_name, color=colors[i % len(colors)], alpha=0.85
        )
        ax.errorbar(
            x + offset, accs, yerr=cis,
            fmt='none', color='black', capsize=3, linewidth=1.5
        )
    
    ax.set_xlabel('Number of Shots')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Few-Shot Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}-shot' for s in shots])
    ax.legend(loc='upper left')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig
