# 🌸 Few-Shot Image Classification — Nhận diện loài hoa

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Project nhận diện loài hoa với rất ít dữ liệu (few-shot learning) sử dụng **Oxford Flowers 102** dataset.

## 📋 Tổng quan

| Level | Model | Mô tả |
|-------|-------|--------|
| 1 | ResNet18 Baseline | Standard supervised classification |
| 2 | Siamese Network | Học similarity giữa các cặp ảnh |
| 3 | Prototypical Network | Few-shot learning với prototype |
| 4 | Experiments | So sánh và phân tích toàn diện |

## 🌺 Dataset: Oxford Flowers 102

- **102 loài hoa** khác nhau
- Train/Val/Test split chuẩn
- Few-shot episodes: 5-way K-shot

## 📦 Cài đặt

```bash
# Clone repo
git clone <repo-url>
cd Few-Shot-Image-Classification

# Tạo môi trường
python -m venv venv
source venv/bin/activate

# Cài dependencies
pip install -r requirements.txt

# Download dataset
python scripts/download_dataset.py
```

## 🚀 Chạy Training

### Level 1: Baseline
```bash
python train_baseline.py --config config.yaml
```

### Level 2: Siamese Network
```bash
python train_siamese.py --config config.yaml
```

### Level 3: Prototypical Network
```bash
python train_protonet.py --config config.yaml
```

### Level 4: Full Experiment
```bash
python run_experiments.py --config config.yaml
```

## 📊 Kết quả

| Model | 1-shot | 5-shot | 10-shot |
|-------|--------|--------|---------|
| ResNet18 | - | - | - |
| Siamese | - | - | - |
| ProtoNet | - | - | - |
| ProtoNet+Aug | - | - | - |

## 📁 Cấu trúc Project

```
├── data/                   # Dataset
├── src/
│   ├── datasets/           # Data loading & sampling
│   ├── models/             # Model definitions
│   ├── losses/             # Loss functions
│   ├── trainers/           # Training loops
│   └── utils/              # Helper functions
├── scripts/                # Utility scripts
├── notebooks/              # Analysis notebooks
├── checkpoints/            # Saved models
├── results/                # Experiment results
├── train_baseline.py
├── train_siamese.py
├── train_protonet.py
├── run_experiments.py
└── config.yaml
```

## 🔬 Phương pháp

### Prototypical Network

Với mỗi class trong support set:

$$p_c = \frac{1}{|S_c|} \sum_{(x_i, y_i) \in S_c} f_\phi(x_i)$$

Phân loại query bằng softmax over negative distances:

$$P(y=c|x) = \frac{\exp(-d(f_\phi(x), p_c))}{\sum_{c'} \exp(-d(f_\phi(x), p_{c'}))}$$

## 📈 Visualization

- Learning curves
- Confusion matrix
- t-SNE embedding visualization
- Prototype visualization
