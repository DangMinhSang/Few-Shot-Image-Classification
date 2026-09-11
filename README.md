# Few‑shot Image Classification of Flowers

This repository implements a small research‑style project for few‑shot visual classification of flower species.

## Levels
1. **Baseline** – Standard supervised training of a ResNet‑18 classifier.
2. **Siamese** – Pairwise similarity learning with contrastive loss.
3. **Prototypical Network** – Episodic training where each class is represented by a prototype in the embedding space.
4. **Experiments** – Run the three models with 1‑, 5‑, 10‑shot settings, log results with Weights & Biases, and produce learning curves / confusion matrices.

## Dataset
We use the **Oxford‑102 Flowers** dataset (102 flower categories).  The dataset can be downloaded via `torchvision.datasets.Flowers102` or manually placed under `data/flowers/` with the typical ImageFolder layout:
```
data/flowers/
    train/<class_name>/img1.jpg
    val/<class_name>/img2.jpg
    test/<class_name>/...
```

## Quick start
```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline (adjust epochs, batch size as needed)
python train_few_shot.py --mode baseline --epochs 20

# Run Siamese network
python train_few_shot.py --mode siamese --epochs 30

# Run Prototypical network (e.g., 5‑shot, 5‑way)
python train_few_shot.py --mode proto --shots 5 --ways 5 --epochs 200
```

All runs are logged to **Weights & Biases** (WANDB) – the default project is `few-shot-flower`.  Set the environment variable `WANDB_API_KEY` if you want to log to a personal account.

## Experiment script (future work)
A separate notebook (`experiments.ipynb`) will be added to aggregate results, plot learning curves, confusion matrices and perform error analysis.

## License
MIT – feel free to adapt and extend.
