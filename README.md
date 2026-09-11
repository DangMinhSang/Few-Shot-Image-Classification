# Few‑shot Image Classification of Flowers

This repository implements a small research‑style project for **few‑shot visual classification** of flower species.  It follows the four levels described in the original problem statement:

1. **Baseline** – Supervised training of a ResNet‑18 classifier.
2. **Siamese** – Pairwise similarity learning with contrastive (or triplet) loss.
3. **Prototypical Network** – Episodic training where each class is represented by a prototype in the embedding space.
4. **Experiment** – Compare the three models under 1‑, 5‑, 10‑shot settings, log results with Weights & Biases, and produce learning curves & confusion matrices.

---

## Repository structure
```
Few-shot Image Classification/
│
├─ code/                     # Core Python package (importable as `code`)
│   ├─ __init__.py
│   ├─ config/               # Global config (paths, hyper‑parameters, WANDB)
│   ├─ data/                 # Data utilities (loader, episodic sampler, set‑seed)
│   ├─ models/               # Model definitions (baseline, siamese, prototypical)
│   ├─ losses/               # Loss functions (contrastive, …)
│   ├─ train/                # Training scripts for each level
│   └─ utils/                # Helper scripts (download dataset)
│
├─ notebooks/                # Jupyter notebooks for exploration & analysis
├─ experiments/              # Scripts to run multiple shot settings & aggregate CSV
├─ reports/                  # Figures, PDFs, final report
│   └─ figures/
│
├─ requirements.txt          # Python dependencies
├─ README.md                 # This file
├─ download_flowers.py       # Simple script to download Oxford‑102 Flowers
├─ main.py                   # Top‑level entry point (dispatches to train scripts)
└─ .git/ …
```

---

## Installation
```bash
# (Optional) create a virtual environment
python -m venv venv
# Activate it
# Windows PowerShell
.\\venv\\Scripts\\activate
# macOS / Linux
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```
If you have a CUDA‑enabled GPU, `torch` will automatically use it; otherwise the code falls back to CPU.

---

## Preparing the dataset
The project uses the **Oxford‑102 Flowers** dataset (102 flower categories).

### Option 1 – Automatic download (requires internet)
```bash
python download_flowers.py
```
The script downloads the three splits (`train`, `val`, `test`) into a folder called `flowers_data/` at the repository root.

### Option 2 – Manual download
1. Go to <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/> and download `102flowers.tgz` plus the annotation files.
2. Extract them so that the following hierarchy exists:
```
flowers_data/
   ├─ train/   # <class_id>/<image>.jpg
   ├─ val/
   └─ test/
```
3. If you store the data elsewhere, edit `code/config/config.py` and change `RAW_DATA_ROOT` / `DATA_ROOT` accordingly.

---

## Quick start – running the models
All training is launched through the **single entry point** `main.py`.  It forwards the arguments to the appropriate script under `code/train/`.

```bash
# 1️⃣ Baseline (ResNet‑18, supervised)
python main.py --mode baseline --epochs 20

# 2️⃣ Siamese (contrastive loss)
python main.py --mode siamese --epochs 30 --shots 5 --ways 5

# 3️⃣ Prototypical Network
python main.py --mode proto --epochs 200 --shots 5 --ways 5
```
- `--epochs` is interpreted as *epochs* for the baseline and as *number of episodes* for Siamese/ProtoNet.
- `--shots` (`k‑shot`) and `--ways` (`n‑way`) are only relevant for Siamese and Prototypical training.
- Each script logs loss/accuracy to **Weights & Biases**.  If you have a WANDB API key, export it before running:
```bash
# Linux / macOS
export WANDB_API_KEY=your_key_here
# Windows PowerShell
set WANDB_API_KEY=your_key_here
```
If no key is set, W&B runs in *offline* mode and stores the run locally.

---

## Notebooks (exploration & analysis)
The `notebooks/` folder contains ready‑to‑run Jupyter notebooks:
| Notebook | Purpose |
|----------|---------|
| `01_explore_dataset.ipynb` | Visualise class distribution, sample images |
| `02_baseline_experiment.ipynb` | Train baseline and plot learning curves |
| `03_siamese_experiment.ipynb` | Train Siamese, visualise pair distances |
| `04_prototypical_experiment.ipynb` | Train ProtoNet, evaluate 1‑/5‑/10‑shot |
| `05_compare_results.ipynb` | Aggregate CSV results, build comparison tables, plot confusion matrices |

Open them with `jupyter lab` (or `jupyter notebook`) and run the cells sequentially.

---

## Running systematic experiments
The `experiments/` directory contains helper scripts (`run_all.sh` for Bash or `run_all.ps1` for PowerShell) that run the three models with several shot settings and write CSV files to `experiments/results/`.  After the runs, launch `05_compare_results.ipynb` to generate the final tables and figures.

---

## License
This project is released under the **MIT License** – feel free to adapt, extend, and share.

---

## Contact
If you encounter any issues or have suggestions, please open an issue on the repository or contact the author.
