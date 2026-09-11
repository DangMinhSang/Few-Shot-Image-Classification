# config.py – central configuration for the project
import os
from pathlib import Path

# Root of the repository (this file is located at <repo>/code/config/config.py)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_ROOT = REPO_ROOT / "data" / "processed"
RAW_DATA_ROOT = REPO_ROOT / "data" / "raw"

# Hyper‑parameters
BATCH_SIZE = 32
NUM_WORKERS = 2
SEED = 42
LR = 1e-3

# WANDB
WANDB_PROJECT = "few-shot-flower"
