#!/usr/bin/env python
"""Entry point for the Few‑shot Flower Classification project.

Usage examples:
    python main.py --mode baseline --epochs 20
    python main.py --mode siamese  --epochs 30 --shots 5 --ways 5
    python main.py --mode proto    --epochs 200 --shots 5 --ways 5
cách 
The script simply forwards the arguments to the appropriate training
module under the ``code/train`` package.  All heavy‑lifting (model
definition, data loading, training loops) lives in those modules.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_script(script_path: Path, extra_args: list):
    """Run a Python script located inside the repository.
    ``script_path`` must be absolute or relative to the current working
    directory. ``extra_args`` are the command‑line arguments that will be
    passed to the script.
    """
    cmd = [sys.executable, str(script_path)] + extra_args
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Few‑shot flower classification – top‑level CLI")
    parser.add_argument("--mode", choices=["baseline", "siamese", "proto"], required=True,
                        help="Which training script to launch")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs / episodes (default depends on mode)")
    parser.add_argument("--shots", type=int, default=5,
                        help="k‑shot value for siamese / proto (default 5)")
    parser.add_argument("--ways", type=int, default=5,
                        help="n‑way value for siamese / proto (default 5)")
    args = parser.parse_args()

    # Resolve the path of the script inside the package
    repo_root = Path(__file__).resolve().parent
    script_map = {
        "baseline": repo_root / "code" / "train" / "baseline_train.py",
        "siamese":  repo_root / "code" / "train" / "siamese_train.py",
        "proto":    repo_root / "code" / "train" / "proto_train.py",
    }
    script_path = script_map[args.mode]

    extra = ["--epochs", str(args.epochs)]
    # Only pass shots/ways to siamese and proto
    if args.mode in {"siamese", "proto"}:
        extra += ["--shots", str(args.shots), "--ways", str(args.ways)]

    run_script(script_path, extra)


if __name__ == "__main__":
    main()
