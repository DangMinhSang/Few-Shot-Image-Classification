"""I/O utilities for configs and results."""
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: Dict[str, Any], output_path: str):
    """Save results dict to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)
