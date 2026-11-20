"""Read params.yaml and load the CSV files."""
import hashlib
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path=ROOT / "params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_csv(name, params=None):
    """name is one of: train, later, holdout (keys under data: in params.yaml)."""
    params = params or load_params()
    return pd.read_csv(ROOT / params["data"][name])


def file_sha256(name, params=None):
    params = params or load_params()
    return hashlib.sha256(open(ROOT / params["data"][name], "rb").read()).hexdigest()
