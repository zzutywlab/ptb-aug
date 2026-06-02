#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TVAE baseline script for phase2 comparison experiments.

Design choices
--------------
1) For a fair comparison with your phase2 GNN-GAN, this script uses the same
   phase2 training source by default:
       original training set + phase1 augmented set
2) It trains a TVAE model on the tabular matrix [X, y].
3) It generates the same number of synthetic samples as the phase2 input size.
4) It saves results in a .npz structure that is intentionally close to your
   phase2 output format, so downstream scripts can reuse it more easily.

Official API basis
------------------
This script uses the standalone `ctgan` package API:
    from ctgan import TVAE
    model = TVAE(...)
    model.fit(dataframe, discrete_columns=[])
    synthetic = model.sample(n)

If your environment does not have ctgan installed, run:
    pip install ctgan
"""

from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from ctgan import TVAE
except ImportError as e:
    raise ImportError(
        "ctgan is not installed. Please run `pip install ctgan` first."
    ) from e

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================
# 1. Basic config
# =========================
SEED = 42
PHASE1_NPZ_PATH = "output/phase1/phase1_augmented_data.npz"
SAVE_DIR = "output/phase2/tvae"

# Train on the same input used by your phase2 GNN-GAN:
#   - "phase2_input": X_train_orig + X_aug1
#   - "orig_only":    X_train_orig only
TRAIN_SOURCE = "phase2_input"

# Generate this many synthetic rows:
#   - "same_as_train_source": same number as TVAE training rows
#   - integer: explicit number of synthetic rows
N_SYNTHETIC = "same_as_train_source"

# TVAE hyperparameters
EPOCHS = 300
EMBEDDING_DIM = 128
COMPRESS_DIMS = (256, 256)
DECOMPRESS_DIMS = (256, 256)
L2SCALE = 1e-5
LOSS_FACTOR = 2
VERBOSE = True
PREFERRED_BATCH_SIZE = 128

# Postprocessing
CLIP_TO_TRAIN_RANGE = True
SAVE_CSV = True
SAVE_SUMMARY_CSV = True
SAVE_LOSS_CSV = True

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# 2. Utilities
# =========================
@dataclass
class TVAERunConfig:
    seed: int
    phase1_npz_path: str
    save_dir: str
    train_source: str
    n_synthetic: str | int
    epochs: int
    embedding_dim: int
    compress_dims: Tuple[int, ...]
    decompress_dims: Tuple[int, ...]
    l2scale: float
    loss_factor: float
    verbose: bool
    preferred_batch_size: int
    clip_to_train_range: bool


CONFIG = TVAERunConfig(
    seed=SEED,
    phase1_npz_path=PHASE1_NPZ_PATH,
    save_dir=SAVE_DIR,
    train_source=TRAIN_SOURCE,
    n_synthetic=N_SYNTHETIC,
    epochs=EPOCHS,
    embedding_dim=EMBEDDING_DIM,
    compress_dims=COMPRESS_DIMS,
    decompress_dims=DECOMPRESS_DIMS,
    l2scale=L2SCALE,
    loss_factor=LOSS_FACTOR,
    verbose=VERBOSE,
    preferred_batch_size=PREFERRED_BATCH_SIZE,
    clip_to_train_range=CLIP_TO_TRAIN_RANGE,
)


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_batch_size(n_rows: int, preferred_batch_size: int = 128) -> int:
    if n_rows < 2:
        raise ValueError("Training rows must be at least 2 for TVAE.")

    batch_size = min(preferred_batch_size, n_rows)
    batch_size = max(2, batch_size)
    return int(batch_size)



def safe_target_name(raw_target) -> str:
    if isinstance(raw_target, np.ndarray):
        if raw_target.ndim == 0:
            return str(raw_target.item())
        if raw_target.size == 1:
            return str(raw_target.reshape(-1)[0])
    return str(raw_target)



def load_phase1_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"phase1 npz not found: {path}")

    data = np.load(path, allow_pickle=True)
    required_keys = [
        "X_train_orig", "y_train_orig",
        "X_test", "y_test",
        "X_aug1", "y_aug1",
        "features", "target",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key in phase1 npz: {key}")

    out = {k: data[k] for k in data.files}
    return out



def build_train_table(data: Dict[str, np.ndarray], train_source: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], List[str], str]:
    X_train_orig = data["X_train_orig"].astype(np.float32)
    y_train_orig = data["y_train_orig"].reshape(-1).astype(np.float32)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].reshape(-1).astype(np.float32)

    X_aug1 = data["X_aug1"].astype(np.float32)
    y_aug1 = data["y_aug1"].reshape(-1).astype(np.float32)

    feature_names = [str(x) for x in data["features"].tolist()]
    target_name = safe_target_name(data["target"])

    if train_source == "phase2_input":
        train_X = np.vstack([X_train_orig, X_aug1]).astype(np.float32)
        train_y = np.concatenate([y_train_orig, y_aug1]).astype(np.float32)
    elif train_source == "orig_only":
        train_X = X_train_orig.copy()
        train_y = y_train_orig.copy()
    else:
        raise ValueError("train_source must be one of {'phase2_input', 'orig_only'}")

    train_xy = np.column_stack([train_X, train_y]).astype(np.float32)
    columns = feature_names + [target_name]
    train_df = pd.DataFrame(train_xy, columns=columns)

    arrays = {
        "X_train_orig": X_train_orig,
        "y_train_orig": y_train_orig,
        "X_test": X_test,
        "y_test": y_test,
        "X_aug1": X_aug1,
        "y_aug1": y_aug1,
        "combined_X": train_X,
        "combined_y": train_y,
        "combined_XY": train_xy,
    }
    return train_df, arrays, feature_names, target_name



def validate_train_df(train_df: pd.DataFrame) -> None:
    if train_df.isnull().any().any():
        null_cols = train_df.columns[train_df.isnull().any()].tolist()
        raise ValueError(
            f"Training table contains missing values in columns: {null_cols}. "
            "Please handle missing values before using standalone TVAE."
        )

    non_numeric = [c for c in train_df.columns if not pd.api.types.is_numeric_dtype(train_df[c])]
    if non_numeric:
        raise TypeError(
            f"This script expects numeric columns only, but found non-numeric columns: {non_numeric}"
        )



def clip_synthetic_to_train_range(synth_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    clipped = synth_df.copy()
    for col in clipped.columns:
        col_min = train_df[col].min()
        col_max = train_df[col].max()
        clipped[col] = clipped[col].clip(lower=col_min, upper=col_max)
    return clipped



def summarize_generation(train_df: pd.DataFrame, synth_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for col in train_df.columns:
        real_mean = float(train_df[col].mean())
        fake_mean = float(synth_df[col].mean())
        real_std = float(train_df[col].std(ddof=1))
        fake_std = float(synth_df[col].std(ddof=1))
        real_min = float(train_df[col].min())
        fake_min = float(synth_df[col].min())
        real_max = float(train_df[col].max())
        fake_max = float(synth_df[col].max())
        rows.append({
            "variable": col,
            "real_mean": real_mean,
            "fake_mean": fake_mean,
            "abs_mean_diff": abs(fake_mean - real_mean),
            "real_std": real_std,
            "fake_std": fake_std,
            "abs_std_diff": abs(fake_std - real_std),
            "real_min": real_min,
            "fake_min": fake_min,
            "real_max": real_max,
            "fake_max": fake_max,
        })

    stats_df = pd.DataFrame(rows)

    real_corr = train_df.corr(method="pearson")
    fake_corr = synth_df.corr(method="pearson")
    corr_delta = fake_corr - real_corr

    return stats_df, corr_delta


# =========================
# 3. Main
# =========================
def main() -> None:
    set_all_seeds(CONFIG.seed)

    data = load_phase1_npz(CONFIG.phase1_npz_path)
    train_df, arrays, feature_names, target_name = build_train_table(data, CONFIG.train_source)
    validate_train_df(train_df)

    n_train = len(train_df)
    batch_size = choose_batch_size(
        n_rows=n_train,
        preferred_batch_size=CONFIG.preferred_batch_size,
    )

    n_synth = n_train if CONFIG.n_synthetic == "same_as_train_source" else int(CONFIG.n_synthetic)
    discrete_columns: List[str] = []  # all columns are continuous in your current project setting

    print("=" * 80)
    print("TVAE baseline for phase2 comparison")
    print(f"Train source         : {CONFIG.train_source}")
    print(f"Training rows        : {n_train}")
    print(f"Synthetic rows       : {n_synth}")
    print(f"Feature count        : {len(feature_names)}")
    print(f"Target               : {target_name}")
    print(f"Batch size           : {batch_size}")
    print(f"GPU available        : {torch.cuda.is_available()}")
    print("=" * 80)

    model = TVAE(
        embedding_dim=CONFIG.embedding_dim,
        compress_dims=CONFIG.compress_dims,
        decompress_dims=CONFIG.decompress_dims,
        l2scale=CONFIG.l2scale,
        batch_size=batch_size,
        epochs=CONFIG.epochs,
        loss_factor=CONFIG.loss_factor,
        enable_gpu=torch.cuda.is_available(),
        verbose=CONFIG.verbose,
    )

    model.fit(train_df, discrete_columns=discrete_columns)
    synth_df = model.sample(n_synth)

    if isinstance(synth_df, np.ndarray):
        synth_df = pd.DataFrame(synth_df, columns=train_df.columns)

    # Keep column order stable
    synth_df = synth_df[train_df.columns].copy()

    if CONFIG.clip_to_train_range:
        synth_df = clip_synthetic_to_train_range(synth_df, train_df)

    synth_xy = synth_df.to_numpy(dtype=np.float32)
    synth_X = synth_xy[:, :-1].astype(np.float32)
    synth_y = synth_xy[:, -1].astype(np.float32)

    # Summary tables
    stats_df, corr_delta = summarize_generation(train_df, synth_df)

    save_stem = f"tvae_{CONFIG.train_source}_epoch_{CONFIG.epochs:04d}"
    npz_path = os.path.join(CONFIG.save_dir, f"{save_stem}.npz")
    csv_path = os.path.join(CONFIG.save_dir, f"{save_stem}.csv")
    stats_csv_path = os.path.join(CONFIG.save_dir, f"{save_stem}_summary_stats.csv")
    corr_csv_path = os.path.join(CONFIG.save_dir, f"{save_stem}_corr_delta.csv")
    config_json_path = os.path.join(CONFIG.save_dir, f"{save_stem}_config.json")
    loss_csv_path = os.path.join(CONFIG.save_dir, f"{save_stem}_loss.csv")

    save_kwargs = {
        "generated_X": synth_X,
        "generated_y": synth_y,
        "generated_XY": synth_xy,
        "X_train_orig": arrays["X_train_orig"],
        "y_train_orig": arrays["y_train_orig"],
        "X_aug1": arrays["X_aug1"],
        "y_aug1": arrays["y_aug1"],
        "X_test": arrays["X_test"],
        "y_test": arrays["y_test"],
        "combined_X": arrays["combined_X"],
        "combined_y": arrays["combined_y"],
        "combined_XY": arrays["combined_XY"],
        "feature_names": np.array(feature_names, dtype=object),
        "target_name": np.array(target_name, dtype=object),
        "train_source": np.array(CONFIG.train_source, dtype=object),
        "n_synthetic": np.array(n_synth),
        "epochs": np.array(CONFIG.epochs),
        "batch_size": np.array(batch_size),
        "seed": np.array(CONFIG.seed),
    }

    if hasattr(model, "loss_values") and model.loss_values is not None:
        save_kwargs["loss_values"] = model.loss_values.to_records(index=False)

    np.savez_compressed(npz_path, **save_kwargs)

    if SAVE_CSV:
        synth_df.to_csv(csv_path, index=False)

    if SAVE_SUMMARY_CSV:
        stats_df.to_csv(stats_csv_path, index=False)
        corr_delta.to_csv(corr_csv_path, index=True)

    if SAVE_LOSS_CSV and hasattr(model, "loss_values") and model.loss_values is not None:
        model.loss_values.to_csv(loss_csv_path, index=False)

    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(CONFIG), f, ensure_ascii=False, indent=2)

    print("Saved files:")
    print(f"  NPZ           : {npz_path}")
    if SAVE_CSV:
        print(f"  Synthetic CSV : {csv_path}")
    if SAVE_SUMMARY_CSV:
        print(f"  Stats CSV     : {stats_csv_path}")
        print(f"  Corr delta    : {corr_csv_path}")
    if SAVE_LOSS_CSV and hasattr(model, "loss_values") and model.loss_values is not None:
        print(f"  Loss CSV      : {loss_csv_path}")
    print(f"  Config JSON   : {config_json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
