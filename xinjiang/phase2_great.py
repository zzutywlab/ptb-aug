import os
import json
import random
import warnings
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# GReaT baseline for phase2 comparison
# Official model package: be-great / from be_great import GReaT
# This script is aligned with the user's phase2 pipeline style:
# - input:  output/phase1/phase1_augmented_data.npz
# - train on: X_train_orig + X_aug1
# - generate same number of synthetic rows as training input
# - save .npz/.csv and auxiliary statistics
# ============================================================


@dataclass
class Config:
    # -------------------------
    # Paths
    # -------------------------
    phase1_npz_path: str = "output/phase1/phase1_augmented_data.npz"
    save_dir: str = "output/phase2/great"
    run_name: str = "great_phase2_input"

    # -------------------------
    # Reproducibility
    # -------------------------
    seed: int = 42

    # -------------------------
    # Training source
    # phase2_input = X_train_orig + X_aug1
    # orig_only     = X_train_orig only
    # -------------------------
    train_source: str = "phase2_input"

    # -------------------------
    # GReaT settings
    # -------------------------
    llm: str = "distilgpt2"
    epochs: int = 100
    batch_size: int = 8
    float_precision: int = 3
    fp16: bool = True
    dataloader_num_workers: int = 0  # safer on Windows

    # -------------------------
    # Sampling settings
    # -------------------------
    sample_same_size_as_train: bool = True
    sample_n_rows: int = 283
    guided_sampling: bool = True
    random_feature_order: bool = True
    temperature: float = 0.7
    max_sampling_attempts: int = 12
    oversample_factor: float = 1.6

    # -------------------------
    # Postprocess / filtering
    # -------------------------
    coerce_numeric: bool = True
    drop_invalid_rows: bool = True
    clip_to_train_range: bool = False

    # -------------------------
    # Saving
    # -------------------------
    save_model_dir: bool = True


CONFIG = Config()


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def load_phase1_data(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)

    required_keys = [
        "X_train_orig", "y_train_orig",
        "X_test", "y_test",
        "X_aug1", "y_aug1",
        "features", "target"
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"phase1 npz is missing key: {key}")

    X_train_orig = data["X_train_orig"].astype(np.float32)
    y_train_orig = data["y_train_orig"].reshape(-1).astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].reshape(-1).astype(np.float32)
    X_aug1 = data["X_aug1"].astype(np.float32)
    y_aug1 = data["y_aug1"].reshape(-1).astype(np.float32)

    feature_names = [str(x) for x in data["features"].tolist()]
    target_name = str(data["target"].tolist() if hasattr(data["target"], "tolist") else data["target"])

    return {
        "X_train_orig": X_train_orig,
        "y_train_orig": y_train_orig,
        "X_test": X_test,
        "y_test": y_test,
        "X_aug1": X_aug1,
        "y_aug1": y_aug1,
        "feature_names": feature_names,
        "target_name": target_name,
    }



def build_training_dataframe(bundle: dict, train_source: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X_train_orig = bundle["X_train_orig"]
    y_train_orig = bundle["y_train_orig"]
    X_aug1 = bundle["X_aug1"]
    y_aug1 = bundle["y_aug1"]
    feature_names = bundle["feature_names"]
    target_name = bundle["target_name"]

    if train_source == "phase2_input":
        combined_X = np.vstack([X_train_orig, X_aug1]).astype(np.float32)
        combined_y = np.concatenate([y_train_orig, y_aug1]).astype(np.float32)
    elif train_source == "orig_only":
        combined_X = X_train_orig.copy().astype(np.float32)
        combined_y = y_train_orig.copy().astype(np.float32)
    else:
        raise ValueError(f"Unsupported train_source: {train_source}")

    df = pd.DataFrame(combined_X, columns=feature_names)
    df[target_name] = combined_y
    return df, combined_X, combined_y



def build_range_dict(df: pd.DataFrame) -> dict:
    ranges = {}
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        ranges[col] = {
            "min": float(vals.min()),
            "max": float(vals.max())
        }
    return ranges



def coerce_and_filter_numeric(df: pd.DataFrame, columns: List[str], train_ranges: dict, clip_to_train_range: bool) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if clip_to_train_range:
            out[col] = out[col].clip(train_ranges[col]["min"], train_ranges[col]["max"])
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=columns).reset_index(drop=True)
    return out



def sample_valid_numeric_rows(model, n_needed: int, all_columns: List[str], train_ranges: dict, cfg: Config) -> pd.DataFrame:
    collected = []
    total_valid = 0

    for attempt in range(1, cfg.max_sampling_attempts + 1):
        request_n = max(int(np.ceil((n_needed - total_valid) * cfg.oversample_factor)), 32)

        try:
            sampled = model.sample(
                n_samples=request_n,
                guided_sampling=cfg.guided_sampling,
                random_feature_order=cfg.random_feature_order,
                temperature=cfg.temperature,
            )
        except TypeError:
            sampled = model.sample(n_samples=request_n)

        if not isinstance(sampled, pd.DataFrame):
            sampled = pd.DataFrame(sampled, columns=all_columns)

        sampled = sampled.copy()
        sampled = sampled[[c for c in all_columns if c in sampled.columns]]

        missing_cols = [c for c in all_columns if c not in sampled.columns]
        for col in missing_cols:
            sampled[col] = np.nan
        sampled = sampled[all_columns]

        sampled = coerce_and_filter_numeric(
            sampled,
            columns=all_columns,
            train_ranges=train_ranges,
            clip_to_train_range=cfg.clip_to_train_range,
        )

        if len(sampled) > 0:
            collected.append(sampled)
            total_valid += len(sampled)

        print(f"Sampling attempt {attempt:02d}: valid rows = {len(sampled)}, accumulated = {total_valid}/{n_needed}")

        if total_valid >= n_needed:
            break

    if total_valid < n_needed:
        raise RuntimeError(
            f"Could not obtain enough valid numeric rows from GReaT. "
            f"Needed {n_needed}, got {total_valid}.\n"
            f"Try increasing epochs, using a smaller LLM, or setting clip_to_train_range=True."
        )

    result = pd.concat(collected, axis=0, ignore_index=True).iloc[:n_needed].reset_index(drop=True)
    return result



def save_summary_stats(real_df: pd.DataFrame, synth_df: pd.DataFrame, output_csv: str) -> None:
    rows = []
    for col in real_df.columns:
        r = pd.to_numeric(real_df[col], errors="coerce")
        s = pd.to_numeric(synth_df[col], errors="coerce")
        rows.append({
            "variable": col,
            "real_mean": float(r.mean()),
            "real_std": float(r.std(ddof=1)),
            "real_min": float(r.min()),
            "real_q25": float(r.quantile(0.25)),
            "real_median": float(r.median()),
            "real_q75": float(r.quantile(0.75)),
            "real_max": float(r.max()),
            "synth_mean": float(s.mean()),
            "synth_std": float(s.std(ddof=1)),
            "synth_min": float(s.min()),
            "synth_q25": float(s.quantile(0.25)),
            "synth_median": float(s.median()),
            "synth_q75": float(s.quantile(0.75)),
            "synth_max": float(s.max()),
            "abs_mean_diff": float(abs(r.mean() - s.mean())),
            "abs_std_diff": float(abs(r.std(ddof=1) - s.std(ddof=1))),
        })
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")



def save_corr_delta(real_df: pd.DataFrame, synth_df: pd.DataFrame, output_csv: str) -> None:
    real_corr = real_df.corr(method="pearson")
    synth_corr = synth_df.corr(method="pearson")
    delta = synth_corr - real_corr
    delta.to_csv(output_csv, encoding="utf-8-sig")



def main() -> None:
    set_all_seeds(CONFIG.seed)
    ensure_dir(CONFIG.save_dir)

    from be_great import GReaT

    bundle = load_phase1_data(CONFIG.phase1_npz_path)
    feature_names = bundle["feature_names"]
    target_name = bundle["target_name"]

    if target_name != "PTB":
        print(f"[Warning] target in phase1 file is {target_name}, not PTB.")

    train_df, combined_X, combined_y = build_training_dataframe(bundle, CONFIG.train_source)
    all_columns = feature_names + [target_name]
    train_ranges = build_range_dict(train_df)

    if CONFIG.sample_same_size_as_train:
        n_generate = len(train_df)
    else:
        n_generate = CONFIG.sample_n_rows

    use_fp16 = bool(CONFIG.fp16 and torch.cuda.is_available())

    print("=" * 80)
    print("GReaT baseline for phase2 comparison")
    print(f"Train source         : {CONFIG.train_source}")
    print(f"Training rows        : {len(train_df)}")
    print(f"Synthetic rows       : {n_generate}")
    print(f"Feature count        : {len(feature_names)}")
    print(f"Target               : {target_name}")
    print(f"LLM                  : {CONFIG.llm}")
    print(f"Epochs               : {CONFIG.epochs}")
    print(f"Batch size           : {CONFIG.batch_size}")
    print(f"GPU available        : {torch.cuda.is_available()}")
    print(f"fp16                 : {use_fp16}")
    print("=" * 80)

    # Official package API: model.fit(df), model.sample(n_samples=...)
    model = GReaT(
        llm=CONFIG.llm,
        batch_size=CONFIG.batch_size,
        epochs=CONFIG.epochs,
        float_precision=CONFIG.float_precision,
        fp16=use_fp16,
        dataloader_num_workers=CONFIG.dataloader_num_workers,
    )

    model.fit(train_df)

    synth_df = sample_valid_numeric_rows(
        model=model,
        n_needed=n_generate,
        all_columns=all_columns,
        train_ranges=train_ranges,
        cfg=CONFIG,
    )

    generated_X = synth_df[feature_names].to_numpy(dtype=np.float32)
    generated_y = synth_df[target_name].to_numpy(dtype=np.float32)
    generated_XY = synth_df[all_columns].to_numpy(dtype=np.float32)

    # Output names
    stem = f"{CONFIG.run_name}_epoch_{CONFIG.epochs:04d}"
    npz_path = os.path.join(CONFIG.save_dir, f"{stem}.npz")
    csv_path = os.path.join(CONFIG.save_dir, f"{stem}.csv")
    stats_csv_path = os.path.join(CONFIG.save_dir, f"{stem}_summary_stats.csv")
    corr_csv_path = os.path.join(CONFIG.save_dir, f"{stem}_corr_delta.csv")
    cfg_json_path = os.path.join(CONFIG.save_dir, f"{stem}_config.json")

    # Save synthetic csv
    synth_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Save aligned npz structure
    np.savez_compressed(
        npz_path,
        generated_X=generated_X,
        generated_y=generated_y,
        generated_XY=generated_XY,

        X_train_orig=bundle["X_train_orig"],
        y_train_orig=bundle["y_train_orig"],
        X_aug1=bundle["X_aug1"],
        y_aug1=bundle["y_aug1"],
        X_test=bundle["X_test"],
        y_test=bundle["y_test"],

        combined_X=combined_X,
        combined_y=combined_y,
        combined_XY=train_df[all_columns].to_numpy(dtype=np.float32),

        feature_names=np.array(feature_names, dtype=object),
        target_name=np.array(target_name, dtype=object),

        model_name=np.array("GReaT", dtype=object),
        llm_name=np.array(CONFIG.llm, dtype=object),
        epochs=np.array(CONFIG.epochs),
        batch_size=np.array(CONFIG.batch_size),
        seed=np.array(CONFIG.seed),
    )

    # Save helper tables
    save_summary_stats(train_df[all_columns], synth_df[all_columns], stats_csv_path)
    save_corr_delta(train_df[all_columns], synth_df[all_columns], corr_csv_path)

    # Save config
    save_cfg = asdict(CONFIG)
    save_cfg["gpu_available"] = bool(torch.cuda.is_available())
    save_cfg["fp16_effective"] = bool(use_fp16)
    with open(cfg_json_path, "w", encoding="utf-8") as f:
        json.dump(save_cfg, f, indent=2, ensure_ascii=False)

    # Optional model save
    if CONFIG.save_model_dir:
        model_dir = os.path.join(CONFIG.save_dir, f"{stem}_model")
        ensure_dir(model_dir)
        try:
            model.save(model_dir)
            print(f"Saved GReaT model to: {model_dir}")
        except Exception as e:
            print(f"[Warning] model.save(...) failed: {e}")

    print("Done.")
    print(f"Saved npz   : {npz_path}")
    print(f"Saved csv   : {csv_path}")
    print(f"Saved stats : {stats_csv_path}")
    print(f"Saved corr  : {corr_csv_path}")
    print(f"Saved config: {cfg_json_path}")


if __name__ == "__main__":
    main()
