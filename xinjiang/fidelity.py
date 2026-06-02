import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance, entropy
from scipy.spatial.distance import cdist

# =========================
# 1. 基础设置
# =========================
MODEL_INFO = [
    ("Full model", "output/phase2/gnn/phase2_cgtgan_epoch_0300.npz"),
    ("CTGAN", "output/phase2/ctgan/ctgan_phase2_input_epoch_0300.npz"),
    ("TVAE", "output/phase2/tvae/tvae_phase2_input_epoch_0300.npz"),
    ("GReaT", "output/phase2/great/great_phase2_input_epoch_0100.npz"),
    ("w/o conditional node", "output/phase2/gnn_ablation/wo_conditional_node/wo_conditional_node_epoch_0300.npz"),
    ("w/o graph module", "output/phase2/gnn_ablation/wo_graph_module/wo_graph_module_epoch_0300.npz"),
    ("w/o learnable adjacency", "output/phase2/gnn_ablation/wo_learnable_adjacency/wo_learnable_adjacency_epoch_0300.npz"),
    ("w/o node-level loss", "output/phase2/gnn_ablation/wo_node_level_loss/wo_node_level_loss_epoch_0300.npz"),
]

SAVE_DIR = "output/phase2/comparison/metrics"
os.makedirs(SAVE_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(SAVE_DIR, "distribution_metrics_summary_generated_only_biased_mmd.csv")
DETAIL_CSV = os.path.join(SAVE_DIR, "distribution_metrics_per_variable_generated_only_biased_mmd.csv")

# 固定：只比较 Original vs Generated
COMPARE_MODE = "generated_only"

# JSD 直方图 bin 数
JSD_BINS = 50

# MMD 参数
MMD_MAX_SAMPLES = 1000
RANDOM_STATE = 42


# =========================
# 2. 工具函数
# =========================
def find_first_existing_key(data, candidate_keys, npz_path):
    for key in candidate_keys:
        if key in data:
            return key
    raise KeyError(f"{npz_path} 中未找到任何候选键: {candidate_keys}")


def load_npz_content(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    # 原始训练集
    if "X_train_orig" not in data or "y_train_orig" not in data:
        raise KeyError(f"{npz_path} 中缺少 X_train_orig 或 y_train_orig")
    X_train_orig = data["X_train_orig"].astype(np.float64)
    y_train_orig = data["y_train_orig"].reshape(-1).astype(np.float64)

    # Phase1（虽然本次 generated_only 不参与比较，但保留读取逻辑方便后续扩展）
    phase1_x_key = find_first_existing_key(data, ["X_aug1", "X_aug1_used"], npz_path)
    phase1_y_key = find_first_existing_key(data, ["y_aug1", "y_aug1_used"], npz_path)
    X_aug1 = data[phase1_x_key].astype(np.float64)
    y_aug1 = data[phase1_y_key].reshape(-1).astype(np.float64)

    # 当前模型生成集
    gen_x_key = find_first_existing_key(
        data,
        ["generated_X", "X_generated", "X_gen", "fake_X", "synthetic_X"],
        npz_path
    )
    gen_y_key = find_first_existing_key(
        data,
        ["generated_y", "y_generated", "y_gen", "fake_y", "synthetic_y"],
        npz_path
    )
    X_generated = data[gen_x_key].astype(np.float64)
    y_generated = data[gen_y_key].reshape(-1).astype(np.float64)

    # 特征名
    if "feature_names" in data:
        feature_names = [str(x) for x in data["feature_names"].tolist()]
    elif "features" in data:
        feature_names = [str(x) for x in data["features"].tolist()]
    else:
        feature_names = [f"X{i+1}" for i in range(X_train_orig.shape[1])]

    # 目标名
    if "target_name" in data:
        target_name = str(data["target_name"].tolist() if hasattr(data["target_name"], "tolist") else data["target_name"])
    elif "target" in data:
        target_name = str(data["target"].tolist() if hasattr(data["target"], "tolist") else data["target"])
    else:
        target_name = "PTB"

    return {
        "X_train_orig": X_train_orig,
        "y_train_orig": y_train_orig,
        "X_aug1": X_aug1,
        "y_aug1": y_aug1,
        "X_generated": X_generated,
        "y_generated": y_generated,
        "feature_names": feature_names,
        "target_name": target_name
    }


def build_xy(X, y):
    y = y.reshape(-1, 1)
    return np.hstack([X, y])


def js_divergence_1d(x, y, bins=50, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    all_vals = np.concatenate([x, y])
    vmin, vmax = np.min(all_vals), np.max(all_vals)

    if np.isclose(vmin, vmax):
        return 0.0

    hist_range = (vmin, vmax)
    px, _ = np.histogram(x, bins=bins, range=hist_range, density=False)
    py, _ = np.histogram(y, bins=bins, range=hist_range, density=False)

    px = px.astype(np.float64)
    py = py.astype(np.float64)

    px = px / (px.sum() + eps)
    py = py / (py.sum() + eps)

    px = np.clip(px, eps, None)
    py = np.clip(py, eps, None)

    px = px / px.sum()
    py = py / py.sum()

    m = 0.5 * (px + py)
    jsd = 0.5 * entropy(px, m) + 0.5 * entropy(py, m)
    return float(jsd)


def mean_abs_delta_r(real_xy, compare_xy):
    corr_real = np.corrcoef(real_xy, rowvar=False)
    corr_cmp = np.corrcoef(compare_xy, rowvar=False)

    delta = np.abs(corr_real - corr_cmp)

    # 只取上三角（不含对角线）
    triu_idx = np.triu_indices_from(delta, k=1)
    return float(np.mean(delta[triu_idx]))


def standardize_two_sets(X, Y, eps=1e-12):
    combined = np.vstack([X, Y])
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (X - mean) / std, (Y - mean) / std


def subsample_rows(X, max_samples=1000, random_state=42):
    n = X.shape[0]
    if n <= max_samples:
        return X
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_samples, replace=False)
    return X[idx]


def compute_mmd_rbf_biased(X, Y, max_samples=1000, random_state=42):
    """
    使用 biased estimator 计算 RBF-kernel MMD^2
    优点：数值更稳定，通常不会出现大量负值或被裁成 0 的情况
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    X = subsample_rows(X, max_samples=max_samples, random_state=random_state)
    Y = subsample_rows(Y, max_samples=max_samples, random_state=random_state + 1)

    X, Y = standardize_two_sets(X, Y)

    pooled = np.vstack([X, Y])

    # 用 median heuristic 估计核带宽
    sq_dists = cdist(pooled, pooled, metric="sqeuclidean")
    upper_vals = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    upper_vals = upper_vals[upper_vals > 0]

    if len(upper_vals) == 0:
        sigma2 = 1.0
    else:
        sigma2 = np.median(upper_vals)
        if (not np.isfinite(sigma2)) or sigma2 <= 0:
            sigma2 = 1.0

    gamma = 1.0 / (2.0 * sigma2)

    Kxx = np.exp(-gamma * cdist(X, X, metric="sqeuclidean"))
    Kyy = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean"))
    Kxy = np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))

    # biased estimator
    mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return float(max(mmd2, 0.0))


# =========================
# 3. 主循环
# =========================
summary_rows = []
detail_rows = []

for model_name, npz_path in MODEL_INFO:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"文件不存在: {npz_path}")

    print(f"正在处理: {model_name}")

    obj = load_npz_content(npz_path)

    X_real = obj["X_train_orig"]
    y_real = obj["y_train_orig"]
    X_gen = obj["X_generated"]
    y_gen = obj["y_generated"]
    feature_names = obj["feature_names"]
    target_name = obj["target_name"]

    real_xy = build_xy(X_real, y_real)
    gen_xy = build_xy(X_gen, y_gen)

    # 固定使用 generated_only
    compare_xy = gen_xy

    variable_names = feature_names + [target_name]

    # -------- 单变量指标 --------
    ks_vals = []
    wd_vals = []
    jsd_vals = []

    for j, var_name in enumerate(variable_names):
        real_col = real_xy[:, j]
        cmp_col = compare_xy[:, j]

        ks_stat = ks_2samp(real_col, cmp_col).statistic
        wd_stat = wasserstein_distance(real_col, cmp_col)
        jsd_stat = js_divergence_1d(real_col, cmp_col, bins=JSD_BINS)

        ks_vals.append(ks_stat)
        wd_vals.append(wd_stat)
        jsd_vals.append(jsd_stat)

        detail_rows.append({
            "Model": model_name,
            "Variable": var_name,
            "KS": ks_stat,
            "Wasserstein": wd_stat,
            "JSD": jsd_stat
        })

    # -------- 相关结构指标 --------
    delta_r_mean = mean_abs_delta_r(real_xy, compare_xy)

    # -------- 全局分布指标 --------
    mmd_val = compute_mmd_rbf_biased(
        real_xy,
        compare_xy,
        max_samples=MMD_MAX_SAMPLES,
        random_state=RANDOM_STATE
    )

    # PTB 单独摘出来
    ptb_idx = len(variable_names) - 1

    summary_rows.append({
        "Model": model_name,
        "Compare_mode": COMPARE_MODE,
        "N_real": real_xy.shape[0],
        "N_compare": compare_xy.shape[0],
        "KS_mean": np.mean(ks_vals),
        "Wasserstein_mean": np.mean(wd_vals),
        "JSD_mean": np.mean(jsd_vals),
        f"KS_{target_name}": ks_vals[ptb_idx],
        f"Wasserstein_{target_name}": wd_vals[ptb_idx],
        f"JSD_{target_name}": jsd_vals[ptb_idx],
        "Mean_|Δr|": delta_r_mean,
        "MMD": mmd_val
    })

# =========================
# 4. 保存结果
# =========================
summary_df = pd.DataFrame(summary_rows)
detail_df = pd.DataFrame(detail_rows)

summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
detail_df.to_csv(DETAIL_CSV, index=False, encoding="utf-8-sig")

print("\n===== 汇总结果 =====")
print(summary_df.round(6))

print(f"\n已保存：\n{SUMMARY_CSV}\n{DETAIL_CSV}")