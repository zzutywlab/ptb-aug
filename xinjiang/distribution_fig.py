import os
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# =========================
# 1. 基础设置（全局字号/线宽适配双栏宽度）
# =========================
MODEL_INFO = [
    ("Full model", "full_model", "output/phase2/gnn/phase2_cgtgan_epoch_0300.npz"),
    ("CTGAN", "ctgan", "output/phase2/ctgan/ctgan_phase2_input_epoch_0300.npz"),
    ("TVAE", "tvae", "output/phase2/tvae/tvae_phase2_input_epoch_0300.npz"),
    ("GReaT", "great", "output/phase2/great/great_phase2_input_epoch_0100.npz"),
    ("w/o conditional node", "wo_conditional_node", "output/phase2/gnn_ablation/wo_conditional_node/wo_conditional_node_epoch_0300.npz"),
    ("w/o graph module", "wo_graph_module", "output/phase2/gnn_ablation/wo_graph_module/wo_graph_module_epoch_0300.npz"),
    ("w/o learnable adjacency", "wo_learnable_adjacency", "output/phase2/gnn_ablation/wo_learnable_adjacency/wo_learnable_adjacency_epoch_0300.npz"),
    ("w/o node-level loss", "wo_node_level_loss", "output/phase2/gnn_ablation/wo_node_level_loss/wo_node_level_loss_epoch_0300.npz"),
]

SAVE_DIR = "output/phase2/comparison/figures/marginal_hist_kde_3x3"
os.makedirs(SAVE_DIR, exist_ok=True)

# 期刊双栏风格全局设置（字号整体缩小）
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#4D4D4D"
plt.rcParams["axes.linewidth"] = 0.7          # 坐标轴线宽调小
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"
plt.rcParams["text.color"] = "#222222"
plt.rcParams["axes.labelcolor"] = "#222222"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"

# 全局默认字号（基准）
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 7

sns.set_style("white")

# =========================
# 2. 配色
# =========================
COLOR_ORIG_FILL = "#B8BEC7"
COLOR_ORIG_LINE = "#4A4A4A"

MODEL_COLOR_MAP = {
    "full_model": "#2B6C99",
    "ctgan": "#E39C45",
    "tvae": "#6A7FDB",
    "great": "#4FAE8B",
    "wo_conditional_node": "#D98C8C",
    "wo_graph_module": "#C7A6D8",
    "wo_learnable_adjacency": "#9FC3C9",
    "wo_node_level_loss": "#D6B26E",
}

# =========================
# 3. 工具函数（保持不变）
# =========================
def find_first_existing_key(data, candidate_keys, npz_path):
    for key in candidate_keys:
        if key in data:
            return key
    raise KeyError(f"{npz_path} 中未找到任何候选键: {candidate_keys}")

def load_original_and_generated(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    if "X_train_orig" not in data:
        raise KeyError(f"{npz_path} 中缺少键: X_train_orig")
    X_train_orig = data["X_train_orig"].astype(np.float32)

    gen_x_key = find_first_existing_key(
        data,
        ["generated_X", "X_generated", "X_gen", "fake_X", "synthetic_X"],
        npz_path
    )
    X_generated = data[gen_x_key].astype(np.float32)

    if "feature_names" in data:
        feature_names = [str(x) for x in data["feature_names"].tolist()]
    elif "features" in data:
        feature_names = [str(x) for x in data["features"].tolist()]
    else:
        feature_names = [f"X{i+1}" for i in range(X_train_orig.shape[1])]

    y_train_orig = None
    y_generated = None
    if "y_train_orig" in data:
        y_train_orig = data["y_train_orig"].reshape(-1).astype(np.float32)

    gen_y_candidates = ["generated_y", "y_generated", "y_gen", "fake_y", "synthetic_y"]
    for k in gen_y_candidates:
        if k in data:
            y_generated = data[k].reshape(-1).astype(np.float32)
            break

    return X_train_orig, y_train_orig, X_generated, y_generated, feature_names

def safe_kdeplot(ax, values, color, linewidth=1.2, alpha=1.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return
    if np.nanstd(values) < 1e-12:
        return
    sns.kdeplot(
        values,
        ax=ax,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        fill=False,
        warn_singular=False
    )

def get_bins(x1, x2, n_bins=28):
    x = np.concatenate([x1, x2]).astype(float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 10
    x_min, x_max = np.min(x), np.max(x)
    if np.isclose(x_min, x_max):
        delta = 0.5 if x_min == 0 else abs(x_min) * 0.05
        return np.linspace(x_min - delta, x_max + delta, 10)
    return np.linspace(x_min, x_max, n_bins)

# =========================
# 4. 主循环：每个模型生成一张 3×3 图（尺寸统一适配双栏）
# =========================
for model_display_name, model_short_name, npz_path in MODEL_INFO:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"文件不存在: {npz_path}")

    print(f"正在处理: {model_display_name} -> {npz_path}")

    X_train_orig, y_train_orig, X_generated, y_generated, feature_names = load_original_and_generated(npz_path)

    variable_names = []
    original_arrays = []
    generated_arrays = []

    # PTB 放在第一个
    if y_train_orig is not None and y_generated is not None:
        variable_names.append("PTB")
        original_arrays.append(y_train_orig)
        generated_arrays.append(y_generated)

    # 再放 8 个特征
    for j, feat in enumerate(feature_names):
        variable_names.append(feat)
        original_arrays.append(X_train_orig[:, j])
        generated_arrays.append(X_generated[:, j])

    # 只取前 9 个变量，保证 3×3
    variable_names = variable_names[:9]
    original_arrays = original_arrays[:9]
    generated_arrays = generated_arrays[:9]

    # 关键修改：尺寸改为双栏兼容 (宽度7.0英寸 ≈ 17.8cm, 高度6.8英寸)
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.8), dpi=300)
    axes = axes.flatten()
    letters = list(string.ascii_lowercase[:9])

    model_color = MODEL_COLOR_MAP[model_short_name]

    for i, ax in enumerate(axes):
        if i >= len(variable_names):
            ax.axis("off")
            continue

        var_name = variable_names[i]
        orig_vals = np.asarray(original_arrays[i], dtype=float)
        gen_vals = np.asarray(generated_arrays[i], dtype=float)

        orig_vals = orig_vals[np.isfinite(orig_vals)]
        gen_vals = gen_vals[np.isfinite(gen_vals)]

        bins = get_bins(orig_vals, gen_vals, n_bins=28)

        # Original histogram (透明度、线宽微调)
        ax.hist(
            orig_vals,
            bins=bins,
            density=True,
            alpha=0.35,
            color=COLOR_ORIG_FILL,
            edgecolor="white",
            linewidth=0.4
        )

        # Current model histogram
        ax.hist(
            gen_vals,
            bins=bins,
            density=True,
            alpha=0.35,
            color=model_color,
            edgecolor="white",
            linewidth=0.4
        )

        # KDE lines (线宽调小)
        safe_kdeplot(ax, orig_vals, color=COLOR_ORIG_LINE, linewidth=1.2, alpha=0.95)
        safe_kdeplot(ax, gen_vals, color=model_color, linewidth=1.3, alpha=0.98)

        # 左上角字母
        ax.text(
            -0.1, 1.08, letters[i],
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,          # 原12
            fontweight="bold",
            color="#222222"
        )

        ax.set_title("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        row = i // 3
        col = i % 3

        # 只保留最左列 y 轴标签
        if col == 0:
            ax.set_ylabel("Density", fontsize=9)
            ax.tick_params(axis="y", labelsize=7, length=2.0, width=0.6)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False, length=2.0, width=0.6)

        # x轴标签用变量名
        ax.set_xlabel(var_name, fontsize=9)

        if row == 2:
            ax.tick_params(axis="x", labelsize=7, length=2.0, width=0.6)
        else:
            ax.tick_params(axis="x", labelsize=7, length=2.0, width=0.6)

        # 留边距
        all_vals = np.concatenate([orig_vals, gen_vals])
        if len(all_vals) > 0:
            x_min, x_max = np.min(all_vals), np.max(all_vals)
            if not np.isclose(x_min, x_max):
                pad = 0.04 * (x_max - x_min)
                ax.set_xlim(x_min - pad, x_max + pad)

        ax.grid(False)

        # 图例只放第一个子图
        if i == 0:
            legend_elements = [
                Line2D([0], [0], color=COLOR_ORIG_LINE, lw=1.4, label="Original"),
                Line2D([0], [0], color=model_color, lw=1.4, label=model_display_name),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                frameon=False,
                fontsize=7.5,
                handlelength=2.0
            )

    # 调整子图间距（紧凑但不粘连）
    plt.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.97,
        bottom=0.08,
        wspace=0.18,
        hspace=0.22
    )

    save_png = os.path.join(SAVE_DIR, f"{model_short_name}_marginal_hist_kde_3x3_xlabel_varname.png")
    save_pdf = os.path.join(SAVE_DIR, f"{model_short_name}_marginal_hist_kde_3x3_xlabel_varname.pdf")

    plt.savefig(save_png, dpi=600, bbox_inches="tight")
    plt.savefig(save_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"已保存:\n{save_png}\n{save_pdf}\n")

print("全部 8 张图已生成完成。")