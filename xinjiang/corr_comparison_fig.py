import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. 基础设置（双栏布局）
# =========================
NPZ_PATHS = [
    "output/phase2/gnn/phase2_cgtgan_epoch_0300.npz",
    "output/phase2/ctgan/ctgan_phase2_input_epoch_0300.npz",
    "output/phase2/tvae/tvae_phase2_input_epoch_0300.npz",
    "output/phase2/great/great_phase2_input_epoch_0100.npz",
    "output/phase2/gnn_ablation/wo_conditional_node/wo_conditional_node_epoch_0300.npz",
    "output/phase2/gnn_ablation/wo_graph_module/wo_graph_module_epoch_0300.npz",
    "output/phase2/gnn_ablation/wo_learnable_adjacency/wo_learnable_adjacency_epoch_0300.npz",
    "output/phase2/gnn_ablation/wo_node_level_loss/wo_node_level_loss_epoch_0300.npz",
]

SAVE_DIR = "output/phase2/comparison/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PNG = os.path.join(SAVE_DIR, "delta_pearson_2x4_full_matrix_compact.png")
SAVE_PDF = os.path.join(SAVE_DIR, "delta_pearson_2x4_full_matrix_compact.pdf")

# 全局字体设置（双栏小字号）
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["font.size"] = 7               # 全局基准字号
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6
plt.rcParams["legend.fontsize"] = 6

# =========================
# 2. 配色
# =========================
cmap = sns.blend_palette(
    ["#F8FBFC", "#DCECEF", "#A8C9D0", "#5F97A8", "#2E5F77"],
    as_cmap=True
)

# =========================
# 3. 工具函数
# =========================
def find_first_existing_key(data, candidate_keys, npz_path):
    for key in candidate_keys:
        if key in data:
            return key
    raise KeyError(f"{npz_path} 中未找到任何候选键: {candidate_keys}")

def load_npz_and_compute_delta_corr(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "X_train_orig" not in data:
        raise KeyError(f"{npz_path} 中缺少键: X_train_orig")
    X_train_orig = data["X_train_orig"].astype(np.float32)

    phase1_key = find_first_existing_key(data, ["X_aug1", "X_aug1_used"], npz_path)
    X_phase1 = data[phase1_key].astype(np.float32)

    gen_key = find_first_existing_key(data, ["generated_X", "X_generated", "X_gen", "fake_X", "synthetic_X"], npz_path)
    X_phase2 = data[gen_key].astype(np.float32)

    if "feature_names" in data:
        feature_names = [str(x) for x in data["feature_names"].tolist()]
    elif "features" in data:
        feature_names = [str(x) for x in data["features"].tolist()]
    else:
        feature_names = [f"X{i+1}" for i in range(X_train_orig.shape[1])]

    X_all = np.vstack([X_train_orig, X_phase1, X_phase2]).astype(np.float32)

    df_orig = pd.DataFrame(X_train_orig, columns=feature_names)
    df_all = pd.DataFrame(X_all, columns=feature_names)

    corr_orig = df_orig.corr(method="pearson")
    corr_all = df_all.corr(method="pearson")

    delta_corr = np.abs(corr_orig.values - corr_all.values)
    delta_corr_df = pd.DataFrame(delta_corr, index=feature_names, columns=feature_names)
    return delta_corr_df, feature_names

# =========================
# 4. 读取所有矩阵并统一色标范围
# =========================
delta_corr_list = []
for npz_path in NPZ_PATHS:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"文件不存在: {npz_path}")
    delta_corr_df, _ = load_npz_and_compute_delta_corr(npz_path)
    delta_corr_list.append(delta_corr_df)

global_vmax = max(np.nanmax(df.values) for df in delta_corr_list)

# =========================
# 5. 绘制两行四列大图（双栏宽度 6.7 英寸）
# =========================
fig = plt.figure(figsize=(6.7, 4.0), dpi=300)   # 宽度 6.7 英寸，高度按比例缩小

gs = fig.add_gridspec(
    nrows=2,
    ncols=5,
    width_ratios=[1, 1, 1, 1, 0.08],           # colorbar 列变窄
    wspace=0.12,
    hspace=0.0
)

axes = []
for r in range(2):
    for c in range(4):
        axes.append(fig.add_subplot(gs[r, c]))

cax = fig.add_subplot(gs[:, 4])

letters = list(string.ascii_lowercase[:8])
first_hm = None

for i, ax in enumerate(axes):
    delta_corr_df = delta_corr_list[i]

    current_hm = sns.heatmap(
        delta_corr_df,
        cmap=cmap,
        vmin=0,
        vmax=global_vmax,
        square=True,
        linewidths=0.5,                 # 网格线变细
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 5.0},    # 标注数字缩小
        cbar=(i == 0),
        cbar_ax=cax if i == 0 else None,
        ax=ax
    )

    if i == 0:
        first_hm = current_hm

    row = i // 4
    col = i % 4

    # 只保留最下行 x 轴标签
    if row == 1:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
        ax.tick_params(axis="x", bottom=True, labelbottom=True, length=2, width=0.5, pad=1)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # 只保留最左列 y 轴标签
    if col == 0:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
        ax.tick_params(axis="y", left=True, labelleft=True, length=2, width=0.5, pad=1)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False, labelleft=False)

    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # 左上角字母标记（字号缩小）
    ax.text(
        -0.08, 1.09, letters[i],
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        fontweight="bold",
        color="#222222"
    )

# =========================
# 6. 共享 colorbar 设置
# =========================
cbar = first_hm.collections[0].colorbar
cbar.set_label("")
cbar.ax.tick_params(labelsize=5.5, length=2, width=0.5)
cbar.ax.set_title(r"$|\Delta r_{ij}|$", fontsize=7, pad=6)

# =========================
# 7. 保存
# =========================
plt.savefig(SAVE_PNG, dpi=600, bbox_inches="tight")
plt.savefig(SAVE_PDF, dpi=600, bbox_inches="tight")
plt.show()

print(f"图已保存到:\n{SAVE_PNG}\n{SAVE_PDF}")