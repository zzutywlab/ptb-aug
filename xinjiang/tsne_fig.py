import os
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D

# =========================
# 1. 基础设置
# =========================
MODEL_PATHS = [
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

SAVE_PNG = os.path.join(SAVE_DIR, "tsne_2x4_models_and_ablations.png")
SAVE_PDF = os.path.join(SAVE_DIR, "tsne_2x4_models_and_ablations.pdf")

RANDOM_STATE = 42

# 期刊双栏风格全局设置
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#4D4D4D"
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"
plt.rcParams["text.color"] = "#222222"
plt.rcParams["axes.labelcolor"] = "#222222"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"

plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 9.5
plt.rcParams["xtick.labelsize"] = 7.5
plt.rcParams["ytick.labelsize"] = 7.5
plt.rcParams["legend.fontsize"] = 7.5  # 底部图例字号

# =========================
# 2. 配色设计
# =========================
color_orig = "#d72e9e"
color_p1 = "#f79173"
color_p2 = "#4bb4d4"
edge_color = "white"

# =========================
# 3. 工具函数（不变）
# =========================
def find_first_existing_key(data, candidate_keys, file_path):
    for key in candidate_keys:
        if key in data:
            return key
    raise KeyError(f"{file_path} 中未找到任何候选键: {candidate_keys}")

def load_npz_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "X_train_orig" not in data:
        raise KeyError(f"{npz_path} 中缺少键: X_train_orig")
    X_train_orig = data["X_train_orig"].astype(np.float32)
    phase1_key = find_first_existing_key(data, ["X_aug1", "X_aug1_used"], npz_path)
    X_phase1 = data[phase1_key].astype(np.float32)
    gen_key = find_first_existing_key(data, ["generated_X", "X_generated", "X_gen", "fake_X", "synthetic_X"], npz_path)
    X_generated = data[gen_key].astype(np.float32)
    return X_train_orig, X_phase1, X_generated

def compute_tsne_three_groups(X1, X2, X3, random_state=42):
    X_all = np.vstack([X1, X2, X3])
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    n_samples = X_all_scaled.shape[0]
    perplexity = min(30, max(5, n_samples // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=random_state, max_iter=2000)
    X_tsne = tsne.fit_transform(X_all_scaled)
    n1, n2, n3 = len(X1), len(X2), len(X3)
    return X_tsne[:n1], X_tsne[n1:n1+n2], X_tsne[n1+n2:n1+n2+n3], X_tsne

def plot_one_panel(ax, X_orig_tsne, X_p1_tsne, X_p2_tsne, X_all_tsne, panel_letter):
    """绘制单个子图，不再包含图例。"""
    ax.scatter(X_orig_tsne[:, 0], X_orig_tsne[:, 1],
               s=20, color=color_orig, alpha=0.82,
               edgecolor=edge_color, linewidth=0.4, label="Original")
    ax.scatter(X_p1_tsne[:, 0], X_p1_tsne[:, 1],
               s=16, color=color_p1, alpha=0.55,
               edgecolor=edge_color, linewidth=0.3, label="Phase 1")
    ax.scatter(X_p2_tsne[:, 0], X_p2_tsne[:, 1],
               s=16, color=color_p2, alpha=0.55,
               edgecolor=edge_color, linewidth=0.3, label="Phase 2")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7.5, length=2.5, width=0.6)

    x_min, x_max = X_all_tsne[:, 0].min(), X_all_tsne[:, 0].max()
    y_min, y_max = X_all_tsne[:, 1].min(), X_all_tsne[:, 1].max()
    x_pad, y_pad = 0.06 * (x_max - x_min), 0.06 * (y_max - y_min)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.text(-0.12, 1.15, panel_letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=10, fontweight="bold")

# =========================
# 4. 绘制 2×4 大图（图例移至底部）
# =========================
# 宽度 6.7 英寸（双栏），高度增加到 5.2 英寸以便为底部图例留出空间
fig, axes = plt.subplots(2, 4, figsize=(6.7, 3.7), dpi=300)
axes = axes.flatten()
letters = list(string.ascii_lowercase[:8])

for i, (ax, npz_path, panel_letter) in enumerate(zip(axes, MODEL_PATHS, letters)):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"文件不存在: {npz_path}")
    print(f"[{panel_letter}] 正在处理: {npz_path}")
    X_train_orig, X_phase1, X_generated = load_npz_data(npz_path)
    X_tsne_orig, X_tsne_p1, X_tsne_p2, X_tsne_all = compute_tsne_three_groups(
        X_train_orig, X_phase1, X_generated, random_state=RANDOM_STATE
    )
    plot_one_panel(ax, X_tsne_orig, X_tsne_p1, X_tsne_p2, X_tsne_all, panel_letter)

    if i % 4 == 0:
        ax.set_ylabel("t-SNE 2", fontsize=9.5)
    else:
        ax.set_ylabel("")
    if i // 4 == 1:
        ax.set_xlabel("t-SNE 1", fontsize=9.5)
    else:
        ax.set_xlabel("")

# 调整子图布局，为底部图例预留空间（bottom 值增大）
plt.subplots_adjust(left=0.06, right=0.99, top=0.98, bottom=0.12,
                    wspace=0.28, hspace=0.32)

# 创建底部图例
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Original',
           markerfacecolor=color_orig, markeredgecolor=edge_color,
           markeredgewidth=0.5, markersize=6, alpha=0.95),
    Line2D([0], [0], marker='o', color='w', label='Phase 1',
           markerfacecolor=color_p1, markeredgecolor=edge_color,
           markeredgewidth=0.5, markersize=6, alpha=0.85),
    Line2D([0], [0], marker='o', color='w', label='Phase 2',
           markerfacecolor=color_p2, markeredgecolor=edge_color,
           markeredgewidth=0.5, markersize=6, alpha=0.85),
]
fig.legend(handles=legend_elements, loc='lower center',
           bbox_to_anchor=(0.5, -0.07), frameon=False,
           fontsize=8, ncol=3, handletextpad=0.5)

# 保存
plt.savefig(SAVE_PNG, dpi=600, bbox_inches="tight")
plt.savefig(SAVE_PDF, dpi=600, bbox_inches="tight")
plt.show()

print(f"\n图已保存到:\n{SAVE_PNG}\n{SAVE_PDF}")