import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#4D4D4D"
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"
plt.rcParams["text.color"] = "#222222"
plt.rcParams["axes.labelcolor"] = "#222222"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"

plt.rcParams["font.size"] = 7.5
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 7

regions = [
    "inner mongolia", "ningxia", "qinghai", "sichuan",
    "xinjiang", "tibet", "yunnan", "guangxi",
    "guizhou", "gansu", "chongqing", "shaanxi"
]

region_to_csv = {
    region: os.path.join(region, "output", "final_train", "figures", "gbdt_cv_test_dualopt_scatter_points.csv")
    for region in regions
}

SAVE_DIR = "output/final_train"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH_PNG = os.path.join(SAVE_DIR, "gbdt_cv_test_dualopt_joint_scatter_12regions_3x4.png")
SAVE_PATH_PDF = os.path.join(SAVE_DIR, "gbdt_cv_test_dualopt_joint_scatter_12regions_3x4.pdf")

color_val_orig = "#4781c1"
color_test_orig = "#0b338b"
color_val_comb = "#cd94b3"
color_test_comb = "#b03966"

def safe_kde(data, grid):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2 or np.std(data) < 1e-12:
        return None
    try:
        kde = gaussian_kde(data)
        return kde(grid)
    except Exception:
        return None

def load_region_scatter_data(csv_path):
    df = pd.read_csv(csv_path)
    df_val_orig = df[df["group"] == "Validation_Orig"]
    df_val_comb = df[df["group"] == "Validation_Orig_P1_P2"]
    df_test_orig = df[df["group"] == "Test_Orig"]
    df_test_comb = df[df["group"] == "Test_Orig_P1_P2"]
    y_val_orig = df_val_orig["observed"].values
    y_val_pred_orig = df_val_orig["predicted"].values
    y_val_comb = df_val_comb["observed"].values
    y_val_pred_comb = df_val_comb["predicted"].values
    y_test = df_test_orig["observed"].values
    y_test_pred_orig = df_test_orig["predicted"].values
    y_test_pred_comb = df_test_comb["predicted"].values
    all_obs = np.concatenate([y_val_orig, y_val_comb, y_test, y_test])
    all_pred = np.concatenate([y_val_pred_orig, y_val_pred_comb, y_test_pred_orig, y_test_pred_comb])
    vmin = min(all_obs.min(), all_pred.min())
    vmax = max(all_obs.max(), all_pred.max())
    pad = 0.05 * (vmax - vmin)
    vmin -= pad
    vmax += pad
    return {
        "y_val_orig": y_val_orig, "y_val_pred_orig": y_val_pred_orig,
        "y_val_comb": y_val_comb, "y_val_pred_comb": y_val_pred_comb,
        "y_test": y_test, "y_test_pred_orig": y_test_pred_orig,
        "y_test_pred_comb": y_test_pred_comb, "vmin": vmin, "vmax": vmax
    }

def plot_single_panel(fig, outer_spec, data, panel_label, row, col, n_rows, n_cols):
    inner_gs = GridSpecFromSubplotSpec(
        4, 4, subplot_spec=outer_spec,
        width_ratios=[1, 1, 1, 0.44],
        height_ratios=[0.44, 1, 1, 1],
        wspace=0.05, hspace=0.05
    )
    ax_top = fig.add_subplot(inner_gs[0, 0:3])
    ax_main = fig.add_subplot(inner_gs[1:4, 0:3])
    ax_right = fig.add_subplot(inner_gs[1:4, 3])

    y_val_orig = data["y_val_orig"]
    y_val_pred_orig = data["y_val_pred_orig"]
    y_val_comb = data["y_val_comb"]
    y_val_pred_comb = data["y_val_pred_comb"]
    y_test = data["y_test"]
    y_test_pred_orig = data["y_test_pred_orig"]
    y_test_pred_comb = data["y_test_pred_comb"]
    vmin = data["vmin"]
    vmax = data["vmax"]

    ax_main.scatter(y_val_orig, y_val_pred_orig, s=14, alpha=0.60,
                    color=color_val_orig, edgecolor="white", linewidth=0.3, zorder=3)
    ax_main.scatter(y_val_comb, y_val_pred_comb, s=14, alpha=0.60,
                    color=color_val_comb, edgecolor="white", linewidth=0.3, zorder=3)
    ax_main.scatter(y_test, y_test_pred_orig, s=22, alpha=0.88,
                    color=color_test_orig, edgecolor="white", linewidth=0.4, zorder=4)
    ax_main.scatter(y_test, y_test_pred_comb, s=22, alpha=0.88,
                    color=color_test_comb, edgecolor="white", linewidth=0.4, zorder=4)

    ax_main.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=0.8, color="#7A7A7A", zorder=2)
    ax_main.set_xlim(vmin, vmax)
    ax_main.set_ylim(vmin, vmax)

    if col == 0:
        ax_main.set_ylabel("Predicted PTB", fontsize=8)
    else:
        ax_main.set_ylabel("")
    if row == n_rows - 1:
        ax_main.set_xlabel("Observed PTB", fontsize=8)
    else:
        ax_main.set_xlabel("")

    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.tick_params(axis="both", labelsize=7, length=2.5, width=0.6)

    top_data = np.concatenate([y_val_orig, y_val_comb])
    bins_x = np.histogram_bin_edges(top_data, bins=20)
    ax_top.hist(y_val_orig, bins=bins_x, density=True, color=color_val_orig, alpha=0.45, edgecolor="white", linewidth=0.4)
    ax_top.hist(y_val_comb, bins=bins_x, density=True, color=color_val_comb, alpha=0.45, edgecolor="white", linewidth=0.4)
    x_grid = np.linspace(vmin, vmax, 400)
    kde_top_orig = safe_kde(y_val_orig, x_grid)
    kde_top_comb = safe_kde(y_val_comb, x_grid)
    if kde_top_orig is not None:
        ax_top.plot(x_grid, kde_top_orig, color=color_test_orig, linewidth=1.2)
    if kde_top_comb is not None:
        ax_top.plot(x_grid, kde_top_comb, color=color_test_comb, linewidth=1.2)
    ax_top.set_xlim(vmin, vmax)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_top.tick_params(axis="y", left=False, labelleft=False)

    right_data = np.concatenate([y_val_pred_orig, y_val_pred_comb])
    bins_y = np.histogram_bin_edges(right_data, bins=20)
    ax_right.hist(y_val_pred_orig, bins=bins_y, density=True, orientation="horizontal",
                  color=color_val_orig, alpha=0.45, edgecolor="white", linewidth=0.4)
    ax_right.hist(y_val_pred_comb, bins=bins_y, density=True, orientation="horizontal",
                  color=color_val_comb, alpha=0.45, edgecolor="white", linewidth=0.4)
    y_grid = np.linspace(vmin, vmax, 400)
    kde_right_orig = safe_kde(y_val_pred_orig, y_grid)
    kde_right_comb = safe_kde(y_val_pred_comb, y_grid)
    if kde_right_orig is not None:
        ax_right.plot(kde_right_orig, y_grid, color=color_test_orig, linewidth=1.2)
    if kde_right_comb is not None:
        ax_right.plot(kde_right_comb, y_grid, color=color_test_comb, linewidth=1.2)
    ax_right.set_ylim(vmin, vmax)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)
    ax_right.tick_params(axis="y", labelleft=False, left=False)
    ax_right.tick_params(axis="x", bottom=False, labelbottom=False)

    ax_main.text(-0.1, 1.05, panel_label, transform=ax_main.transAxes,
                 ha="left", va="bottom", fontsize=10, fontweight="bold", color="#222222")


def main():
    data_dict = {}
    for region, csv_path in region_to_csv.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"未找到文件：{csv_path}")
        data_dict[region] = load_region_scatter_data(csv_path)

    n_rows, n_cols = 3, 4

    fig = plt.figure(figsize=(7.0, 5.0))

    outer_gs = GridSpec(
        n_rows, n_cols, figure=fig,
        wspace=0.18, hspace=0.12,
        bottom=0.08, top=0.97, left=0.05, right=0.99
    )

    panel_labels = list("abcdefghijkl")

    for idx, region in enumerate(regions):
        row = idx // n_cols
        col = idx % n_cols
        plot_single_panel(
            fig=fig,
            outer_spec=outer_gs[row, col],
            data=data_dict[region],
            panel_label=panel_labels[idx],
            row=row, col=col,
            n_rows=n_rows, n_cols=n_cols
        )

    legend_handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=color_val_orig, markeredgecolor='white',
               markeredgewidth=0.4, markersize=5,
               label='Validation (Orig)'),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=color_val_comb, markeredgecolor='white',
               markeredgewidth=0.4, markersize=5,
               label='Validation (Orig+P1+P2)'),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=color_test_orig, markeredgecolor='white',
               markeredgewidth=0.4, markersize=6,
               label='Test (Orig)'),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=color_test_comb, markeredgecolor='white',
               markeredgewidth=0.4, markersize=6,
               label='Test (Orig+P1+P2)')
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        fontsize=7,
        handletextpad=0.4,
        ncol=4
    )

    plt.savefig(SAVE_PATH_PNG, dpi=600, bbox_inches="tight")
    plt.savefig(SAVE_PATH_PDF, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"\nFigure saved to:\n{SAVE_PATH_PNG}\n{SAVE_PATH_PDF}")

if __name__ == "__main__":
    main()