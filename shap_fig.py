import os
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import shap

warnings.filterwarnings("ignore")

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
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

plt.rcParams["font.size"] = 7.5
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 6.5

regions = [
    "inner mongolia", "ningxia", "qinghai", "sichuan",
    "xinjiang", "tibet", "yunnan", "guangxi",
    "guizhou", "gansu", "chongqing", "shaanxi"
]

bundle_paths = {
    region: os.path.join(
        region,
        "output",
        "final_train",
        "shap_phase1_phase2",
        "gbdt_shap_bundle_train_combined.npz",
    )
    for region in regions
}

SAVE_DIR = os.path.join("output", "final_train", "shap_phase1_phase2")
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_BAR_PNG = os.path.join(SAVE_DIR, "gbdt_shap_importance_bar_12regions_3x4_native.png")
SAVE_BAR_PDF = os.path.join(SAVE_DIR, "gbdt_shap_importance_bar_12regions_3x4_native.pdf")
SAVE_BAR_TIFF = os.path.join(SAVE_DIR, "gbdt_shap_importance_bar_12regions_3x4_native.tiff")

SAVE_BEE_PNG = os.path.join(SAVE_DIR, "gbdt_shap_beeswarm_12regions_3x4_native.png")
SAVE_BEE_PDF = os.path.join(SAVE_DIR, "gbdt_shap_beeswarm_12regions_3x4_native.pdf")
SAVE_BEE_TIFF = os.path.join(SAVE_DIR, "gbdt_shap_beeswarm_12regions_3x4_native.tiff")

bar_cmap = LinearSegmentedColormap.from_list(
    "bar_cmap",
    ["#2f4159", "#4f7ca5", "#90b6d8", "#efc9b4", "#f1be90"]
)

beeswarm_cmap = LinearSegmentedColormap.from_list(
    "beeswarm_cmap",
    ["#4781c1", "#adcfea", "#dccbde", "#a97eba"]
)


def decode_feature_names(arr):
    names = []
    for x in arr:
        if isinstance(x, bytes):
            names.append(x.decode("utf-8"))
        else:
            names.append(str(x))
    return names


def load_region_bundle(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "shap_values" not in data or "X_explain" not in data or "feature_names" not in data:
        raise KeyError(f"Missing required arrays in {npz_path}")
    shap_values = np.asarray(data["shap_values"], dtype=float)
    X_explain = np.asarray(data["X_explain"], dtype=float)
    feature_names = decode_feature_names(data["feature_names"])
    if shap_values.ndim != 2:
        raise ValueError(f"shap_values must be 2D in {npz_path}, got shape {shap_values.shape}")
    if X_explain.ndim != 2:
        raise ValueError(f"X_explain must be 2D in {npz_path}, got shape {X_explain.shape}")
    if shap_values.shape != X_explain.shape:
        raise ValueError(
            f"Shape mismatch in {npz_path}: shap_values {shap_values.shape} vs X_explain {X_explain.shape}")
    if len(feature_names) != X_explain.shape[1]:
        raise ValueError(f"Feature name count mismatch in {npz_path}: {len(feature_names)} vs {X_explain.shape[1]}")
    exp = shap.Explanation(values=shap_values, data=X_explain, feature_names=feature_names)
    return exp


def style_native_bar_ax(ax, n_features):
    ax.set_title("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#7A7A7A")
    ax.tick_params(axis="x", labelsize=7, length=2.5, width=0.6)
    ax.tick_params(axis="y", labelsize=7, length=0)
    ax.set_axisbelow(True)

    patches = ax.patches
    n = len(patches)
    denom = max(n - 1, 1)
    for i, p in enumerate(patches):
        p.set_facecolor(bar_cmap(i / denom))
        p.set_edgecolor("none")
        p.set_alpha(0.98)

    for text in ax.texts:
        text.set_fontsize(6)
        text.set_ha("left")


def style_native_beeswarm_ax(ax, row, col, n_rows, n_cols):
    ax.set_title("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#7A7A7A")
    ax.tick_params(axis="x", labelsize=7, length=2.5, width=0.6)
    ax.tick_params(axis="y", labelsize=7, length=0)
    ax.grid(False)
    if row != n_rows - 1:
        ax.set_xlabel("")
    else:
        ax.set_xlabel("SHAP value", fontsize=8)
    if col != 0:
        ax.set_ylabel("")


def add_panel_label(ax, label):
    ax.text(
        -0.10, 1.02, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=10, fontweight="bold", color="#222222"
    )


def main():
    region_explanations = {}
    for region, path in bundle_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        region_explanations[region] = load_region_bundle(path)

    n_rows, n_cols = 3, 4
    panel_labels = list("abcdefghijkl")

    # Figure 1: bar plot
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(7.0, 6.2))
    axes1 = np.asarray(axes1)

    for idx, region in enumerate(regions):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes1[row, col]
        exp = region_explanations[region]
        n_features = exp.values.shape[1]

        shap.plots.bar(
            exp,
            max_display=n_features,
            show=False,
            ax=ax,
        )

        style_native_bar_ax(ax, n_features=n_features)
        add_panel_label(ax, panel_labels[idx])

        if row != n_rows - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("mean(|SHAP value|)", fontsize=8)

    fig1.subplots_adjust(left=0.10, right=0.99, top=0.98, bottom=0.08, wspace=0.45, hspace=0.30)
    fig1.savefig(SAVE_BAR_PNG, dpi=600, bbox_inches="tight")
    fig1.savefig(SAVE_BAR_PDF, dpi=600, bbox_inches="tight")
    fig1.savefig(SAVE_BAR_TIFF, dpi=600, bbox_inches="tight")
    plt.close(fig1)

    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(7.0, 6.2))
    axes2 = np.asarray(axes2)

    for idx, region in enumerate(regions):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes2[row, col]
        exp = region_explanations[region]
        n_features = exp.values.shape[1]

        shap.plots.beeswarm(
            exp,
            max_display=n_features,
            order=shap.Explanation.abs.mean(0),
            color=beeswarm_cmap,
            axis_color="#333333",
            alpha=0.95,
            ax=ax,
            show=False,
            plot_size=None,
            color_bar=False,
            group_remaining_features=False,
            s=10,
        )

        style_native_beeswarm_ax(ax, row=row, col=col, n_rows=n_rows, n_cols=n_cols)
        add_panel_label(ax, panel_labels[idx])

    fig2.subplots_adjust(left=0.08, right=0.88, top=0.98, bottom=0.08, wspace=0.40, hspace=0.30)

    cax = fig2.add_axes([0.90, 0.18, 0.012, 0.64])
    sm = mpl.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=beeswarm_cmap)
    sm.set_array([])
    cbar = fig2.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=8, length=0)
    cbar.outline.set_visible(False)
    cbar.set_label("Feature value", fontsize=8)

    fig2.savefig(SAVE_BEE_PNG, dpi=600, bbox_inches="tight")
    fig2.savefig(SAVE_BEE_PDF, dpi=600, bbox_inches="tight")
    fig2.savefig(SAVE_BEE_TIFF, dpi=600, bbox_inches="tight")
    plt.close(fig2)

    print("Saved files:")
    print(SAVE_BAR_PNG)
    print(SAVE_BAR_PDF)
    print(SAVE_BAR_TIFF)
    print(SAVE_BEE_PNG)
    print(SAVE_BEE_PDF)
    print(SAVE_BEE_TIFF)


if __name__ == "__main__":
    main()