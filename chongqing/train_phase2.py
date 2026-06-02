import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#4D4D4D"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"
plt.rcParams["text.color"] = "#222222"
plt.rcParams["axes.labelcolor"] = "#222222"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"

PHASE1_PATH = "output/phase1/phase1_augmented_data.npz"
PHASE2_PATH = "output/phase2/gnn/phase2_cgtgan_epoch_0800.npz"

SAVE_DIR = "output/final_train"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PATH_PNG = os.path.join(SAVE_DIR, "gbdt_cv_test_dualopt_joint_scatter_phase1_phase2.png")
SAVE_PATH_PDF = os.path.join(SAVE_DIR, "gbdt_cv_test_dualopt_joint_scatter_phase1_phase2.pdf")
SAVE_PATH_SCATTER_CSV = os.path.join(SAVE_DIR, "gbdt_cv_test_dualopt_scatter_points.csv")

RANDOM_STATE = 42

ORIGINAL_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3
}

COMB_CV_PARAMS = {
    "n_estimators": 191,
    "learning_rate": 0.09023641438824008,
    "max_depth": 3
}

COMB_TEST_PARAMS = {
    "n_estimators": 127,
    "learning_rate": 0.20003734686890992,
    "max_depth": 10
}

def build_gbdt_model(params):
    return GradientBoostingRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=RANDOM_STATE
    )

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

def cross_validated_predictions(X, y, params, n_splits=5, random_state=42):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = build_gbdt_model(params)
    y_pred = cross_val_predict(model, X, y, cv=cv)
    return y, y_pred

def plot_joint_scatter_with_marginals(
    y_val_orig, y_val_pred_orig,
    y_val_comb, y_val_pred_comb,
    y_test, y_test_pred_orig,
    y_test_pred_comb,
    save_path_png,
    save_path_pdf
):
    color_val_orig = "#9CC9E3"
    color_test_orig = "#2E6F9E"
    color_val_comb = "#F2A7A0"
    color_test_comb = "#B33C3C"

    all_obs = np.concatenate([y_val_orig, y_val_comb, y_test, y_test])
    all_pred = np.concatenate([y_val_pred_orig, y_val_pred_comb, y_test_pred_orig, y_test_pred_comb])

    vmin = min(all_obs.min(), all_pred.min())
    vmax = max(all_obs.max(), all_pred.max())
    pad = 0.05 * (vmax - vmin)
    vmin -= pad
    vmax += pad

    fig = plt.figure(figsize=(9.2, 8.4))
    gs = GridSpec(
        4, 4, figure=fig,
        width_ratios=[1, 1, 1, 0.44],
        height_ratios=[0.44, 1, 1, 1],
        wspace=0.05, hspace=0.05
    )

    ax_top = fig.add_subplot(gs[0, 0:3])
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_right = fig.add_subplot(gs[1:4, 3])

    ax_main.scatter(
        y_val_orig, y_val_pred_orig,
        s=28, alpha=0.55,
        color=color_val_orig,
        edgecolor="white", linewidth=0.4
    )
    ax_main.scatter(
        y_val_comb, y_val_pred_comb,
        s=28, alpha=0.55,
        color=color_val_comb,
        edgecolor="white", linewidth=0.4
    )
    ax_main.scatter(
        y_test, y_test_pred_orig,
        s=46, alpha=0.85,
        color=color_test_orig,
        edgecolor="white", linewidth=0.5
    )
    ax_main.scatter(
        y_test, y_test_pred_comb,
        s=46, alpha=0.85,
        color=color_test_comb,
        edgecolor="white", linewidth=0.5
    )

    ax_main.plot(
        [vmin, vmax], [vmin, vmax],
        linestyle="--", linewidth=1.1, color="#7A7A7A"
    )

    ax_main.set_xlim(vmin, vmax)
    ax_main.set_ylim(vmin, vmax)
    ax_main.set_xlabel("Observed PTB", fontsize=11)
    ax_main.set_ylabel("Predicted PTB", fontsize=11)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.tick_params(axis="both", labelsize=10, length=3, width=0.8)

    top_data = np.concatenate([y_val_orig, y_val_comb])
    bins_x = np.histogram_bin_edges(top_data, bins=20)

    ax_top.hist(
        y_val_orig,
        bins=bins_x,
        density=True,
        color=color_val_orig,
        alpha=0.35,
        edgecolor="white",
        linewidth=0.5
    )
    ax_top.hist(
        y_val_comb,
        bins=bins_x,
        density=True,
        color=color_val_comb,
        alpha=0.35,
        edgecolor="white",
        linewidth=0.5
    )

    x_grid = np.linspace(vmin, vmax, 400)
    kde_top_orig = safe_kde(y_val_orig, x_grid)
    kde_top_comb = safe_kde(y_val_comb, x_grid)

    if kde_top_orig is not None:
        ax_top.plot(x_grid, kde_top_orig, color=color_test_orig, linewidth=1.8)
    if kde_top_comb is not None:
        ax_top.plot(x_grid, kde_top_comb, color=color_test_comb, linewidth=1.8)

    ax_top.set_xlim(vmin, vmax)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_top.tick_params(axis="y", left=False, labelleft=False)

    right_data = np.concatenate([y_val_pred_orig, y_val_pred_comb])
    bins_y = np.histogram_bin_edges(right_data, bins=20)

    ax_right.hist(
        y_val_pred_orig,
        bins=bins_y,
        density=True,
        orientation="horizontal",
        color=color_val_orig,
        alpha=0.35,
        edgecolor="white",
        linewidth=0.5
    )
    ax_right.hist(
        y_val_pred_comb,
        bins=bins_y,
        density=True,
        orientation="horizontal",
        color=color_val_comb,
        alpha=0.35,
        edgecolor="white",
        linewidth=0.5
    )

    y_grid = np.linspace(vmin, vmax, 400)
    kde_right_orig = safe_kde(y_val_pred_orig, y_grid)
    kde_right_comb = safe_kde(y_val_pred_comb, y_grid)

    if kde_right_orig is not None:
        ax_right.plot(kde_right_orig, y_grid, color=color_test_orig, linewidth=1.8)
    if kde_right_comb is not None:
        ax_right.plot(kde_right_comb, y_grid, color=color_test_comb, linewidth=1.8)

    ax_right.set_ylim(vmin, vmax)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)
    ax_right.tick_params(axis="y", labelleft=False, left=False)
    ax_right.tick_params(axis="x", bottom=False, labelbottom=False)

    ax_main.text(
        0.03, 0.97, "Validation (Orig, 5-fold)",
        transform=ax_main.transAxes,
        ha="left", va="top", fontsize=10, color=color_test_orig
    )
    ax_main.text(
        0.03, 0.92, "Validation (Orig+P1+P2, 5-fold)",
        transform=ax_main.transAxes,
        ha="left", va="top", fontsize=10, color=color_test_comb
    )
    ax_main.text(
        0.03, 0.87, "Test (Orig)",
        transform=ax_main.transAxes,
        ha="left", va="top", fontsize=10, color=color_test_orig
    )
    ax_main.text(
        0.03, 0.82, "Test (Orig+P1+P2)",
        transform=ax_main.transAxes,
        ha="left", va="top", fontsize=10, color=color_test_comb
    )

    plt.savefig(save_path_png, dpi=600, bbox_inches="tight")
    plt.savefig(save_path_pdf, dpi=600, bbox_inches="tight")
    plt.show()

def main():
    phase1 = np.load(PHASE1_PATH, allow_pickle=True)
    X_train_orig = phase1["X_train_orig"]
    y_train_orig = phase1["y_train_orig"]
    X_test = phase1["X_test"]
    y_test = phase1["y_test"]
    X_aug1 = phase1["X_aug1"]
    y_aug1 = phase1["y_aug1"]

    phase2 = np.load(PHASE2_PATH, allow_pickle=True)
    X_aug2 = phase2["generated_X"]
    y_aug2 = phase2["generated_y"]

    X_train_combined = np.vstack([X_train_orig, X_aug1, X_aug2])
    y_train_combined = np.concatenate([y_train_orig, y_aug1, y_aug2])

    y_val_orig_all, y_val_pred_orig_all = cross_validated_predictions(
        X_train_orig, y_train_orig,
        params=ORIGINAL_PARAMS,
        n_splits=5,
        random_state=RANDOM_STATE
    )

    model_orig = build_gbdt_model(ORIGINAL_PARAMS)
    model_orig.fit(X_train_orig, y_train_orig)
    y_test_pred_orig = model_orig.predict(X_test)

    y_val_comb_cv, y_val_pred_comb_cv = cross_validated_predictions(
        X_train_combined, y_train_combined,
        params=COMB_CV_PARAMS,
        n_splits=5,
        random_state=RANDOM_STATE
    )

    model_comb_test = build_gbdt_model(COMB_TEST_PARAMS)
    model_comb_test.fit(X_train_combined, y_train_combined)
    y_test_pred_comb_test = model_comb_test.predict(X_test)

    scatter_df = pd.DataFrame({
        "group": (
            ["Validation_Orig"] * len(y_val_orig_all) +
            ["Validation_Orig_P1_P2"] * len(y_val_comb_cv) +
            ["Test_Orig"] * len(y_test) +
            ["Test_Orig_P1_P2"] * len(y_test)
        ),
        "observed": np.concatenate([
            y_val_orig_all,
            y_val_comb_cv,
            y_test,
            y_test
        ]),
        "predicted": np.concatenate([
            y_val_pred_orig_all,
            y_val_pred_comb_cv,
            y_test_pred_orig,
            y_test_pred_comb_test
        ])
    })
    scatter_df.to_csv(SAVE_PATH_SCATTER_CSV, index=False, encoding="utf-8-sig")

    plot_joint_scatter_with_marginals(
        y_val_orig=y_val_orig_all,
        y_val_pred_orig=y_val_pred_orig_all,
        y_val_comb=y_val_comb_cv,
        y_val_pred_comb=y_val_pred_comb_cv,
        y_test=y_test,
        y_test_pred_orig=y_test_pred_orig,
        y_test_pred_comb=y_test_pred_comb_test,
        save_path_png=SAVE_PATH_PNG,
        save_path_pdf=SAVE_PATH_PDF
    )

if __name__ == "__main__":
    main()