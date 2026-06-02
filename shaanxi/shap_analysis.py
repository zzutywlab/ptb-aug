import os
import warnings
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# =========================
# Paths
# =========================
PHASE1_PATH = "output/phase1/phase1_augmented_data.npz"
PHASE2_PATH = "output/phase2/gnn/phase2_cgtgan_epoch_0300.npz"

SAVE_DIR = "output/final_train/shap_phase1_phase2"
os.makedirs(SAVE_DIR, exist_ok=True)

# If your npz file does not contain feature names, fill them here.
MANUAL_FEATURE_NAMES = None

# Explain on the full Orig + P1 + P2 training set
EXPLAIN_ON = "train_combined"   # options: "test" or "train_combined"
MAX_BACKGROUND_SAMPLES = 300
RANDOM_STATE = 42

# Fixed parameters only, no optimization
FIXED_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}


# =========================
# Utilities
# =========================
def flatten_y(y):
    return np.asarray(y).reshape(-1)


def build_gbdt_model(params):
    return GradientBoostingRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        subsample=float(params["subsample"]),
        random_state=int(params["random_state"]),
    )


def evaluate_regression(y_true, y_pred, dataset_name="Dataset"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{dataset_name} performance")
    print("-" * 50)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def sample_rows(X, max_n, random_state=42):
    if (max_n is None) or (len(X) <= max_n):
        return X
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx]


def to_string_list(arr):
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    if not isinstance(arr, (list, tuple)):
        return None

    names = []
    for x in arr:
        if isinstance(x, bytes):
            names.append(x.decode("utf-8"))
        else:
            names.append(str(x))
    return names


def extract_feature_names(npz_obj, n_features, manual_names=None):
    candidate_keys = [
        "feature_names", "features", "selected_features", "input_features",
        "columns", "x_columns", "X_columns"
    ]

    for key in candidate_keys:
        if key in npz_obj.files:
            names = to_string_list(npz_obj[key])
            if names is not None and len(names) == n_features:
                return names

    if manual_names is not None and len(manual_names) == n_features:
        return [str(x) for x in manual_names]

    print("[Warning] Feature names were not found in npz. Using generic names.")
    print("[Warning] If needed, set MANUAL_FEATURE_NAMES manually.")
    return [f"Feature {i+1}" for i in range(n_features)]


# =========================
# Data loading
# =========================
def load_data():
    phase1 = np.load(PHASE1_PATH, allow_pickle=True)
    phase2 = np.load(PHASE2_PATH, allow_pickle=True)

    X_train_orig = np.asarray(phase1["X_train_orig"], dtype=float)
    y_train_orig = flatten_y(phase1["y_train_orig"])
    X_test = np.asarray(phase1["X_test"], dtype=float)
    y_test = flatten_y(phase1["y_test"])
    X_aug1 = np.asarray(phase1["X_aug1"], dtype=float)
    y_aug1 = flatten_y(phase1["y_aug1"])

    X_aug2 = np.asarray(phase2["generated_X"], dtype=float)
    y_aug2 = flatten_y(phase2["generated_y"])

    X_train_combined = np.vstack([X_train_orig, X_aug1, X_aug2])
    y_train_combined = np.concatenate([y_train_orig, y_aug1, y_aug2])

    feature_names = extract_feature_names(
        phase1,
        n_features=X_train_orig.shape[1],
        manual_names=MANUAL_FEATURE_NAMES,
    )

    print("Data loaded successfully.")
    print(f"X_train_orig shape    : {X_train_orig.shape}")
    print(f"y_train_orig shape    : {y_train_orig.shape}")
    print(f"X_aug1 shape          : {X_aug1.shape}")
    print(f"y_aug1 shape          : {y_aug1.shape}")
    print(f"X_aug2 shape          : {X_aug2.shape}")
    print(f"y_aug2 shape          : {y_aug2.shape}")
    print(f"X_train_combined shape: {X_train_combined.shape}")
    print(f"y_train_combined shape: {y_train_combined.shape}")
    print(f"X_test shape          : {X_test.shape}")
    print(f"y_test shape          : {y_test.shape}")
    print(f"Feature names         : {feature_names}")

    return {
        "X_train_orig": X_train_orig,
        "y_train_orig": y_train_orig,
        "X_test": X_test,
        "y_test": y_test,
        "X_aug1": X_aug1,
        "y_aug1": y_aug1,
        "X_aug2": X_aug2,
        "y_aug2": y_aug2,
        "X_train_combined": X_train_combined,
        "y_train_combined": y_train_combined,
        "feature_names": feature_names,
    }


# =========================
# SHAP calculation
# =========================
def compute_shap_values(model, X_background, X_explain):
    explainer = shap.TreeExplainer(
        model,
        data=X_background,
        feature_perturbation="interventional",
    )

    shap_values = explainer.shap_values(X_explain)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    shap_values = np.asarray(shap_values, dtype=float)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.ravel(expected_value)[0])
    else:
        expected_value = float(expected_value)

    return shap_values, expected_value


# =========================
# Main
# =========================
def main():
    data = load_data()

    X_train_combined = data["X_train_combined"]
    y_train_combined = data["y_train_combined"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    model = build_gbdt_model(FIXED_PARAMS)
    model.fit(X_train_combined, y_train_combined)

    if EXPLAIN_ON == "test":
        X_explain = X_test.copy()
        y_explain = y_test.copy()
        explain_tag = "test"
    elif EXPLAIN_ON == "train_combined":
        X_explain = X_train_combined.copy()
        y_explain = y_train_combined.copy()
        explain_tag = "train_combined"
    else:
        raise ValueError("EXPLAIN_ON must be 'test' or 'train_combined'.")

    X_background = sample_rows(X_train_combined, MAX_BACKGROUND_SAMPLES, random_state=RANDOM_STATE)

    y_pred_test = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred_test, dataset_name="Held-out test set")

    shap_values, expected_value = compute_shap_values(
        model=model,
        X_background=X_background,
        X_explain=X_explain,
    )

    X_explain_df = pd.DataFrame(X_explain, columns=feature_names)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP": mean_abs_shap,
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)
    importance_df["Rank"] = np.arange(1, len(importance_df) + 1)

    pred_df = pd.DataFrame({
        "Observed": y_explain,
        "Predicted": model.predict(X_explain),
    })

    detail_df = pd.concat(
        [
            pred_df,
            X_explain_df.add_prefix("FeatureValue_"),
            shap_df.add_prefix("SHAP_"),
        ],
        axis=1,
    )

    bundle_npz = os.path.join(SAVE_DIR, f"gbdt_shap_bundle_{explain_tag}.npz")
    importance_csv = os.path.join(SAVE_DIR, f"gbdt_shap_importance_{explain_tag}.csv")
    detail_csv = os.path.join(SAVE_DIR, f"gbdt_shap_detail_{explain_tag}.csv")
    metrics_csv = os.path.join(SAVE_DIR, "gbdt_final_test_metrics.csv")

    np.savez_compressed(
        bundle_npz,
        shap_values=shap_values,
        X_explain=X_explain,
        y_explain=y_explain,
        y_pred_explain=model.predict(X_explain),
        feature_names=np.array(feature_names, dtype=object),
        mean_abs_shap=mean_abs_shap,
        expected_value=np.array([expected_value], dtype=float),
        explain_tag=np.array([explain_tag], dtype=object),
        fixed_params=np.array([str(FIXED_PARAMS)], dtype=object),
    )

    importance_df.to_csv(importance_csv, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "Dataset": "Test",
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "R2": metrics["R2"],
        "ExpectedValue": expected_value,
        **FIXED_PARAMS,
    }]).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    print("\nSaved files:")
    print(f"1) {bundle_npz}")
    print(f"2) {importance_csv}")
    print(f"3) {detail_csv}")
    print(f"4) {metrics_csv}")
    print("\nNext step:")
    print("Run the plotting script to generate the custom SHAP figures without recomputing SHAP.")


if __name__ == "__main__":
    main()
