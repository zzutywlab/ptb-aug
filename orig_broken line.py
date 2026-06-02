import os
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("请先安装 xgboost：pip install xgboost")

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("请先安装 lightgbm：pip install lightgbm")

try:
    import catboost as cb
except ImportError:
    raise ImportError("请先安装 catboost：pip install catboost")


DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "output_9regions_cv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REGION_FILES = {
    "内蒙古": "inner mongolia.xlsx",
    "宁夏": "ningxia.xlsx",
    "青海": "qinghai.xlsx",
    "四川": "sichuan.xlsx",
    "新疆": "tibet.xlsx",
    "西藏": "xizang.xlsx",
    "云南": "yunnan.xlsx",
    "广西": "guangxi.xlsx",
    "贵州": "guizhou.xlsx",
    "甘肃": "gansu.xlsx",
    "重庆": "chongqing.xlsx",
    "陕西": "shaanxi.xlsx",
}

TARGET = "PTB"
N_SPLITS = 5
RANDOM_STATE = 42
SCALE_MODELS = {"LinearRegression", "SVR", "KNN", "MLP"}

def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=3),
        "RandomForest": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=1,
        ),
        "LightGBM": lgb.LGBMRegressor(
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=1,
        ),
        "CatBoost": cb.CatBoostRegressor(
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=1,
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation='relu',
            alpha=0.01,
            batch_size=16,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=RANDOM_STATE
        ),
    }

def load_and_preprocess(file_path, target):
    df = pd.read_excel(file_path, sheet_name="Sheet1")

    if target not in df.columns:
        raise KeyError(f"{os.path.basename(file_path)} 缺失目标列: {target}")

    features = [col for col in df.columns if col != target]
    if len(features) == 0:
        raise ValueError(f"{os.path.basename(file_path)} 中除目标变量 {target} 外没有可用特征列")

    X = df[features].copy()
    y = df[target].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    valid_cols = [col for col in X.columns if not X[col].isna().all()]
    X = X[valid_cols]

    if X.shape[1] == 0:
        raise ValueError(f"{os.path.basename(file_path)} 所有特征列在数值化后都不可用")

    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(y.median())

    return X.reset_index(drop=True), y.reset_index(drop=True), list(X.columns)

def evaluate_models_5fold(X, y, models):
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    region_result = {}

    for model_name, model in models.items():
        fold_r2_list = []

        for train_idx, val_idx in kfold.split(X):
            X_train_raw = X.iloc[train_idx]
            X_val_raw = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            current_model = clone(model)

            if model_name in SCALE_MODELS:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train_raw)
                X_val = scaler.transform(X_val_raw)
            else:
                X_train = X_train_raw
                X_val = X_val_raw

            current_model.fit(X_train, y_train)
            y_val_pred = current_model.predict(X_val)
            r2 = r2_score(y_val, y_val_pred)
            fold_r2_list.append(r2)

        region_result[model_name] = np.mean(fold_r2_list)

    return region_result

def main():
    summary_rows = []
    feature_info_rows = []

    for region_name, file_name in REGION_FILES.items():
        file_path = os.path.join(DATA_DIR, file_name)
        print(f"\n{'=' * 50}")
        print(f"开始处理地区：{region_name}")
        print(f"文件路径：{file_path}")
        print(f"{'=' * 50}")

        if not os.path.exists(file_path):
            print(f"警告：文件不存在，跳过 {region_name}")
            continue

        X, y, features = load_and_preprocess(file_path, TARGET)
        print(f"数据形状: {X.shape}")
        print(f"自动识别特征数: {len(features)}")
        print(f"特征列: {features}")

        feature_info_rows.append({
            "Region": region_name,
            "Feature_Count": len(features),
            "Features": ", ".join(features)
        })

        if len(X) < N_SPLITS:
            print(f"警告：{region_name} 样本数不足 {N_SPLITS}，无法做5折交叉验证，跳过。")
            continue

        region_scores = evaluate_models_5fold(X, y, get_models())

        row = {"Region": region_name}
        row.update(region_scores)
        summary_rows.append(row)

        print("该地区5折均值 R²：")
        for model_name, mean_r2 in region_scores.items():
            print(f"  {model_name:<18s}: {mean_r2:.4f}")

    if not summary_rows:
        print("没有成功处理任何地区，未生成结果文件。")
        return

    summary_df = pd.DataFrame(summary_rows)
    column_order = [
        "Region",
        "LinearRegression",
        "DecisionTree",
        "SVR",
        "KNN",
        "RandomForest",
        "GradientBoosting",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "MLP",
    ]
    summary_df = summary_df[column_order]

    output_csv = os.path.join(OUTPUT_DIR, "nine_regions_10models_5fold_mean_r2.csv")
    summary_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    feature_info_df = pd.DataFrame(feature_info_rows)
    feature_info_csv = os.path.join(OUTPUT_DIR, "nine_regions_feature_info.csv")
    feature_info_df.to_csv(feature_info_csv, index=False, encoding="utf-8-sig")

    print("\n全部地区处理完成！")
    print(summary_df)
    print(f"\n九个地区、10个模型的5折交叉验证均值已保存到：{output_csv}")
    print(f"各地区自动识别到的特征信息已保存到：{feature_info_csv}")


if __name__ == "__main__":
    main()