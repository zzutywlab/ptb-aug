import os
import random
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# =========================================================
# 全局随机种子
# =========================================================
def set_all_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)


# =========================================================
# 配置类
# =========================================================
class Phase1AugmentationConfig:
    def __init__(self):
        self.seed = 42

        # 数据路径
        self.data_path = "data/chongqing.xlsx"

        # 特征与目标
        self.features = ['PM2.5', 'O3', 'WS', 'NDVI', 'PRE', 'NPBMI', 'TEP']
        self.target = "PTB"

        # 输出路径
        self.output_dir = os.path.join("output", "phase1")
        os.makedirs(self.output_dir, exist_ok=True)

        # 训练测试划分
        self.train_ratio = 0.75

        # 增强倍数
        self.local_cov_factor = 2.0
        self.smogn_factor = 2.0

        # 特征类型
        self.environmental_features = ['PM2.5', 'O3', 'WS', 'PRE']
        self.remote_sensing_features = ["NDVI"]
        self.socioeconomic_features = ['NPBMI', 'TEP']

        # 物理约束
        self.physical_constraints = {
            "NDVI": (0.0, 1.0),
            "PM2.5": (0.0, None),
            "O3": (0.0, None),
            "WS": (0.0, None),
            "PRE": (0.0, None),
            "NPBMI": (0.0, None),
            "TEP": (0.0, None),
        }

        # ---------- 局部协方差条件高斯增强 ----------
        self.local_cov_k_neighbors = 8
        self.local_cov_noise_ratio = 0.08
        self.local_cov_jitter = 1e-6
        self.local_cov_env_ratio = 1.15
        self.local_cov_socio_ratio = 0.60
        self.local_cov_y_perturb_ratio = 0.08
        self.local_cov_alpha_range = (0.10, 0.35)

        # ---------- 改进 SMOGN ----------
        self.smogn_k_neighbors = 6
        self.smogn_perturb_ratio = 0.08
        self.smogn_env_ratio = 1.15
        self.smogn_socio_ratio = 0.60
        self.smogn_y_noise_ratio = 0.05
        self.smogn_min_samples_per_bin = 3
        self.smogn_min_bins = 3
        self.smogn_max_bins = 5
        self.smogn_minority_threshold_ratio = 0.85
        self.smogn_neighbor_bin_span = 1  # 同箱/邻箱
        self.smogn_max_expand_multiplier = 2.5  # 每个稀疏箱最大扩增倍数

        # ---------- 样本质量筛选 ----------
        self.filter_knn_k = 5
        self.filter_distance_quantile = 0.95
        self.filter_y_std_multiplier = 2.5
        self.enable_y_consistency_filter = True


# =========================================================
# 数据预处理
# =========================================================
class DataPreprocessor:
    @staticmethod
    def load_data(config: Phase1AugmentationConfig):
        df = pd.read_excel(config.data_path)

        missing_features = [f for f in config.features if f not in df.columns]
        if missing_features:
            raise ValueError(f"缺失特征: {missing_features}")
        if config.target not in df.columns:
            raise ValueError(f"缺失目标变量: {config.target}")

        X = df[config.features].values.astype(float)
        y = df[config.target].values.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=config.train_ratio,
            random_state=config.seed,
            shuffle=True,
        )

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_scaled = x_scaler.fit_transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)

        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

        feature_indices = {f: i for i, f in enumerate(config.features)}
        env_indices = [feature_indices[f] for f in config.environmental_features if f in feature_indices]
        remote_indices = [feature_indices[f] for f in config.remote_sensing_features if f in feature_indices]
        socio_indices = [feature_indices[f] for f in config.socioeconomic_features if f in feature_indices]

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "y_train_scaled": y_train_scaled,
            "y_test_scaled": y_test_scaled,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "feature_indices": feature_indices,
            "env_indices": env_indices,
            "remote_indices": remote_indices,
            "socio_indices": socio_indices,
        }


# =========================================================
# 物理约束
# =========================================================
class PhysicalConstraintProcessor:
    def __init__(self, config: Phase1AugmentationConfig):
        self.constraints = config.physical_constraints
        self.feature_indices = {f: i for i, f in enumerate(config.features)}

    def apply_constraints(self, X):
        X = X.copy()
        for feature_name, (min_val, max_val) in self.constraints.items():
            if feature_name not in self.feature_indices:
                continue
            idx = self.feature_indices[feature_name]
            if min_val is not None:
                X[:, idx] = np.maximum(X[:, idx], min_val)
            if max_val is not None:
                X[:, idx] = np.minimum(X[:, idx], max_val)
        return X


# =========================================================
# 工具函数
# =========================================================
def safe_local_covariance(X_local, jitter=1e-6):
    cov = np.cov(X_local.T)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    cov = np.atleast_2d(cov)
    cov = cov + np.eye(cov.shape[0]) * jitter
    return cov


def sample_adaptive_n_bins(n_samples, min_bins=3, max_bins=5):
    # 小样本保守自适应
    if n_samples < 80:
        return min_bins
    elif n_samples < 150:
        return min(max_bins, 4)
    else:
        return max_bins


def compute_bin_edges_quantile(y, n_bins):
    try:
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(y, quantiles)
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.histogram_bin_edges(y, bins=n_bins)
    except Exception:
        edges = np.histogram_bin_edges(y, bins=n_bins)
    return edges


def assign_bins(y, bin_edges, n_bins):
    indices = np.digitize(y, bin_edges) - 1
    indices = np.clip(indices, 0, n_bins - 1)
    return indices


# =========================================================
# 局部协方差条件高斯增强
# =========================================================
class LocalCovarianceConditionalGaussianAugmenter:
    def __init__(self, config, env_indices, remote_indices, socio_indices):
        self.config = config
        self.env_indices = env_indices
        self.remote_indices = remote_indices
        self.socio_indices = socio_indices

    def _build_feature_noise_scale(self, X_local):
        feature_std = np.std(X_local, axis=0)
        feature_std = np.where(feature_std < 1e-8, 1e-8, feature_std)

        scale = np.ones(X_local.shape[1]) * self.config.local_cov_noise_ratio

        for idx in self.env_indices:
            scale[idx] *= self.config.local_cov_env_ratio
        for idx in self.remote_indices:
            scale[idx] *= 1.0
        for idx in self.socio_indices:
            scale[idx] *= self.config.local_cov_socio_ratio

        return scale * feature_std

    def augment(self, X, y, num_samples):
        n = len(X)
        k = min(self.config.local_cov_k_neighbors, n)

        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)

        X_augmented = []
        y_augmented = []

        for _ in range(num_samples):
            base_idx = np.random.randint(n)
            x_base = X[base_idx]
            y_base = y[base_idx]

            _, neighbor_indices = knn.kneighbors([x_base], return_distance=True)
            neighbor_indices = neighbor_indices[0]

            X_local = X[neighbor_indices]
            y_local = y[neighbor_indices]

            # 局部协方差
            cov_local = safe_local_covariance(X_local, jitter=self.config.local_cov_jitter)

            try:
                L = np.linalg.cholesky(cov_local)
                z = np.random.normal(0, 1, size=X.shape[1])
                correlated_noise = L @ z
            except np.linalg.LinAlgError:
                local_std = np.std(X_local, axis=0)
                local_std = np.where(local_std < 1e-8, 1e-8, local_std)
                correlated_noise = np.random.normal(0, local_std, size=X.shape[1])

            noise_scale = self._build_feature_noise_scale(X_local)
            x_noise = correlated_noise * noise_scale
            x_new = x_base + x_noise

            # y 做局部条件扰动：邻居插值 + 小噪声
            if len(neighbor_indices) > 1:
                candidate_neighbors = neighbor_indices[neighbor_indices != base_idx]
                if len(candidate_neighbors) == 0:
                    nbr_idx = base_idx
                else:
                    nbr_idx = np.random.choice(candidate_neighbors)
            else:
                nbr_idx = base_idx

            alpha = np.random.uniform(*self.config.local_cov_alpha_range)
            y_interp = y_base + alpha * (y[nbr_idx] - y_base)

            y_local_std = np.std(y_local)
            if y_local_std < 1e-8:
                y_local_std = 1e-8
            y_noise = np.random.normal(0, self.config.local_cov_y_perturb_ratio * y_local_std)
            y_new = y_interp + y_noise

            X_augmented.append(x_new)
            y_augmented.append(y_new)

        return np.array(X_augmented), np.array(y_augmented)


# =========================================================
# 改进版 SMOGN
# =========================================================
class ImprovedSMOGNAugmenter:
    def __init__(self, config, env_indices, remote_indices, socio_indices):
        self.config = config
        self.env_indices = env_indices
        self.remote_indices = remote_indices
        self.socio_indices = socio_indices

    def _build_feature_noise_scale(self, X_local):
        local_std = np.std(X_local, axis=0)
        local_std = np.where(local_std < 1e-8, 1e-8, local_std)

        scale = np.ones(X_local.shape[1]) * self.config.smogn_perturb_ratio
        for idx in self.env_indices:
            scale[idx] *= self.config.smogn_env_ratio
        for idx in self.remote_indices:
            scale[idx] *= 1.0
        for idx in self.socio_indices:
            scale[idx] *= self.config.smogn_socio_ratio

        return scale * local_std

    def augment(self, X, y, num_samples):
        n = len(X)
        n_bins = sample_adaptive_n_bins(
            n_samples=n,
            min_bins=self.config.smogn_min_bins,
            max_bins=self.config.smogn_max_bins,
        )

        bin_edges = compute_bin_edges_quantile(y, n_bins)
        bin_indices = assign_bins(y, bin_edges, n_bins)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)

        valid_bins = [i for i in range(n_bins) if bin_counts[i] >= self.config.smogn_min_samples_per_bin]
        if len(valid_bins) < 2:
            valid_bins = list(range(n_bins))

        valid_counts = np.array([bin_counts[i] for i in valid_bins], dtype=float)
        median_count = np.median(valid_counts)
        minority_bins = [
            i for i in valid_bins
            if bin_counts[i] < median_count * self.config.smogn_minority_threshold_ratio
        ]
        if len(minority_bins) == 0:
            minority_bins = valid_bins.copy()

        # 每个稀疏箱最多允许扩增多少
        max_expand_per_bin = {}
        for b in minority_bins:
            original_count = max(1, bin_counts[b])
            cap = int(max(original_count * self.config.smogn_max_expand_multiplier, median_count))
            max_expand_per_bin[b] = cap

        generated_per_bin = {b: 0 for b in minority_bins}

        X_augmented = []
        y_augmented = []

        # 全局近邻器，仅用于候选检索
        k_global = min(max(self.config.smogn_k_neighbors * 3, self.config.smogn_k_neighbors + 1), n)
        knn_global = NearestNeighbors(n_neighbors=k_global)
        knn_global.fit(X)

        # 为防死循环
        max_trials = num_samples * 20
        trials = 0

        while len(X_augmented) < num_samples and trials < max_trials:
            trials += 1

            available_bins = [
                b for b in minority_bins
                if generated_per_bin[b] < max_expand_per_bin[b]
            ]
            if len(available_bins) == 0:
                break

            chosen_bin = np.random.choice(available_bins)
            base_candidates = np.where(bin_indices == chosen_bin)[0]
            if len(base_candidates) == 0:
                continue

            base_idx = np.random.choice(base_candidates)
            x_base = X[base_idx]
            y_base = y[base_idx]

            # 候选邻居：同箱/邻箱 + y约束
            _, neighbor_candidates = knn_global.kneighbors([x_base], return_distance=True)
            neighbor_candidates = neighbor_candidates[0]

            lower_bin = max(0, chosen_bin - self.config.smogn_neighbor_bin_span)
            upper_bin = min(n_bins - 1, chosen_bin + self.config.smogn_neighbor_bin_span)

            filtered_neighbors = []
            for idx in neighbor_candidates:
                b = bin_indices[idx]
                if lower_bin <= b <= upper_bin and idx != base_idx:
                    filtered_neighbors.append(idx)

            if len(filtered_neighbors) == 0:
                # 退化：直接从同箱或邻箱抽
                span_mask = (bin_indices >= lower_bin) & (bin_indices <= upper_bin)
                filtered_neighbors = [idx for idx in np.where(span_mask)[0] if idx != base_idx]

            if len(filtered_neighbors) == 0:
                continue

            neighbor_idx = np.random.choice(filtered_neighbors)
            x_neighbor = X[neighbor_idx]
            y_neighbor = y[neighbor_idx]

            # 取局部邻域用于局部扰动
            local_pool = np.unique(np.concatenate([[base_idx, neighbor_idx], filtered_neighbors[:self.config.smogn_k_neighbors]]))
            X_local = X[local_pool]
            y_local = y[local_pool]

            # 插值
            alpha = np.random.uniform(0.25, 0.75)
            x_new = x_base + alpha * (x_neighbor - x_base)
            y_new = y_base + alpha * (y_neighbor - y_base)

            # 局部噪声
            noise_scale = self._build_feature_noise_scale(X_local)
            x_noise = np.random.normal(0, noise_scale, size=X.shape[1])
            x_new = x_new + x_noise

            y_local_std = np.std(y_local)
            if y_local_std < 1e-8:
                y_local_std = 1e-8
            y_new = y_new + np.random.normal(0, self.config.smogn_y_noise_ratio * y_local_std)

            X_augmented.append(x_new)
            y_augmented.append(y_new)
            generated_per_bin[chosen_bin] += 1

        # 如果因上限没补满，做保守补齐：仍只在 minority bins 附近轻微插值
        while len(X_augmented) < num_samples:
            chosen_bin = np.random.choice(minority_bins)
            base_candidates = np.where(bin_indices == chosen_bin)[0]
            if len(base_candidates) == 0:
                base_idx = np.random.randint(n)
            else:
                base_idx = np.random.choice(base_candidates)

            x_base = X[base_idx]
            y_base = y[base_idx]

            _, neighbor_candidates = knn_global.kneighbors([x_base], return_distance=True)
            neighbor_candidates = neighbor_candidates[0]
            valid_neighbors = [idx for idx in neighbor_candidates if idx != base_idx]
            if len(valid_neighbors) == 0:
                valid_neighbors = [np.random.randint(n)]

            neighbor_idx = np.random.choice(valid_neighbors)

            X_local = X[valid_neighbors[:self.config.smogn_k_neighbors]]
            y_local = y[valid_neighbors[:self.config.smogn_k_neighbors]]

            alpha = np.random.uniform(0.2, 0.5)
            x_new = x_base + alpha * (X[neighbor_idx] - x_base)
            y_new = y_base + alpha * (y[neighbor_idx] - y_base)

            noise_scale = self._build_feature_noise_scale(X_local)
            x_noise = np.random.normal(0, noise_scale, size=X.shape[1])
            x_new = x_new + x_noise

            y_local_std = np.std(y_local) if len(y_local) > 0 else np.std(y)
            y_local_std = max(y_local_std, 1e-8)
            y_new = y_new + np.random.normal(0, self.config.smogn_y_noise_ratio * y_local_std)

            X_augmented.append(x_new)
            y_augmented.append(y_new)

        return np.array(X_augmented), np.array(y_augmented)


# =========================================================
# 样本质量筛选：保留接近原始数据的样本
# =========================================================
class ConservativeSampleFilter:
    def __init__(self, config: Phase1AugmentationConfig):
        self.config = config

    def filter(self, X_orig, y_orig, X_gen, y_gen):
        if len(X_gen) == 0:
            return X_gen, y_gen, np.array([], dtype=bool), {}

        k = min(self.config.filter_knn_k + 1, len(X_orig))
        knn_orig = NearestNeighbors(n_neighbors=k)
        knn_orig.fit(X_orig)

        # 原始训练集内部最近邻距离分布
        dists_orig, idxs_orig = knn_orig.kneighbors(X_orig, return_distance=True)
        # 第一个是自己，取第二个
        if dists_orig.shape[1] > 1:
            orig_nn_dist = dists_orig[:, 1]
        else:
            orig_nn_dist = dists_orig[:, 0]

        distance_threshold = np.quantile(orig_nn_dist, self.config.filter_distance_quantile)

        # 生成样本到原始训练集的最近邻距离
        dists_gen, idxs_gen = knn_orig.kneighbors(X_gen, return_distance=True)
        gen_nn_dist = dists_gen[:, 0]
        distance_mask = gen_nn_dist <= distance_threshold

        # y 一致性筛选
        if self.config.enable_y_consistency_filter:
            y_mask = np.ones(len(X_gen), dtype=bool)
            for i in range(len(X_gen)):
                neighbor_y = y_orig[idxs_gen[i]]
                y_mean = np.mean(neighbor_y)
                y_std = np.std(neighbor_y)
                if y_std < 1e-8:
                    y_std = 1e-8

                lower = y_mean - self.config.filter_y_std_multiplier * y_std
                upper = y_mean + self.config.filter_y_std_multiplier * y_std

                if not (lower <= y_gen[i] <= upper):
                    y_mask[i] = False
        else:
            y_mask = np.ones(len(X_gen), dtype=bool)

        final_mask = distance_mask & y_mask

        stats = {
            "distance_threshold": float(distance_threshold),
            "n_input": int(len(X_gen)),
            "n_after_distance": int(np.sum(distance_mask)),
            "n_after_y": int(np.sum(y_mask)),
            "n_final": int(np.sum(final_mask)),
            "retention_rate": float(np.mean(final_mask) * 100.0),
        }

        return X_gen[final_mask], y_gen[final_mask], final_mask, stats


# =========================================================
# 主增强器
# =========================================================
class Phase1UpgradedAugmenter:
    def __init__(self, config, data_dict):
        self.config = config
        self.data_dict = data_dict

        self.X_train_scaled = data_dict["X_train_scaled"]
        self.y_train_scaled = data_dict["y_train_scaled"]

        self.local_cov_augmenter = LocalCovarianceConditionalGaussianAugmenter(
            config,
            data_dict["env_indices"],
            data_dict["remote_indices"],
            data_dict["socio_indices"],
        )

        self.smogn_augmenter = ImprovedSMOGNAugmenter(
            config,
            data_dict["env_indices"],
            data_dict["remote_indices"],
            data_dict["socio_indices"],
        )

    def generate(self):
        n_train = len(self.X_train_scaled)

        n_local_cov = int(self.config.local_cov_factor * n_train)
        n_smogn = int(self.config.smogn_factor * n_train)

        print(f"局部协方差条件高斯增强目标样本数: {n_local_cov}")
        print(f"改进版 SMOGN 增强目标样本数: {n_smogn}")

        X_cov, y_cov = self.local_cov_augmenter.augment(
            self.X_train_scaled, self.y_train_scaled, n_local_cov
        )
        X_smogn, y_smogn = self.smogn_augmenter.augment(
            self.X_train_scaled, self.y_train_scaled, n_smogn
        )

        X_aug_scaled = np.vstack([X_cov, X_smogn])
        y_aug_scaled = np.concatenate([y_cov, y_smogn])

        method_labels = np.array(
            ["local_cov"] * len(X_cov) + ["smogn"] * len(X_smogn),
            dtype=object
        )

        return X_aug_scaled, y_aug_scaled, method_labels


# =========================================================
# 保存结果
# =========================================================
def save_phase1_npz(config, data_dict, X_aug, y_aug, method_labels, filter_stats):
    save_path = os.path.join(config.output_dir, "phase1_augmented_data.npz")

    np.savez_compressed(
        save_path,
        X_train_orig=data_dict["X_train"],
        y_train_orig=data_dict["y_train"],
        X_test=data_dict["X_test"],
        y_test=data_dict["y_test"],
        X_aug1=X_aug,
        y_aug1=y_aug,
        aug_method_labels=method_labels,
        features=np.array(config.features, dtype=object),
        target=config.target,
        filter_stats=np.array([filter_stats], dtype=object),
    )

    return save_path


# =========================================================
# 主函数
# =========================================================
def main():
    set_all_seeds(42)
    config = Phase1AugmentationConfig()

    print("=== Phase 1 升级版数据增强开始 ===")
    print(f"数据路径: {config.data_path}")
    print(f"特征: {config.features}")
    print(f"目标变量: {config.target}")
    print(f"输出目录: {config.output_dir}")
    print("增强策略: 局部协方差条件高斯增强(2倍) + 改进版SMOGN(2倍) + 保守筛选")

    # 1. 读取数据
    data_dict = DataPreprocessor.load_data(config)

    # 2. 生成增强样本（标准化空间）
    augmenter = Phase1UpgradedAugmenter(config, data_dict)
    X_aug_scaled, y_aug_scaled, method_labels = augmenter.generate()

    print(f"筛选前增强样本数: {len(X_aug_scaled)}")

    # 3. 逆标准化
    X_aug = data_dict["x_scaler"].inverse_transform(X_aug_scaled)
    y_aug = data_dict["y_scaler"].inverse_transform(y_aug_scaled.reshape(-1, 1)).ravel()

    # 4. 应用物理约束
    constraint_processor = PhysicalConstraintProcessor(config)
    X_aug = constraint_processor.apply_constraints(X_aug)

    # 5. 保守型样本筛选
    print("开始执行保守型样本筛选...")
    sample_filter = ConservativeSampleFilter(config)
    X_aug_filtered, y_aug_filtered, mask, filter_stats = sample_filter.filter(
        X_orig=data_dict["X_train"],
        y_orig=data_dict["y_train"],
        X_gen=X_aug,
        y_gen=y_aug,
    )
    method_labels_filtered = method_labels[mask]

    print(f"筛选后增强样本数: {len(X_aug_filtered)}")
    print(f"保留率: {filter_stats['retention_rate']:.2f}%")
    print(f"最近邻距离阈值: {filter_stats['distance_threshold']:.6f}")

    # 6. 保存
    save_path = save_phase1_npz(
        config=config,
        data_dict=data_dict,
        X_aug=X_aug_filtered,
        y_aug=y_aug_filtered,
        method_labels=method_labels_filtered,
        filter_stats=filter_stats,
    )

    print("=== Phase 1 升级版数据增强完成 ===")
    print(f"原始训练集样本数: {len(data_dict['X_train'])}")
    print(f"筛选前增强样本数: {len(X_aug)}")
    print(f"筛选后增强样本数: {len(X_aug_filtered)}")
    print(f"测试集样本数: {len(data_dict['X_test'])}")
    print(f"保存文件: {save_path}")


if __name__ == "__main__":
    main()