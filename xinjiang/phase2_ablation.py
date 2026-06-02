import os
import math
import json
import csv
import random
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# 1. 基础设置
# =========================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PHASE1_NPZ_PATH = "output/phase1/phase1_augmented_data.npz"
SAVE_DIR = "output/phase2/gnn_ablation"

EPOCHS = 800
BATCH_SIZE = 32
RAND_DIM = 32
EMBED_DIM = 32
MLP_HIDDEN_DIM = 128

LR_G = 1e-4
LR_D = 1e-4
LR_A = 1e-4
LR_P = 1e-4
LR_R = 1e-4

N_CRITIC = 5
LAMBDA_GP = 10.0
NODE_LOSS_WEIGHT = 1.0

# PTB 条件分箱数
N_COND_BINS = 5

# 是否保留中间 checkpoint（默认只保存每个消融实验的最终结果）
SAVE_INTERMEDIATE = True
SAVE_START = 300
SAVE_EVERY = 100

os.makedirs(SAVE_DIR, exist_ok=True)


EXPERIMENTS = [
    {
        "name": "wo_learnable_adjacency",
        "label": "w/o learnable adjacency",
        "use_graph_module": True,
        "use_conditioning": True,
        "use_node_level_loss": True,
        "use_learnable_adjacency": False,
    },
    {
        "name": "wo_conditional_node",
        "label": "w/o conditional node",
        "use_graph_module": True,
        "use_conditioning": False,
        "use_node_level_loss": True,
        "use_learnable_adjacency": True,
    },
    {
        "name": "wo_node_level_loss",
        "label": "w/o node-level loss",
        "use_graph_module": True,
        "use_conditioning": True,
        "use_node_level_loss": False,
        "use_learnable_adjacency": True,
    },
    {
        "name": "wo_graph_module",
        "label": "w/o graph module",
        "use_graph_module": False,
        "use_conditioning": True,
        "use_node_level_loss": True,
        "use_learnable_adjacency": False,
    },
]


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(SEED)


# =========================
# 2. 数据读取
# =========================

def _extract_scalar(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        flat = x.reshape(-1)
        if flat.size == 1:
            return flat[0].item() if hasattr(flat[0], "item") else flat[0]
        return x.tolist()
    return x


data = np.load(PHASE1_NPZ_PATH, allow_pickle=True)

required_keys = [
    "X_train_orig", "y_train_orig",
    "X_test", "y_test",
    "X_aug1", "y_aug1",
    "features", "target"
]
for k in required_keys:
    if k not in data:
        raise KeyError(f"phase1 npz 中缺少键: {k}")

X_train_orig = data["X_train_orig"].astype(np.float32)
y_train_orig = data["y_train_orig"].reshape(-1).astype(np.float32)

X_test = data["X_test"].astype(np.float32)
y_test = data["y_test"].reshape(-1).astype(np.float32)

X_aug1 = data["X_aug1"].astype(np.float32)
y_aug1 = data["y_aug1"].reshape(-1).astype(np.float32)

FEATURE_NAMES = [str(x) for x in data["features"].tolist()]
TARGET_NAME = str(_extract_scalar(data["target"]))

if TARGET_NAME != "PTB":
    raise ValueError(f"phase1 文件中的目标变量不是 PTB，而是: {TARGET_NAME}")

# phase2 使用：原始训练集 + phase1 全部增强数据
combined_X = np.vstack([X_train_orig, X_aug1]).astype(np.float32)
combined_y = np.concatenate([y_train_orig, y_aug1]).astype(np.float32)
combined_xy = np.column_stack([combined_X, combined_y]).astype(np.float32)

print(f"Original samples      : {len(X_train_orig)}")
print(f"Phase1 total samples  : {len(X_aug1)}")
print(f"Phase1 used in phase2 : {len(X_aug1)} (all)")
print(f"Combined training shape: {combined_xy.shape}")
print(f"Feature count: {len(FEATURE_NAMES)} | Target: {TARGET_NAME}")

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_xy = scaler.fit_transform(combined_xy).astype(np.float32)

# 目标列索引
TARGET_IDX = scaled_xy.shape[1] - 1
NUM_REAL_NODES = scaled_xy.shape[1]


# =========================
# 3. 条件向量：将 PTB 分箱
# =========================

def build_target_bins(y_scaled: np.ndarray, n_bins: int = 5):
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y_scaled, quantiles)
    edges = np.unique(edges)

    if len(edges) <= 2:
        edges = np.linspace(float(y_scaled.min()), float(y_scaled.max()) + 1e-8, 3)

    bin_ids = np.digitize(y_scaled, edges[1:-1], right=False)
    n_effective_bins = int(bin_ids.max()) + 1
    return edges, bin_ids.astype(np.int64), n_effective_bins


BIN_EDGES, COND_BIN_IDS, COND_DIM = build_target_bins(scaled_xy[:, TARGET_IDX], n_bins=N_COND_BINS)


def sample_conditional_indices(bin_ids: np.ndarray, batch_size: int):
    unique_bins, counts = np.unique(bin_ids, return_counts=True)
    probs = counts / counts.sum()

    chosen_bins = np.random.choice(unique_bins, size=batch_size, p=probs)
    sample_indices = []

    for b in chosen_bins:
        idx_pool = np.where(bin_ids == b)[0]
        idx = np.random.choice(idx_pool)
        sample_indices.append(idx)

    chosen_bins = np.array(chosen_bins, dtype=np.int64)
    sample_indices = np.array(sample_indices, dtype=np.int64)

    cond_vec = np.zeros((batch_size, COND_DIM), dtype=np.float32)
    cond_vec[np.arange(batch_size), chosen_bins] = 1.0
    return cond_vec, chosen_bins, sample_indices


# =========================
# 4. 图结构模块
# =========================
class LearnableAdjacency(nn.Module):
    def __init__(self, num_nodes: int):
        super().__init__()
        init = (np.ones((num_nodes, num_nodes), dtype=np.float32) - np.eye(num_nodes, dtype=np.float32))
        init = init / np.sqrt(num_nodes)
        self.weight = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def get_AP(self):
        A = 0.5 * (self.weight + self.weight.T)
        eye = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        A = A + 1e-3 * eye
        return A

    def get_AG(self):
        return self.get_AP()

    def get_AD(self):
        A = self.get_AP()
        eye = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        return torch.linalg.inv(A + 1e-4 * eye)


class StaticPearsonAdjacency(nn.Module):
    """
    基于 Pearson 相关系数预先构建静态图。
    为了保证图卷积中的边权非负且数值稳定，这里使用 |r| 作为边权强度。
    若包含条件节点，则通过“每个变量与 PTB 分箱 one-hot 各维度之间的 Pearson 相关”的平均绝对值
    来连接条件节点与真实节点。
    """
    def __init__(self, real_data_scaled: np.ndarray, cond_bin_ids: np.ndarray, cond_dim: int, use_conditioning: bool):
        super().__init__()
        A = self.build_static_adjacency(real_data_scaled, cond_bin_ids, cond_dim, use_conditioning)
        self.register_buffer("weight", torch.tensor(A, dtype=torch.float32))

    @staticmethod
    def _safe_abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        r = np.corrcoef(x, y)[0, 1]
        if np.isnan(r):
            return 0.0
        return float(abs(r))

    @classmethod
    def build_static_adjacency(cls, real_data_scaled: np.ndarray, cond_bin_ids: np.ndarray,
                               cond_dim: int, use_conditioning: bool) -> np.ndarray:
        corr = np.corrcoef(real_data_scaled, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.abs(corr).astype(np.float32)
        np.fill_diagonal(corr, 0.0)

        if not use_conditioning:
            A = corr
        else:
            cond_onehot = np.eye(cond_dim, dtype=np.float32)[cond_bin_ids]
            cond_strength = []
            for i in range(real_data_scaled.shape[1]):
                vals = [cls._safe_abs_pearson(real_data_scaled[:, i], cond_onehot[:, j]) for j in range(cond_dim)]
                cond_strength.append(float(np.mean(vals)))
            cond_strength = np.array(cond_strength, dtype=np.float32)

            A = np.zeros((real_data_scaled.shape[1] + 1, real_data_scaled.shape[1] + 1), dtype=np.float32)
            A[:-1, :-1] = corr
            A[-1, :-1] = cond_strength
            A[:-1, -1] = cond_strength
            A[-1, -1] = 0.0

        max_val = float(A.max()) if A.size > 0 else 0.0
        if max_val > 0:
            A = A / max_val
        return A.astype(np.float32)

    def get_AP(self):
        A = 0.5 * (self.weight + self.weight.T)
        eye = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        return A + 1e-3 * eye

    def get_AG(self):
        return self.get_AP()

    def get_AD(self):
        A = self.get_AP()
        eye = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        return torch.linalg.inv(A + 1e-3 * eye)


# =========================
# 5. 模型核心：图版
# =========================
class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        out = torch.matmul(adj, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    def __init__(self, embed_dim: int, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.gc1 = GraphConvolution(embed_dim, embed_dim * 2)
        self.gc2 = GraphConvolution(embed_dim * 2, embed_dim * 2)
        self.gc3 = GraphConvolution(embed_dim * 2, embed_dim)
        self.prelu = nn.PReLU()

    def forward(self, x, adj):
        x = x.reshape(x.shape[0], self.num_nodes, -1)
        x_init = x.clone()
        x_init_2 = x_init.repeat(1, 1, 2)

        x = self.prelu(self.gc1(x, adj) + x_init_2)
        x = self.prelu(self.gc2(x, adj) + x_init_2)
        x = self.prelu(self.gc3(x, adj) + x_init)
        return x


class ScalarProjection(nn.Module):
    def __init__(self, num_real_nodes: int, cond_dim: int, embed_dim: int, use_conditioning: bool):
        super().__init__()
        self.num_real_nodes = num_real_nodes
        self.embed_dim = embed_dim
        self.use_conditioning = use_conditioning

        self.node_proj = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_real_nodes)])
        self.cond_proj = nn.Linear(cond_dim, embed_dim) if use_conditioning else None

    def forward_real(self, x, cond_vec=None):
        nodes = []
        for i in range(self.num_real_nodes):
            nodes.append(self.node_proj[i](x[:, i:i + 1]).unsqueeze(1))
        nodes = torch.cat(nodes, dim=1)

        if self.use_conditioning:
            if cond_vec is None:
                raise ValueError("use_conditioning=True 时，cond_vec 不能为 None")
            cond_node = self.cond_proj(cond_vec).unsqueeze(1)
            graph = torch.cat([nodes, cond_node], dim=1)
            return graph
        return nodes

    def cond_node_only(self, cond_vec=None):
        if not self.use_conditioning:
            return None
        if cond_vec is None:
            raise ValueError("use_conditioning=True 时，cond_vec 不能为 None")
        return self.cond_proj(cond_vec).unsqueeze(1)


class NodewiseGenerator(nn.Module):
    def __init__(self, rand_dim: int, cond_dim: int, num_graph_nodes: int,
                 embed_dim: int, num_real_nodes: int, use_conditioning: bool):
        super().__init__()
        self.num_graph_nodes = num_graph_nodes
        self.num_real_nodes = num_real_nodes
        self.embed_dim = embed_dim
        self.use_conditioning = use_conditioning

        input_dim = rand_dim + (cond_dim if use_conditioning else 0)
        latent_nodes = num_graph_nodes - (1 if use_conditioning else 0)
        self.rand_to_proj = nn.Linear(input_dim, latent_nodes * embed_dim)
        self.gcn = GCN(embed_dim, num_graph_nodes)
        self.proj_to_final = nn.Linear(num_graph_nodes * embed_dim, num_real_nodes)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, z_input, cond_tensor_node, adj):
        batch_size = z_input.shape[0]
        output_1 = self.leakyrelu(self.rand_to_proj(z_input))

        latent_nodes = self.num_graph_nodes - (1 if self.use_conditioning else 0)
        output_1 = output_1.reshape(batch_size, latent_nodes, self.embed_dim)

        if self.use_conditioning:
            output_w_cond = torch.cat([output_1, cond_tensor_node], dim=1)
        else:
            output_w_cond = output_1

        gnn_output = self.gcn(output_w_cond, adj)
        gnn_output = gnn_output.reshape(batch_size, -1)

        out = self.proj_to_final(gnn_output)
        out = torch.sigmoid(out)
        return out


class Critic(nn.Module):
    def __init__(self, embed_dim: int, num_graph_nodes: int):
        super().__init__()
        self.layers = GCN(embed_dim, num_graph_nodes)
        self.linear = nn.Linear(num_graph_nodes * embed_dim, 1)

    def forward(self, x_graph, adj):
        batch_size = x_graph.shape[0]
        x_repr = self.layers(x_graph, adj)
        x_repr = x_repr.reshape(batch_size, -1)
        return self.linear(x_repr)


class TargetRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x)


# =========================
# 6. 模型核心：MLP 版（w/o graph module）
# =========================
class MLPGenerator(nn.Module):
    def __init__(self, rand_dim: int, cond_dim: int, output_dim: int, use_conditioning: bool):
        super().__init__()
        input_dim = rand_dim + (cond_dim if use_conditioning else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z_input):
        return self.net(z_input)


class MLPCritic(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int, use_conditioning: bool):
        super().__init__()
        total_in = input_dim + (cond_dim if use_conditioning else 0)
        self.use_conditioning = use_conditioning
        self.feature = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x, cond_vec=None, return_hidden: bool = False):
        if self.use_conditioning:
            if cond_vec is None:
                raise ValueError("MLPCritic 需要 cond_vec，但收到 None")
            inp = torch.cat([x, cond_vec], dim=1)
        else:
            inp = x
        h = self.feature(inp)
        s = self.score(h)
        if return_hidden:
            return s, h
        return s


# =========================
# 7. 训练辅助函数
# =========================

def set_requires_grad(module: nn.Module, requires_grad: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(requires_grad)


# 图版 gradient penalty

def gradient_penalty_graph(critic, real_graph, fake_graph, adj):
    batch_size = real_graph.shape[0]
    epsilon = torch.rand((batch_size, 1, 1), device=real_graph.device).repeat(
        1, real_graph.shape[1], real_graph.shape[2]
    )
    interpolated = real_graph * epsilon + fake_graph * (1 - epsilon)
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated, adj)
    grad = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad = grad.view(grad.shape[0], -1)
    grad_norm = grad.norm(2, dim=1)
    gp = torch.mean((grad_norm - 1.0) ** 2)
    return gp


# MLP 版 gradient penalty

def gradient_penalty_mlp(critic, real_x, fake_x, cond_vec=None):
    batch_size = real_x.shape[0]
    epsilon = torch.rand((batch_size, 1), device=real_x.device)
    interpolated = real_x * epsilon + fake_x * (1 - epsilon)
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated, cond_vec)
    grad = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad = grad.view(grad.shape[0], -1)
    grad_norm = grad.norm(2, dim=1)
    gp = torch.mean((grad_norm - 1.0) ** 2)
    return gp



def sample_real_batch(train_tensor: torch.Tensor, batch_size: int, use_conditioning: bool):
    if use_conditioning:
        cond_vec_np, chosen_bins, sample_idx = sample_conditional_indices(COND_BIN_IDS, batch_size)
        real_x = train_tensor[sample_idx]
        cond_vec = torch.tensor(cond_vec_np, dtype=torch.float32, device=DEVICE)
        return real_x, cond_vec, chosen_bins, sample_idx

    sample_idx = np.random.choice(len(train_tensor), size=batch_size, replace=len(train_tensor) < batch_size)
    real_x = train_tensor[sample_idx]
    return real_x, None, None, sample_idx


@torch.no_grad()
def generate_samples_graph(generator, projection, adj_module, n_samples: int, use_conditioning: bool):
    generator.eval()
    projection.eval()

    outputs = []
    remain = n_samples

    while remain > 0:
        bs = min(256, remain)
        if use_conditioning:
            cond_vec_np, _, _ = sample_conditional_indices(COND_BIN_IDS, bs)
            cond_vec = torch.tensor(cond_vec_np, dtype=torch.float32, device=DEVICE)
            cond_node = projection.cond_node_only(cond_vec)
            z = torch.randn(bs, RAND_DIM, device=DEVICE)
            z_input = torch.cat([z, cond_vec], dim=1)
        else:
            cond_vec = None
            cond_node = None
            z_input = torch.randn(bs, RAND_DIM, device=DEVICE)

        fake_scaled = generator(z_input, cond_node, adj_module.get_AG())
        outputs.append(fake_scaled.cpu().numpy())
        remain -= bs

    fake_scaled = np.vstack(outputs)
    fake_original = scaler.inverse_transform(fake_scaled)
    return fake_scaled, fake_original


@torch.no_grad()
def generate_samples_mlp(generator, n_samples: int, use_conditioning: bool):
    generator.eval()

    outputs = []
    remain = n_samples
    while remain > 0:
        bs = min(256, remain)
        if use_conditioning:
            cond_vec_np, _, _ = sample_conditional_indices(COND_BIN_IDS, bs)
            cond_vec = torch.tensor(cond_vec_np, dtype=torch.float32, device=DEVICE)
            z = torch.randn(bs, RAND_DIM, device=DEVICE)
            z_input = torch.cat([z, cond_vec], dim=1)
        else:
            z_input = torch.randn(bs, RAND_DIM, device=DEVICE)

        fake_scaled = generator(z_input)
        outputs.append(fake_scaled.cpu().numpy())
        remain -= bs

    fake_scaled = np.vstack(outputs)
    fake_original = scaler.inverse_transform(fake_scaled)
    return fake_scaled, fake_original


# =========================
# 8. 保存函数
# =========================

def save_checkpoint(experiment_dir: str, config: dict, epoch: int, history: dict,
                    generator, projection=None, adj_module=None):
    if config["use_graph_module"]:
        fake_scaled, fake_original = generate_samples_graph(
            generator=generator,
            projection=projection,
            adj_module=adj_module,
            n_samples=len(train_tensor),
            use_conditioning=config["use_conditioning"],
        )
    else:
        fake_scaled, fake_original = generate_samples_mlp(
            generator=generator,
            n_samples=len(train_tensor),
            use_conditioning=config["use_conditioning"],
        )

    save_path = os.path.join(experiment_dir, f"{config['name']}_epoch_{epoch:04d}.npz")
    np.savez_compressed(
        save_path,
        generated_X=fake_original[:, :-1],
        generated_y=fake_original[:, -1],
        generated_XY=fake_original,
        generated_X_scaled=fake_scaled[:, :-1],
        generated_y_scaled=fake_scaled[:, -1],
        generated_XY_scaled=fake_scaled,

        X_train_orig=X_train_orig,
        y_train_orig=y_train_orig,
        X_aug1=X_aug1,
        y_aug1=y_aug1,
        X_test=X_test,
        y_test=y_test,

        combined_X=combined_X,
        combined_y=combined_y,
        combined_XY=combined_xy,

        feature_names=np.array(FEATURE_NAMES, dtype=object),
        target_name=np.array(TARGET_NAME, dtype=object),

        cond_bin_edges=BIN_EDGES,
        cond_dim=COND_DIM,

        scaler_min_=scaler.min_,
        scaler_scale_=scaler.scale_,

        history=history,
        config_json=np.array(json.dumps(config, ensure_ascii=False), dtype=object),
        epoch=epoch,
    )
    print(f"已保存: {save_path}")
    return save_path


# =========================
# 9. 初始化数据 Tensor
# =========================
train_tensor = torch.tensor(scaled_xy, dtype=torch.float32, device=DEVICE)


# =========================
# 10. 单个实验训练入口
# =========================

def build_modules(config: dict):
    use_graph_module = config["use_graph_module"]
    use_conditioning = config["use_conditioning"]
    use_learnable_adjacency = config["use_learnable_adjacency"]

    if use_graph_module:
        num_graph_nodes = NUM_REAL_NODES + (1 if use_conditioning else 0)

        if use_learnable_adjacency:
            adj_module = LearnableAdjacency(num_graph_nodes).to(DEVICE)
        else:
            adj_module = StaticPearsonAdjacency(
                real_data_scaled=scaled_xy,
                cond_bin_ids=COND_BIN_IDS,
                cond_dim=COND_DIM,
                use_conditioning=use_conditioning,
            ).to(DEVICE)

        projection = ScalarProjection(
            num_real_nodes=NUM_REAL_NODES,
            cond_dim=COND_DIM,
            embed_dim=EMBED_DIM,
            use_conditioning=use_conditioning,
        ).to(DEVICE)
        generator = NodewiseGenerator(
            rand_dim=RAND_DIM,
            cond_dim=COND_DIM,
            num_graph_nodes=num_graph_nodes,
            embed_dim=EMBED_DIM,
            num_real_nodes=NUM_REAL_NODES,
            use_conditioning=use_conditioning,
        ).to(DEVICE)
        critic = Critic(embed_dim=EMBED_DIM, num_graph_nodes=num_graph_nodes).to(DEVICE)
        regressor = TargetRegressor(input_dim=EMBED_DIM).to(DEVICE)

        opt_g = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.9))
        opt_d = torch.optim.Adam(critic.parameters(), lr=LR_D, betas=(0.5, 0.9))
        opt_p = torch.optim.Adam(projection.parameters(), lr=LR_P, betas=(0.5, 0.9))
        opt_r = torch.optim.Adam(regressor.parameters(), lr=LR_R, betas=(0.5, 0.9))
        opt_a = (
            torch.optim.Adam(adj_module.parameters(), lr=LR_A, betas=(0.5, 0.9))
            if use_learnable_adjacency else None
        )

        return {
            "adj_module": adj_module,
            "projection": projection,
            "generator": generator,
            "critic": critic,
            "regressor": regressor,
            "opt_g": opt_g,
            "opt_d": opt_d,
            "opt_p": opt_p,
            "opt_r": opt_r,
            "opt_a": opt_a,
        }

    generator = MLPGenerator(
        rand_dim=RAND_DIM,
        cond_dim=COND_DIM,
        output_dim=NUM_REAL_NODES,
        use_conditioning=use_conditioning,
    ).to(DEVICE)
    critic = MLPCritic(
        input_dim=NUM_REAL_NODES,
        cond_dim=COND_DIM,
        hidden_dim=MLP_HIDDEN_DIM,
        use_conditioning=use_conditioning,
    ).to(DEVICE)
    regressor = TargetRegressor(input_dim=MLP_HIDDEN_DIM).to(DEVICE)

    opt_g = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=LR_D, betas=(0.5, 0.9))
    opt_r = torch.optim.Adam(regressor.parameters(), lr=LR_R, betas=(0.5, 0.9))

    return {
        "adj_module": None,
        "projection": None,
        "generator": generator,
        "critic": critic,
        "regressor": regressor,
        "opt_g": opt_g,
        "opt_d": opt_d,
        "opt_p": None,
        "opt_r": opt_r,
        "opt_a": None,
    }



def maybe_zero(optimizer):
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)



def maybe_step(optimizer):
    if optimizer is not None:
        optimizer.step()



def train_one_experiment(config: dict):
    print("\n" + "=" * 100)
    print(f"开始实验: {config['label']}")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print("=" * 100)

    set_all_seeds(SEED)
    modules = build_modules(config)

    adj_module = modules["adj_module"]
    projection = modules["projection"]
    generator = modules["generator"]
    critic = modules["critic"]
    regressor = modules["regressor"]

    opt_g = modules["opt_g"]
    opt_d = modules["opt_d"]
    opt_p = modules["opt_p"]
    opt_r = modules["opt_r"]
    opt_a = modules["opt_a"]

    mse_loss = nn.MSELoss()

    history = {
        "d_loss_graph": [],
        "d_loss_node": [],
        "g_loss_graph": [],
        "g_loss_node": [],
        "w_dist": []
    }

    experiment_dir = os.path.join(SAVE_DIR, config["name"])
    os.makedirs(experiment_dir, exist_ok=True)

    final_save_path = None

    for epoch in range(1, EPOCHS + 1):
        d_graph_losses = []
        d_node_losses = []
        g_graph_losses = []
        g_node_losses = []

        steps_per_epoch = max(1, len(train_tensor) // BATCH_SIZE)

        for _ in range(steps_per_epoch):
            # -------------------------------------------------
            # A. 判别器：图级 / 对抗任务
            # -------------------------------------------------
            set_requires_grad(critic, True)
            set_requires_grad(regressor, True)
            if config["use_graph_module"]:
                set_requires_grad(projection, True)
                if opt_a is not None:
                    set_requires_grad(adj_module, True)

            generator.eval()
            critic.train()
            regressor.train()

            for _ in range(N_CRITIC):
                maybe_zero(opt_d)
                maybe_zero(opt_p)
                maybe_zero(opt_a)

                real_x, cond_vec, _, _ = sample_real_batch(train_tensor, BATCH_SIZE, config["use_conditioning"])

                if config["use_graph_module"]:
                    real_graph = projection.forward_real(real_x, cond_vec)

                    if config["use_conditioning"]:
                        cond_node = projection.cond_node_only(cond_vec)
                        z = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)
                        z_input = torch.cat([z, cond_vec], dim=1)
                    else:
                        cond_node = None
                        z_input = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)

                    fake_x = generator(z_input, cond_node, adj_module.get_AG()).detach()
                    fake_graph = projection.forward_real(fake_x, cond_vec)

                    critic_real = critic(real_graph, adj_module.get_AD())
                    critic_fake = critic(fake_graph, adj_module.get_AD())
                    gp = gradient_penalty_graph(critic, real_graph, fake_graph, adj_module.get_AD())
                else:
                    if config["use_conditioning"]:
                        z = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)
                        z_input = torch.cat([z, cond_vec], dim=1)
                    else:
                        z_input = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)

                    fake_x = generator(z_input).detach()
                    critic_real = critic(real_x, cond_vec)
                    critic_fake = critic(fake_x, cond_vec)
                    gp = gradient_penalty_mlp(critic, real_x, fake_x, cond_vec)

                d_loss_graph = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                d_loss_graph.backward()

                maybe_step(opt_d)
                maybe_step(opt_p)
                maybe_step(opt_a)

                d_graph_losses.append(float(d_loss_graph.item()))

            # -------------------------------------------------
            # B. 判别器：节点级任务（可选）
            # -------------------------------------------------
            if config["use_node_level_loss"]:
                maybe_zero(opt_d)
                maybe_zero(opt_p)
                maybe_zero(opt_r)
                maybe_zero(opt_a)

                real_x, cond_vec, _, _ = sample_real_batch(train_tensor, BATCH_SIZE, config["use_conditioning"])

                if config["use_graph_module"]:
                    real_graph = projection.forward_real(real_x, cond_vec)
                    real_repr = critic.layers(real_graph, adj_module.get_AD())
                    y_pred = regressor(real_repr[:, TARGET_IDX])
                else:
                    _, real_hidden = critic(real_x, cond_vec, return_hidden=True)
                    y_pred = regressor(real_hidden)

                y_true = real_x[:, TARGET_IDX:TARGET_IDX + 1]
                d_loss_node = mse_loss(y_pred, y_true)
                d_loss_node.backward()

                maybe_step(opt_d)
                maybe_step(opt_p)
                maybe_step(opt_r)
                maybe_step(opt_a)

                d_node_losses.append(float(d_loss_node.item()))
            else:
                d_node_losses.append(0.0)

            # -------------------------------------------------
            # C. 生成器：图级 / 对抗任务
            # -------------------------------------------------
            set_requires_grad(critic, False)
            set_requires_grad(regressor, False)
            if config["use_graph_module"] and opt_a is not None:
                set_requires_grad(adj_module, True)

            generator.train()
            critic.eval()

            maybe_zero(opt_g)
            maybe_zero(opt_p)
            maybe_zero(opt_a)

            real_x, cond_vec, _, _ = sample_real_batch(train_tensor, BATCH_SIZE, config["use_conditioning"])

            if config["use_graph_module"]:
                if config["use_conditioning"]:
                    cond_node = projection.cond_node_only(cond_vec)
                    z = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)
                    z_input = torch.cat([z, cond_vec], dim=1)
                else:
                    cond_node = None
                    z_input = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)

                fake_x = generator(z_input, cond_node, adj_module.get_AG())
                fake_graph = projection.forward_real(fake_x, cond_vec)
                g_loss_graph = -torch.mean(critic(fake_graph, adj_module.get_AD()))
            else:
                if config["use_conditioning"]:
                    z = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)
                    z_input = torch.cat([z, cond_vec], dim=1)
                else:
                    z_input = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)

                fake_x = generator(z_input)
                g_loss_graph = -torch.mean(critic(fake_x, cond_vec))

            g_loss_graph.backward()
            maybe_step(opt_g)
            maybe_step(opt_p)
            maybe_step(opt_a)
            g_graph_losses.append(float(g_loss_graph.item()))

            # -------------------------------------------------
            # D. 生成器：节点级任务（可选）
            # -------------------------------------------------
            if config["use_node_level_loss"]:
                maybe_zero(opt_g)
                maybe_zero(opt_p)
                maybe_zero(opt_a)

                real_x, cond_vec, _, _ = sample_real_batch(train_tensor, BATCH_SIZE, config["use_conditioning"])

                if config["use_graph_module"]:
                    if config["use_conditioning"]:
                        cond_node = projection.cond_node_only(cond_vec)
                        z = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)
                        z_input = torch.cat([z, cond_vec], dim=1)
                    else:
                        cond_node = None
                        z_input = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)

                    fake_x = generator(z_input, cond_node, adj_module.get_AG())
                    fake_graph = projection.forward_real(fake_x, cond_vec)
                    fake_repr = critic.layers(fake_graph, adj_module.get_AD())
                    fake_target_pred = regressor(fake_repr[:, TARGET_IDX])
                else:
                    if config["use_conditioning"]:
                        z = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)
                        z_input = torch.cat([z, cond_vec], dim=1)
                    else:
                        z_input = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)

                    fake_x = generator(z_input)
                    _, fake_hidden = critic(fake_x, cond_vec, return_hidden=True)
                    fake_target_pred = regressor(fake_hidden)

                fake_target = fake_x[:, TARGET_IDX:TARGET_IDX + 1]
                g_loss_node = NODE_LOSS_WEIGHT * mse_loss(fake_target_pred, fake_target)
                g_loss_node.backward()

                maybe_step(opt_g)
                maybe_step(opt_p)
                maybe_step(opt_a)
                g_node_losses.append(float(g_loss_node.item()))
            else:
                g_node_losses.append(0.0)

            # 恢复 critic / regressor 参数可训练状态
            set_requires_grad(critic, True)
            set_requires_grad(regressor, True)

        avg_dg = float(np.mean(d_graph_losses)) if d_graph_losses else 0.0
        avg_dn = float(np.mean(d_node_losses)) if d_node_losses else 0.0
        avg_gg = float(np.mean(g_graph_losses)) if g_graph_losses else 0.0
        avg_gn = float(np.mean(g_node_losses)) if g_node_losses else 0.0

        history["d_loss_graph"].append(avg_dg)
        history["d_loss_node"].append(avg_dn)
        history["g_loss_graph"].append(avg_gg)
        history["g_loss_node"].append(avg_gn)
        history["w_dist"].append(-avg_dg)

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"[{config['name']}] Epoch {epoch:4d}/{EPOCHS} | "
                f"D_graph: {avg_dg:.4f} | D_node: {avg_dn:.4f} | "
                f"G_graph: {avg_gg:.4f} | G_node: {avg_gn:.4f} | "
                f"W: {-avg_dg:.4f}"
            )

        if SAVE_INTERMEDIATE and epoch >= SAVE_START and epoch % SAVE_EVERY == 0:
            final_save_path = save_checkpoint(
                experiment_dir=experiment_dir,
                config=config,
                epoch=epoch,
                history=history,
                generator=generator,
                projection=projection,
                adj_module=adj_module,
            )

    # 保存最终结果
    final_save_path = save_checkpoint(
        experiment_dir=experiment_dir,
        config=config,
        epoch=EPOCHS,
        history=history,
        generator=generator,
        projection=projection,
        adj_module=adj_module,
    )

    summary = {
        "experiment": config["name"],
        "label": config["label"],
        "final_epoch": EPOCHS,
        "final_d_loss_graph": history["d_loss_graph"][-1],
        "final_d_loss_node": history["d_loss_node"][-1],
        "final_g_loss_graph": history["g_loss_graph"][-1],
        "final_g_loss_node": history["g_loss_node"][-1],
        "final_w_dist": history["w_dist"][-1],
        "save_path": final_save_path,
    }
    return summary


# =========================
# 11. 主程序：依次运行 4 个消融实验
# =========================

def save_summary_csv(summary_rows, csv_path: str):
    fieldnames = [
        "experiment", "label", "final_epoch",
        "final_d_loss_graph", "final_d_loss_node",
        "final_g_loss_graph", "final_g_loss_node",
        "final_w_dist", "save_path"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"共需运行 {len(EXPERIMENTS)} 个消融实验。")

    all_summaries = []
    for exp_cfg in EXPERIMENTS:
        summary = train_one_experiment(deepcopy(exp_cfg))
        all_summaries.append(summary)

    summary_csv = os.path.join(SAVE_DIR, "ablation_summary.csv")
    save_summary_csv(all_summaries, summary_csv)

    print("\n四个消融实验已全部完成。")
    print(f"汇总文件已保存: {summary_csv}")
    for row in all_summaries:
        print(f"- {row['label']}: {row['save_path']}")
