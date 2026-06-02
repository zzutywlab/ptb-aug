import os
import math
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# 1. 基础设置
# =========================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PHASE1_NPZ_PATH = "output/phase1/phase1_augmented_data.npz"
SAVE_DIR = "output/phase2/gnn"

EPOCHS = 800
BATCH_SIZE = 32
RAND_DIM = 32
EMBED_DIM = 32

LR_G = 1e-4
LR_D = 1e-4
LR_A = 1e-4
LR_P = 1e-4
LR_R = 1e-4

N_CRITIC = 5
LAMBDA_GP = 10.0
NODE_LOSS_WEIGHT = 1.0

# 多版本保存设置
SAVE_START = 300
SAVE_EVERY = 100

# PTB 条件分箱数
N_COND_BINS = 5

os.makedirs(SAVE_DIR, exist_ok=True)


def set_all_seeds(seed=42):
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
TARGET_NAME = str(data["target"].tolist() if hasattr(data["target"], "tolist") else data["target"])

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
target_idx = scaled_xy.shape[1] - 1
num_real_nodes = scaled_xy.shape[1]          # 8特征 + 1目标
num_graph_nodes = num_real_nodes + 1         # 再加1个条件节点


# =========================
# 3. 条件向量：将 PTB 分箱
# =========================
def build_target_bins(y_scaled, n_bins=5):
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y_scaled, quantiles)
    edges = np.unique(edges)

    if len(edges) <= 2:
        edges = np.linspace(float(y_scaled.min()), float(y_scaled.max()) + 1e-8, 3)

    bin_ids = np.digitize(y_scaled, edges[1:-1], right=False)
    n_effective_bins = int(bin_ids.max()) + 1
    return edges, bin_ids.astype(np.int64), n_effective_bins


bin_edges, cond_bin_ids, cond_dim = build_target_bins(scaled_xy[:, target_idx], n_bins=N_COND_BINS)


def sample_conditional_indices(bin_ids, batch_size):
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

    cond_vec = np.zeros((batch_size, cond_dim), dtype=np.float32)
    cond_vec[np.arange(batch_size), chosen_bins] = 1.0

    return cond_vec, chosen_bins, sample_indices


# =========================
# 4. 动态可学习图
# =========================
class LearnableAdjacency(nn.Module):
    def __init__(self, num_nodes):
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


# =========================
# 5. 模型核心
# =========================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
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
    def __init__(self, embed_dim, num_nodes):
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
    def __init__(self, num_real_nodes, cond_dim, embed_dim):
        super().__init__()
        self.num_real_nodes = num_real_nodes
        self.embed_dim = embed_dim

        self.node_proj = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(num_real_nodes)
        ])
        self.cond_proj = nn.Linear(cond_dim, embed_dim)

    def forward_real(self, x, cond_vec):
        nodes = []
        for i in range(self.num_real_nodes):
            nodes.append(self.node_proj[i](x[:, i:i+1]).unsqueeze(1))
        nodes = torch.cat(nodes, dim=1)

        cond_node = self.cond_proj(cond_vec).unsqueeze(1)
        graph = torch.cat([nodes, cond_node], dim=1)
        return graph

    def cond_node_only(self, cond_vec):
        return self.cond_proj(cond_vec).unsqueeze(1)


class NodewiseGenerator(nn.Module):
    def __init__(self, rand_dim, cond_dim, num_graph_nodes, embed_dim, num_real_nodes):
        super().__init__()
        self.num_graph_nodes = num_graph_nodes
        self.num_real_nodes = num_real_nodes
        self.embed_dim = embed_dim

        self.rand_to_proj = nn.Linear(rand_dim + cond_dim, (num_graph_nodes - 1) * embed_dim)
        self.gcn = GCN(embed_dim, num_graph_nodes)
        self.proj_to_final = nn.Linear(num_graph_nodes * embed_dim, num_real_nodes)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, z, cond_tensor_node, adj):
        batch_size = z.shape[0]
        output_1 = self.leakyrelu(self.rand_to_proj(z))
        output_1 = output_1.reshape(batch_size, self.num_graph_nodes - 1, self.embed_dim)

        output_w_cond = torch.cat([output_1, cond_tensor_node], dim=1)
        gnn_output = self.gcn(output_w_cond, adj)
        gnn_output = gnn_output.reshape(batch_size, -1)

        out = self.proj_to_final(gnn_output)
        out = torch.sigmoid(out)
        return out


class Critic(nn.Module):
    def __init__(self, embed_dim, num_graph_nodes):
        super().__init__()
        self.layers = GCN(embed_dim, num_graph_nodes)
        self.linear = nn.Linear(num_graph_nodes * embed_dim, 1)

    def forward(self, x_graph, adj):
        batch_size = x_graph.shape[0]
        x_repr = self.layers(x_graph, adj)
        x_repr = x_repr.reshape(batch_size, -1)
        return self.linear(x_repr)


class TargetRegressor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, node_repr):
        return self.head(node_repr)


# =========================
# 6. 训练辅助函数
# =========================
def gradient_penalty(critic, real_graph, fake_graph, adj):
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
        retain_graph=True
    )[0]

    grad = grad.view(grad.shape[0], -1)
    grad_norm = grad.norm(2, dim=1)
    gp = torch.mean((grad_norm - 1.0) ** 2)
    return gp


def sample_real_batch(train_tensor, batch_size):
    cond_vec_np, chosen_bins, sample_idx = sample_conditional_indices(cond_bin_ids, batch_size)
    real_x = train_tensor[sample_idx]
    cond_vec = torch.tensor(cond_vec_np, dtype=torch.float32, device=DEVICE)
    return real_x, cond_vec, chosen_bins, sample_idx


@torch.no_grad()
def generate_samples(generator, projection, adj_module, n_samples):
    generator.eval()
    projection.eval()

    outputs = []
    remain = n_samples

    while remain > 0:
        bs = min(256, remain)
        cond_vec_np, _, _ = sample_conditional_indices(cond_bin_ids, bs)
        cond_vec = torch.tensor(cond_vec_np, dtype=torch.float32, device=DEVICE)
        cond_node = projection.cond_node_only(cond_vec)

        z = torch.randn(bs, RAND_DIM, device=DEVICE)
        z_in = torch.cat([z, cond_vec], dim=1)

        fake_scaled = generator(z_in, cond_node, adj_module.get_AG())
        outputs.append(fake_scaled.cpu().numpy())
        remain -= bs

    fake_scaled = np.vstack(outputs)
    fake_original = scaler.inverse_transform(fake_scaled)

    return fake_scaled, fake_original


# =========================
# 7. 初始化模型
# =========================
train_tensor = torch.tensor(scaled_xy, dtype=torch.float32, device=DEVICE)

adj_module = LearnableAdjacency(num_graph_nodes).to(DEVICE)
projection = ScalarProjection(num_real_nodes, cond_dim, EMBED_DIM).to(DEVICE)
generator = NodewiseGenerator(
    rand_dim=RAND_DIM,
    cond_dim=cond_dim,
    num_graph_nodes=num_graph_nodes,
    embed_dim=EMBED_DIM,
    num_real_nodes=num_real_nodes
).to(DEVICE)
critic = Critic(embed_dim=EMBED_DIM, num_graph_nodes=num_graph_nodes).to(DEVICE)
regressor = TargetRegressor(embed_dim=EMBED_DIM).to(DEVICE)

opt_g = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.9))
opt_d = torch.optim.Adam(critic.parameters(), lr=LR_D, betas=(0.5, 0.9))
opt_p = torch.optim.Adam(projection.parameters(), lr=LR_P, betas=(0.5, 0.9))
opt_r = torch.optim.Adam(regressor.parameters(), lr=LR_R, betas=(0.5, 0.9))
opt_a = torch.optim.Adam(adj_module.parameters(), lr=LR_A, betas=(0.5, 0.9))

mse_loss = nn.MSELoss()

history = {
    "d_loss_graph": [],
    "d_loss_node": [],
    "g_loss_graph": [],
    "g_loss_node": [],
    "w_dist": []
}


# =========================
# 8. 保存函数
# =========================
def save_checkpoint(epoch):
    fake_scaled, fake_original = generate_samples(generator, projection, adj_module, len(train_tensor))

    save_path = os.path.join(SAVE_DIR, f"phase2_cgtgan_epoch_{epoch:04d}.npz")
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

        cond_bin_edges=bin_edges,
        cond_dim=cond_dim,

        scaler_min_=scaler.min_,
        scaler_scale_=scaler.scale_,

        history=history,
        epoch=epoch
    )
    print(f"已保存: {save_path}")


# =========================
# 9. 训练
# =========================
print(f"Using device: {DEVICE}")

for epoch in range(1, EPOCHS + 1):
    d_graph_losses = []
    d_node_losses = []
    g_graph_losses = []
    g_node_losses = []

    steps_per_epoch = max(1, len(train_tensor) // BATCH_SIZE)

    for _ in range(steps_per_epoch):
        # 1) 判别器：图级任务
        critic.train()
        generator.eval()
        regressor.train()

        for _ in range(N_CRITIC):
            opt_d.zero_grad()
            opt_p.zero_grad()
            opt_a.zero_grad()

            real_x, cond_vec, _, _ = sample_real_batch(train_tensor, BATCH_SIZE)
            real_graph = projection.forward_real(real_x, cond_vec)

            cond_node = projection.cond_node_only(cond_vec)
            z = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)
            z_in = torch.cat([z, cond_vec], dim=1)

            fake_x = generator(z_in, cond_node, adj_module.get_AG()).detach()
            fake_graph = projection.forward_real(fake_x, cond_vec)

            critic_real = critic(real_graph, adj_module.get_AD())
            critic_fake = critic(fake_graph, adj_module.get_AD())

            gp = gradient_penalty(critic, real_graph, fake_graph, adj_module.get_AD())
            d_loss_graph = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp

            d_loss_graph.backward()
            opt_d.step()
            opt_p.step()
            opt_a.step()

            d_graph_losses.append(d_loss_graph.item())

        # 2) 判别器：节点级任务
        opt_d.zero_grad()
        opt_p.zero_grad()
        opt_r.zero_grad()
        opt_a.zero_grad()

        real_x, cond_vec, _, _ = sample_real_batch(train_tensor, BATCH_SIZE)
        real_graph = projection.forward_real(real_x, cond_vec)
        real_repr = critic.layers(real_graph, adj_module.get_AD())

        y_true = real_x[:, target_idx:target_idx + 1]
        y_pred = regressor(real_repr[:, target_idx])

        d_loss_node = mse_loss(y_pred, y_true)
        d_loss_node.backward()

        opt_d.step()
        opt_p.step()
        opt_r.step()
        opt_a.step()

        d_node_losses.append(d_loss_node.item())

        # 3) 生成器：图级任务
        critic.eval()
        generator.train()
        opt_g.zero_grad()
        opt_p.zero_grad()
        opt_a.zero_grad()

        real_x, cond_vec, _, _ = sample_real_batch(train_tensor, BATCH_SIZE)
        cond_node = projection.cond_node_only(cond_vec)

        z = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)
        z_in = torch.cat([z, cond_vec], dim=1)
        fake_x = generator(z_in, cond_node, adj_module.get_AG())
        fake_graph = projection.forward_real(fake_x, cond_vec)

        g_loss_graph = -torch.mean(critic(fake_graph, adj_module.get_AD()))
        g_loss_graph.backward()

        opt_g.step()
        opt_p.step()
        opt_a.step()

        g_graph_losses.append(g_loss_graph.item())

        # 4) 生成器：节点级任务
        opt_g.zero_grad()
        opt_p.zero_grad()
        opt_a.zero_grad()

        real_x, cond_vec, _, _ = sample_real_batch(train_tensor, BATCH_SIZE)
        cond_node = projection.cond_node_only(cond_vec)

        z = torch.randn(real_x.size(0), RAND_DIM, device=DEVICE)
        z_in = torch.cat([z, cond_vec], dim=1)
        fake_x = generator(z_in, cond_node, adj_module.get_AG())
        fake_graph = projection.forward_real(fake_x, cond_vec)
        fake_repr = critic.layers(fake_graph, adj_module.get_AD())

        fake_target = fake_x[:, target_idx:target_idx + 1]
        fake_target_pred = regressor(fake_repr[:, target_idx])

        g_loss_node = NODE_LOSS_WEIGHT * mse_loss(fake_target_pred, fake_target)
        g_loss_node.backward()

        opt_g.step()
        opt_p.step()
        opt_a.step()

        g_node_losses.append(g_loss_node.item())

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
            f"Epoch {epoch:4d}/{EPOCHS} | "
            f"D_graph: {avg_dg:.4f} | D_node: {avg_dn:.4f} | "
            f"G_graph: {avg_gg:.4f} | G_node: {avg_gn:.4f} | "
            f"W: {-avg_dg:.4f}"
        )

    if epoch >= SAVE_START and epoch % SAVE_EVERY == 0:
        save_checkpoint(epoch)
