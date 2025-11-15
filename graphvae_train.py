# graphvae_train.py

import os
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv

# ============================================================
# USER SETTINGS
# ============================================================
PADDING_SIZE = 12        # <-- YOU CAN CHANGE THIS
SAVE_EVERY = 25          # save checkpoint every N epochs
STATUS_EVERY = 20        # print status every N epochs

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset_graph")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
NPY_DIR = os.path.join(OUTPUT_DIR, "npy_dataset")
CKPT_PATH = os.path.join(OUTPUT_DIR, "model_gatmlp.pt")
NORM_PATH = os.path.join(NPY_DIR, "normalization.npz")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(NPY_DIR, exist_ok=True)

# ============================================================
# CONSTANTS
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HID_DIM = 64
Z_DIM = 32
GAT_HEADS = 4
BATCH_SIZE = 1
EPOCHS = 300
LR = 1e-3
BETA_KL = 1e-4
WARMUP_EPOCHS = 20
GRAD_CLIP_NORM = 5.0

# ============================================================
# GraphVAE MODEL
# ============================================================
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, z_dim, heads=4):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hid_dim)
        self.gat1 = GATConv(hid_dim, hid_dim, heads=heads, concat=False)
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
        )
        self.mu_lin = nn.Linear(hid_dim, z_dim)
        self.logvar_lin = nn.Linear(hid_dim, z_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.gcn1(x, edge_index))
        h = F.relu(self.gat1(h, edge_index))
        h = self.mlp(h)
        mu = self.mu_lin(h)
        logvar = self.logvar_lin(h)
        return mu, logvar


class GraphDecoder(nn.Module):
    def __init__(self, z_dim, hid_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, z_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = self.fc2(h) + z
        logits = h @ h.T
        return logits


class GraphVAE(nn.Module):
    def __init__(self, in_dim, hid_dim, z_dim, heads=4):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hid_dim, z_dim, heads=heads)
        self.decoder = GraphDecoder(z_dim, hid_dim)

    def encode(self, data):
        return self.encoder(data.x, data.edge_index)

    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar

# ============================================================
# DATA LOADING
# ============================================================
def load_graph_json(path):
    with open(path, "r") as f:
        obj = json.load(f)

    X = np.array(obj["X"], dtype=np.float32)
    N, F_dim = X.shape

    edges = np.array(obj["A"], dtype=np.int64).reshape(-1, 2)
    if edges.size > 0:
        if edges.min() >= 1 and edges.max() <= N:
            edges -= 1
    else:
        edges = np.zeros((0, 2), dtype=np.int64)

    adj = np.zeros((N, N), dtype=np.float32)
    if edges.size > 0:
        u, v = edges[:, 0], edges[:, 1]
        adj[u, v] = 1
        adj[v, u] = 1

    d = Data(
        x=torch.from_numpy(X),
        edge_index=torch.from_numpy(edges.T.copy()).long()
        if edges.size > 0 else torch.empty((2, 0), dtype=torch.long)
    )
    d.adj = torch.from_numpy(adj)
    d.name = os.path.basename(path)
    return d

def build_dataset(folder):
    out = []
    for f in sorted(os.listdir(folder)):
        if f.endswith("_graph.json"):
            d = load_graph_json(os.path.join(folder, f))
            out.append(d)
    return out

# ============================================================
# NORMALIZATION
# ============================================================
def normalize_dataset(dataset):
    xs = torch.cat([d.x for d in dataset], dim=0)
    mean = xs.mean(0, keepdim=True)
    std = xs.std(0, keepdim=True)
    std[std < 1e-6] = 1

    for d in dataset:
        d.x = (d.x - mean) / std

    return mean.squeeze(0), std.squeeze(0)

# ============================================================
# NPY EXPORT (with user-chosen padding)
# ============================================================
def export_dataset_to_npy(dataset, out_dir, mean, std):
    F_dim = dataset[0].x.size(1)

    for idx, d in enumerate(dataset):
        N = d.x.size(0)

        x_pad = torch.zeros(PADDING_SIZE, F_dim)
        adj_pad = torch.zeros(PADDING_SIZE, PADDING_SIZE)
        mask = torch.zeros(PADDING_SIZE)

        n_use = min(N, PADDING_SIZE)
        x_pad[:n_use] = d.x[:n_use]
        adj_pad[:n_use, :n_use] = d.adj[:n_use, :n_use]
        mask[:n_use] = 1

        np.savez(
            os.path.join(out_dir, f"graph_{idx:05d}.npz"),
            x=x_pad.numpy(),
            adj=adj_pad.numpy(),
            mask=mask.numpy(),
            name=d.name,
        )

    np.savez(NORM_PATH, mean=mean.numpy(), std=std.numpy())

# ============================================================
# LOSSES
# ============================================================
def recon_loss(logits, adj_true):
    adj_true = adj_true.float().to(logits.device)
    pos = adj_true.sum()
    total = adj_true.numel()
    pos_weight = (total - pos) / pos if pos > 0 else 1.0
    return F.binary_cross_entropy_with_logits(logits, adj_true, pos_weight=pos_weight)

def kl_loss(mu, logvar):
    logvar = logvar.clamp(min=-10, max=10)
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# ============================================================
# TRAINING
# ============================================================
def main():
    dataset = build_dataset(DATA_DIR)
    in_dim = dataset[0].x.size(1)

    mean, std = normalize_dataset(dataset)
    export_dataset_to_npy(dataset, NPY_DIR, mean, std)

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = GraphVAE(in_dim, HID_DIM, Z_DIM, heads=GAT_HEADS).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        beta = BETA_KL * min(epoch / WARMUP_EPOCHS, 1.0)

        for data in loader:
            data = data.to(DEVICE)
            logits, mu, logvar = model(data)
            rec = recon_loss(logits, data.adj)
            kl = kl_loss(mu, logvar)
            loss = rec + beta * kl

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optim.step()

        if epoch % STATUS_EVERY == 0:
            print(f"[epoch {epoch}] loss={loss.item():.4f} rec={rec.item():.4f} kl={kl.item():.4f}")

        if epoch % SAVE_EVERY == 0:
            ckpt = {"cfg": {"in_dim": in_dim}, "model": model.state_dict()}
            torch.save(ckpt, os.path.join(OUTPUT_DIR, f"model_epoch_{epoch}.pt"))
            print(f"Saved checkpoint at epoch {epoch}")

    torch.save({"cfg": {"in_dim": in_dim}, "model": model.state_dict()}, CKPT_PATH)
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    main()
