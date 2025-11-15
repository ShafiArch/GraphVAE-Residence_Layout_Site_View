# graphvae_generate.py
#
# Decode / regenerate graphs from a trained GraphVAE model.
# For each *_graph.json in dataset_graph, this script:
#   1) loads the graph and normalizes X using saved mean/std
#   2) encodes it with the trained VAE
#   3) decodes a new adjacency matrix
#   4) writes a new JSON file with the same nodes/X but new A/E
#
# Output: outputs/generated_graphs/plan_XXXXX_graph_decoded.json

import os
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

# ============================================================
# PATHS + SETTINGS (match graphvae_train.py)
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "dataset_graph")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
NPY_DIR    = os.path.join(OUTPUT_DIR, "npy_dataset")
CKPT_PATH  = os.path.join(OUTPUT_DIR, "model_gatmlp.pt")
NORM_PATH  = os.path.join(NPY_DIR, "normalization.npz")

GEN_DIR    = os.path.join(OUTPUT_DIR, "generated_graphs")
os.makedirs(GEN_DIR, exist_ok=True)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HID_DIM      = 64
Z_DIM        = 32
GAT_HEADS    = 4
EDGE_THR     = 0.5      # threshold on sigmoid(logits) for edges
USE_MU_ONLY  = True     # deterministic decode: z = mu (no sampling)

# ============================================================
# Model (copied from graphvae_train.py)
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
# Helpers: normalization + data loading (JSON → Data)
# ============================================================
def load_normalization(norm_path):
    """Load feature mean/std used during training."""
    arr = np.load(norm_path)
    mean = torch.from_numpy(arr["mean"]).float()  # [F]
    std = torch.from_numpy(arr["std"]).float()    # [F]
    return mean, std


def load_graph_for_inference(path, mean, std):
    """
    Load one *_graph.json, normalize X with given mean/std,
    and return (Data, raw_json_obj).
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    X = np.array(obj["X"], dtype=np.float32)          # [N, F]
    N, F_dim = X.shape

    # A is stored as edge list [[u,v], ...]
    edges = np.array(obj["A"], dtype=np.int64).reshape(-1, 2)
    if edges.size > 0:
        # handle possible 1-based indices
        if edges.min() >= 1 and edges.max() <= N:
            edges -= 1
    else:
        edges = np.zeros((0, 2), dtype=np.int64)

    # Dense adj not strictly needed here, but may be useful
    adj = np.zeros((N, N), dtype=np.float32)
    if edges.size > 0:
        u, v = edges[:, 0], edges[:, 1]
        adj[u, v] = 1.0
        adj[v, u] = 1.0

    # Normalize X with training stats
    mean_t = mean.view(1, -1)
    std_t = std.view(1, -1)
    x = torch.from_numpy(X)
    x_norm = (x - mean_t) / std_t

    edge_index = (
        torch.from_numpy(edges.T.copy()).long()
        if edges.size > 0
        else torch.empty((2, 0), dtype=torch.long)
    )

    data = Data(x=x_norm, edge_index=edge_index)
    data.adj = torch.from_numpy(adj)
    data.name = os.path.basename(path)

    return data, obj


def infer_edge_type(u, v, X):
    """
    Edge type heuristic:
      0 = room-room adjacency (both inside)
      1 = view edge (any edge involving an outside node)
    inside_flag = X[:,0], outside_flag = X[:,1]
    """
    inside_u = X[u, 0] > X[u, 1]
    inside_v = X[v, 0] > X[v, 1]
    return 0 if (inside_u and inside_v) else 1


# ============================================================
# Main generation routine
# ============================================================
def decode_graphs():
    # Load mean/std and checkpoint
    mean, std = load_normalization(NORM_PATH)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    in_dim = ckpt["cfg"]["in_dim"]

    model = GraphVAE(in_dim, HID_DIM, Z_DIM, heads=GAT_HEADS).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    files = [
        f for f in sorted(os.listdir(DATA_DIR))
        if f.endswith("_graph.json")
    ]

    if not files:
        print(f"[warn] no *_graph.json files found in {DATA_DIR}")
        return

    print(f"[info] Loaded model from {CKPT_PATH}")
    print(f"[info] Using normalization from {NORM_PATH}")
    print(f"[info] Decoding {len(files)} graphs...")

    for fname in files:
        in_path = os.path.join(DATA_DIR, fname)
        data, raw = load_graph_for_inference(in_path, mean, std)

        X_raw = np.array(raw["X"], dtype=np.float32)  # for edge-type inference
        N = X_raw.shape[0]

        data = data.to(DEVICE)
        with torch.no_grad():
            mu, logvar = model.encode(data)
            if USE_MU_ONLY:
                z = mu
            else:
                z = model.reparameterize(mu, logvar)
            logits = model.decoder(z)
            probs = torch.sigmoid(logits).cpu().numpy()

        # Symmetrize & remove self-loops
        probs = (probs + probs.T) / 2.0
        np.fill_diagonal(probs, 0.0)

        # Keep upper triangle to avoid duplicates, then add both directions
        mask = np.triu(probs, k=1) > EDGE_THR
        src, dst = np.where(mask)

        A_pred = []
        E_pred = []

        for u, v in zip(src.tolist(), dst.tolist()):
            etype = infer_edge_type(u, v, X_raw)
            # directed edges in both directions (like original JSON)
            A_pred.append([int(u), int(v)])
            E_pred.append([int(etype)])
            A_pred.append([int(v), int(u)])
            E_pred.append([int(etype)])

        # Build output JSON (keep nodes + features, change edges)
        Z = raw.get("Z", [float(N), 0.0])
        out_obj = {
            "Z": Z,
            "node_names": raw["node_names"],
            "node_meta": raw["node_meta"],
            "X": raw["X"],
            "A": A_pred,
            "E": E_pred,
        }

        out_name = os.path.splitext(fname)[0] + "_decoded.json"
        out_path = os.path.join(GEN_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2)

        print(f"[gen] {fname} -> {out_name}   (N={N}, edges={len(A_pred)})")


if __name__ == "__main__":
    decode_graphs()
