import os, json, math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# =================== PATHS ===================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
NPY_DIR    = os.path.join(OUTPUT_DIR, "npy_dataset")
CKPT_PATH  = os.path.join(OUTPUT_DIR, "model_gatmlp.pt")
NORM_PATH  = os.path.join(NPY_DIR, "normalization.npz")

GEN_DIR    = os.path.join(OUTPUT_DIR, "generated_conditional")
os.makedirs(GEN_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# match training hyperparams from graphvae_train.py
HID_DIM   = 64
Z_DIM     = 32
GAT_HEADS = 4


# =================== MODEL (same as training) ===================
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

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)


# =================== HELPERS ===================
def load_normalization(path):
    arr = np.load(path)
    mean = torch.from_numpy(arr["mean"]).float()
    std  = torch.from_numpy(arr["std"]).float()
    return mean, std


def build_features(num_nodes, total_area, view_score_percent, wwr=0.4):
    """
    Build synthetic node features in the SAME 11-dim format:

    0: inside_flag
    1: outside_flag
    2: area
    3: bw
    4: bh
    5: bminx
    6: bminy
    7: bmaxx
    8: bmaxy
    9: WWR
    10: view_score

    - First (num_nodes - 4) nodes = inside rooms (if possible)
    - Last up to 4 nodes = outside façade nodes
    - total_area distributed across inside rooms
    - view_score of inside rooms chosen to match target graph score
    """
    # assume up to 4 outside nodes
    num_outside = 4 if num_nodes >= 4 else max(0, num_nodes - 1)
    num_inside = num_nodes - num_outside
    if num_inside <= 0:
        raise ValueError("Need at least 1 inside node")

    target_view = view_score_percent / 100.0 if view_score_percent > 1.0 else view_score_percent
    target_view = max(0.0, min(1.0, target_view))

    area_per_room = float(total_area) / float(num_inside)

    X = []
    node_names = []
    node_meta = []

    # place rooms along x axis
    cursor_x = 0.0
    side = math.sqrt(area_per_room)
    spacing = side * 0.1

    for i in range(num_inside):
        inside_flag = 1.0
        outside_flag = 0.0
        area = area_per_room

        bminx = cursor_x
        bminy = 0.0
        bmaxx = cursor_x + side
        bmaxy = side
        cursor_x = bmaxx + spacing

        bw = side
        bh = side
        wwr_val = float(wwr)
        view_score = target_view

        name = f"room_{i}"
        node_names.append(name)
        node_meta.append({
            "name": name,
            "kind": "inside",
            "room_type": "room",
            "view_type": ""
        })
        X.append([
            inside_flag, outside_flag,
            area, bw, bh,
            bminx, bminy, bmaxx, bmaxy,
            wwr_val, view_score
        ])

    # outside nodes (similar to prep_gui semantics)
    outside_types = ["sky", "vegetation", "context", "building"]
    view_weights = {"sky": 1.0, "vegetation": 0.8, "context": 0.4, "building": 0.2}

    for j in range(num_outside):
        vtype = outside_types[j % len(outside_types)]
        inside_flag = 0.0
        outside_flag = 1.0
        area = 0.0
        bw = bh = 0.0
        bminx = bminy = bmaxx = bmaxy = 0.0
        wwr_val = 0.0
        view_score = view_weights.get(vtype, 0.0)

        name = f"{vtype}_{j}"
        node_names.append(name)
        node_meta.append({
            "name": name,
            "kind": "outside",
            "room_type": "",
            "view_type": vtype
        })
        X.append([
            inside_flag, outside_flag,
            area, bw, bh,
            bminx, bminy, bmaxx, bmaxy,
            wwr_val, view_score
        ])

    X = np.array(X, dtype=np.float32)

    # graph-level score = area-weighted avg of inside nodes (like compute_graph_score)
    areas = X[:num_inside, 2]
    v_scores = X[:num_inside, 10]
    if areas.sum() > 0:
        graph_view = float((areas * v_scores).sum() / areas.sum())
    else:
        graph_view = 0.0

    Z = [float(num_nodes), float(graph_view)]
    return X, node_names, node_meta, Z


def generate_graph():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    if not os.path.exists(NORM_PATH):
        raise FileNotFoundError(f"Normalization not found: {NORM_PATH}")

    mean, std = load_normalization(NORM_PATH)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    in_dim = ckpt["cfg"]["in_dim"]

    model = GraphVAE(in_dim, HID_DIM, Z_DIM, heads=GAT_HEADS).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("=== Conditional graph generation ===")
    num_nodes = int(input("Total number of nodes (inside + outside): ").strip())
    total_area = float(input("Total room area (same unit as training, e.g. px^2): ").strip())
    view_score = float(input("Target graph view score (0–1 or 0–100%): ").strip())

    X_raw, node_names, node_meta, Z = build_features(num_nodes, total_area, view_score)

    # normalize features
    mean_v = mean.view(1, -1)
    std_v = std.view(1, -1)
    X_norm = (torch.from_numpy(X_raw) - mean_v) / std_v
    X_norm = X_norm.to(DEVICE)

    # sample latent z ~ N(0, I) and decode adjacency
    z = torch.randn((num_nodes, Z_DIM), device=DEVICE)
    with torch.no_grad():
        logits = model.decoder(z)
        probs = torch.sigmoid(logits).cpu().numpy()

    # symmetric adjacency, no self-loops
    probs = (probs + probs.T) / 2.0
    np.fill_diagonal(probs, 0.0)
    threshold = 0.5
    mask = np.triu(probs, k=1) > threshold
    us, vs = np.where(mask)

    A = []
    E = []
    for u, v in zip(us.tolist(), vs.tolist()):
        # edge type: 0 = room-room, 1 = involves outside (same semantics as prep script)
        inside_u = X_raw[u, 0] > X_raw[u, 1]
        inside_v = X_raw[v, 0] > X_raw[v, 1]
        etype = 0 if (inside_u and inside_v) else 1

        A.append([int(u), int(v)])
        E.append([int(etype)])
        A.append([int(v), int(u)])
        E.append([int(etype)])

    out_obj = {
        "Z": Z,
        "node_names": node_names,
        "node_meta": node_meta,
        "X": X_raw.tolist(),
        "A": A,
        "E": E,
    }

    # pick new filename
    idx = 0
    while True:
        out_name = f"cond_plan_{idx:04d}_graph.json"
        out_path = os.path.join(GEN_DIR, out_name)
        if not os.path.exists(out_path):
            break
        idx += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"[done] generated graph saved to: {out_path}")
    print(f"    nodes={len(node_names)}, edges={len(A)} (directed)")


if __name__ == "__main__":
    generate_graph()
