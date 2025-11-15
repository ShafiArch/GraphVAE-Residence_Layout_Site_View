from glob import glob
import numpy as np
import json
import os

folder = r"D:\ARCHITECTURE\00.Article CADRIA\GraphVAE\Dataset\dataset_graph"

for fname in sorted(os.listdir(folder)):
    if not fname.lower().endswith("_graph.json"):
        continue
    path = os.path.join(folder, fname)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    X = np.array(obj["X"], dtype=np.float32)
    A = np.array(obj["A"], dtype=np.float32)

    n_x = X.shape[0]
    n_a0, n_a1 = A.shape

    print("----", fname, "----")
    print("X nodes:", n_x, "A shape:", A.shape)

    if n_a0 != n_x or n_a1 != n_x:
        print("  ❌ mismatch between X and A sizes!")

    src, dst = np.nonzero(A)
    if src.size > 0:
        max_idx = max(src.max(), dst.max())
        print("  max edge index:", max_idx)
        if max_idx >= n_x:
            print("  ❌ edge index out of range!")
    else:
        print("  (no edges)")
