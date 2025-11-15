import os
import json
import numpy as np

# ------------------------------------------------------------
# INSERT YOUR FOLDER PATH HERE
# Example: FOLDER = r"C:\Users\Shafiul\Desktop\dataset_graph"
# ------------------------------------------------------------
FOLDER = r"D:\ARCHITECTURE\00.Article CADRIA\GraphVAE\Dataset\dataset_graph"


def get_max_nodes(folder):
    max_nodes = 0

    print("Checking JSON files in:", folder)
    print("-" * 50)

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(folder, filename)

        try:
            with open(path, "r") as f:
                data = json.load(f)

            # X is [N, F]
            X = np.array(data["X"])
            node_count = X.shape[0]

            print(f"{filename}: nodes = {node_count}")

            if node_count > max_nodes:
                max_nodes = node_count

        except Exception as e:
            print(f"{filename}: ERROR reading file → {e}")

    print("-" * 50)
    print(f"Maximum nodes in folder = {max_nodes}")
    return max_nodes


if __name__ == "__main__":
    get_max_nodes(FOLDER)
