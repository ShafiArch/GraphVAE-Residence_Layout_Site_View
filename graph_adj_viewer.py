import os
import json
import tkinter as tk
from tkinter import filedialog
import numpy as np
import math
import random

# Node feature index meaning
IDX_INSIDE  = 0
IDX_OUTSIDE = 1

NODE_R = 18          # radius of node circle
CANVAS_W = 900
CANVAS_H = 700


def load_graph_json(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    X = np.array(obj["X"], dtype=np.float32)
    A = np.array(obj["A"], dtype=np.int32)
    node_names = obj["node_names"]
    return X, A, node_names


def compute_layout(n):
    """
    Simple circular layout.
    Places n nodes evenly around a circle.
    """
    cx, cy = CANVAS_W // 2, CANVAS_H // 2
    radius = min(CANVAS_W, CANVAS_H) * 0.35

    positions = {}
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        positions[i] = (x, y)
    return positions


class GraphViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Adjacency Graph Viewer")
        self.geometry(f"{CANVAS_W}x{CANVAS_H}")

        self.canvas = tk.Canvas(self, width=CANVAS_W, height=CANVAS_H, bg="white")
        self.canvas.pack(fill="both", expand=True)

        self.create_menu()

    def create_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Graph JSON", command=self.open_file)
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

    def open_file(self):
        fpath = filedialog.askopenfilename(
            title="Open *_graph.json",
            filetypes=[("Graph JSON", "*_graph.json"), ("JSON files", "*.json")]
        )
        if not fpath:
            return

        X, A, node_names = load_graph_json(fpath)
        self.draw_graph(X, A, node_names, os.path.basename(fpath))

    def draw_graph(self, X, A, node_names, title):
        self.canvas.delete("all")

        n = len(node_names)
        positions = compute_layout(n)

        # Draw edges (undirected - avoid duplicates)
        seen = set()
        for (u, v) in A:
            u, v = int(u), int(v)
            key = tuple(sorted((u, v)))
            if key in seen:
                continue
            seen.add(key)

            x1, y1 = positions[u]
            x2, y2 = positions[v]
            self.canvas.create_line(x1, y1, x2, y2, fill="#999", width=2)

        # Draw nodes
        for i in range(n):
            x, y = positions[i]
            inside = X[i, IDX_INSIDE] > X[i, IDX_OUTSIDE]
            color = "#4f8ef7" if inside else "#34a853"

            self.canvas.create_oval(
                x - NODE_R, y - NODE_R,
                x + NODE_R, y + NODE_R,
                fill=color, outline="black", width=2
            )

            self.canvas.create_text(
                x, y,
                text=node_names[i],
                fill="white",
                font=("Arial", 10, "bold")
            )

        # Title
        self.canvas.create_text(
            CANVAS_W // 2, 20,
            text=title,
            fill="black",
            font=("Arial", 16, "bold")
        )


if __name__ == "__main__":
    app = GraphViewer()
    app.mainloop()
