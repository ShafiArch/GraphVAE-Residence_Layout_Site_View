#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_prep_gui.py

Prepare graph JSON from plan_*.json floorplans.

GUI:
- Select folder with plan JSONs.
- Choose outside node 'view types' for N / S / E / W.
- Enter a global window-to-wall ratio (WWR) used as a node feature.
- Batch convert → writes one *_graph.json per input file.

Graph JSON structure (per file):

{
  "Z": [num_nodes, 0.0],          # graph-level features (2nd is placeholder)
  "node_names": [...],            # list of node names (rooms + outside nodes)
  "node_meta": [                  # list of dicts with string meta
      {"name": "...", "kind": "inside"/"outside", "room_type": "...", "view_type": "..."},
      ...
  ],
  "X": [                          # node features (numeric)
      [inside_flag, outside_flag,
       area, bbox_w, bbox_h, minx, miny, maxx, maxy,
       wwr],
      ...
  ],
  "A": [ [u, v], ... ],           # undirected edges (indices in node_names)
  "E": [ [edge_type], ... ]       # edge features: 0 = room-room adj, 1 = room-outside view
}

This is intentionally simple & robust; you can extend the features later
(e.g. add view scores, one-hot room types, etc.) without changing this GUI.
"""

import json
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from shapely.geometry import shape as shp_shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from view_calc import (
    OutsideNode,
    compute_grid_view,
    compute_room_scores,
    compute_graph_score,
    VIEW_WEIGHTS,
)

# ---------------------------------------------------------------------
# Helpers to read plan JSON and build graphs
# ---------------------------------------------------------------------

NON_ROOM_LAYERS = {
    "wall", "window", "door", "front_door",
    "neighbor", "inner", "land", "pool", "parking",
}

OUTSIDE_VIEW_TYPES = ["sky", "vegetation", "context", "building", "balcony", "none"]


def _safe_iter_geoms(g):
    """Yield polygons from Polygon or MultiPolygon."""
    if isinstance(g, Polygon):
        yield g
    elif isinstance(g, MultiPolygon):
        for p in g.geoms:
            yield p


def _collect_rooms(plan):
    """
    Return dict name -> shapely geometry for all 'room' layers,
    skipping walls/windows/etc and empty / zero-area geometries.
    """
    rooms = {}
    for name, obj in plan.items():
        if name in NON_ROOM_LAYERS:
            continue
        if not isinstance(obj, dict):
            continue
        if "type" not in obj or "coordinates" not in obj:
            continue
        try:
            g = shp_shape(obj)
        except Exception:
            continue
        if g.is_empty:
            # <-- this skips your empty 'garden' layer
            continue
        if g.area <= 0:
            continue
        rooms[name] = g
    return rooms


def _building_bounds(plan, rooms):
    """Get overall bounding box of building from 'inner' or union of rooms."""
    if "inner" in plan and isinstance(plan["inner"], dict):
        try:
            g_inner = shp_shape(plan["inner"])
            if not g_inner.is_empty:
                return g_inner.bounds
        except Exception:
            pass
    if rooms:
        try:
            u = unary_union(list(rooms.values()))
            return u.bounds
        except Exception:
            pass
    # fallback
    return (0.0, 0.0, 1.0, 1.0)


def _room_bbox(geom):
    """Return (minx, miny, maxx, maxy, w, h, area) for a room geometry."""
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    area = geom.area
    return minx, miny, maxx, maxy, w, h, area


def _build_graph_from_plan(plan, side_cfg, global_wwr):
    """
    Convert floorplan JSON + outside side config into a graph dict.
    side_cfg: dict like {"north": "sky", "south": "vegetation", ...}
    global_wwr: float

    New:
    - calls view_calc to compute:
        * per-room view_score (0–1)
        * graph-level view_score (0–1)
    - node features X gain an extra 'view_score' column at the end
    - Z = [num_nodes, graph_view_score]
    """
    # -----------------------------
    # 1) Collect rooms (inside nodes)
    # -----------------------------
    rooms = _collect_rooms(plan)

    # If there are no rooms, return an empty graph
    if not rooms:
        return {
            "Z": [0.0, 0.0],
            "node_names": [],
            "node_meta": [],
            "X": [],
            "A": [],
            "E": [],
        }

    # Building bounds
    minx_b, miny_b, maxx_b, maxy_b = _building_bounds(plan, rooms)
    tol = 1e-6  # tolerance for matching edges to building boundary

    # -----------------------------
    # 2) Build layers for view_calc
    # -----------------------------
    layers_for_view = {}

    # inner
    if "inner" in plan and isinstance(plan["inner"], dict):
        try:
            g_inner = shp_shape(plan["inner"])
            if not g_inner.is_empty:
                layers_for_view["inner"] = g_inner
        except Exception:
            pass

    # window
    if "window" in plan and isinstance(plan["window"], dict):
        try:
            g_win = shp_shape(plan["window"])
            if not g_win.is_empty:
                layers_for_view["window"] = g_win
        except Exception:
            pass

    # rooms (re-use the geometries we already have)
    for rname, geom in rooms.items():
        layers_for_view[rname] = geom

    # -----------------------------
    # 3) Build OutsideNode objects for view_calc
    # -----------------------------
    outside_nodes_for_view = []
    cx = 0.5 * (minx_b + maxx_b)
    cy = 0.5 * (miny_b + maxy_b)
    span = max(maxx_b - minx_b, maxy_b - miny_b)
    margin = 0.1 * span if span > 0 else 1.0

    def _add_outside(side: str, x: float, y: float):
        vtype = side_cfg.get(side, "none")
        if vtype is None or vtype == "none":
            return
        outside_nodes_for_view.append(
            OutsideNode(name=side, category=vtype, position=(x, y))
        )

    # place one node per façade
    _add_outside("north", cx, maxy_b + margin)
    _add_outside("south", cx, miny_b - margin)
    _add_outside("west",  minx_b - margin, cy)
    _add_outside("east",  maxx_b + margin, cy)

    # -----------------------------
    # 4) View calculation (grid → room_scores → graph_score)
    # -----------------------------
    room_scores = {}
    graph_view_score = 0.0

    if "inner" in layers_for_view and "window" in layers_for_view and outside_nodes_for_view:
        # pick a grid step so we get ~20 samples across the largest dimension
        span = max(maxx_b - minx_b, maxy_b - miny_b)
        grid_step = span / 20.0 if span > 0 else 1.0

        grid_samples = compute_grid_view(
            layers_for_view,
            outside_nodes_for_view,
            grid_step=grid_step
        )
        if grid_samples:
            room_scores = compute_room_scores(
                layers_for_view,
                grid_samples,
                room_layer_names=list(rooms.keys())
            )
            graph_view_score = compute_graph_score(room_scores)

    # -----------------------------
    # 5) Node list: inside rooms first
    # -----------------------------
    room_names = sorted(rooms.keys())
    node_names = list(room_names)

    node_meta = []
    X = []

    for rname in room_names:
        geom = rooms[rname]
        bminx, bminy, bmaxx, bmaxy, bw, bh, area = _room_bbox(geom)
        inside_flag = 1.0
        outside_flag = 0.0
        wwr = float(global_wwr)

        # view_score for this room (0–1)
        rs = room_scores.get(rname)
        view_score = float(rs.score) if rs is not None else 0.0

        node_meta.append({
            "name": rname,
            "kind": "inside",
            "room_type": rname,
            "view_type": "",
        })
        X.append([
            inside_flag, outside_flag,
            float(area),
            float(bw), float(bh),
            float(bminx), float(bminy),
            float(bmaxx), float(bmaxy),
            wwr,
            view_score,          # ← NEW: room view score
        ])

    # -----------------------------
    # 6) Outside nodes (graph nodes, with weight as view feature)
    # -----------------------------
    outside_nodes = []  # list of (node_name, side, view_type)
    for side in ["north", "south", "east", "west"]:
        vtype = side_cfg.get(side, "none")
        if vtype is None or vtype == "none":
            continue
        node_name = f"{side}_{vtype}"
        outside_nodes.append((node_name, side, vtype))

    for node_name, side, vtype in outside_nodes:
        inside_flag = 0.0
        outside_flag = 1.0
        area = 0.0
        bw = bh = 0.0
        bminx = bminy = bmaxx = bmaxy = 0.0

        # use VIEW_WEIGHTS as a simple view_score for outside nodes
        view_score = float(VIEW_WEIGHTS.get(vtype, 0.0))

        node_meta.append({
            "name": node_name,
            "kind": "outside",
            "room_type": "",
            "view_type": vtype,
        })
        X.append([
            inside_flag, outside_flag,
            float(area),
            float(bw), float(bh),
            float(bminx), float(bminy),
            float(bmaxx), float(bmaxy),
            0.0,           # WWR not relevant for outside node
            view_score,    # ← NEW: façade weight
        ])
        node_names.append(node_name)

    node_index = {name: idx for idx, name in enumerate(node_names)}

    # -----------------------------
    # 7) Edges: room-room adjacency
    # -----------------------------
    A = []
    E = []  # [edge_type]; 0 = room-room adjacency, 1 = view edge

    ritems = list(rooms.items())
    for i in range(len(ritems)):
        rname_i, geom_i = ritems[i]
        idx_i = node_index.get(rname_i)
        if idx_i is None:
            continue
        for j in range(i + 1, len(ritems)):
            rname_j, geom_j = ritems[j]
            idx_j = node_index.get(rname_j)
            if idx_j is None:
                continue
            try:
                if geom_i.touches(geom_j) or geom_i.intersects(geom_j):
                    inter = geom_i.intersection(geom_j)
                    if not inter.is_empty and (inter.area > 0 or inter.length > 0):
                        A.append([idx_i, idx_j])
                        E.append([0])
                        A.append([idx_j, idx_i])
                        E.append([0])
            except Exception:
                continue

    # -----------------------------
    # 8) Room–outside view edges (same heuristic as before)
    # -----------------------------
    for rname, geom in rooms.items():
        idx_r = node_index.get(rname)
        if idx_r is None:
            continue
        bminx, bminy, bmaxx, bmaxy, bw, bh, area = _room_bbox(geom)

        touches = set()
        if abs(bmaxy - maxy_b) <= tol:
            touches.add("north")
        if abs(bminy - miny_b) <= tol:
            touches.add("south")
        if abs(bminx - minx_b) <= tol:
            touches.add("west")
        if abs(bmaxx - maxx_b) <= tol:
            touches.add("east")

        for node_name, side, vtype in outside_nodes:
            if side not in touches:
                continue
            idx_o = node_index.get(node_name)
            if idx_o is None:
                continue
            A.append([idx_r, idx_o])
            E.append([1])  # view edge
            A.append([idx_o, idx_r])
            E.append([1])

    # -----------------------------
    # 9) Graph-level features Z
    # -----------------------------
    num_nodes = len(node_names)
    Z = [float(num_nodes), float(graph_view_score)]

    graph = {
        "Z": Z,
        "node_names": node_names,
        "node_meta": node_meta,
        "X": X,
        "A": A,
        "E": E,
    }
    return graph



# ---------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------

class GraphPrepGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Graph JSON Prep")
        self.geometry("800x450")

        self.folder = None
        self.files = []

        # outside node type variables
        self.side_vars = {
            "north": tk.StringVar(value="sky"),
            "south": tk.StringVar(value="vegetation"),
            "east":  tk.StringVar(value="context"),
            "west":  tk.StringVar(value="building"),
        }

        self.wwr_var = tk.StringVar(value="0.4")
        self.info_var = tk.StringVar(value="No folder selected")

        self._build_ui()

    def _build_ui(self):
        # top bar
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=8, pady=6)

        ttk.Button(top, text="Open folder...", command=self.choose_folder)\
            .pack(side="left")
        self.lbl_folder = ttk.Label(top, text="", anchor="w")
        self.lbl_folder.pack(side="left", padx=8)

        # outside node types frame
        frm_out = ttk.LabelFrame(self, text="Outside node types")
        frm_out.pack(side="top", fill="x", padx=8, pady=(4, 4))

        for row, side in enumerate(["North", "South", "East", "West"]):
            sname = side.lower()
            ttk.Label(frm_out, text=side, width=8)\
                .grid(row=row, column=0, padx=4, pady=2, sticky="w")
            cb = ttk.Combobox(
                frm_out,
                state="readonly",
                values=OUTSIDE_VIEW_TYPES,
                width=15,
                textvariable=self.side_vars[sname],
            )
            cb.grid(row=row, column=1, padx=4, pady=2, sticky="w")

        # global WWR
        frm_wwr = ttk.Frame(self)
        frm_wwr.pack(side="top", fill="x", padx=8, pady=(4, 4))
        ttk.Label(frm_wwr, text="Global WWR (0–1):", width=16)\
            .pack(side="left")
        ttk.Entry(frm_wwr, textvariable=self.wwr_var, width=10)\
            .pack(side="left", padx=4)

        # bottom: info + button
        frm_bottom = ttk.Frame(self)
        frm_bottom.pack(side="top", fill="x", padx=8, pady=(4, 4))

        self.lbl_info = ttk.Label(frm_bottom, textvariable=self.info_var,
                                  anchor="w")
        self.lbl_info.pack(side="left", fill="x", expand=True)

        ttk.Button(frm_bottom,
                   text="Batch convert → graph JSON",
                   command=self.batch_convert)\
            .pack(side="right")

        # spacer for future log output
        self.txt_log = tk.Text(self, height=12, wrap="word")
        self.txt_log.pack(side="top", fill="both", expand=True,
                          padx=8, pady=(4, 8))
        self.txt_log.insert("end", "Select a folder to begin...\n")
        self.txt_log.config(state="disabled")

    # ------------------------------------------------------------ IO

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select folder with plan JSONs")
        if not folder:
            return
        self.folder = folder
        self.lbl_folder.config(text=folder)

        self.files = [f for f in os.listdir(folder)
                      if f.lower().endswith(".json")]
        self.files.sort()
        self.info_var.set(f"{len(self.files)} JSON files found")
        self._log(f"Folder: {folder}\nNumber of JSON files: {len(self.files)}\n")

    def _parse_wwr(self):
        try:
            v = float(self.wwr_var.get())
        except ValueError:
            raise ValueError("WWR must be a number between 0 and 1.")
        if not (0.0 <= v <= 1.0):
            raise ValueError("WWR must be between 0 and 1.")
        return v

    def batch_convert(self):
        if not self.folder:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        if not self.files:
            messagebox.showwarning("No JSON files", "No *.json files found.")
            return
        try:
            wwr = self._parse_wwr()
        except ValueError as e:
            messagebox.showerror("Invalid WWR", str(e))
            return

        side_cfg = {k: v.get() for k, v in self.side_vars.items()}
        self._log(f"Side types: {side_cfg}, WWR={wwr}\n")

        count_ok = 0
        for fname in self.files:
            in_path = os.path.join(self.folder, fname)
            try:
                with open(in_path, "r", encoding="utf-8") as f:
                    plan = json.load(f)
            except Exception as e:
                self._log(f"ERROR reading {fname}: {e}\n")
                continue

            try:
                graph = _build_graph_from_plan(plan, side_cfg, wwr)
            except Exception as e:
                self._log(f"ERROR converting {fname}: {e}\n")
                continue

            out_name = os.path.splitext(fname)[0] + "_graph.json"
            out_path = os.path.join(self.folder, out_name)
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(graph, f, indent=2)
                self._log(f"Written: {out_name}\n")
                count_ok += 1
            except Exception as e:
                self._log(f"ERROR writing {out_name}: {e}\n")

        self._log(f"\nDone. {count_ok} / {len(self.files)} files converted.\n")

    def _log(self, msg: str):
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", msg)
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    app = GraphPrepGUI()
    app.mainloop()
