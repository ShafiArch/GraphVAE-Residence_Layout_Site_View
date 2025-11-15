#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResPlan Axes Viewer — with live sliders
- Grid grey slider (0–255)
- Pixel scale slider (1.00–10.00 inches per pixel)
- Everything updates dynamically
- Secondary axes in feet (auto updates with scale)
- Minor (sub) grids
- Opening colors configurable via constants
"""

import os, json, math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# plotting
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# geometry
from shapely.geometry import shape as shp_shape, Polygon, MultiPolygon
from shapely.ops import unary_union

# ──────────────────────────────────────────────────────────────────────────────
# PALETTES (Pastel default)
PALETTE_PASTEL = ["#d3c18d", "#ffffb3", "#bebada", "#fb8072", "#a0356e",
                  "#fdb462", "#de8069", "#fccde5", "#d9d9d9", "#bc80bd"]
PALETTE_MUTED  = ["#c4840e", "#ffd92f", "#8da0cb", "#e78ac3", "#c63800",
                  "#fc8d62","#e5c494","#b3b3b3","#9e1b4b","#7570b3"]
PALETTE_TAB10  = ["#b199b0","#572F3E","#a02c51","#996e6e","#9467bd",
                  "#8c564b","#c5aabd","#cacaca","#ff825d","#eab189"]  # fix invalid 8-digit hex

REDDISH_LIGHT = ["#8B5E3C","#FFB347","#FF6B6B","#C39BD3","#FFE066"]
REDDISH_DARK  = ["#5A3C27","#D97A00","#CC4C4C","#8E5AA8","#C9B400"]
FALLBACK_GREY = "#CFCFCF"

ROOM_TYPES_ORDER = ["living","living_room","bedroom","bathroom","kitchen","storage",
                    "stair","balcony","veranda","garden","parking","land"]
NON_ROOMS = {"wall","window","door","front_door","neighbor","inner","land"}

# Use Pastel for matplotlib's default cycle so anything uncolored still matches
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PALETTE_PASTEL)

# ──────────────────────────────────────────────────────────────────────────────
# OPENING COLORS (edit these to control window/door colors)
WINDOW_FACE = "#6b6b6b"   # ← your previous green; change to what you want
WINDOW_EDGE = "#3b3b3b"
DOOR_FACE   = "#C5C5C5"
DOOR_EDGE   = "#505050"
FRONT_DOOR_FACE = "#616161"
FRONT_DOOR_EDGE = "#000000"
# Example alternatives:
# WINDOW_FACE = "#9ad0f5"; WINDOW_EDGE = "#3876a4"     # bluish windows
# WINDOW_FACE = "#c7ffd8"; WINDOW_EDGE = "#3aa774"     # pale mint

# ──────────────────────────────────────────────────────────────────────────────

def palette_map(palette_name: str, present_types):
    pts = list(present_types) if present_types else ROOM_TYPES_ORDER
    if palette_name == "Reddish (Light)":
        base = REDDISH_LIGHT
    elif palette_name == "Reddish (Dark)":
        base = REDDISH_DARK
    elif palette_name == "Muted":
        base = PALETTE_MUTED
    elif palette_name == "Tab10":
        base = PALETTE_TAB10
    elif palette_name == "Monochrome":
        return {rt: "#C7C7C7" for rt in pts}
    elif palette_name == "Grayscale":
        import numpy as np, matplotlib.cm as cm
        shades = [cm.Greys(v) for v in np.linspace(0.3, 0.85, max(1,len(pts)))]
        def rgba2hex(rgba):
            r,g,b,_ = rgba; return "#{:02x}{:02x}{:02x}".format(int(r*255),int(g*255),int(b*255))
        cols = [rgba2hex(s) for s in shades]
        return {rt: cols[i % len(cols)] for i, rt in enumerate(pts)}
    else:
        base = PALETTE_TAB10
    mapping, pri = {}, ["living_room","kitchen","bedroom","bathroom","closet"]
    for i, rt in enumerate(pri):
        if rt in pts: mapping[rt] = base[i % len(base)]
    idx = 0
    for rt in pts:
        if rt not in mapping:
            mapping[rt] = base[idx % len(base)] if idx < len(base) else FALLBACK_GREY
            idx += 1
    return mapping




def _safe_iter_geoms(g):
    if isinstance(g, Polygon):
        yield g
    elif isinstance(g, MultiPolygon):
        for p in g.geoms:
            yield p




def _collect_layers(plan):
    layers = {}
    for k, v in plan.items():
        if isinstance(v, dict) and "type" in v and "coordinates" in v:
            try:
                g = shp_shape(v)
                if isinstance(g, (Polygon, MultiPolygon)):
                    layers[k] = g
            except Exception:
                pass
    return layers




def _bbox_size(layers):
    if not layers: return (0,0,256,256)
    try:
        u = unary_union([g for g in layers.values()])
        return u.bounds
    except Exception:
        return (0,0,256,256)




def grey_hex(val_0_255: int) -> str:
    v = max(0, min(255, int(round(val_0_255))))
    return "#{:02x}{:02x}{:02x}".format(v, v, v)




class AxesViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ResPlan Axes Viewer (sliders)")
        self.geometry("1250x900")
        self.folder = None
        self.files = []
        self.plan = None
        self.layers = {}
        self.palette_name = tk.StringVar(value="Pastel")

        # dynamic units
        self.in_per_px = tk.DoubleVar(value=2.4)   # inches per pixel (1.00–10.00)
        self.grid_grey = tk.DoubleVar(value=220.0) # 0–255
        self.show_total_area = tk.BooleanVar(value=True)

        self._build_topbar()
        self._build_controls()
        self._build_body()




    def _build_topbar(self):
        frm = ttk.Frame(self); frm.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10,4))
        ttk.Button(frm, text="Open Folder…", command=self.choose_folder).pack(side=tk.LEFT)
        self.lbl_folder = ttk.Label(frm, text="No folder selected")
        self.lbl_folder.pack(side=tk.LEFT, padx=12)

        ttk.Label(frm, text="Palette:").pack(side=tk.LEFT, padx=(20,6))
        cb = ttk.Combobox(frm, state="readonly", width=18,
                          values=["Pastel","Muted","Tab10","Reddish (Light)","Reddish (Dark)","Monochrome","Grayscale"],
                          textvariable=self.palette_name)
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda e: self.render())


        ttk.Checkbutton(
            frm, text="Total area",
            variable=self.show_total_area,
            command=self.render
        ).pack(side=tk.LEFT, padx=(12,0))

        ttk.Button(frm, text="Save PNG…", command=self.save_png).pack(side=tk.RIGHT)



    def _build_controls(self):
        ctl = ttk.Frame(self); ctl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0,8))

        # Grid grey slider
        row1 = ttk.Frame(ctl); row1.pack(fill=tk.X, pady=4)
        ttk.Label(row1, text="Grid Grey").pack(side=tk.LEFT)
        self.lbl_grey_val = ttk.Label(row1, text=f"{int(self.grid_grey.get())}")
        self.lbl_grey_val.pack(side=tk.RIGHT)
        s1 = ttk.Scale(row1, from_=0.0, to=255.0, orient=tk.HORIZONTAL, variable=self.grid_grey,
                       command=self._on_slider_change)
        s1.pack(fill=tk.X, expand=True, padx=10)

        # Pixel scale slider (inches per pixel)
        row2 = ttk.Frame(ctl); row2.pack(fill=tk.X, pady=4)
        ttk.Label(row2, text="Pixel Scale (inches per pixel)").pack(side=tk.LEFT)
        self.lbl_scale_val = ttk.Label(row2, text=f"{self.in_per_px.get():.2f} in/px")
        self.lbl_scale_val.pack(side=tk.RIGHT)
        s2 = ttk.Scale(row2, from_=1.00, to=10.00, orient=tk.HORIZONTAL, variable=self.in_per_px,
                       command=self._on_slider_change)
        s2.pack(fill=tk.X, expand=True, padx=10)




    def _build_body(self):
        body = ttk.Frame(self); body.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        # Left listbox
        left = ttk.Frame(body, width=300); left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="Plans").pack(anchor="w")
        self.lst = tk.Listbox(left, selectmode=tk.SINGLE); self.lst.pack(fill=tk.BOTH, expand=True)
        self.lst.bind("<<ListboxSelect>>", lambda e: self.on_file_select())

        # Right plot
        right = ttk.Frame(body); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12,0))
        self.fig = plt.Figure(figsize=(8.0,7.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)





    # --- Events ---
    def _on_slider_change(self, *_):
        self.lbl_grey_val.config(text=f"{int(self.grid_grey.get())}")
        self.lbl_scale_val.config(text=f"{self.in_per_px.get():.2f} in/px")
        self.render()




    # --- IO ---
    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select folder with plan_*.json")
        if not folder: return
        self.folder = folder
        self.lbl_folder.config(text=folder)
        self.files = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
        self.files.sort()
        self.lst.delete(0, tk.END)
        for f in self.files: self.lst.insert(tk.END, f)
        if self.files:
            self.lst.selection_set(0)
            self.on_file_select()




    def on_file_select(self):
        if not self.folder: return
        sel = self.lst.curselection()
        if not sel: return
        fname = self.files[sel[0]]
        path = os.path.join(self.folder, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.plan = json.load(f)
        except Exception as e:
            messagebox.showerror("Read error", f"{fname}\n{e}")
            return
        self.layers = self._collect_layers(self.plan)
        self.render()




    # --- Geometry helpers ---
    def _collect_layers(self, plan):
        return _collect_layers(plan)

    def _bbox_size(self, layers):
        return _bbox_size(layers)

    def _draw_polycollection(self, ax, geom, face, edge="#333333", alpha=0.75, lw=0.8):
        from matplotlib.patches import Polygon as MplPoly
        from matplotlib.collections import PatchCollection
        patches = []
        for p in _safe_iter_geoms(geom):
            patches.append(MplPoly(list(p.exterior.coords), closed=True))
        if patches:
            pc = PatchCollection(patches, facecolor=face, edgecolor=edge, linewidths=lw, alpha=alpha)
            ax.add_collection(pc)




    def _label_room_pieces(self, ax, name, geom, color):
        """Label each polygon piece separately; show true per-piece W×H from its bbox.
        Also compute total area across all pieces for a single small header label."""
        try:
            ft_per_px = self.in_per_px.get() / 12.0
            sqft_per_sqpx = ft_per_px * ft_per_px

            pieces = list(_safe_iter_geoms(geom))
            if not pieces:
                return

            # Total area across all pieces (shown once, near the largest piece)
            total_area_ft2 = sum(p.area for p in pieces) * sqft_per_sqpx
            largest = max(pieces, key=lambda p: p.area)

            # Label each piece with its own W×H and area
            for idx, p in enumerate(pieces, 1):
                minx, miny, maxx, maxy = p.bounds
                w_ft = (maxx - minx) * ft_per_px
                h_ft = (maxy - miny) * ft_per_px
                piece_area_ft2 = p.area * sqft_per_sqpx

                rp = p.representative_point()
                x, y = rp.x, rp.y

                # If there are multiple pieces, append -1, -2 … to the name
                piece_name = f"{name}-{idx}" if len(pieces) > 1 else name

                txt = f"{piece_name}\n{w_ft:.1f}×{h_ft:.1f} ft\n{piece_area_ft2:.1f} ft²"
                ax.text(
                    x, y, txt,
                    ha="center", va="center", fontsize=9, color="#111111",
                    bbox=dict(
                        boxstyle="round,pad=0.25", facecolor="white",
                        edgecolor=color, linewidth=0.6, alpha=0.85
                    ),
                    zorder=10, clip_on=False
                )

            # Put a small "total" tag at the largest piece so you see combined area once
            if self.show_total_area.get():
                rpL = largest.representative_point()
                ax.text(
                    rpL.x, rpL.y, f"(total {total_area_ft2:.1f} ft²)",
                    ha="center", va="bottom", fontsize=8, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                            edgecolor=color, linewidth=0.4, alpha=0.75),
                    zorder=11, clip_on=False
                )
        except Exception:
            pass





    # --- Render ---
    def render(self):
        self.ax.clear()

        # grid color from slider (major)
        grid_color = grey_hex(self.grid_grey.get())
        self.ax.grid(True, which="major", linestyle="-", linewidth=0.3, color=grid_color)

        # minor (sub) grids — thinner & lighter
        from matplotlib.ticker import AutoMinorLocator
        self.ax.minorticks_on()
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # 4 minor intervals → 5 ticks per major
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.grid(True, which="minor", linestyle="-", linewidth=0.15, alpha=0.35, color=grid_color)

        self.ax.set_axis_on()

        if not self.layers:
            self.canvas.draw(); return

        # Prepare color map
        room_layers = [k for k in self.layers if k not in NON_ROOMS]
        cmap = palette_map("Pastel", room_layers) if self.palette_name.get() == "Pastel" \
               else palette_map(self.palette_name.get(), room_layers)

        # Draw walls
        if "wall" in self.layers:
            self._draw_polycollection(self.ax, self.layers["wall"], face="#000000", edge="#000000", alpha=0.5, lw=1.0)

        # Draw rooms + labels
        for k in room_layers:
            col = cmap.get(k, FALLBACK_GREY)
            self._draw_polycollection(self.ax, self.layers[k], face=col, edge="#333333", alpha=0.65, lw=0.8)
            self._label_room_pieces(self.ax, k, self.layers[k], col)

        # Openings (now controlled by constants at top)
        if "window" in self.layers:
            self._draw_polycollection(self.ax, self.layers["window"], face=WINDOW_FACE, edge=WINDOW_EDGE, alpha=0.9, lw=0.6)
        if "door" in self.layers:
            self._draw_polycollection(self.ax, self.layers["door"], face=DOOR_FACE, edge=DOOR_EDGE, alpha=0.9, lw=0.6)
        if "front_door" in self.layers:
            self._draw_polycollection(self.ax, self.layers["front_door"], face=FRONT_DOOR_FACE, edge=FRONT_DOOR_EDGE, alpha=1.0, lw=1.2)

        # Axis labels with dynamic scale
        in_per_px = self.in_per_px.get()
        ft_per_px = in_per_px / 12.0
        self.ax.set_xlabel(f"X (px)   —   1 px = {in_per_px:.2f} in = {ft_per_px:.3f} ft")
        self.ax.set_ylabel("Y (px)")

        # Secondary axes in feet (top & right) — auto update each render
        def px_to_ft(x): return x * ft_per_px
        def ft_to_px(x): return x / ft_per_px
        secx = self.ax.secondary_xaxis('top', functions=(px_to_ft, ft_to_px))
        secy = self.ax.secondary_yaxis('right', functions=(px_to_ft, ft_to_px))
        secx.set_xlabel("X (ft)")
        secy.set_ylabel("Y (ft)")

        # Limits & aspect
        minx, miny, maxx, maxy = self._bbox_size(self.layers)
        pad = 10
        self.ax.set_xlim(minx - pad, maxx + pad)
        self.ax.set_ylim(miny - pad, maxy + pad)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Floor Plan (Axes, Grid and Labels)")

        self.canvas.draw()



    def save_png(self):
        if not self.layers:
            messagebox.showwarning("Nothing to save", "Open a plan first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".png",
                                           initialfile="plan.png",
                                           filetypes=[("PNG image","*.png")])
        if not out: return
        try:
            self.fig.savefig(out, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Saved PNG to:\n{out}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

if __name__ == "__main__":
    AxesViewer().mainloop()
