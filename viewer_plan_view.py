#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viewer_plan_view_ctk.py  (all-white background)

Requires:
    pip install customtkinter shapely matplotlib
"""

import os, json
import customtkinter as ctk
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator

from shapely.geometry import shape as shp_shape, Polygon, MultiPolygon
from shapely.ops import unary_union

from view_calc import (
    OutsideNode,
    compute_grid_view,
    compute_room_scores,
    compute_graph_score,
)

PALETTE_PASTEL = [
    "#d3c18d", "#ffffb3", "#bebada", "#fb8072", "#a0356e",
    "#fdb462", "#de8069", "#fccde5", "#d9d9d9", "#bc80bd"
]
PALETTE_TAB10 = [
    "#b199b0", "#572F3E", "#a02c51", "#996e6e", "#9467bd",
    "#8c564b", "#c5aabd", "#cacaca", "#ff825d", "#eab189"
]
PALETTE_MUTED = [
    "#c4840e", "#ffd92f", "#8da0cb", "#e78ac3", "#c63800",
    "#fc8d62", "#e5c494", "#b3b3b3", "#9e1b4b", "#7570b3"
]
REDDISH_LIGHT = ["#8B5E3C", "#FFB347", "#FF6B6B", "#C39BD3", "#FFE066"]
REDDISH_DARK  = ["#5A3C27", "#D97A00", "#CC4C4C", "#8E5AA8", "#C9B400"]
FALLBACK_GREY = "#CFCFCF"

NON_ROOMS = {"wall", "window", "door", "front_door", "neighbor", "inner", "land"}

WINDOW_FACE = "#6b6b6b"
WINDOW_EDGE = "#3b3b3b"
DOOR_FACE   = "#C5C5C5"
DOOR_EDGE   = "#505050"
FRONT_DOOR_FACE = "#616161"
FRONT_DOOR_EDGE = "#000000"

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PALETTE_PASTEL)


def palette_map(name: str, present_types):
    pts = list(present_types) if present_types else ["room"]
    if name == "Muted":
        base = PALETTE_MUTED
    elif name == "Reddish (Light)":
        base = REDDISH_LIGHT
    elif name == "Reddish (Dark)":
        base = REDDISH_DARK
    else:
        base = PALETTE_TAB10
    mapping = {}
    for i, rt in enumerate(pts):
        mapping[rt] = base[i % len(base)] if i < len(base) else FALLBACK_GREY
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
    if not layers:
        return (0, 0, 256, 256)
    try:
        u = unary_union([g for g in layers.values()])
        return u.bounds
    except Exception:
        return (0, 0, 256, 256)


def grey_hex(v):
    v = max(0, min(255, int(round(v))))
    return "#{:02x}{:02x}{:02x}".format(v, v, v)


class Viewer(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title("ResPlan – Modern View Quality Viewer")
        self.geometry("1400x900")
        self.configure(fg_color="white")  # root background

        # ttk Treeview style → white
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                        background="white",
                        fieldbackground="white",
                        foreground="black")
        style.configure("Treeview.Heading",
                        background="white",
                        foreground="black")

        # data
        self.folder = None
        self.files = []
        self.plan = None
        self.layers = {}
        self.palette_name = tk.StringVar(value="Pastel")

        self.grid_grey = tk.DoubleVar(value=220.0)
        self.in_per_px = tk.DoubleVar(value=2.4)
        self.cmin = tk.DoubleVar(value=0.0)
        self.cmax = tk.DoubleVar(value=1.0)
        self.grid_step = tk.DoubleVar(value=10.0)
        self.cell_scale = tk.DoubleVar(value=0.4)
        self.show_total_area = tk.BooleanVar(value=True)

        self.view_samples = []
        self.room_scores = {}
        self.view_avg = 0.0

        self._build_top()
        self._build_sliders()
        self._build_tabs()

    # ───────────────────────────────── UI BUILDING

    def _build_top(self):
        top = ctk.CTkFrame(self, corner_radius=0, fg_color="white")
        top.pack(side="top", fill="x", padx=10, pady=(8, 4))

        ctk.CTkButton(
            top, text="Open folder…", width=120,
            command=self.choose_folder, fg_color="#1f6aa5"
        ).pack(side="left", padx=(4, 8))

        self.lbl_folder = ctk.CTkLabel(
            top, text="No folder selected", anchor="w", fg_color="white")
        self.lbl_folder.pack(side="left", padx=4)

        ctk.CTkLabel(top, text="Palette:", fg_color="white").pack(
            side="left", padx=(20, 4))
        self.cmb_palette = ctk.CTkComboBox(
            top, width=140, values=["Pastel", "Muted", "Tab10",
                                    "Reddish (Light)", "Reddish (Dark)"],
            command=lambda _: (self.render_plan(), self.render_view()),
            fg_color="white", button_color="#1f6aa5",
            border_color="#cccccc")
        self.cmb_palette.set("Pastel")
        self.cmb_palette.pack(side="left")

        self.chk_total = ctk.CTkCheckBox(
            top, text="Total area", variable=self.show_total_area,
            command=lambda: (self.render_plan(), self.render_view()),
            fg_color="white")
        self.chk_total.pack(side="left", padx=(20, 4))

        ctk.CTkButton(
            top, text="Save Plan PNG", width=120,
            command=self.save_plan_png, fg_color="#1f6aa5"
        ).pack(side="right", padx=4)
        ctk.CTkButton(
            top, text="Save View PNG", width=120,
            command=self.save_view_png, fg_color="#1f6aa5"
        ).pack(side="right", padx=4)

    def _build_sliders(self):
        bar = ctk.CTkFrame(self, corner_radius=0, fg_color="white")
        bar.pack(side="top", fill="x", padx=10, pady=(0, 4))

        def add_slider(label, var, from_, to_, callback, fmt=None):
            frame = ctk.CTkFrame(bar, corner_radius=0, fg_color="white")
            frame.pack(fill="x", pady=2)
            ctk.CTkLabel(frame, text=label, width=160,
                         anchor="w", fg_color="white").pack(
                side="left", padx=(2, 6))
            slider = ctk.CTkSlider(
                frame, from_=from_, to=to_,
                variable=var, command=callback,
                fg_color="white", progress_color="#1f6aa5", button_color="#1f6aa5")
            slider.pack(side="left", fill="x", expand=True, padx=(0, 6))
            txt = fmt(var.get()) if fmt else f"{var.get():.2f}"
            lbl = ctk.CTkLabel(frame, text=txt, width=80,
                               anchor="e", fg_color="white")
            lbl.pack(side="right", padx=2)
            return lbl

        self.lbl_grey = add_slider(
            "Grid grey", self.grid_grey, 0, 255,
            self._on_basic_slider, fmt=lambda v: f"{int(v)}")
        self.lbl_scale = add_slider(
            "Pixel scale (in/px)", self.in_per_px, 1.0, 10.0,
            self._on_basic_slider, fmt=lambda v: f"{v:.2f} in/px")
        self.lbl_cmin = add_slider(
            "View color min", self.cmin, 0.0, 1.0,
            self._on_color_slider, fmt=lambda v: f"{v:.2f}")
        self.lbl_cmax = add_slider(
            "View color max", self.cmax, 0.0, 1.0,
            self._on_color_slider, fmt=lambda v: f"{v:.2f}")
        self.lbl_step = add_slider(
            "Grid spacing", self.grid_step, 5.0, 40.0,
            self._on_step_slider, fmt=lambda v: f"{v:.1f}")
        self.lbl_cell = add_slider(
            "Cell size factor", self.cell_scale, 0.2, 1.0,
            self._on_cell_slider, fmt=lambda v: f"{v:.2f}")

    def _build_tabs(self):
        tabs = ctk.CTkTabview(
            self,
            fg_color="white",
            segmented_button_fg_color="white",
            segmented_button_selected_color="#1f6aa5",
            segmented_button_unselected_color="#e0e0e0",
        )
        tabs.pack(fill="both", expand=True, padx=10, pady=6)

        tab_plan = tabs.add("Plan")
        tab_view = tabs.add("View Quality")
        tab_plan.configure(fg_color="white")
        tab_view.configure(fg_color="white")

        # Plan tab
        left = ctk.CTkFrame(tab_plan, fg_color="white")
        left.pack(side="left", fill="y", padx=(0, 8), pady=4)
        ctk.CTkLabel(left, text="Plans",
                     fg_color="white").pack(anchor="w", padx=4, pady=(4, 0))

        self.lst = tk.Listbox(left, height=20, bg="white", fg="black",
                              highlightthickness=0, bd=1)
        self.lst.pack(fill="both", expand=True, padx=4, pady=4)
        self.lst.bind("<<ListboxSelect>>", lambda e: self.on_file_select())

        right = ctk.CTkFrame(tab_plan, fg_color="white")
        right.pack(side="left", fill="both", expand=True, pady=4)

        self.fig_plan = plt.Figure(figsize=(7.5, 6.5),
                                   dpi=100, facecolor="white")
        self.ax_plan = self.fig_plan.add_subplot(111)
        self.canvas_plan = FigureCanvasTkAgg(self.fig_plan, master=right)
        self.canvas_plan.get_tk_widget().pack(fill="both", expand=True)

        # View tab
        top_v = ctk.CTkFrame(tab_view, fg_color="white")
        top_v.pack(side="top", fill="both", expand=True, pady=(4, 2))

        self.fig_view = plt.Figure(figsize=(7.5, 5.0),
                                   dpi=100, facecolor="white")
        self.ax_view = self.fig_view.add_subplot(111)
        self.canvas_view = FigureCanvasTkAgg(self.fig_view, master=top_v)
        self.canvas_view.get_tk_widget().pack(fill="both", expand=True)

        bottom_v = ctk.CTkFrame(tab_view, fg_color="white")
        bottom_v.pack(side="top", fill="both", expand=True, pady=(2, 4))

        self.tree = ttk.Treeview(
            bottom_v, columns=("id", "x", "y", "score"),
            show="headings", height=7, style="Treeview")
        for c, txt in zip(("id", "x", "y", "score"),
                          ("ID", "X", "Y", "Score (%)")):
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=80, anchor="center")
        vsb = ttk.Scrollbar(
            bottom_v, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")

        self.lbl_avg = ctk.CTkLabel(
            bottom_v, text="Average view score: n/a", fg_color="white")
        self.lbl_avg.pack(side="left", padx=10, pady=4)

    # ────────────────────────────── SLIDER CALLBACKS

    def _on_basic_slider(self, *_):
        self.lbl_grey.configure(text=f"{int(self.grid_grey.get())}")
        self.lbl_scale.configure(
            text=f"{self.in_per_px.get():.2f} in/px"
        )
        self.render_plan()
        self.render_view()

    def _on_color_slider(self, *_):
        self.lbl_cmin.configure(text=f"{self.cmin.get():.2f}")
        self.lbl_cmax.configure(text=f"{self.cmax.get():.2f}")
        self.render_view()

    def _on_step_slider(self, *_):
        self.lbl_step.configure(text=f"{self.grid_step.get():.1f}")
        self.compute_view_grid()
        self.render_view()

    def _on_cell_slider(self, *_):
        self.lbl_cell.configure(text=f"{self.cell_scale.get():.2f}")
        self.render_view()

    # ────────────────────────────── IO

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select folder with JSON plans")
        if not folder:
            return
        self.folder = folder
        self.lbl_folder.configure(text=folder)
        self.files = [f for f in os.listdir(folder)
                      if f.lower().endswith(".json")]
        self.files.sort()
        self.lst.delete(0, tk.END)
        for f in self.files:
            self.lst.insert(tk.END, f)
        if self.files:
            self.lst.selection_set(0)
            self.on_file_select()

    def on_file_select(self):
        if not self.folder:
            return
        sel = self.lst.curselection()
        if not sel:
            return
        fname = self.files[sel[0]]
        path = os.path.join(self.folder, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.plan = json.load(f)
        except Exception as e:
            messagebox.showerror("Read error", f"{fname}\n{e}")
            return
        self.layers = _collect_layers(self.plan)
        self.compute_view_grid()
        self.render_plan()
        self.render_view()

    # ────────────────────────────── DRAW HELPERS

    def _draw_polycollection(self, ax, geom, face, edge="#333333",
                             alpha=0.75, lw=0.8):
        from matplotlib.collections import PatchCollection
        patches = []
        for p in _safe_iter_geoms(geom):
            patches.append(mpatches.Polygon(
                list(p.exterior.coords), closed=True))
        if patches:
            pc = PatchCollection(
                patches, facecolor=face, edgecolor=edge,
                linewidths=lw, alpha=alpha
            )
            ax.add_collection(pc)

    def _label_room_pieces(self, ax, name, geom, color):
        try:
            ft_per_px = self.in_per_px.get() / 12.0
            sqft_per_sqpx = ft_per_px * ft_per_px
            pieces = list(_safe_iter_geoms(geom))
            if not pieces:
                return
            total_area_ft2 = sum(p.area for p in pieces) * sqft_per_sqpx
            largest = max(pieces, key=lambda p: p.area)

            for idx, p in enumerate(pieces, 1):
                minx, miny, maxx, maxy = p.bounds
                w_ft = (maxx - minx) * ft_per_px
                h_ft = (maxy - miny) * ft_per_px
                piece_area_ft2 = p.area * sqft_per_sqpx
                rp = p.representative_point()
                x, y = rp.x, rp.y
                name_i = f"{name}-{idx}" if len(pieces) > 1 else name
                txt = f"{name_i}\n{w_ft:.1f}×{h_ft:.1f} ft\n{piece_area_ft2:.1f} ft²"
                ax.text(
                    x, y, txt,
                    ha="center", va="center", fontsize=9, color="#111111",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor=color, linewidth=0.6, alpha=0.85
                    ),
                    zorder=10, clip_on=False
                )

            if self.show_total_area.get():
                rpL = largest.representative_point()
                ax.text(
                    rpL.x, rpL.y, f"(total {total_area_ft2:.1f} ft²)",
                    ha="center", va="bottom", fontsize=8, color="#333333",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor=color, linewidth=0.4, alpha=0.75),
                    zorder=11, clip_on=False
                )
        except Exception:
            pass

    # ────────────────────────────── PLAN RENDER

    def render_plan(self):
        ax = self.ax_plan
        ax.clear()
        ax.set_facecolor("white")

        grid_color = grey_hex(self.grid_grey.get())
        ax.grid(True, which="major",
                linestyle="-", linewidth=0.3, color=grid_color)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(True, which="minor",
                linestyle="-", linewidth=0.15,
                alpha=0.35, color=grid_color)

        if not self.layers:
            self.canvas_plan.draw()
            return

        room_layers = [k for k in self.layers if k not in NON_ROOMS]
        cmap = palette_map(self.cmb_palette.get(), room_layers)

        if "wall" in self.layers:
            self._draw_polycollection(
                ax, self.layers["wall"],
                face="#000000", edge="#000000",
                alpha=0.5, lw=1.0
            )

        for k in room_layers:
            col = cmap.get(k, FALLBACK_GREY)
            self._draw_polycollection(
                ax, self.layers[k],
                face=col, edge="#333333",
                alpha=0.65, lw=0.8
            )
            self._label_room_pieces(ax, k, self.layers[k], col)

        if "window" in self.layers:
            self._draw_polycollection(
                ax, self.layers["window"],
                face=WINDOW_FACE, edge=WINDOW_EDGE,
                alpha=0.9, lw=0.6
            )
        if "door" in self.layers:
            self._draw_polycollection(
                ax, self.layers["door"],
                face=DOOR_FACE, edge=DOOR_EDGE,
                alpha=0.9, lw=0.6
            )
        if "front_door" in self.layers:
            self._draw_polycollection(
                ax, self.layers["front_door"],
                face=FRONT_DOOR_FACE, edge=FRONT_DOOR_EDGE,
                alpha=1.0, lw=1.2
            )

        in_per_px = self.in_per_px.get()
        ft_per_px = in_per_px / 12.0
        ax.set_xlabel(
            f"X (px)   —   1 px = {in_per_px:.2f} in = {ft_per_px:.3f} ft"
        )
        ax.set_ylabel("Y (px)")

        def px_to_ft(x): return x * ft_per_px
        def ft_to_px(x): return x / ft_per_px
        secx = ax.secondary_xaxis("top", functions=(px_to_ft, ft_to_px))
        secy = ax.secondary_yaxis("right", functions=(px_to_ft, ft_to_px))
        secx.set_xlabel("X (ft)")
        secy.set_ylabel("Y (ft)")

        minx, miny, maxx, maxy = _bbox_size(self.layers)
        pad = 10
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Floor Plan")

        self.canvas_plan.draw()

    # ────────────────────────────── VIEW CALC + RENDER

    def compute_view_grid(self):
        self.view_samples = []
        self.room_scores = {}
        self.view_avg = 0.0

        if not self.layers:
            return

        minx, miny, maxx, maxy = _bbox_size(self.layers)
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)

        outside_nodes = [
            OutsideNode("north", "sky",        (cx, maxy + 1)),
            OutsideNode("south", "vegetation", (cx, miny - 1)),
            OutsideNode("west",  "context",    (minx - 1, cy)),
            OutsideNode("east",  "building",   (maxx + 1, cy)),
        ]
        step = max(1.0, float(self.grid_step.get()))
        samples = compute_grid_view(self.layers, outside_nodes, step)
        self.view_samples = [
            {"id": s.id, "x": s.x, "y": s.y, "v": s.v_norm} for s in samples
        ]
        room_layer_names = [k for k in self.layers if k not in NON_ROOMS]
        self.room_scores = compute_room_scores(
            self.layers, samples, room_layer_names
        )
        self.view_avg = compute_graph_score(self.room_scores)

    def render_view(self):
        ax = self.ax_view
        ax.clear()
        ax.set_facecolor("white")

        grid_color = grey_hex(self.grid_grey.get())
        ax.grid(True, which="major",
                linestyle="-", linewidth=0.3, color=grid_color)

        if not self.layers:
            ax.set_title("No plan loaded")
            self.canvas_view.draw()
            self.update_table()
            return

        if "inner" in self.layers:
            self._draw_polycollection(
                ax, self.layers["inner"],
                face="#ffffff", edge="#000000",
                alpha=0.1, lw=1.0
            )
        if "window" in self.layers:
            self._draw_polycollection(
                ax, self.layers["window"],
                face=WINDOW_FACE, edge=WINDOW_EDGE,
                alpha=0.9, lw=0.6
            )

        if self.view_samples:
            step = max(1.0, float(self.grid_step.get()))
            half = step * float(self.cell_scale.get())
            cmap = cm.get_cmap("jet")
            vmin = self.cmin.get()
            vmax = self.cmax.get()
            if vmax <= vmin + 1e-6:
                vmax = vmin + 1e-6
            levels = 20
            for s in self.view_samples:
                v = s["v"]
                t = (v - vmin) / (vmax - vmin)
                t = max(0.0, min(1.0, t))
                k = round(t * (levels - 1))
                t_q = k / (levels - 1)
                r, g, b, _ = cmap(t_q)
                color = "#{:02x}{:02x}{:02x}".format(
                    int(r * 255), int(g * 255), int(b * 255)
                )
                ax.add_patch(
                    mpatches.Rectangle(
                        (s["x"] - half, s["y"] - half),
                        2 * half, 2 * half,
                        facecolor=color, edgecolor=None, alpha=0.9
                    )
                )

        minx, miny, maxx, maxy = _bbox_size(self.layers)
        pad = 10
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("View Quality Heatmap (0–100%)")

        self.canvas_view.draw()
        self.update_table()

    def update_table(self):
        for r in self.tree.get_children():
            self.tree.delete(r)
        for s in self.view_samples:
            self.tree.insert(
                "", "end",
                values=(
                    s["id"],
                    f"{s['x']:.1f}",
                    f"{s['y']:.1f}",
                    f"{100.0 * s['v']:.1f}",
                ),
            )
        if self.view_samples:
            self.lbl_avg.configure(
                text=f"Average view score: {100.0 * self.view_avg:.1f}%"
            )
        else:
            self.lbl_avg.configure(text="Average view score: n/a")

    # ────────────────────────────── SAVE

    def save_plan_png(self):
        if not self.layers:
            messagebox.showwarning("Nothing to save", "Open a plan first.")
            return
        out = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile="plan.png",
            filetypes=[("PNG image", "*.png")])
        if out:
            self.fig_plan.savefig(out, dpi=150, bbox_inches="tight")

    def save_view_png(self):
        if not self.layers:
            messagebox.showwarning("Nothing to save", "Open a plan first.")
            return
        out = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile="view_quality.png",
            filetypes=[("PNG image", "*.png")])
        if out:
            self.fig_view.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    app = Viewer()
    app.mainloop()
