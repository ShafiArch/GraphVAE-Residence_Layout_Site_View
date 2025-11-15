import json
import math
import tkinter as tk

# ---------- CONFIG ----------
JSON_FILE = "plan_00038.json"
GRID_STEP = 10.0      # world units between sample points (smaller = finer grid)
CANVAS_W = 900
CANVAS_H = 700
# ----------------------------

# --- Geometry helpers ---

def extract_polygons(geom_obj):
    """
    Climate-style JSON stores coordinates like:
    "coordinates": [ [ [ [x,y], ... ] ], [ [ [x,y], ... ] ], ... ]
    We just want the outer ring of each polygon.
    """
    polys = []
    for poly in geom_obj["coordinates"]:
        ring = poly[0]   # first ring
        polys.append(ring)
    return polys

def polygon_area_and_centroid(poly):
    """
    Shoelace formula for area & centroid of a 2D polygon.
    poly = list of [x,y]; last point may or may not repeat first.
    Returns (area, (cx, cy))
    """
    area = 0.0
    cx = 0.0
    cy = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    area *= 0.5
    if area == 0:
        # fallback: just return first point as "centroid"
        return 0.0, poly[0]
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return abs(area), (cx, cy)

def point_in_polygon(x, y, poly):
    """
    Ray casting algorithm for point-in-polygon.
    poly is list of [px, py].
    """
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # check if edge crosses horizontal ray to +infinity
        if ((y1 > y) != (y2 > y)):
            t = (y - y1) / (y2 - y1)
            x_cross = x1 + t * (x2 - x1)
            if x_cross > x:
                inside = not inside
    return inside

# --- Load JSON and precompute things ---

with open(JSON_FILE, "r") as f:
    plan = json.load(f)

inner_polys = extract_polygons(plan["inner"])
window_polys = extract_polygons(plan["window"])

# We'll just use the first inner polygon as the "floor area"
inner_poly = inner_polys[0]

# Build window list with area + center
windows = []
for poly in window_polys:
    area, center = polygon_area_and_centroid(poly)
    windows.append({"area": area, "center": center})

# Compute bounding box of inner area for grid + drawing
xs = [p[0] for p in inner_poly]
ys = [p[1] for p in inner_poly]
min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)

width_world = max_x - min_x
height_world = max_y - min_y

# scale to canvas (keep aspect ratio, leave margin)
scale = 0.9 * min(CANVAS_W / width_world, CANVAS_H / height_world)
offset_x = (CANVAS_W - width_world * scale) / 2.0
offset_y = (CANVAS_H - height_world * scale) / 2.0

def world_to_screen(x, y):
    """
    Convert plan coordinates to Tkinter canvas coordinates.
    We also flip Y so the plan isn't upside down.
    """
    sx = offset_x + (x - min_x) * scale
    sy = CANVAS_H - (offset_y + (y - min_y) * scale)
    return sx, sy

# --- Sample grid + compute view index ---

samples = []   # list of (px, py, V_raw)

y = min_y
while y <= max_y:
    x = min_x
    while x <= max_x:
        # only consider points inside the inner polygon
        if point_in_polygon(x, y, inner_poly):
            # simple view index based on sum(area / d^2)
            v_raw = 0.0
            for w in windows:
                area = w["area"]
                cx, cy = w["center"]
                dx = cx - x
                dy = cy - y
                d2 = dx * dx + dy * dy
                if d2 < 1e-6:
                    continue
                v_raw += area / d2
            samples.append((x, y, v_raw))
        x += GRID_STEP
    y += GRID_STEP

# Normalize V to [0, 1]
if samples:
    max_v = max(v for _, _, v in samples)
else:
    max_v = 1.0

norm_samples = []
for x, y, v in samples:
    if max_v > 0:
        vn = v / max_v
    else:
        vn = 0.0
    norm_samples.append((x, y, vn))

print(f"Number of sample points: {len(norm_samples)}")
print("First 10 samples (x, y, V):")
for s in norm_samples[:10]:
    print(s)

# --- Tkinter visualization ---

root = tk.Tk()
root.title("Simple View Index Heatmap")

canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg="white")
canvas.pack()

# Draw inner boundary
coords = []
for px, py in inner_poly:
    sx, sy = world_to_screen(px, py)
    coords.extend([sx, sy])
canvas.create_polygon(*coords, outline="black", fill="", width=2)

# Draw windows for reference (in blue)
for poly in window_polys:
    coords = []
    for px, py in poly:
        sx, sy = world_to_screen(px, py)
        coords.extend([sx, sy])
    canvas.create_polygon(*coords, outline="blue", fill="lightblue")

# Draw heatmap squares for each sample
half = GRID_STEP * scale * 0.4  # visual size of each tile in screen units

for x, y, vn in norm_samples:
    sx, sy = world_to_screen(x, y)
    # simple color map: black (0) to red (1)
    c = int(255 * vn)
    color = f"#{c:02x}0000"
    canvas.create_rectangle(
        sx - half, sy - half, sx + half, sy + half,
        fill=color, outline=""
    )

# Optional: show value on click
def on_click(event):
    # find nearest sample and print its value
    if not norm_samples:
        return
    mx, my = event.x, event.y
    best = None
    best_d2 = None
    for x, y, vn in norm_samples:
        sx, sy = world_to_screen(x, y)
        d2 = (sx - mx) ** 2 + (sy - my) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best = (x, y, vn)
    if best:
        xw, yw, vval = best
        print(f"Closest sample at ({xw:.2f}, {yw:.2f}) → V = {vval:.3f}")

canvas.bind("<Button-1>", on_click)

root.mainloop()
