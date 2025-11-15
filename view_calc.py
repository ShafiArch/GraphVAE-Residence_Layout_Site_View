# view_calc.py
"""
View calculation module.

Responsibility:
- Take geometric layers (shapely) + outside-node config
- Sample a grid inside the inner area
- Compute view score at each grid point
- Aggregate per-room and whole-graph scores

This file is PURE calculation. No Tkinter, no Matplotlib.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

# ──────────────────────────────────────────────────────────────
# Data structures

@dataclass
class OutsideNode:
    name: str                 # "north", "south", ...
    category: str             # "sky", "vegetation", "context", "building", "balcony"
    position: Tuple[float,float]  # (x, y) point somewhere on that side
    # you can add more later (solid angle, etc.)

@dataclass
class GridSample:
    id: int
    x: float
    y: float
    v_raw: float
    v_norm: float

@dataclass
class RoomScore:
    room_name: str
    score: float              # 0–1
    area: float               # in plan units^2
    num_points: int

# ──────────────────────────────────────────────────────────────
# Weights for outside categories

VIEW_WEIGHTS = {
    "sky": 1.0,
    "vegetation": 0.8,
    "context": 0.6,
    "building": 0.4,
    "balcony": 0.2
}

# ──────────────────────────────────────────────────────────────

def _iter_polys(g):
    if isinstance(g, Polygon):
        yield g
    elif isinstance(g, MultiPolygon):
        for p in g.geoms:
            yield p

def compute_grid_view(
    layers: Dict[str, Polygon],
    outside_nodes: List[OutsideNode],
    grid_step: float
) -> List[GridSample]:
    """
    Returns a list of GridSample with v_norm in [0,1].

    layers: dict from your floorplan JSON (already shapely-ified)
      must contain "inner" and "window".
    outside_nodes: 4 nodes (N,S,E,W) with categories + positions.
    grid_step: spacing between grid centers (same as slider).
    """
    if "inner" not in layers or "window" not in layers:
        return []

    inner_union = unary_union([g for g in _iter_polys(layers["inner"])])
    window_geom = layers["window"]

    # For now we just use window centroid and area.
    windows = []
    for pg in _iter_polys(window_geom):
        area = pg.area
        c = pg.centroid
        windows.append((area, c.x, c.y))

    if not windows:
        return []

    # bounds of inner region
    minx, miny, maxx, maxy = inner_union.bounds

    samples_raw: List[Tuple[float,float,float]] = []
    y = miny
    while y <= maxy:
        x = minx
        while x <= maxx:
            pt = Point(x, y)
            if inner_union.contains(pt):
                v_raw = 0.0

                # base window-based part (area / d^2)
                for area, cx, cy in windows:
                    dx = cx - x
                    dy = cy - y
                    d2 = dx*dx + dy*dy
                    if d2 < 1e-6:
                        continue
                    v_raw += area / d2

                # extra “outside node” weight: assume each grid point
                # is influenced by all outside nodes; later you can
                # restrict it by direction.
                for on in outside_nodes:
                    w = VIEW_WEIGHTS.get(on.category, 0.0)
                    # simple distance weighting to outside-node position
                    dx = on.position[0] - x
                    dy = on.position[1] - y
                    d2 = dx*dx + dy*dy
                    if d2 < 1e-6:
                        continue
                    v_raw += w / d2

                samples_raw.append((x, y, v_raw))
            x += grid_step
        y += grid_step

    if not samples_raw:
        return []

    max_v = max(v for _,_,v in samples_raw) or 1.0
    min_v = min(v for _,_,v in samples_raw)

    out: List[GridSample] = []
    for i, (x, y, v) in enumerate(samples_raw, start=1):
        v_norm = (v - min_v) / (max_v - min_v) if max_v > min_v else 0.0
        out.append(GridSample(id=i, x=x, y=y, v_raw=v, v_norm=v_norm))

    return out

# ──────────────────────────────────────────────────────────────

def compute_room_scores(
    layers: Dict[str, Polygon],
    grid_samples: List[GridSample],
    room_layer_names: List[str]
) -> Dict[str, RoomScore]:
    """
    Aggregate grid samples into per-room scores.
    Assumes each room layer is one polygon or multipolygon.
    """
    # precompute union + point index
    room_scores: Dict[str, RoomScore] = {}
    # map from grid sample to which room it belongs
    from shapely.geometry import Point

    room_geoms = {name: layers[name] for name in room_layer_names if name in layers}

    for name, geom in room_geoms.items():
        polys = list(_iter_polys(geom))
        if not polys:
            continue
        # area
        total_area = sum(p.area for p in polys)
        total_v = 0.0
        count = 0
        for s in grid_samples:
            p = Point(s.x, s.y)
            if any(p.within(poly) for poly in polys):
                total_v += s.v_norm
                count += 1

        score = (total_v / count) if count > 0 else 0.0
        room_scores[name] = RoomScore(
            room_name=name,
            score=score,
            area=total_area,
            num_points=count
        )

    return room_scores

def compute_graph_score(room_scores: Dict[str, RoomScore]) -> float:
    """Area-weighted average of room scores (0–1)."""
    total_area = sum(r.area for r in room_scores.values())
    if total_area == 0:
        return 0.0
    acc = sum(r.score * r.area for r in room_scores.values())
    return acc / total_area
