from __future__ import annotations

import numpy as np
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay


def build_triangulation(x: np.ndarray, y: np.ndarray) -> Triangulation:
    points = np.column_stack([x, y])
    delaunay = Delaunay(points)
    return Triangulation(x, y, triangles=delaunay.simplices)


def mirror_fields(ap: np.ndarray, ac: np.ndarray, enforce_mirror: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if not enforce_mirror:
        return ap, ac

    def _normalize(values: np.ndarray) -> np.ndarray:
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        span = v_max - v_min
        if span == 0:
            return np.zeros_like(values, dtype=float)
        return (values - v_min) / span

    ap_norm = _normalize(ap)
    ac_norm = _normalize(ac)

    # Base field merges both channels and enforces mirrored relation for contours.
    base = (ap_norm + (1.0 - ac_norm)) / 2.0

    common_min = float(min(np.min(ap), np.min(ac)))
    common_max = float(max(np.max(ap), np.max(ac)))
    span = common_max - common_min

    ap_plot = common_min + base * span
    ac_plot = common_min + (1.0 - base) * span
    return ap_plot, ac_plot


def build_levels(ap_plot: np.ndarray, ac_plot: np.ndarray, levels_count: int) -> tuple[np.ndarray, float, float]:
    vmin = float(min(np.min(ap_plot), np.min(ac_plot)))
    vmax = float(max(np.max(ap_plot), np.max(ac_plot)))
    if vmin == vmax:
        vmax = vmin + 1.0
    levels = np.linspace(vmin, vmax, levels_count)
    return levels, vmin, vmax
