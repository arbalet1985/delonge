from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class TriangulationData:
    x: np.ndarray
    y: np.ndarray
    triangles: np.ndarray


def build_triangulation(x: np.ndarray, y: np.ndarray) -> TriangulationData:
    points = np.column_stack([x, y])
    delaunay = Delaunay(points)
    return TriangulationData(x=x, y=y, triangles=delaunay.simplices)


def interpolate_to_grid(
    triangulation: TriangulationData,
    values: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi = np.linspace(float(np.min(triangulation.x)), float(np.max(triangulation.x)), grid_size)
    yi = np.linspace(float(np.min(triangulation.y)), float(np.max(triangulation.y)), grid_size)
    xg, yg = np.meshgrid(xi, yi)

    points = np.column_stack([triangulation.x, triangulation.y])
    interpolator = LinearNDInterpolator(points, values)
    zg = interpolator(xg, yg)
    zg_data = np.asarray(zg, dtype=float)
    return xg, yg, zg_data


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
