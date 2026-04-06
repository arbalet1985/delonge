from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter

from matplotlib.tri import Triangulation

from core.interpolation import build_levels, interpolate_to_grid, mirror_fields


@dataclass(frozen=True)
class PlotParams:
    levels_count: int
    smooth_contours: bool
    smooth_sigma: float
    grid_size: int
    show_points: bool
    show_coordinates: bool
    show_rn_labels: bool
    point_size: int
    annotation_font_size: int
    show_scale_bar: bool
    show_contour_lines: bool
    axis_margin: float
    invert_x: bool
    invert_y: bool
    x_label: str
    y_label: str
    vertical_layout: bool


def _axis_range(values: np.ndarray, margin: float, invert: bool) -> list[float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin
    pad = span * float(np.clip(margin, 0.0, 0.3))
    lo = vmin - pad
    hi = vmax + pad
    return [hi, lo] if invert else [lo, hi]


def _smooth_grid(z: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return z
    valid = np.isfinite(z)
    safe = np.where(valid, z, 0.0)
    weights = valid.astype(float)
    safe_blur = gaussian_filter(safe, sigma=sigma, mode="nearest")
    weights_blur = gaussian_filter(weights, sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = safe_blur / np.where(weights_blur == 0, np.nan, weights_blur)
    return np.where(valid, smoothed, np.nan)


def _nice_scale_length_meters(target_m: float) -> int:
    target = max(float(target_m), 1.0)
    power = int(np.floor(np.log10(target)))
    nice = 10 ** power
    if nice > target:
        nice = max(1, nice // 10)
    return int(max(1, nice))


def _compute_scale_bar(tr: Triangulation) -> tuple[float, float, float, int] | None:
    x = tr.x
    y = tr.y
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 0 or y_span <= 0:
        return None

    x_is_geo = float(np.max(np.abs(x))) <= 180.0
    y_is_geo = float(np.max(np.abs(y))) <= 90.0
    if x_is_geo and y_is_geo:
        lat_mid = float(np.mean(y))
        meters_per_unit = 111_320.0 * np.cos(np.deg2rad(lat_mid))
        meters_per_unit = float(np.clip(meters_per_unit, 1_000.0, 111_320.0))
    else:
        meters_per_unit = 1.0

    x_span_m = x_span * meters_per_unit
    target_len_m = max(x_span_m * 0.2, 1.0)
    bar_len_m = _nice_scale_length_meters(target_len_m)
    if bar_len_m >= x_span_m:
        bar_len_m = _nice_scale_length_meters(max(x_span_m * 0.5, 1.0))
    bar_len_units = bar_len_m / meters_per_unit

    bar_x0 = x_min + x_span * 0.06
    bar_y0 = y_min + y_span * 0.08
    return bar_x0, bar_y0, bar_len_units, bar_len_m


def _add_points(
    fig: go.Figure,
    row: int | None,
    col: int | None,
    tr: Triangulation,
    params: PlotParams,
    rn_labels: list[str] | None,
) -> None:
    if not (params.show_points or params.show_coordinates or params.show_rn_labels):
        return

    text = None
    if params.show_coordinates:
        text = [f"({x:.2f}, {y:.2f})" for x, y in zip(tr.x, tr.y)]
    if params.show_rn_labels and rn_labels is not None:
        text = rn_labels

    mode_parts = []
    if params.show_points:
        mode_parts.append("markers")
    if text is not None:
        mode_parts.append("text")
    mode = "+".join(mode_parts) if mode_parts else "markers"

    scatter = go.Scatter(
        x=tr.x,
        y=tr.y,
        mode=mode,
        text=text,
        textposition="top right",
        textfont={"size": params.annotation_font_size, "color": "#101010"},
        marker={
            "size": max(4, int(params.point_size / 3)),
            "color": "white",
            "line": {"color": "black", "width": 1},
        },
        hovertemplate="x=%{x}<br>y=%{y}<extra></extra>",
        showlegend=False,
    )
    if row is None or col is None:
        fig.add_trace(scatter)
    else:
        fig.add_trace(scatter, row=row, col=col)

def _add_scale_bar(fig: go.Figure, row: int | None, col: int | None, rows_total: int, cols_total: int, tr: Triangulation) -> None:
    bar = _compute_scale_bar(tr)
    if not bar:
        return
    x0, y0, length_units, length_m = bar
    x1 = x0 + length_units

    if row is None or col is None:
        xref, yref = "x", "y"
    else:
        idx = (row - 1) * cols_total + col
        xref = "x" if idx == 1 else f"x{idx}"
        yref = "y" if idx == 1 else f"y{idx}"

    fig.add_shape(
        type="line",
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y0,
        xref=xref,
        yref=yref,
        line={"color": "black", "width": 3},
    )
    fig.add_annotation(
        x=(x0 + x1) / 2,
        y=y0,
        xref=xref,
        yref=yref,
        text=f"{int(length_m)} м",
        showarrow=False,
        yshift=14,
        font={"size": 12, "color": "black"},
        bgcolor="rgba(255,255,255,0.65)",
        bordercolor="rgba(0,0,0,0)",
    )


def _contour_traces(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    levels: np.ndarray,
    zmin: float,
    zmax: float,
    colorscale: str,
    show_scale: bool,
    colorbar_x: float,
    opacity: float,
    show_lines: bool,
    line_color: str,
    line_width: float,
    line_dash: str | None,
) -> tuple[go.Contour, go.Contour | None]:
    start = float(levels[0])
    end = float(levels[-1])
    size = float(levels[1] - levels[0]) if len(levels) > 1 else 1.0

    fill = go.Contour(
        x=xg[0, :],
        y=yg[:, 0],
        z=zg,
        contours={"start": start, "end": end, "size": size, "coloring": "fill"},
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        opacity=opacity,
        showscale=show_scale,
        colorbar={"x": colorbar_x, "len": 0.9, "thickness": 18},
        line={"width": 0},
        hoverinfo="skip",
    )

    if not show_lines:
        return fill, None

    lines = go.Contour(
        x=xg[0, :],
        y=yg[:, 0],
        z=zg,
        contours={"start": start, "end": end, "size": size, "coloring": "none"},
        showscale=False,
        line={
            "color": line_color,
            "width": line_width,
            "dash": (line_dash or "solid"),
        },
        hoverinfo="skip",
    )
    return fill, lines


def render_dual_maps_plotly(
    triangulation: Triangulation,
    ap: np.ndarray,
    ac: np.ndarray,
    params: PlotParams,
    rn_labels: list[str] | None,
    enforce_mirror: bool,
) -> go.Figure:
    ap_plot, ac_plot = mirror_fields(ap, ac, enforce_mirror=enforce_mirror)
    levels, vmin, vmax = build_levels(ap_plot, ac_plot, params.levels_count)

    rows, cols = (2, 1) if params.vertical_layout else (1, 2)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=("Карта Ap", "Карта Ac") if cols == 2 else ("Карта Ap", "Карта Ac"),
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )

    sigma = float(params.smooth_sigma) if params.smooth_contours else 0.0
    xg_ap, yg_ap, zg_ap = interpolate_to_grid(triangulation, ap_plot, grid_size=params.grid_size)
    xg_ac, yg_ac, zg_ac = interpolate_to_grid(triangulation, ac_plot, grid_size=params.grid_size)
    zg_ap = _smooth_grid(zg_ap, sigma=sigma)
    zg_ac = _smooth_grid(zg_ac, sigma=sigma)

    # Colorbar positions: to the right of each subplot.
    if cols == 2:
        cb1x, cb2x = 0.47, 1.02
        (r1, c1), (r2, c2) = (1, 1), (1, 2)
    else:
        cb1x, cb2x = 1.02, 1.02
        (r1, c1), (r2, c2) = (1, 1), (2, 1)

    ap_fill, ap_lines = _contour_traces(
        xg_ap,
        yg_ap,
        zg_ap,
        levels=levels,
        zmin=vmin,
        zmax=vmax,
        colorscale="Greys",
        show_scale=True,
        colorbar_x=cb1x,
        opacity=1.0,
        show_lines=params.show_contour_lines,
        line_color="#101010",
        line_width=1.0,
        line_dash=None,
    )
    fig.add_trace(ap_fill, row=r1, col=c1)
    if ap_lines is not None:
        fig.add_trace(ap_lines, row=r1, col=c1)

    ac_fill, ac_lines = _contour_traces(
        xg_ac,
        yg_ac,
        zg_ac,
        levels=levels,
        zmin=vmin,
        zmax=vmax,
        colorscale="Greys",
        show_scale=True,
        colorbar_x=cb2x,
        opacity=1.0,
        show_lines=params.show_contour_lines,
        line_color="#101010",
        line_width=1.0,
        line_dash=None,
    )
    fig.add_trace(ac_fill, row=r2, col=c2)
    if ac_lines is not None:
        fig.add_trace(ac_lines, row=r2, col=c2)

    _add_points(fig, row=r1, col=c1, tr=triangulation, params=params, rn_labels=rn_labels)
    _add_points(fig, row=r2, col=c2, tr=triangulation, params=params, rn_labels=rn_labels)

    if params.show_scale_bar:
        _add_scale_bar(fig, row=r1, col=c1, rows_total=rows, cols_total=cols, tr=triangulation)
        _add_scale_bar(fig, row=r2, col=c2, rows_total=rows, cols_total=cols, tr=triangulation)

    fig.update_xaxes(title_text=params.x_label, range=_axis_range(triangulation.x, params.axis_margin, params.invert_x))
    fig.update_yaxes(title_text=params.y_label, range=_axis_range(triangulation.y, params.axis_margin, params.invert_y), scaleanchor=None)

    fig.update_layout(
        height=900 if params.vertical_layout else 520,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        template="plotly_white",
        showlegend=False,
    )
    return fig


def render_overlay_plotly(
    triangulation: Triangulation,
    ap: np.ndarray,
    ac: np.ndarray,
    params: PlotParams,
    rn_labels: list[str] | None,
    enforce_mirror: bool,
    alpha: float,
) -> go.Figure:
    ap_plot, ac_plot = mirror_fields(ap, ac, enforce_mirror=enforce_mirror)
    levels, vmin, vmax = build_levels(ap_plot, ac_plot, params.levels_count)

    sigma = float(params.smooth_sigma) if params.smooth_contours else 0.0
    xg_ap, yg_ap, zg_ap = interpolate_to_grid(triangulation, ap_plot, grid_size=params.grid_size)
    xg_ac, yg_ac, zg_ac = interpolate_to_grid(triangulation, ac_plot, grid_size=params.grid_size)
    zg_ap = _smooth_grid(zg_ap, sigma=sigma)
    zg_ac = _smooth_grid(zg_ac, sigma=sigma)

    fig = go.Figure()

    ap_fill_alpha = max(0.2, min(0.85, float(alpha) * 0.8))
    ac_fill_alpha = max(0.2, min(0.85, (1.0 - float(alpha)) * 0.8))

    ap_fill, ap_lines = _contour_traces(
        xg_ap,
        yg_ap,
        zg_ap,
        levels=levels,
        zmin=vmin,
        zmax=vmax,
        colorscale="Greys",
        show_scale=True,
        colorbar_x=1.02,
        opacity=ap_fill_alpha,
        show_lines=params.show_contour_lines,
        line_color="#000000",
        line_width=1.2,
        line_dash=None,
    )
    fig.add_trace(ap_fill)
    if ap_lines is not None:
        fig.add_trace(ap_lines)

    ac_fill, ac_lines = _contour_traces(
        xg_ac,
        yg_ac,
        zg_ac,
        levels=levels,
        zmin=vmin,
        zmax=vmax,
        colorscale="Greys_r",
        show_scale=False,
        colorbar_x=1.02,
        opacity=ac_fill_alpha,
        show_lines=params.show_contour_lines,
        line_color="#222222",
        line_width=1.1,
        line_dash="dash",
    )
    fig.add_trace(ac_fill)
    if ac_lines is not None:
        fig.add_trace(ac_lines)

    _add_points(fig, row=None, col=None, tr=triangulation, params=params, rn_labels=rn_labels)
    if params.show_scale_bar:
        _add_scale_bar(fig, row=None, col=None, rows_total=1, cols_total=1, tr=triangulation)

    fig.update_xaxes(title_text=params.x_label, range=_axis_range(triangulation.x, params.axis_margin, params.invert_x))
    fig.update_yaxes(title_text=params.y_label, range=_axis_range(triangulation.y, params.axis_margin, params.invert_y))

    fig.update_layout(
        title="Overlay Ap/Ac (проверка совпадения изолиний)",
        height=520,
        margin={"l": 40, "r": 60, "t": 60, "b": 40},
        template="plotly_white",
        showlegend=False,
    )
    return fig

