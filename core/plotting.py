from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.path import Path as MPath
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter
from matplotlib.tri import LinearTriInterpolator, Triangulation
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

from core.basemap import add_satellite_basemap, compute_mercator_square_extent, web_mercator_to_lon_lat
from core.interpolation import build_levels, mirror_fields


def _tfm_kw(tfm: mtransforms.Transform | None) -> dict:
    return {"transform": tfm} if tfm is not None else {}


def _mercator_rotation_center(
    triangulation: Triangulation,
    axis_margin: float,
) -> tuple[float, float]:
    """Центр квадратного extent подложки (как в compute_mercator_square_extent), не среднее точек."""
    xlim, ylim = compute_mercator_square_extent(triangulation.x, triangulation.y, axis_margin=axis_margin)
    cx = 0.5 * (xlim[0] + xlim[1])
    cy = 0.5 * (ylim[0] + ylim[1])
    return cx, cy


def _view_data_transform(
    ax,
    cx: float,
    cy: float,
    view_rotation_deg: float,
) -> mtransforms.Transform | None:
    """Поворот вокруг центра той же рамки, что и подложка (imshow extent), иначе снимок «плывёт»."""
    if abs(view_rotation_deg) < 1e-9:
        return None
    rot = mtransforms.Affine2D().translate(-cx, -cy).rotate_deg(float(view_rotation_deg)).translate(cx, cy)
    return rot + ax.transData


def _triangulation_hull_path(triangulation: Triangulation) -> MPath:
    """Выпуклая оболочка точек сетки (контур области данных)."""
    x, y = triangulation.x, triangulation.y
    pts = np.column_stack((x, y))
    n = pts.shape[0]
    if n < 1:
        return MPath(np.zeros((1, 2)), closed=True)
    if n < 3:
        cx, cy = float(np.mean(x)), float(np.mean(y))
        pad = max(float(np.ptp(x)) if n > 1 else 1.0, float(np.ptp(y)) if n > 1 else 1.0) * 0.5 + 5.0
        q = np.array(
            [
                [cx - pad, cy - pad],
                [cx + pad, cy - pad],
                [cx + pad, cy + pad],
                [cx - pad, cy + pad],
                [cx - pad, cy - pad],
            ]
        )
        return MPath(q, closed=True)
    hull = ConvexHull(pts)
    v = pts[hull.vertices]
    closed = np.vstack([v, v[0:1]])
    return MPath(closed, closed=True)


def _mercator_lon_formatter(ax):
    def fmt(xv: float, pos: int | None) -> str:
        y0, y1 = ax.get_ylim()
        y_edge = y1 if ax.yaxis_inverted() else y0
        lon, _ = web_mercator_to_lon_lat(np.array([xv], dtype=float), np.array([y_edge], dtype=float))
        return f"{float(lon[0]):.4f}"

    return fmt


def _mercator_lat_formatter(ax):
    def fmt(yv: float, pos: int | None) -> str:
        x0, x1 = ax.get_xlim()
        x_edge = x1 if ax.xaxis_inverted() else x0
        _, lat = web_mercator_to_lon_lat(np.array([x_edge], dtype=float), np.array([yv], dtype=float))
        return f"{float(lat[0]):.4f}"

    return fmt


def _style_axis_mercator_degrees(
    ax,
    title: str,
    axis_margin: float = 0.05,
    skip_margins: bool = False,
) -> None:
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("Долгота (°)")
    ax.set_ylabel("Широта (°)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.xaxis.set_major_formatter(FuncFormatter(_mercator_lon_formatter(ax)))
    ax.yaxis.set_major_formatter(FuncFormatter(_mercator_lat_formatter(ax)))
    ax.use_sticky_edges = False
    if not skip_margins:
        margin = float(np.clip(axis_margin, 0.0, 0.3))
        ax.margins(x=margin, y=margin)
    ax.grid(alpha=0.15, linewidth=0.6)


def _style_axis(
    ax,
    title: str,
    x_label: str = "X",
    y_label: str = "Y",
    axis_margin: float = 0.05,
    skip_margins: bool = False,
) -> None:
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    x_fmt = ScalarFormatter(useOffset=False)
    y_fmt = ScalarFormatter(useOffset=False)
    x_fmt.set_scientific(False)
    y_fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(x_fmt)
    ax.yaxis.set_major_formatter(y_fmt)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    margin = float(np.clip(axis_margin, 0.0, 0.3))
    # Contour/contourf artists can set sticky edges that effectively ignore margins.
    # Disable them so user-configured padding is always visible.
    ax.use_sticky_edges = False
    if not skip_margins:
        ax.margins(x=margin, y=margin)
    ax.grid(alpha=0.15, linewidth=0.6)


def _apply_axis_inversion(ax, invert_x: bool, invert_y: bool) -> None:
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()


def _build_cmap(start_hex: str, end_hex: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("custom_gray", [start_hex, end_hex], N=256)


def _compute_levels(
    ap_plot: np.ndarray,
    ac_plot: np.ndarray,
    levels_count: int,
    levels_step: float | None,
) -> tuple[np.ndarray, float, float]:
    if levels_step is None or levels_step <= 0:
        return build_levels(ap_plot, ac_plot, levels_count)

    vmin = float(min(np.min(ap_plot), np.min(ac_plot)))
    vmax = float(max(np.max(ap_plot), np.max(ac_plot)))
    if vmin == vmax:
        vmax = vmin + levels_step

    step = float(levels_step)
    levels = np.arange(vmin, vmax + step * 0.999, step)
    if levels.size < 2:
        return build_levels(ap_plot, ac_plot, levels_count)
    return levels, vmin, vmax


def _draw_points_and_labels(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    point_size: int,
    show_points: bool,
    show_coordinates: bool,
    show_rn_labels: bool,
    rn_labels: list[str] | None,
    annotation_font_size: int,
    coordinate_degrees_lon_lat: tuple[np.ndarray, np.ndarray] | None = None,
    data_transform: mtransforms.Transform | None = None,
) -> None:
    kw = _tfm_kw(data_transform)
    if show_points:
        ax.scatter(x, y, s=point_size, c="white", edgecolors="black", linewidths=0.35, zorder=5, **kw)
    if show_coordinates:
        if coordinate_degrees_lon_lat is not None:
            lon_arr, lat_arr = coordinate_degrees_lon_lat
            for xi, yi, lo, la in zip(x, y, lon_arr, lat_arr):
                ax.text(
                    xi,
                    yi,
                    f"({lo:.6f}°, {la:.6f}°)",
                    fontsize=annotation_font_size,
                    color="#202020",
                    ha="left",
                    va="bottom",
                    alpha=0.9,
                    zorder=6,
                    **kw,
                )
        else:
            for xi, yi in zip(x, y):
                ax.text(
                    xi,
                    yi,
                    f"({xi:.2f}, {yi:.2f})",
                    fontsize=annotation_font_size,
                    color="#202020",
                    ha="left",
                    va="bottom",
                    alpha=0.9,
                    zorder=6,
                    **kw,
                )
    if show_rn_labels and rn_labels is not None:
        for xi, yi, label in zip(x, y, rn_labels):
            ax.text(
                xi,
                yi,
                f"{label}",
                fontsize=annotation_font_size,
                color="#101010",
                ha="right",
                va="top",
                alpha=0.95,
                zorder=7,
                **kw,
            )


def _draw_scale_bar(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    enabled: bool,
    web_mercator: bool = False,
    data_transform: mtransforms.Transform | None = None,
) -> None:
    if not enabled:
        return
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 0 or y_span <= 0:
        return

    bar_x0 = x_min + x_span * 0.06
    bar_y0 = y_min + y_span * 0.08

    # Web Mercator (EPSG:3857): axis units are meters.
    if web_mercator:
        meters_per_unit = 1.0
    else:
        # Convert scale length to meters for geographic coordinates (degrees).
        x_is_geo = float(np.max(np.abs(x))) <= 180.0
        y_is_geo = float(np.max(np.abs(y))) <= 90.0
        if x_is_geo and y_is_geo:
            lat_mid = float(np.mean(y))
            meters_per_unit = 111_320.0 * np.cos(np.deg2rad(lat_mid))
            meters_per_unit = float(np.clip(meters_per_unit, 1_000.0, 111_320.0))
        else:
            # Assume incoming coordinates are already meters.
            meters_per_unit = 1.0
    x_span_m = x_span * meters_per_unit
    target_len_m = max(x_span_m * 0.2, 1.0)
    bar_len_m = _nice_scale_length_meters(target_len_m)
    if bar_len_m >= x_span_m:
        bar_len_m = _nice_scale_length_meters(max(x_span_m * 0.5, 1.0))
    bar_len = bar_len_m / meters_per_unit

    kw = _tfm_kw(data_transform)
    ax.plot([bar_x0, bar_x0 + bar_len], [bar_y0, bar_y0], color="black", linewidth=2.2, solid_capstyle="butt", zorder=8, **kw)
    ax.plot([bar_x0, bar_x0], [bar_y0 - y_span * 0.01, bar_y0 + y_span * 0.01], color="black", linewidth=1.4, zorder=8, **kw)
    ax.plot(
        [bar_x0 + bar_len, bar_x0 + bar_len],
        [bar_y0 - y_span * 0.01, bar_y0 + y_span * 0.01],
        color="black",
        linewidth=1.4,
        zorder=8,
        **kw,
    )
    ax.text(
        bar_x0 + bar_len / 2.0,
        bar_y0 + y_span * 0.02,
        _format_meters_text(bar_len_m),
        fontsize=8.2,
        ha="center",
        va="bottom",
        color="black",
        zorder=8,
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        **kw,
    )


def _format_meters_text(length_m: float) -> str:
    value = max(1, int(round(float(abs(length_m)))))
    return f"{value} м"


def _nice_scale_length_meters(target_m: float) -> int:
    # Use powers of 10 to keep scale bar labels like 10, 100, 1000 m.
    target = max(float(target_m), 1.0)
    power = int(np.floor(np.log10(target)))
    nice = 10 ** power
    if nice > target:
        nice = max(1, nice // 10)
    return int(max(1, nice))


def _interpolate_to_grid(
    triangulation: Triangulation,
    values: np.ndarray,
    grid_size: int,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = triangulation.x
    y = triangulation.y
    xi = np.linspace(float(np.min(x)), float(np.max(x)), grid_size)
    yi = np.linspace(float(np.min(y)), float(np.max(y)), grid_size)
    xg, yg = np.meshgrid(xi, yi)

    interpolator = LinearTriInterpolator(triangulation, values)
    zg = interpolator(xg, yg)
    if np.ma.isMaskedArray(zg):
        zg_data = zg.filled(np.nan)
    else:
        zg_data = np.asarray(zg, dtype=float)

    if smooth_sigma > 0:
        valid_mask = np.isfinite(zg_data)
        safe = np.where(valid_mask, zg_data, 0.0)
        weights = valid_mask.astype(float)
        safe_blur = gaussian_filter(safe, sigma=smooth_sigma, mode="nearest")
        weights_blur = gaussian_filter(weights, sigma=smooth_sigma, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = safe_blur / np.where(weights_blur == 0, np.nan, weights_blur)
        zg_data = np.where(valid_mask, smoothed, np.nan)

    return xg, yg, zg_data


def render_dual_maps(
    triangulation: Triangulation,
    ap: np.ndarray,
    ac: np.ndarray,
    levels_count: int = 10,
    levels_step: float | None = None,
    point_size: int = 18,
    enforce_mirror: bool = True,
    smooth_contours: bool = True,
    smooth_sigma: float = 0.6,
    grid_size: int = 260,
    show_points: bool = True,
    show_coordinates: bool = False,
    show_rn_labels: bool = False,
    rn_labels: list[str] | None = None,
    show_scale_bar: bool = True,
    invert_x: bool = False,
    invert_y: bool = False,
    x_label: str = "X",
    y_label: str = "Y",
    show_contour_lines: bool = True,
    contour_line_width: float = 0.9,
    show_contour_labels: bool = False,
    contour_label_font_size: int = 8,
    cmap_start: str = "#ffffff",
    cmap_end: str = "#000000",
    vertical_layout: bool = False,
    annotation_font_size: int = 7,
    axis_margin: float = 0.05,
    basemap_enabled: bool = False,
    map_layer_alpha: float = 0.85,
    web_mercator: bool = False,
    basemap_source: str = "esri",
    coordinate_degrees_lon_lat: tuple[np.ndarray, np.ndarray] | None = None,
    google_maps_api_key: str | None = None,
    view_rotation_deg: float = 0.0,
) -> Figure:
    ap_plot, ac_plot = mirror_fields(ap, ac, enforce_mirror=enforce_mirror)
    levels, vmin, vmax = _compute_levels(ap_plot, ac_plot, levels_count=levels_count, levels_step=levels_step)

    if vertical_layout:
        fig, axes = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=False)
    else:
        # constrained_layout breaks with dual colorbars + basemap (axes collapse to zero width).
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=False)
    cmap = _build_cmap(cmap_start, cmap_end)
    cmap_ac = cmap.reversed()

    fill_alpha = float(np.clip(map_layer_alpha, 0.05, 1.0)) if basemap_enabled else 1.0

    cx, cy = _mercator_rotation_center(triangulation, axis_margin)
    clip_path: MPath | None = None
    if basemap_enabled and abs(view_rotation_deg) > 1e-6:
        clip_path = _triangulation_hull_path(triangulation)

    tfm0 = _view_data_transform(axes[0], cx, cy, view_rotation_deg)
    tfm1 = _view_data_transform(axes[1], cx, cy, view_rotation_deg)
    kw0 = _tfm_kw(tfm0)
    kw1 = _tfm_kw(tfm1)

    if basemap_enabled:
        add_satellite_basemap(
            axes[0],
            x=triangulation.x,
            y=triangulation.y,
            axis_margin=axis_margin,
            zorder=0,
            basemap_source_key=basemap_source,
            google_maps_api_key=google_maps_api_key,
            display_transform=tfm0,
            clip_path=clip_path,
            clip_transform=tfm0,
        )
        add_satellite_basemap(
            axes[1],
            x=triangulation.x,
            y=triangulation.y,
            axis_margin=axis_margin,
            zorder=0,
            basemap_source_key=basemap_source,
            google_maps_api_key=google_maps_api_key,
            display_transform=tfm1,
            clip_path=clip_path,
            clip_transform=tfm1,
        )

    if smooth_contours:
        xg_ap, yg_ap, zg_ap = _interpolate_to_grid(triangulation, ap_plot, grid_size=grid_size, smooth_sigma=smooth_sigma)
        cf_ap = axes[0].contourf(
            xg_ap,
            yg_ap,
            zg_ap,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=fill_alpha,
            zorder=1,
            **kw0,
        )
        if show_contour_lines:
            cs = axes[0].contour(
                xg_ap,
                yg_ap,
                zg_ap,
                levels=levels,
                colors="#101010",
                linewidths=float(contour_line_width),
                alpha=1.0,
                **kw0,
            )
            if show_contour_labels:
                axes[0].clabel(cs, inline=True, fontsize=contour_label_font_size, fmt="%.2f")
    else:
        cf_ap = axes[0].tricontourf(
            triangulation,
            ap_plot,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=fill_alpha,
            zorder=1,
            **kw0,
        )
        if show_contour_lines:
            cs = axes[0].tricontour(
                triangulation,
                ap_plot,
                levels=levels,
                colors="#101010",
                linewidths=float(contour_line_width),
                alpha=1.0,
                **kw0,
            )
            if show_contour_labels:
                axes[0].clabel(cs, inline=True, fontsize=contour_label_font_size, fmt="%.2f")
    _draw_points_and_labels(
        axes[0],
        triangulation.x,
        triangulation.y,
        point_size=point_size,
        show_points=show_points,
        show_coordinates=show_coordinates,
        show_rn_labels=show_rn_labels,
        rn_labels=rn_labels,
        annotation_font_size=annotation_font_size,
        coordinate_degrees_lon_lat=coordinate_degrees_lon_lat if web_mercator else None,
        data_transform=tfm0,
    )
    _draw_scale_bar(
        axes[0],
        triangulation.x,
        triangulation.y,
        enabled=show_scale_bar,
        web_mercator=web_mercator,
        data_transform=tfm0,
    )
    if web_mercator:
        _style_axis_mercator_degrees(
            axes[0],
            "Карта Ap",
            axis_margin=axis_margin,
            skip_margins=basemap_enabled,
        )
    else:
        _style_axis(
            axes[0],
            "Карта Ap",
            x_label=x_label,
            y_label=y_label,
            axis_margin=axis_margin,
            skip_margins=basemap_enabled,
        )
    _apply_axis_inversion(axes[0], invert_x=invert_x, invert_y=invert_y)
    fig.colorbar(cf_ap, ax=axes[0], location="right", shrink=0.95)

    if smooth_contours:
        xg_ac, yg_ac, zg_ac = _interpolate_to_grid(triangulation, ac_plot, grid_size=grid_size, smooth_sigma=smooth_sigma)
        cf_ac = axes[1].contourf(
            xg_ac,
            yg_ac,
            zg_ac,
            levels=levels,
            cmap=cmap_ac,
            vmin=vmin,
            vmax=vmax,
            alpha=fill_alpha,
            zorder=1,
            **kw1,
        )
        if show_contour_lines:
            cs = axes[1].contour(
                xg_ac,
                yg_ac,
                zg_ac,
                levels=levels,
                colors="#101010",
                linewidths=float(contour_line_width),
                alpha=1.0,
                **kw1,
            )
            if show_contour_labels:
                axes[1].clabel(cs, inline=True, fontsize=contour_label_font_size, fmt="%.2f")
    else:
        cf_ac = axes[1].tricontourf(
            triangulation,
            ac_plot,
            levels=levels,
            cmap=cmap_ac,
            vmin=vmin,
            vmax=vmax,
            alpha=fill_alpha,
            zorder=1,
            **kw1,
        )
        if show_contour_lines:
            cs = axes[1].tricontour(
                triangulation,
                ac_plot,
                levels=levels,
                colors="#101010",
                linewidths=float(contour_line_width),
                alpha=1.0,
                **kw1,
            )
            if show_contour_labels:
                axes[1].clabel(cs, inline=True, fontsize=contour_label_font_size, fmt="%.2f")
    _draw_points_and_labels(
        axes[1],
        triangulation.x,
        triangulation.y,
        point_size=point_size,
        show_points=show_points,
        show_coordinates=show_coordinates,
        show_rn_labels=show_rn_labels,
        rn_labels=rn_labels,
        annotation_font_size=annotation_font_size,
        coordinate_degrees_lon_lat=coordinate_degrees_lon_lat if web_mercator else None,
        data_transform=tfm1,
    )
    _draw_scale_bar(
        axes[1],
        triangulation.x,
        triangulation.y,
        enabled=show_scale_bar,
        web_mercator=web_mercator,
        data_transform=tfm1,
    )
    if web_mercator:
        _style_axis_mercator_degrees(
            axes[1],
            "Карта Ac",
            axis_margin=axis_margin,
            skip_margins=basemap_enabled,
        )
    else:
        _style_axis(
            axes[1],
            "Карта Ac",
            x_label=x_label,
            y_label=y_label,
            axis_margin=axis_margin,
            skip_margins=basemap_enabled,
        )
    _apply_axis_inversion(axes[1], invert_x=invert_x, invert_y=invert_y)
    fig.colorbar(cf_ac, ax=axes[1], location="right", shrink=0.95)

    if vertical_layout:
        fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.07, hspace=0.3)
    else:
        fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.11, wspace=0.32)

    return fig


def render_overlay_map(
    triangulation: Triangulation,
    ap: np.ndarray,
    ac: np.ndarray,
    alpha: float = 0.5,
    levels_count: int = 10,
    levels_step: float | None = None,
    enforce_mirror: bool = True,
    smooth_contours: bool = True,
    smooth_sigma: float = 0.6,
    grid_size: int = 260,
    show_points: bool = True,
    show_coordinates: bool = False,
    show_rn_labels: bool = False,
    rn_labels: list[str] | None = None,
    show_scale_bar: bool = True,
    invert_x: bool = False,
    invert_y: bool = False,
    x_label: str = "X",
    y_label: str = "Y",
    show_contour_lines: bool = True,
    contour_line_width: float = 1.2,
    show_contour_labels: bool = False,
    contour_label_font_size: int = 8,
    annotation_font_size: int = 7,
    axis_margin: float = 0.05,
    cmap_start: str = "#ffffff",
    cmap_end: str = "#000000",
    basemap_enabled: bool = False,
    map_layer_alpha: float = 0.85,
    web_mercator: bool = False,
    basemap_source: str = "esri",
    coordinate_degrees_lon_lat: tuple[np.ndarray, np.ndarray] | None = None,
    google_maps_api_key: str | None = None,
    view_rotation_deg: float = 0.0,
) -> Figure:
    ap_plot, ac_plot = mirror_fields(ap, ac, enforce_mirror=enforce_mirror)
    levels, vmin, vmax = _compute_levels(ap_plot, ac_plot, levels_count=levels_count, levels_step=levels_step)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=False)
    cmap = _build_cmap(cmap_start, cmap_end)
    overlay_cmap_ac = cmap.reversed()

    map_alpha_factor = float(np.clip(map_layer_alpha, 0.05, 1.0)) if basemap_enabled else 1.0

    cx, cy = _mercator_rotation_center(triangulation, axis_margin)
    clip_path: MPath | None = None
    if basemap_enabled and abs(view_rotation_deg) > 1e-6:
        clip_path = _triangulation_hull_path(triangulation)
    tfm = _view_data_transform(ax, cx, cy, view_rotation_deg)
    kw = _tfm_kw(tfm)

    if basemap_enabled:
        add_satellite_basemap(
            ax,
            x=triangulation.x,
            y=triangulation.y,
            axis_margin=axis_margin,
            zorder=0,
            basemap_source_key=basemap_source,
            google_maps_api_key=google_maps_api_key,
            display_transform=tfm,
            clip_path=clip_path,
            clip_transform=tfm,
        )

    if smooth_contours:
        xg_ap, yg_ap, zg_ap = _interpolate_to_grid(triangulation, ap_plot, grid_size=grid_size, smooth_sigma=smooth_sigma)
        xg_ac, yg_ac, zg_ac = _interpolate_to_grid(triangulation, ac_plot, grid_size=grid_size, smooth_sigma=smooth_sigma)
        ap_fill_alpha = max(0.2, min(0.85, alpha * 0.8)) * map_alpha_factor
        ac_fill_alpha = max(0.2, min(0.85, (1.0 - alpha) * 0.8)) * map_alpha_factor
        ax.contourf(
            xg_ap,
            yg_ap,
            zg_ap,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=ap_fill_alpha,
            antialiased=False,
            zorder=1,
            **kw,
        )
        ax.contourf(
            xg_ac,
            yg_ac,
            zg_ac,
            levels=levels,
            cmap=overlay_cmap_ac,
            vmin=vmin,
            vmax=vmax,
            alpha=ac_fill_alpha,
            antialiased=False,
            zorder=1,
            **kw,
        )
        if show_contour_lines:
            lw_main = float(contour_line_width)
            cs1 = ax.contour(
                xg_ap, yg_ap, zg_ap, levels=levels, colors="#000000", linewidths=lw_main, alpha=1.0, **kw
            )
            cs2 = ax.contour(
                xg_ac,
                yg_ac,
                zg_ac,
                levels=levels,
                colors="#222222",
                linewidths=max(0.3, lw_main * 0.9),
                alpha=1.0,
                linestyles="dashed",
                **kw,
            )
            if show_contour_labels:
                ax.clabel(cs1, inline=True, fontsize=contour_label_font_size, fmt="%.2f")
                ax.clabel(cs2, inline=True, fontsize=contour_label_font_size, fmt="%.2f")
    else:
        ap_fill_alpha = max(0.2, min(0.85, alpha * 0.8)) * map_alpha_factor
        ac_fill_alpha = max(0.2, min(0.85, (1.0 - alpha) * 0.8)) * map_alpha_factor
        ax.tricontourf(
            triangulation,
            ap_plot,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=ap_fill_alpha,
            antialiased=False,
            zorder=1,
            **kw,
        )
        ax.tricontourf(
            triangulation,
            ac_plot,
            levels=levels,
            cmap=overlay_cmap_ac,
            vmin=vmin,
            vmax=vmax,
            alpha=ac_fill_alpha,
            antialiased=False,
            zorder=1,
            **kw,
        )
        if show_contour_lines:
            lw_main = float(contour_line_width)
            cs1 = ax.tricontour(
                triangulation, ap_plot, levels=levels, colors="#000000", linewidths=lw_main, alpha=1.0, **kw
            )
            cs2 = ax.tricontour(
                triangulation,
                ac_plot,
                levels=levels,
                colors="#222222",
                linewidths=max(0.3, lw_main * 0.9),
                alpha=1.0,
                linestyles="dashed",
                **kw,
            )
            if show_contour_labels:
                ax.clabel(cs1, inline=True, fontsize=contour_label_font_size, fmt="%.2f")
                ax.clabel(cs2, inline=True, fontsize=contour_label_font_size, fmt="%.2f")
    _draw_points_and_labels(
        ax,
        triangulation.x,
        triangulation.y,
        point_size=15,
        show_points=show_points,
        show_coordinates=show_coordinates,
        show_rn_labels=show_rn_labels,
        rn_labels=rn_labels,
        annotation_font_size=annotation_font_size,
        coordinate_degrees_lon_lat=coordinate_degrees_lon_lat if web_mercator else None,
        data_transform=tfm,
    )
    _draw_scale_bar(
        ax,
        triangulation.x,
        triangulation.y,
        enabled=show_scale_bar,
        web_mercator=web_mercator,
        data_transform=tfm,
    )
    if web_mercator:
        _style_axis_mercator_degrees(
            ax,
            "Overlay Ap/Ac (проверка совпадения изолиний)",
            axis_margin=axis_margin,
            skip_margins=basemap_enabled,
        )
    else:
        _style_axis(
            ax,
            "Overlay Ap/Ac (проверка совпадения изолиний)",
            x_label=x_label,
            y_label=y_label,
            axis_margin=axis_margin,
            skip_margins=basemap_enabled,
        )
    _apply_axis_inversion(ax, invert_x=invert_x, invert_y=invert_y)

    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_clim(vmin, vmax)
    fig.colorbar(mappable, ax=ax, location="right", shrink=0.95)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.93, bottom=0.11)
    return fig
