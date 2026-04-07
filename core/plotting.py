from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter
from matplotlib.tri import LinearTriInterpolator, Triangulation
from scipy.ndimage import gaussian_filter

from core.basemap import (
    add_satellite_basemap,
    compute_mercator_axis_extent,
    expand_mercator_extent_for_view_rotation,
    web_mercator_to_lon_lat,
)
from core.interpolation import build_levels, mirror_fields


def _seal_filled_contours(cs) -> None:
    """Убрать видимые «швы» между полигонами contourf/tricontourf.

    На спутниковой подложке при ``alpha < 1`` Matplotlib иногда показывает тонкие светлые линии
    на стыках полигонов (особенно в Web Mercator). Это лечится отключением обводок/AA и
    принудительным rasterization для коллекций.
    """
    try:
        collections = list(getattr(cs, "collections", []))
    except Exception:  # noqa: BLE001
        collections = []
    for col in collections:
        try:
            col.set_edgecolor("face")
            col.set_linewidth(0.0)
            col.set_antialiased(False)
            col.set_rasterized(True)
        except Exception:  # noqa: BLE001
            continue


def _imshow_fill(
    ax,
    *,
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    cmap: LinearSegmentedColormap,
    alpha: float,
    zorder: float,
    extras: dict,
    interpolation: str,
    tfm_kw: dict,
):
    """Raster fill to avoid polygon seam artifacts.

    Used primarily when basemap is enabled and alpha < 1, where contourf seams become visible.
    Returns the image (mappable) for colorbar.
    """
    xmin = float(np.nanmin(xg))
    xmax = float(np.nanmax(xg))
    ymin = float(np.nanmin(yg))
    ymax = float(np.nanmax(yg))
    # Ensure NaNs are transparent.
    cmap2 = cmap.copy()
    cmap2.set_bad((0.0, 0.0, 0.0, 0.0))
    zmask = np.ma.masked_invalid(zg)
    # "interpolation" controls smoothness without introducing polygon seams.
    # ``contourf`` accepts ``extend=``, but ``imshow``/AxesImage doesn't.
    # Keep only parameters applicable to images: norm/vmin/vmax.
    img_extras: dict = {}
    if isinstance(extras, dict):
        if "norm" in extras and extras["norm"] is not None:
            img_extras["norm"] = extras["norm"]
        else:
            if "vmin" in extras:
                img_extras["vmin"] = extras["vmin"]
            if "vmax" in extras:
                img_extras["vmax"] = extras["vmax"]

    return ax.imshow(
        zmask,
        extent=(xmin, xmax, ymin, ymax),
        origin="lower",
        cmap=cmap2,
        alpha=float(alpha),
        zorder=float(zorder),
        interpolation=str(interpolation),
        **img_extras,
        **tfm_kw,
    )


def _tfm_kw(tfm: mtransforms.Transform | None) -> dict:
    return {"transform": tfm} if tfm is not None else {}


def _mercator_rotation_center(
    triangulation: Triangulation,
    axis_margin: float,
    *,
    mercator_force_square: bool = True,
    mercator_span_scale_x: float = 1.0,
    mercator_span_scale_y: float = 1.0,
) -> tuple[float, float]:
    """Центр охвата подложки (как в compute_mercator_axis_extent), не среднее точек."""
    xlim, ylim = compute_mercator_axis_extent(
        triangulation.x,
        triangulation.y,
        axis_margin,
        scale_x=mercator_span_scale_x,
        scale_y=mercator_span_scale_y,
        force_square=mercator_force_square,
    )
    cx = 0.5 * (xlim[0] + xlim[1])
    cy = 0.5 * (ylim[0] + ylim[1])
    return cx, cy


def _mercator_axis_limits_tuple(
    triangulation: Triangulation,
    axis_margin: float,
    *,
    mercator_force_square: bool,
    mercator_span_scale_x: float,
    mercator_span_scale_y: float,
    view_rotation_deg: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Охват осей в EPSG:3857 (м): inner по данным, при повороте — AABB повёрнутого прямоугольника.

    При ненулевом угле расширяем ``xlim``/``ylim``, чтобы повёрнутые изолинии не обрезались по краям
    осей; численный масштаб по X/Y меняется, зато карта заполняет доступную область без «сжатия».
    """
    inner_xlim, inner_ylim = compute_mercator_axis_extent(
        triangulation.x,
        triangulation.y,
        axis_margin,
        scale_x=mercator_span_scale_x,
        scale_y=mercator_span_scale_y,
        force_square=mercator_force_square,
    )
    return expand_mercator_extent_for_view_rotation(inner_xlim, inner_ylim, view_rotation_deg)


def _set_mercator_view_axis_limits(
    ax,
    triangulation: Triangulation,
    axis_margin: float,
    view_rotation_deg: float,
    *,
    mercator_force_square: bool,
    mercator_span_scale_x: float,
    mercator_span_scale_y: float,
) -> None:
    """Пределы осей: inner при 0°, при повороте — расширенный AABB (как в basemap)."""
    xlim, ylim = _mercator_axis_limits_tuple(
        triangulation,
        axis_margin,
        mercator_force_square=mercator_force_square,
        mercator_span_scale_x=mercator_span_scale_x,
        mercator_span_scale_y=mercator_span_scale_y,
        view_rotation_deg=view_rotation_deg,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _set_cartesian_axis_limits(
    ax,
    triangulation: Triangulation,
    axis_margin: float,
    *,
    span_scale_x: float = 1.0,
    span_scale_y: float = 1.0,
) -> None:
    """Явные пределы осей в исходных координатах.

    Нужны для управления «охватом» по X/Y (аналог mercator_span_*), чтобы можно было
    увеличивать/уменьшать кадр по каждой оси независимо от подложки/проекции.
    """
    x = triangulation.x
    y = triangulation.y
    x0 = float(np.min(x))
    x1 = float(np.max(x))
    y0 = float(np.min(y))
    y1 = float(np.max(y))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    margin = float(np.clip(axis_margin, 0.0, 0.3))
    dx = max(1e-12, (x1 - x0))
    dy = max(1e-12, (y1 - y0))
    half_x = 0.5 * dx * (1.0 + 2.0 * margin) * float(span_scale_x)
    half_y = 0.5 * dy * (1.0 + 2.0 * margin) * float(span_scale_y)

    ax.set_xlim((cx - half_x, cx + half_x))
    ax.set_ylim((cy - half_y, cy + half_y))


def _finalize_web_mercator_aspect_after_layout(
    axes: Iterable,
    *,
    triangulation: Triangulation,
    axis_margin: float,
    mercator_force_square: bool,
    mercator_span_scale_x: float,
    mercator_span_scale_y: float,
    view_rotation_deg: float = 0.0,
) -> None:
    """Равный масштаб по осям (1 м = 1 м в EPSG:3857).

    Используем ``adjustable='box'``, а не ``datalim``: иначе при смене ширины оси (например,
    из‑за colorbar рядом с картой изолиний) Matplotlib пересчитывает ``xlim``/``ylim`` и подложка
    визуально «плывёт» относительно данных при скрытии изолиний.
    Подложку рисуем **после** этой функции с ``preserve_axes_limits=True``, чтобы растр не сдвигал
    уже согласованные пределы осей.
    """
    ax_list = list(axes)
    if not ax_list:
        return
    fig = ax_list[0].figure
    mx, my = _mercator_axis_limits_tuple(
        triangulation,
        axis_margin,
        mercator_force_square=mercator_force_square,
        mercator_span_scale_x=mercator_span_scale_x,
        mercator_span_scale_y=mercator_span_scale_y,
        view_rotation_deg=view_rotation_deg,
    )
    for ax in ax_list:
        ax.set_xlim(mx)
        ax.set_ylim(my)
        ax.set_aspect("equal", adjustable="box", anchor="C")
    fig.canvas.draw()
    for ax in ax_list:
        ax.apply_aspect()


def _view_data_transform(
    ax,
    cx: float,
    cy: float,
    view_rotation_deg: float,
) -> mtransforms.Transform | None:
    """Поворот **содержимого** (контуры, точки, подложка) вокруг центра inner-охвата.

    Пределы осей при ненулевом угле **расширяются** до осевого AABB повёрнутого прямоугольника,
    чтобы изолинии не обрезались и занимали область графика; подписи в градусах остаются
    по краям прямоугольника в метрах Mercator.
    """
    if abs(view_rotation_deg) < 1e-9:
        return None
    rot = mtransforms.Affine2D().translate(-cx, -cy).rotate_deg(float(view_rotation_deg)).translate(cx, cy)
    return rot + ax.transData


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
    axis_tick_fontsize_x: float = 9.0,
    axis_tick_fontsize_y: float = 9.0,
    show_coordinate_grid: bool = True,
) -> None:
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("Долгота (°)")
    ax.set_ylabel("Широта (°)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.xaxis.set_major_formatter(FuncFormatter(_mercator_lon_formatter(ax)))
    ax.yaxis.set_major_formatter(FuncFormatter(_mercator_lat_formatter(ax)))
    ax.tick_params(axis="x", labelsize=float(axis_tick_fontsize_x))
    ax.tick_params(axis="y", labelsize=float(axis_tick_fontsize_y))
    ax.use_sticky_edges = False
    if not skip_margins:
        margin = float(np.clip(axis_margin, 0.0, 0.3))
        ax.margins(x=margin, y=margin)
    if show_coordinate_grid:
        ax.grid(alpha=0.15, linewidth=0.6)
    else:
        ax.grid(False)


def _style_axis(
    ax,
    title: str,
    x_label: str = "X",
    y_label: str = "Y",
    axis_margin: float = 0.05,
    skip_margins: bool = False,
    axis_tick_fontsize_x: float = 9.0,
    axis_tick_fontsize_y: float = 9.0,
    show_coordinate_grid: bool = True,
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
    ax.tick_params(axis="x", labelsize=float(axis_tick_fontsize_x))
    ax.tick_params(axis="y", labelsize=float(axis_tick_fontsize_y))
    margin = float(np.clip(axis_margin, 0.0, 0.3))
    # Contour/contourf artists can set sticky edges that effectively ignore margins.
    # Disable them so user-configured padding is always visible.
    ax.use_sticky_edges = False
    if not skip_margins:
        ax.margins(x=margin, y=margin)
    if show_coordinate_grid:
        ax.grid(alpha=0.15, linewidth=0.6)
    else:
        ax.grid(False)


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


def _build_fill_cmap_and_norm(
    ap_plot: np.ndarray,
    ac_plot: np.ndarray,
    levels_count: int,
    levels_step: float | None,
    cmap_start: str,
    cmap_end: str,
    custom_gradient_colors: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, float, float, LinearSegmentedColormap, Normalize | None]:
    """Уровни заливки и colormap.

    При ``custom_gradient_colors`` — непрерывная интерполяция между выбранными цветами (как в легенде),
    плюс плотная сетка уровней заливки, чтобы на карте переходы были плавными, а не из N полос.
    Второй массив — уровни для изолиний (тот же, что задаёт пользователь ``levels_count`` / шаг).
    """
    contour_levels, vmin, vmax = _compute_levels(
        ap_plot, ac_plot, levels_count=levels_count, levels_step=levels_step
    )
    if custom_gradient_colors and len(custom_gradient_colors) >= 2:
        n = len(custom_gradient_colors)
        positions = np.linspace(0.0, 1.0, n)
        cmap = LinearSegmentedColormap.from_list(
            "custom_gradient",
            list(zip(positions.tolist(), custom_gradient_colors)),
            N=256,
        )
        n_fill = int(max(128, min(512, 32 * n)))
        fill_levels = np.linspace(float(vmin), float(vmax), n_fill)
        norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
        return fill_levels, contour_levels, vmin, vmax, cmap, norm
    cmap = _build_cmap(cmap_start, cmap_end)
    return contour_levels, contour_levels, vmin, vmax, cmap, None


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


def _meters_per_data_unit_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    web_mercator: bool,
) -> tuple[float, float]:
    """Метры на единицу данных по X (восток) и Y (север) для подписи масштаба."""
    if web_mercator:
        return 1.0, 1.0
    x_is_geo = float(np.max(np.abs(x))) <= 180.0
    y_is_geo = float(np.max(np.abs(y))) <= 90.0
    if x_is_geo and y_is_geo:
        lat_mid = float(np.mean(y))
        mux = 111_320.0 * np.cos(np.deg2rad(lat_mid))
        mux = float(np.clip(mux, 1_000.0, 111_320.0))
        return mux, 111_320.0
    return 1.0, 1.0


def _draw_scale_bars(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    show_scale_bar_x: bool,
    show_scale_bar_y: bool,
    web_mercator: bool = False,
) -> None:
    """Шкалы в долях оси (``transAxes``): горизонтальная по X и вертикальная по Y на экране, без поворота с картой."""
    if not show_scale_bar_x and not show_scale_bar_y:
        return
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 0 or y_span <= 0:
        return

    mux, muy = _meters_per_data_unit_xy(x, y, web_mercator=web_mercator)
    x_span_m = abs(x_span) * mux
    y_span_m = abs(y_span) * muy

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xw = abs(xmax - xmin)
    yw = abs(ymax - ymin)
    if xw <= 0 or yw <= 0:
        return

    trans = ax.transAxes
    ax_px, ax_py = 0.08, 0.07
    frac_x_drawn = 0.0

    if show_scale_bar_x:
        target_len_m = max(x_span_m * 0.2, 1.0)
        bar_len_m = _nice_scale_length_meters(target_len_m)
        if bar_len_m >= x_span_m:
            bar_len_m = _nice_scale_length_meters(max(x_span_m * 0.5, 1.0))
        bar_len_data = bar_len_m / mux
        frac_x = bar_len_data / xw
        frac_x_drawn = frac_x
        y_line = ax_py
        ax.plot(
            [ax_px, ax_px + frac_x],
            [y_line, y_line],
            color="black",
            linewidth=2.2,
            solid_capstyle="butt",
            zorder=8,
            transform=trans,
            clip_on=False,
        )
        ax.plot([ax_px, ax_px], [y_line - 0.012, y_line + 0.012], color="black", linewidth=1.4, zorder=8, transform=trans, clip_on=False)
        ax.plot(
            [ax_px + frac_x, ax_px + frac_x],
            [y_line - 0.012, y_line + 0.012],
            color="black",
            linewidth=1.4,
            zorder=8,
            transform=trans,
            clip_on=False,
        )
        ax.text(
            ax_px + frac_x / 2.0,
            y_line + 0.02,
            _format_meters_text(bar_len_m),
            fontsize=8.2,
            ha="center",
            va="bottom",
            color="black",
            zorder=8,
            transform=trans,
            clip_on=False,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        )

    if show_scale_bar_y:
        target_len_m = max(y_span_m * 0.2, 1.0)
        bar_len_m_y = _nice_scale_length_meters(target_len_m)
        if bar_len_m_y >= y_span_m:
            bar_len_m_y = _nice_scale_length_meters(max(y_span_m * 0.5, 1.0))
        bar_len_data_y = bar_len_m_y / muy
        frac_y = bar_len_data_y / yw
        vx0 = ax_px + (frac_x_drawn + 0.03) if show_scale_bar_x else ax_px
        vy0 = ax_py
        ax.plot([vx0, vx0], [vy0, vy0 + frac_y], color="black", linewidth=2.2, solid_capstyle="butt", zorder=8, transform=trans, clip_on=False)
        ax.plot([vx0 - 0.012, vx0 + 0.012], [vy0, vy0], color="black", linewidth=1.4, zorder=8, transform=trans, clip_on=False)
        ax.plot(
            [vx0 - 0.012, vx0 + 0.012],
            [vy0 + frac_y, vy0 + frac_y],
            color="black",
            linewidth=1.4,
            zorder=8,
            transform=trans,
            clip_on=False,
        )
        ax.text(
            vx0 + 0.024,
            vy0 + frac_y / 2.0,
            _format_meters_text(bar_len_m_y),
            fontsize=8.2,
            ha="left",
            va="center",
            color="black",
            zorder=8,
            rotation=90,
            transform=trans,
            clip_on=False,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
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
    show_coordinate_grid: bool = True,
    show_scale_bar_x: bool = True,
    show_scale_bar_y: bool = False,
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
    custom_gradient_colors: list[str] | None = None,
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
    axis_tick_fontsize_x: float = 9.0,
    axis_tick_fontsize_y: float = 9.0,
    mercator_force_square: bool = True,
    mercator_span_scale_x: float = 1.0,
    mercator_span_scale_y: float = 1.0,
    basemap_offset_east_m: float = 0.0,
    basemap_offset_north_m: float = 0.0,
    show_isoline_map: bool = True,
    span_scale_x: float = 1.0,
    span_scale_y: float = 1.0,
) -> Figure:
    ap_plot, ac_plot = mirror_fields(ap, ac, enforce_mirror=enforce_mirror)
    fill_levels, contour_levels, vmin, vmax, cmap, bnorm = _build_fill_cmap_and_norm(
        ap_plot,
        ac_plot,
        levels_count=levels_count,
        levels_step=levels_step,
        cmap_start=cmap_start,
        cmap_end=cmap_end,
        custom_gradient_colors=custom_gradient_colors,
    )
    _cf_extras = {"norm": bnorm, "extend": "neither"} if bnorm is not None else {"vmin": vmin, "vmax": vmax}

    if vertical_layout:
        fig, axes = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=False)
    else:
        # constrained_layout breaks with dual colorbars + basemap (axes collapse to zero width).
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=False)

    fill_alpha = float(np.clip(map_layer_alpha, 0.05, 1.0)) if basemap_enabled else 1.0

    cx, cy = _mercator_rotation_center(
        triangulation,
        axis_margin,
        mercator_force_square=mercator_force_square,
        mercator_span_scale_x=mercator_span_scale_x,
        mercator_span_scale_y=mercator_span_scale_y,
    )

    tfm0 = _view_data_transform(axes[0], cx, cy, view_rotation_deg)
    tfm1 = _view_data_transform(axes[1], cx, cy, view_rotation_deg)
    kw0 = _tfm_kw(tfm0)
    kw1 = _tfm_kw(tfm1)

    if basemap_enabled and not web_mercator:
        add_satellite_basemap(
            axes[0],
            x=triangulation.x,
            y=triangulation.y,
            axis_margin=axis_margin,
            zorder=0,
            basemap_source_key=basemap_source,
            google_maps_api_key=google_maps_api_key,
            display_transform=tfm0,
            view_rotation_deg=view_rotation_deg,
            mercator_force_square=mercator_force_square,
            mercator_span_scale_x=mercator_span_scale_x,
            mercator_span_scale_y=mercator_span_scale_y,
            basemap_offset_east_m=basemap_offset_east_m,
            basemap_offset_north_m=basemap_offset_north_m,
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
            view_rotation_deg=view_rotation_deg,
            mercator_force_square=mercator_force_square,
            mercator_span_scale_x=mercator_span_scale_x,
            mercator_span_scale_y=mercator_span_scale_y,
            basemap_offset_east_m=basemap_offset_east_m,
            basemap_offset_north_m=basemap_offset_north_m,
        )

    if show_isoline_map:
        if smooth_contours:
            xg_ap, yg_ap, zg_ap = _interpolate_to_grid(
                triangulation, ap_plot, grid_size=grid_size, smooth_sigma=smooth_sigma
            )
            if basemap_enabled:
                cf_ap = _imshow_fill(
                    axes[0],
                    xg=xg_ap,
                    yg=yg_ap,
                    zg=zg_ap,
                    cmap=cmap,
                    alpha=fill_alpha,
                    zorder=1,
                    extras=_cf_extras,
                    interpolation="bilinear",
                    tfm_kw=kw0,
                )
            else:
                cf_ap = axes[0].contourf(
                    xg_ap,
                    yg_ap,
                    zg_ap,
                    levels=fill_levels,
                    cmap=cmap,
                    alpha=fill_alpha,
                    antialiased=False,
                    linewidths=0.0,
                    zorder=1,
                    **_cf_extras,
                    **kw0,
                )
                _seal_filled_contours(cf_ap)
            if show_contour_lines:
                cs = axes[0].contour(
                    xg_ap,
                    yg_ap,
                    zg_ap,
                    levels=contour_levels,
                    colors="#101010",
                    linewidths=float(contour_line_width),
                    alpha=1.0,
                    **kw0,
                )
                if show_contour_labels:
                    axes[0].clabel(cs, inline=True, fontsize=contour_label_font_size, fmt="%.2f")
        else:
            if basemap_enabled:
                xg_ap, yg_ap, zg_ap = _interpolate_to_grid(
                    triangulation, ap_plot, grid_size=grid_size, smooth_sigma=0.0
                )
                cf_ap = _imshow_fill(
                    axes[0],
                    xg=xg_ap,
                    yg=yg_ap,
                    zg=zg_ap,
                    cmap=cmap,
                    alpha=fill_alpha,
                    zorder=1,
                    extras=_cf_extras,
                    interpolation="nearest",
                    tfm_kw=kw0,
                )
            else:
                cf_ap = axes[0].tricontourf(
                    triangulation,
                    ap_plot,
                    levels=fill_levels,
                    cmap=cmap,
                    alpha=fill_alpha,
                    antialiased=False,
                    zorder=1,
                    **_cf_extras,
                    **kw0,
                )
                _seal_filled_contours(cf_ap)
            if show_contour_lines:
                cs = axes[0].tricontour(
                    triangulation,
                    ap_plot,
                    levels=contour_levels,
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
    if web_mercator:
        _style_axis_mercator_degrees(
            axes[0],
            "Карта Ap",
            axis_margin=axis_margin,
            skip_margins=True,
            axis_tick_fontsize_x=axis_tick_fontsize_x,
            axis_tick_fontsize_y=axis_tick_fontsize_y,
            show_coordinate_grid=show_coordinate_grid,
        )
        _set_mercator_view_axis_limits(
            axes[0],
            triangulation,
            axis_margin,
            view_rotation_deg,
            mercator_force_square=mercator_force_square,
            mercator_span_scale_x=mercator_span_scale_x,
            mercator_span_scale_y=mercator_span_scale_y,
        )
    else:
        _style_axis(
            axes[0],
            "Карта Ap",
            x_label=x_label,
            y_label=y_label,
            axis_margin=axis_margin,
            skip_margins=True,
            axis_tick_fontsize_x=axis_tick_fontsize_x,
            axis_tick_fontsize_y=axis_tick_fontsize_y,
            show_coordinate_grid=show_coordinate_grid,
        )
        _set_cartesian_axis_limits(
            axes[0],
            triangulation,
            axis_margin,
            span_scale_x=span_scale_x,
            span_scale_y=span_scale_y,
        )
    if not web_mercator:
        _apply_axis_inversion(axes[0], invert_x=invert_x, invert_y=invert_y)
    if show_isoline_map:
        fig.colorbar(cf_ap, ax=axes[0], location="right", shrink=0.95)

    if show_isoline_map:
        if smooth_contours:
            xg_ac, yg_ac, zg_ac = _interpolate_to_grid(
                triangulation, ac_plot, grid_size=grid_size, smooth_sigma=smooth_sigma
            )
            if basemap_enabled:
                cf_ac = _imshow_fill(
                    axes[1],
                    xg=xg_ac,
                    yg=yg_ac,
                    zg=zg_ac,
                    cmap=cmap,
                    alpha=fill_alpha,
                    zorder=1,
                    extras=_cf_extras,
                    interpolation="bilinear",
                    tfm_kw=kw1,
                )
            else:
                cf_ac = axes[1].contourf(
                    xg_ac,
                    yg_ac,
                    zg_ac,
                    levels=fill_levels,
                    cmap=cmap,
                    alpha=fill_alpha,
                    antialiased=False,
                    linewidths=0.0,
                    zorder=1,
                    **_cf_extras,
                    **kw1,
                )
                _seal_filled_contours(cf_ac)
            if show_contour_lines:
                cs = axes[1].contour(
                    xg_ac,
                    yg_ac,
                    zg_ac,
                    levels=contour_levels,
                    colors="#101010",
                    linewidths=float(contour_line_width),
                    alpha=1.0,
                    **kw1,
                )
                if show_contour_labels:
                    axes[1].clabel(cs, inline=True, fontsize=contour_label_font_size, fmt="%.2f")
        else:
            if basemap_enabled:
                xg_ac, yg_ac, zg_ac = _interpolate_to_grid(
                    triangulation, ac_plot, grid_size=grid_size, smooth_sigma=0.0
                )
                cf_ac = _imshow_fill(
                    axes[1],
                    xg=xg_ac,
                    yg=yg_ac,
                    zg=zg_ac,
                    cmap=cmap,
                    alpha=fill_alpha,
                    zorder=1,
                    extras=_cf_extras,
                    interpolation="nearest",
                    tfm_kw=kw1,
                )
            else:
                cf_ac = axes[1].tricontourf(
                    triangulation,
                    ac_plot,
                    levels=fill_levels,
                    cmap=cmap,
                    alpha=fill_alpha,
                    antialiased=False,
                    zorder=1,
                    **_cf_extras,
                    **kw1,
                )
                _seal_filled_contours(cf_ac)
            if show_contour_lines:
                cs = axes[1].tricontour(
                    triangulation,
                    ac_plot,
                    levels=contour_levels,
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
    if web_mercator:
        _style_axis_mercator_degrees(
            axes[1],
            "Карта Ac",
            axis_margin=axis_margin,
            skip_margins=True,
            axis_tick_fontsize_x=axis_tick_fontsize_x,
            axis_tick_fontsize_y=axis_tick_fontsize_y,
            show_coordinate_grid=show_coordinate_grid,
        )
        _set_mercator_view_axis_limits(
            axes[1],
            triangulation,
            axis_margin,
            view_rotation_deg,
            mercator_force_square=mercator_force_square,
            mercator_span_scale_x=mercator_span_scale_x,
            mercator_span_scale_y=mercator_span_scale_y,
        )
    else:
        _style_axis(
            axes[1],
            "Карта Ac",
            x_label=x_label,
            y_label=y_label,
            axis_margin=axis_margin,
            skip_margins=True,
            axis_tick_fontsize_x=axis_tick_fontsize_x,
            axis_tick_fontsize_y=axis_tick_fontsize_y,
            show_coordinate_grid=show_coordinate_grid,
        )
        _set_cartesian_axis_limits(
            axes[1],
            triangulation,
            axis_margin,
            span_scale_x=span_scale_x,
            span_scale_y=span_scale_y,
        )
    if not web_mercator:
        _apply_axis_inversion(axes[1], invert_x=invert_x, invert_y=invert_y)
    if show_isoline_map:
        fig.colorbar(cf_ac, ax=axes[1], location="right", shrink=0.95)

    if vertical_layout:
        fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.07, hspace=0.3)
    else:
        fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.11, wspace=0.32)

    # Одинаковый м/пиксель по X и Y в EPSG:3857; иначе «auto» даёт разное растяжение на разных фигурах
    # и подложка на overlay визуально не совпадает с отдельными картами при том же охвате.
    if web_mercator:
        _finalize_web_mercator_aspect_after_layout(
            axes,
            triangulation=triangulation,
            axis_margin=axis_margin,
            mercator_force_square=mercator_force_square,
            mercator_span_scale_x=mercator_span_scale_x,
            mercator_span_scale_y=mercator_span_scale_y,
            view_rotation_deg=view_rotation_deg,
        )
        if basemap_enabled:
            for ax, tfm in ((axes[0], tfm0), (axes[1], tfm1)):
                add_satellite_basemap(
                    ax,
                    x=triangulation.x,
                    y=triangulation.y,
                    axis_margin=axis_margin,
                    zorder=0,
                    basemap_source_key=basemap_source,
                    google_maps_api_key=google_maps_api_key,
                    display_transform=tfm,
                    view_rotation_deg=view_rotation_deg,
                    mercator_force_square=mercator_force_square,
                    mercator_span_scale_x=mercator_span_scale_x,
                    mercator_span_scale_y=mercator_span_scale_y,
                    basemap_offset_east_m=basemap_offset_east_m,
                    basemap_offset_north_m=basemap_offset_north_m,
                    preserve_axes_limits=True,
                )
        for ax in axes:
            _apply_axis_inversion(ax, invert_x=invert_x, invert_y=invert_y)

    for ax in axes:
        _draw_scale_bars(
            ax,
            triangulation.x,
            triangulation.y,
            show_scale_bar_x=show_scale_bar_x,
            show_scale_bar_y=show_scale_bar_y,
            web_mercator=web_mercator,
        )

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
    show_coordinate_grid: bool = True,
    show_scale_bar_x: bool = True,
    show_scale_bar_y: bool = False,
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
    custom_gradient_colors: list[str] | None = None,
    basemap_enabled: bool = False,
    map_layer_alpha: float = 0.85,
    web_mercator: bool = False,
    basemap_source: str = "esri",
    coordinate_degrees_lon_lat: tuple[np.ndarray, np.ndarray] | None = None,
    google_maps_api_key: str | None = None,
    view_rotation_deg: float = 0.0,
    axis_tick_fontsize_x: float = 9.0,
    axis_tick_fontsize_y: float = 9.0,
    mercator_force_square: bool = True,
    mercator_span_scale_x: float = 1.0,
    mercator_span_scale_y: float = 1.0,
    basemap_offset_east_m: float = 0.0,
    basemap_offset_north_m: float = 0.0,
    show_isoline_map: bool = True,
    span_scale_x: float = 1.0,
    span_scale_y: float = 1.0,
) -> Figure:
    ap_plot, ac_plot = mirror_fields(ap, ac, enforce_mirror=enforce_mirror)
    fill_levels, contour_levels, vmin, vmax, cmap, bnorm = _build_fill_cmap_and_norm(
        ap_plot,
        ac_plot,
        levels_count=levels_count,
        levels_step=levels_step,
        cmap_start=cmap_start,
        cmap_end=cmap_end,
        custom_gradient_colors=custom_gradient_colors,
    )
    _cf_extras = {"norm": bnorm, "extend": "neither"} if bnorm is not None else {"vmin": vmin, "vmax": vmax}

    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5), constrained_layout=False)

    map_alpha_factor = float(np.clip(map_layer_alpha, 0.05, 1.0)) if basemap_enabled else 1.0

    cx, cy = _mercator_rotation_center(
        triangulation,
        axis_margin,
        mercator_force_square=mercator_force_square,
        mercator_span_scale_x=mercator_span_scale_x,
        mercator_span_scale_y=mercator_span_scale_y,
    )
    tfm = _view_data_transform(ax, cx, cy, view_rotation_deg)
    kw = _tfm_kw(tfm)

    if basemap_enabled and not web_mercator:
        add_satellite_basemap(
            ax,
            x=triangulation.x,
            y=triangulation.y,
            axis_margin=axis_margin,
            zorder=0,
            basemap_source_key=basemap_source,
            google_maps_api_key=google_maps_api_key,
            display_transform=tfm,
            view_rotation_deg=view_rotation_deg,
            mercator_force_square=mercator_force_square,
            mercator_span_scale_x=mercator_span_scale_x,
            mercator_span_scale_y=mercator_span_scale_y,
            basemap_offset_east_m=basemap_offset_east_m,
            basemap_offset_north_m=basemap_offset_north_m,
        )

    if show_isoline_map:
        if smooth_contours:
            xg_ap, yg_ap, zg_ap = _interpolate_to_grid(
                triangulation, ap_plot, grid_size=grid_size, smooth_sigma=smooth_sigma
            )
            xg_ac, yg_ac, zg_ac = _interpolate_to_grid(
                triangulation, ac_plot, grid_size=grid_size, smooth_sigma=smooth_sigma
            )
            ap_fill_alpha = max(0.2, min(0.85, alpha * 0.8)) * map_alpha_factor
            ac_fill_alpha = max(0.2, min(0.85, (1.0 - alpha) * 0.8)) * map_alpha_factor
            if basemap_enabled:
                _imshow_fill(
                    ax,
                    xg=xg_ap,
                    yg=yg_ap,
                    zg=zg_ap,
                    cmap=cmap,
                    alpha=ap_fill_alpha,
                    zorder=1,
                    extras=_cf_extras,
                    interpolation="bilinear",
                    tfm_kw=kw,
                )
                _imshow_fill(
                    ax,
                    xg=xg_ac,
                    yg=yg_ac,
                    zg=zg_ac,
                    cmap=cmap,
                    alpha=ac_fill_alpha,
                    zorder=1,
                    extras=_cf_extras,
                    interpolation="bilinear",
                    tfm_kw=kw,
                )
            else:
                cf1 = ax.contourf(
                    xg_ap,
                    yg_ap,
                    zg_ap,
                    levels=fill_levels,
                    cmap=cmap,
                    alpha=ap_fill_alpha,
                    antialiased=False,
                    linewidths=0.0,
                    zorder=1,
                    **_cf_extras,
                    **kw,
                )
                cf2 = ax.contourf(
                    xg_ac,
                    yg_ac,
                    zg_ac,
                    levels=fill_levels,
                    cmap=cmap,
                    alpha=ac_fill_alpha,
                    antialiased=False,
                    linewidths=0.0,
                    zorder=1,
                    **_cf_extras,
                    **kw,
                )
                _seal_filled_contours(cf1)
                _seal_filled_contours(cf2)
            if show_contour_lines:
                lw_main = float(contour_line_width)
                cs1 = ax.contour(
                    xg_ap,
                    yg_ap,
                    zg_ap,
                    levels=contour_levels,
                    colors="#000000",
                    linewidths=lw_main,
                    alpha=1.0,
                    **kw,
                )
                cs2 = ax.contour(
                    xg_ac,
                    yg_ac,
                    zg_ac,
                    levels=contour_levels,
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
            if basemap_enabled:
                xg_ap, yg_ap, zg_ap = _interpolate_to_grid(
                    triangulation, ap_plot, grid_size=grid_size, smooth_sigma=0.0
                )
                xg_ac, yg_ac, zg_ac = _interpolate_to_grid(
                    triangulation, ac_plot, grid_size=grid_size, smooth_sigma=0.0
                )
                _imshow_fill(
                    ax,
                    xg=xg_ap,
                    yg=yg_ap,
                    zg=zg_ap,
                    cmap=cmap,
                    alpha=ap_fill_alpha,
                    zorder=1,
                    extras=_cf_extras,
                    interpolation="nearest",
                    tfm_kw=kw,
                )
                _imshow_fill(
                    ax,
                    xg=xg_ac,
                    yg=yg_ac,
                    zg=zg_ac,
                    cmap=cmap,
                    alpha=ac_fill_alpha,
                    zorder=1,
                    extras=_cf_extras,
                    interpolation="nearest",
                    tfm_kw=kw,
                )
            else:
                cf1 = ax.tricontourf(
                    triangulation,
                    ap_plot,
                    levels=fill_levels,
                    cmap=cmap,
                    alpha=ap_fill_alpha,
                    antialiased=False,
                    zorder=1,
                    **_cf_extras,
                    **kw,
                )
                cf2 = ax.tricontourf(
                    triangulation,
                    ac_plot,
                    levels=fill_levels,
                    cmap=cmap,
                    alpha=ac_fill_alpha,
                    antialiased=False,
                    zorder=1,
                    **_cf_extras,
                    **kw,
                )
                _seal_filled_contours(cf1)
                _seal_filled_contours(cf2)
            if show_contour_lines:
                lw_main = float(contour_line_width)
                cs1 = ax.tricontour(
                    triangulation,
                    ap_plot,
                    levels=contour_levels,
                    colors="#000000",
                    linewidths=lw_main,
                    alpha=1.0,
                    **kw,
                )
                cs2 = ax.tricontour(
                    triangulation,
                    ac_plot,
                    levels=contour_levels,
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
    if web_mercator:
        _style_axis_mercator_degrees(
            ax,
            "Overlay Ap/Ac (проверка совпадения изолиний)",
            axis_margin=axis_margin,
            skip_margins=True,
            axis_tick_fontsize_x=axis_tick_fontsize_x,
            axis_tick_fontsize_y=axis_tick_fontsize_y,
            show_coordinate_grid=show_coordinate_grid,
        )
        _set_mercator_view_axis_limits(
            ax,
            triangulation,
            axis_margin,
            view_rotation_deg,
            mercator_force_square=mercator_force_square,
            mercator_span_scale_x=mercator_span_scale_x,
            mercator_span_scale_y=mercator_span_scale_y,
        )
    else:
        _style_axis(
            ax,
            "Overlay Ap/Ac (проверка совпадения изолиний)",
            x_label=x_label,
            y_label=y_label,
            axis_margin=axis_margin,
            skip_margins=True,
            axis_tick_fontsize_x=axis_tick_fontsize_x,
            axis_tick_fontsize_y=axis_tick_fontsize_y,
            show_coordinate_grid=show_coordinate_grid,
        )
        _set_cartesian_axis_limits(
            ax,
            triangulation,
            axis_margin,
            span_scale_x=span_scale_x,
            span_scale_y=span_scale_y,
        )
    if not web_mercator:
        _apply_axis_inversion(ax, invert_x=invert_x, invert_y=invert_y)

    if show_isoline_map:
        if bnorm is not None:
            mappable = plt.cm.ScalarMappable(norm=bnorm, cmap=cmap)
        else:
            mappable = plt.cm.ScalarMappable(cmap=cmap)
            mappable.set_clim(vmin, vmax)
        fig.colorbar(mappable, ax=ax, location="right", shrink=0.95)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.93, bottom=0.11)
    if web_mercator:
        _finalize_web_mercator_aspect_after_layout(
            (ax,),
            triangulation=triangulation,
            axis_margin=axis_margin,
            mercator_force_square=mercator_force_square,
            mercator_span_scale_x=mercator_span_scale_x,
            mercator_span_scale_y=mercator_span_scale_y,
            view_rotation_deg=view_rotation_deg,
        )
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
                view_rotation_deg=view_rotation_deg,
                mercator_force_square=mercator_force_square,
                mercator_span_scale_x=mercator_span_scale_x,
                mercator_span_scale_y=mercator_span_scale_y,
                basemap_offset_east_m=basemap_offset_east_m,
                basemap_offset_north_m=basemap_offset_north_m,
                preserve_axes_limits=True,
            )
        _apply_axis_inversion(ax, invert_x=invert_x, invert_y=invert_y)
    _draw_scale_bars(
        ax,
        triangulation.x,
        triangulation.y,
        show_scale_bar_x=show_scale_bar_x,
        show_scale_bar_y=show_scale_bar_y,
        web_mercator=web_mercator,
    )
    return fig
