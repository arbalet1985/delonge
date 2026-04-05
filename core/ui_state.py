"""Сериализация параметров интерфейса (JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

UI_STATE_VERSION = 1

# Файл по умолчанию рядом с домашней папкой пользователя
def default_ui_config_path() -> Path:
    return Path.home() / ".delaunay_maps_ui.json"


def default_ui_state_dict() -> dict[str, Any]:
    """Значения по умолчанию (совпадают с начальной инициализацией MainWindow)."""
    n = 8
    grad: list[str] = []
    for i in range(n):
        t = i / max(n - 1, 1)
        v = int(round(255 * (1.0 - t)))
        grad.append(f"#{v:02x}{v:02x}{v:02x}")
    return {
        "version": UI_STATE_VERSION,
        "cmap_start": "#ffffff",
        "cmap_end": "#000000",
        "levels_spin": 20,
        "use_levels_step": False,
        "levels_step": 1.0,
        "point_size": 18,
        "annotation_font": 7,
        "axis_tick_font_x": 9,
        "axis_tick_font_y": 9,
        "axis_margin_pct": 5,
        "mercator_square": False,
        "mercator_span_x_pct": 100,
        "mercator_span_y_pct": 100,
        "map_extent_zoom_pct": 100,
        "dual_canvas_width_px": 1200,
        "dual_canvas_height_px": 550,
        "overlay_canvas_width_px": 600,
        "overlay_canvas_height_px": 550,
        "basemap_offset_e": 0.0,
        "basemap_offset_n": 0.0,
        "smoothing_pct": 20,
        "map_view_rotation": 0.0,
        "basemap_enabled": False,
        "basemap_source_key": "yandex_hybrid",
        "map_opacity_pct": 75,
        "overlay_alpha_pct": 50,
        "use_custom_gradient": False,
        "gradient_steps": 8,
        "custom_gradient_colors": grad,
        "gradient_user_edited_indices": [],
        "smooth_contours": True,
        "show_points": True,
        "show_coordinates": False,
        "show_rn": False,
        "show_coordinate_grid": True,
        "show_scale_bar_x": True,
        "show_scale_bar_y": False,
        "show_contour_lines": True,
        "show_contour_labels": False,
        "contour_label_font": 8,
        "contour_line_width": 1.0,
        "invert_x": False,
        "invert_y": False,
        "swap_xy": False,
        "enforce_mirror": True,
        "horizontal_align_vertical": False,
        "tabs_index": 0,
        "settings_panel_expanded": True,
        "last_excel_path": None,
        "column_selections": {"rn": "", "x": "", "y": "", "ap": "", "ac": ""},
    }


def load_ui_state_from_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def save_ui_state_to_file(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(state)
    out["version"] = UI_STATE_VERSION
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
