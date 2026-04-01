from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from typing import TYPE_CHECKING

import contextily as ctx
import numpy as np
from matplotlib import pyplot as plt
from pyproj import Transformer

if TYPE_CHECKING:
    from matplotlib.path import Path as MplPath
    from matplotlib.transforms import Transform

try:
    from xyzservices import TileProvider

    _GOOGLE_SATELLITE_UNOFFICIAL = TileProvider(
        name="GoogleSatelliteUnofficial",
        url="https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attribution="© Google (неофициально; возможны ограничения ToS)",
        max_zoom=21,
    )
except Exception:  # noqa: BLE001
    # Fallback: contextily accepts URL template strings for some versions.
    _GOOGLE_SATELLITE_UNOFFICIAL = "https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

try:
    _ESRI_WORLD_IMAGERY = ctx.providers.Esri.WorldImagery
except AttributeError:
    import xyzservices.providers as xyz  # type: ignore[import-not-found]

    _ESRI_WORLD_IMAGERY = xyz.Esri.WorldImagery

_transformer_4326_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_transformer_3857_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

# Google tile servers often reject contextily's default User-Agent; browser-like headers help.
# Keys lowercase so they override contextily's {"user-agent": USER_AGENT, **headers} merge.
_GOOGLE_TILE_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "referer": "https://www.google.com/maps",
}

_STATIC_MAP_UA = _GOOGLE_TILE_HEADERS["user-agent"]


def _tile_request_headers(basemap_source_key: str | None) -> dict[str, str] | None:
    k = (basemap_source_key or "esri").strip().lower()
    if k == "google":
        return dict(_GOOGLE_TILE_HEADERS)
    return None


class BasemapError(Exception):
    """Failed to load satellite tiles (network, provider, or CRS issue)."""


def resolve_basemap_source(key: str | None):
    """Return tile provider for Esri (default) or unofficial Google satellite."""
    k = (key or "esri").strip().lower()
    if k == "google":
        return _GOOGLE_SATELLITE_UNOFFICIAL
    return _ESRI_WORLD_IMAGERY


def looks_like_wgs84_degrees(x: np.ndarray, y: np.ndarray) -> bool:
    if x.size == 0 or y.size == 0:
        return False
    return float(np.max(np.abs(x))) <= 180.0 and float(np.max(np.abs(y))) <= 90.0


def lon_lat_to_web_mercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    east, north = _transformer_4326_to_3857.transform(
        np.asarray(lon, dtype=float),
        np.asarray(lat, dtype=float),
    )
    return np.asarray(east, dtype=float), np.asarray(north, dtype=float)


def web_mercator_to_lon_lat(east: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = _transformer_3857_to_4326.transform(
        np.asarray(east, dtype=float),
        np.asarray(north, dtype=float),
    )
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def compute_mercator_square_extent(
    x: np.ndarray,
    y: np.ndarray,
    axis_margin: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Square axis limits in EPSG:3857 (meters), same logic as former add_satellite_basemap."""
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span_x = x_max - x_min
    span_y = y_max - y_min

    margin = float(np.clip(axis_margin, 0.0, 0.3))
    min_span_m = 50.0
    span_x = max(span_x, min_span_m)
    span_y = max(span_y, min_span_m)
    span = max(span_x, span_y)
    xc = 0.5 * (x_min + x_max)
    yc = 0.5 * (y_min + y_max)
    half = 0.5 * span * (1.0 + 2.0 * margin)
    xlim = (xc - half, xc + half)
    ylim = (yc - half, yc + half)
    return xlim, ylim


def view_rotation_basemap_extent_scale(view_rotation_deg: float) -> float:
    """При повороте вида квадрат подложки в экранных координатах нужно больше «запаса» в данных.

    Для квадрата со стороной :math:`L`, повёрнутого на угол θ, ось-ориентированный охват
    имеет сторону :math:`L(|\\cos\\theta|+|\\sin\\theta|)`. Без увеличения extent тайлы
    не покрывают углы прямоугольника осей (белые треугольники).
    """
    rad = np.radians(float(view_rotation_deg))
    return float(max(1.0, abs(np.cos(rad)) + abs(np.sin(rad))))


def expand_mercator_square_extent(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    scale: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Тот же центр, полуширина квадрата умножена на ``scale`` (≥ 1)."""
    scale = float(max(1.0, scale))
    xc = 0.5 * (xlim[0] + xlim[1])
    yc = 0.5 * (ylim[0] + ylim[1])
    half = 0.5 * (xlim[1] - xlim[0]) * scale
    return (xc - half, xc + half), (yc - half, yc + half)


def _imread_image_bytes(data: bytes) -> np.ndarray:
    """Декодирует PNG/JPEG/WebP и т.д. Yandex для спутника часто отдаёт JPEG; без Pillow
    matplotlib умеет только встроенный PNG — отсюда ошибка «не PNG file»."""
    if not data:
        raise BasemapError("Пустой ответ от сервера подложки.")
    stripped = data.lstrip()
    if stripped.startswith(b"<?xml") or stripped.startswith(b"<error"):
        raise BasemapError(
            "Сервис вернул XML вместо изображения: " + data[:800].decode("utf-8", errors="replace")
        )
    try:
        from PIL import Image
    except ImportError as exc:
        raise BasemapError(
            "Для подложки (JPEG/PNG) установите Pillow: pip install pillow"
        ) from exc
    try:
        with Image.open(BytesIO(data)) as im:
            im = im.convert("RGBA")
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except Exception as exc:  # noqa: BLE001
        raise BasemapError(f"Не удалось разобрать изображение подложки: {exc}") from exc
    return arr


def _lon_lat_bbox_from_mercator_square(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Min/max lon/lat (degrees) for the square extent in EPSG:3857."""
    corners_e_n = [
        (xlim[0], ylim[0]),
        (xlim[0], ylim[1]),
        (xlim[1], ylim[0]),
        (xlim[1], ylim[1]),
    ]
    lon, lat = web_mercator_to_lon_lat(
        np.array([c[0] for c in corners_e_n], dtype=float),
        np.array([c[1] for c in corners_e_n], dtype=float),
    )
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    return lon_min, lon_max, lat_min, lat_max


def _fetch_google_static_hybrid_png(api_key: str, visible_lat_lon: str) -> bytes:
    """visible: 'lat1,lon1|lat2,lon2|...' per Static Maps API."""
    params = {
        "visible": visible_lat_lon,
        "size": "640x640",
        "maptype": "hybrid",
        "key": api_key,
    }
    url = "https://maps.googleapis.com/maps/api/staticmap?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _STATIC_MAP_UA})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise BasemapError(f"Google Static Maps HTTP {exc.code}: {body}") from exc
    except OSError as exc:
        raise BasemapError(f"Google Static Maps: сеть или таймаут: {exc}") from exc


def add_google_hybrid_static_basemap(
    ax,
    *,
    x: np.ndarray,
    y: np.ndarray,
    axis_margin: float = 0.05,
    zorder: float = 0,
    api_key: str,
    display_transform: "Transform | None" = None,
    clip_path: "MplPath | None" = None,
    clip_transform: "Transform | None" = None,
    view_rotation_deg: float = 0.0,
) -> None:
    """Hybrid (satellite + labels) via official Static Maps API; axes in EPSG:3857."""
    inner_xlim, inner_ylim = compute_mercator_square_extent(x, y, axis_margin=axis_margin)
    k = view_rotation_basemap_extent_scale(view_rotation_deg)
    fetch_xlim, fetch_ylim = expand_mercator_square_extent(inner_xlim, inner_ylim, k)

    ax.set_xlim(*inner_xlim)
    ax.set_ylim(*inner_ylim)
    ax.set_aspect("auto")

    corners_e_n = [
        (fetch_xlim[0], fetch_ylim[0]),
        (fetch_xlim[0], fetch_ylim[1]),
        (fetch_xlim[1], fetch_ylim[0]),
        (fetch_xlim[1], fetch_ylim[1]),
    ]
    lon_c, lat_c = web_mercator_to_lon_lat(
        np.array([c[0] for c in corners_e_n], dtype=float),
        np.array([c[1] for c in corners_e_n], dtype=float),
    )
    parts = [f"{float(lat_c[i])},{float(lon_c[i])}" for i in range(4)]
    visible = "|".join(parts)

    png = _fetch_google_static_hybrid_png(api_key, visible)
    img = _imread_image_bytes(png)
    tfm = display_transform if display_transform is not None else ax.transData
    im = ax.imshow(
        img,
        extent=(fetch_xlim[0], fetch_xlim[1], fetch_ylim[0], fetch_ylim[1]),
        origin="upper",
        zorder=zorder,
        aspect="auto",
        transform=tfm,
    )
    if clip_path is not None:
        ct = clip_transform if clip_transform is not None else tfm
        im.set_clip_path(clip_path, transform=ct)
    ax.set_xlim(*inner_xlim)
    ax.set_ylim(*inner_ylim)
    ax.set_aspect("auto")


def _fetch_yandex_static_png(
    lon_c: float,
    lat_c: float,
    lon_span: float,
    lat_span: float,
    layer: str,
    width: int = 650,
    height: int = 450,
) -> bytes:
    """Yandex Static Maps API (как weather-bot/src/plotter.py: ll + spn + l + size)."""
    w = min(int(width), 650)
    h = min(int(height), 450)
    if w < 1:
        w = 450
    if h < 1:
        h = 450
    # Как в weather-bot/plotter.py: size=ширина,высота (через запятую, не "x").
    url = (
        f"https://static-maps.yandex.ru/1.x/"
        f"?ll={lon_c},{lat_c}&spn={lon_span},{lat_span}&l={layer}&size={w},{h}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": _STATIC_MAP_UA})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise BasemapError(f"Yandex Static Maps HTTP {exc.code}: {body}") from exc
    except OSError as exc:
        raise BasemapError(f"Yandex Static Maps: сеть или таймаут: {exc}") from exc


def add_yandex_static_basemap(
    ax,
    *,
    x: np.ndarray,
    y: np.ndarray,
    axis_margin: float = 0.05,
    zorder: float = 0,
    layer: str = "map",
    display_transform: "Transform | None" = None,
    clip_path: "MplPath | None" = None,
    clip_transform: "Transform | None" = None,
    view_rotation_deg: float = 0.0,
) -> None:
    """Подложка через Yandex Static API (схема ``map`` или гибрид ``sat,skl``), оси EPSG:3857."""
    inner_xlim, inner_ylim = compute_mercator_square_extent(x, y, axis_margin=axis_margin)
    k = view_rotation_basemap_extent_scale(view_rotation_deg)
    fetch_xlim, fetch_ylim = expand_mercator_square_extent(inner_xlim, inner_ylim, k)

    ax.set_xlim(*inner_xlim)
    ax.set_ylim(*inner_ylim)
    ax.set_aspect("auto")

    lon_min, lon_max, lat_min, lat_max = _lon_lat_bbox_from_mercator_square(fetch_xlim, fetch_ylim)
    lon_span = max(lon_max - lon_min, 1e-5)
    lat_span = max(lat_max - lat_min, 1e-5)
    lon_c = 0.5 * (lon_min + lon_max)
    lat_c = 0.5 * (lat_min + lat_max)

    w = min(650, max(1, int(650 * min(k, 2.0))))
    h = min(450, max(1, int(450 * min(k, 2.0))))

    raw = _fetch_yandex_static_png(lon_c, lat_c, lon_span, lat_span, layer=layer, width=w, height=h)
    img = _imread_image_bytes(raw)
    tfm = display_transform if display_transform is not None else ax.transData
    im = ax.imshow(
        img,
        extent=(fetch_xlim[0], fetch_xlim[1], fetch_ylim[0], fetch_ylim[1]),
        origin="upper",
        zorder=zorder,
        aspect="auto",
        transform=tfm,
    )
    if clip_path is not None:
        ct = clip_transform if clip_transform is not None else tfm
        im.set_clip_path(clip_path, transform=ct)
    ax.set_xlim(*inner_xlim)
    ax.set_ylim(*inner_ylim)
    ax.set_aspect("auto")


def add_satellite_basemap(
    ax,
    *,
    x: np.ndarray,
    y: np.ndarray,
    axis_margin: float = 0.05,
    source=None,
    zorder: float = 0,
    basemap_source_key: str | None = None,
    google_maps_api_key: str | None = None,
    display_transform: "Transform | None" = None,
    clip_path: "MplPath | None" = None,
    clip_transform: "Transform | None" = None,
    view_rotation_deg: float = 0.0,
) -> None:
    """Draw satellite/hybrid under data in EPSG:3857."""
    k = (basemap_source_key or "esri").strip().lower()

    if k == "yandex":
        add_yandex_static_basemap(
            ax,
            x=x,
            y=y,
            axis_margin=axis_margin,
            zorder=zorder,
            layer="map",
            display_transform=display_transform,
            clip_path=clip_path,
            clip_transform=clip_transform,
            view_rotation_deg=view_rotation_deg,
        )
        return

    if k == "yandex_hybrid":
        add_yandex_static_basemap(
            ax,
            x=x,
            y=y,
            axis_margin=axis_margin,
            zorder=zorder,
            layer="sat,skl",
            display_transform=display_transform,
            clip_path=clip_path,
            clip_transform=clip_transform,
            view_rotation_deg=view_rotation_deg,
        )
        return

    if k == "google_hybrid":
        if not google_maps_api_key or not google_maps_api_key.strip():
            raise BasemapError(
                "Для Google Hybrid нужен API-ключ: скопируйте config.example.yaml в "
                "config.local.yaml и задайте google_maps_api_key, либо переменную окружения "
                "GOOGLE_MAPS_API_KEY."
            )
        add_google_hybrid_static_basemap(
            ax,
            x=x,
            y=y,
            axis_margin=axis_margin,
            zorder=zorder,
            api_key=google_maps_api_key.strip(),
            display_transform=display_transform,
            clip_path=clip_path,
            clip_transform=clip_transform,
            view_rotation_deg=view_rotation_deg,
        )
        return

    if source is None:
        source = resolve_basemap_source(basemap_source_key)

    inner_xlim, inner_ylim = compute_mercator_square_extent(x, y, axis_margin=axis_margin)
    scale = view_rotation_basemap_extent_scale(view_rotation_deg)
    fetch_xlim, fetch_ylim = expand_mercator_square_extent(inner_xlim, inner_ylim, scale)

    ax.set_xlim(*fetch_xlim)
    ax.set_ylim(*fetch_ylim)
    ax.set_aspect("auto")

    headers = _tile_request_headers(basemap_source_key)
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=source,
            zorder=zorder,
            reset_extent=True,
            headers=headers,
        )
    except TypeError:
        # Older contextily: no headers/reset_extent keyword — still patch Google UA for tile fetches.
        import contextily.tile as ctile

        old_ua = ctile.USER_AGENT
        if headers and "user-agent" in headers:
            ctile.USER_AGENT = headers["user-agent"]
        try:
            try:
                ctx.add_basemap(ax, crs="EPSG:3857", source=source, zorder=zorder, reset_extent=True)
            except TypeError:
                ctx.add_basemap(ax, crs="EPSG:3857", source=source, zorder=zorder)
        finally:
            ctile.USER_AGENT = old_ua
    except Exception as exc:  # noqa: BLE001
        raise BasemapError(f"Не удалось загрузить спутниковую подложку: {exc}") from exc

    if display_transform is not None or clip_path is not None:
        tfm = display_transform if display_transform is not None else ax.transData
        ct = clip_transform if clip_transform is not None else tfm
        for im in ax.images:
            if display_transform is not None:
                im.set_transform(display_transform)
            if clip_path is not None:
                im.set_clip_path(clip_path, transform=ct)

    ax.set_xlim(*inner_xlim)
    ax.set_ylim(*inner_ylim)
    ax.set_aspect("auto")
