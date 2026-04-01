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
from scipy.ndimage import map_coordinates

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


def compute_mercator_axis_extent(
    x: np.ndarray,
    y: np.ndarray,
    axis_margin: float = 0.05,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    force_square: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Пределы осей в EPSG:3857 (м).

    По умолчанию (``force_square=False``) охват по X и Y независим по размаху данных —
    без «лишних» полей сверху/снизу у вытянутых полос. При ``force_square=True`` —
    квадрат со стороной ``max(span_x, span_y)`` (старое поведение).
    """
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
    xc = 0.5 * (x_min + x_max)
    yc = 0.5 * (y_min + y_max)
    sx = float(np.clip(scale_x, 0.05, 10.0))
    sy = float(np.clip(scale_y, 0.05, 10.0))
    pad = 1.0 + 2.0 * margin
    if force_square:
        span = max(span_x, span_y)
        s = max(sx, sy)
        half = 0.5 * span * pad * s
        xlim = (xc - half, xc + half)
        ylim = (yc - half, yc + half)
    else:
        half_x = 0.5 * span_x * pad * sx
        half_y = 0.5 * span_y * pad * sy
        xlim = (xc - half_x, xc + half_x)
        ylim = (yc - half_y, yc + half_y)
    return xlim, ylim


def compute_mercator_square_extent(
    x: np.ndarray,
    y: np.ndarray,
    axis_margin: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Квадратный охват в EPSG:3857 (как раньше: max из размахов X/Y)."""
    return compute_mercator_axis_extent(
        x, y, axis_margin, scale_x=1.0, scale_y=1.0, force_square=True
    )


def expand_mercator_extent_for_view_rotation(
    inner_xlim: tuple[float, float],
    inner_ylim: tuple[float, float],
    view_rotation_deg: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Минимальный ось-ориентированный прямоугольник в данных, чтобы после поворота вида
    подложка покрыла весь внутренний охват (аналог :math:`|\\cos|+|\\sin|` для квадрата).

    Для внутреннего полуразмера :math:`h_x, h_y` полуразмеры внешнего охвата:
    :math:`h_x' = |\\cos\\theta| h_x + |\\sin\\theta| h_y`, и симметрично по Y.
    """
    if abs(float(view_rotation_deg)) < 1e-9:
        return inner_xlim, inner_ylim
    xc = 0.5 * (inner_xlim[0] + inner_xlim[1])
    yc = 0.5 * (inner_ylim[0] + inner_ylim[1])
    hx = 0.5 * (inner_xlim[1] - inner_xlim[0])
    hy = 0.5 * (inner_ylim[1] - inner_ylim[0])
    rad = np.radians(float(view_rotation_deg))
    c, s = abs(np.cos(rad)), abs(np.sin(rad))
    hx_out = c * hx + s * hy
    hy_out = s * hx + c * hy
    return (xc - hx_out, xc + hx_out), (yc - hy_out, yc + hy_out)


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


def _extent_with_basemap_offset_m(
    extent_xyxy: tuple[float, float, float, float],
    offset_east_m: float,
    offset_north_m: float,
) -> tuple[float, float, float, float]:
    """Сдвиг подложки в метрах EPSG:3857 (восток / север), оси и данные не трогаем."""
    x0, x1, y0, y1 = extent_xyxy
    ox, oy = float(offset_east_m), float(offset_north_m)
    return (x0 + ox, x1 + ox, y0 + oy, y1 + oy)


def _merge_mercator_fetch_extents(
    a: tuple[tuple[float, float], tuple[float, float]],
    b: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    ax0, ax1 = sorted((float(a[0][0]), float(a[0][1])))
    ay0, ay1 = sorted((float(a[1][0]), float(a[1][1])))
    bx0, bx1 = sorted((float(b[0][0]), float(b[0][1])))
    by0, by1 = sorted((float(b[1][0]), float(b[1][1])))
    return (min(ax0, bx0), max(ax1, bx1)), (min(ay0, by0), max(ay1, by1))


def _inflate_mercator_fetch_extent(
    fetch_xlim: tuple[float, float],
    fetch_ylim: tuple[float, float],
    *,
    rel_pad: float = 0.002,
    min_pad_m: float = 2.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Небольшой запас по краю после warp/тайлов при больших сдвигах и повороте."""
    x0, x1 = sorted((float(fetch_xlim[0]), float(fetch_xlim[1])))
    y0, y1 = sorted((float(fetch_ylim[0]), float(fetch_ylim[1])))
    sx, sy = x1 - x0, y1 - y0
    px = max(float(rel_pad) * sx, float(min_pad_m))
    py = max(float(rel_pad) * sy, float(min_pad_m))
    return (x0 - px, x1 + px), (y0 - py, y1 + py)


def expand_fetch_mercator_for_view_rotation_and_offset(
    inner_xlim: tuple[float, float],
    inner_ylim: tuple[float, float],
    view_rotation_deg: float,
    offset_east_m: float,
    offset_north_m: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Охват загрузки подложки с учётом поворота вида и сдвига E/N в метрах Mercator.

    Недостаточно отдельно расширять под поворот (окно осей) и отдельно под сдвиг
    (прообраз ``inner − offset``): при совместных −20° и +90 m углы окна требуют
    расширения и для сдвинутого прообраза с тем же правилом ``|cos|+|sin|``.
    """
    r_view = expand_mercator_extent_for_view_rotation(
        inner_xlim, inner_ylim, view_rotation_deg
    )
    ox, oy = float(offset_east_m), float(offset_north_m)
    ix0, ix1 = sorted((float(inner_xlim[0]), float(inner_xlim[1])))
    iy0, iy1 = sorted((float(inner_ylim[0]), float(inner_ylim[1])))
    shifted_xlim = (ix0 - ox, ix1 - ox)
    shifted_ylim = (iy0 - oy, iy1 - oy)
    r_shift = expand_mercator_extent_for_view_rotation(
        shifted_xlim, shifted_ylim, view_rotation_deg
    )
    merged = _merge_mercator_fetch_extents(r_view, r_shift)
    return _inflate_mercator_fetch_extent(merged[0], merged[1])


def _static_map_pixel_size_for_mercator_extent(
    fetch_xlim: tuple[float, float],
    fetch_ylim: tuple[float, float],
    *,
    max_w: int,
    max_h: int,
    k_pix: float = 1.0,
    min_short_side: int = 300,
) -> tuple[int, int]:
    """Подбирает ширину/высоту запроса так, чтобы отношение сторон совпадало с охватом в метрах Mercator.

    Ограничивает слишком «плоские» запросы (650×30 и т.п.): при малом числе строк warp даёт
    заметный дрейф относительно изолиний при повороте и нестандартном масштабе охвата.
    """
    dx = abs(float(fetch_xlim[1] - fetch_xlim[0]))
    dy = abs(float(fetch_ylim[1] - fetch_ylim[0]))
    k = float(max(1.0, min(k_pix, 2.0)))
    cap_w = max(1, int(max_w * k))
    cap_h = max(1, int(max_h * k))
    if dx < 1e-9 or dy < 1e-9:
        return min(cap_w, max_w), min(cap_h, max_h)
    merc_aspect = dx / dy
    box_aspect = max_w / max(max_h, 1)
    if merc_aspect >= box_aspect:
        w = min(max_w, cap_w)
        h = max(1, min(max_h, int(round(w / merc_aspect))))
    else:
        h = min(max_h, cap_h)
        w = max(1, min(max_w, int(round(h * merc_aspect))))
    w, h = max(1, w), max(1, h)
    m = min(w, h)
    if m < min_short_side:
        f = float(min_short_side) / float(m)
        w = min(max_w, max(1, int(round(w * f))))
        h = min(max_h, max(1, int(round(h * f))))
    return max(1, w), max(1, h)


def _warp_static_rgba_lonlat_linear_to_mercator_extent(
    img: np.ndarray,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    mercator_xlim: tuple[float, float],
    mercator_ylim: tuple[float, float],
    min_grid_short_side: int = 560,
) -> np.ndarray:
    """Снимок Static API в линейных lon/lat по пикселям приводит к сетке Web Mercator (как изолинии).

    Без этого снимок растягивается по ``extent`` в метрах, а содержимое остаётся «географически
    линейным» — расхождение с данными в EPSG:3857, заметное при неквадратном охвате.

    Сетку в метрах строим с достаточным разрешением по короткой стороне (не равным размеру PNG),
    иначе при узком охвате и повороте bilinear даёт видимый «дрейф» подложки.
    """
    if img.ndim != 3 or img.shape[2] < 3:
        return img
    h0, w0 = int(img.shape[0]), int(img.shape[1])
    if h0 < 2 or w0 < 2:
        return img
    x_min = min(float(mercator_xlim[0]), float(mercator_xlim[1]))
    x_max = max(float(mercator_xlim[0]), float(mercator_xlim[1]))
    y_min = min(float(mercator_ylim[0]), float(mercator_ylim[1]))
    y_max = max(float(mercator_ylim[0]), float(mercator_ylim[1]))
    dlon = max(float(lon_max - lon_min), 1e-12)
    dlat = max(float(lat_max - lat_min), 1e-12)
    merc_w = x_max - x_min
    merc_h = y_max - y_min
    aspect = merc_w / max(merc_h, 1e-12)
    # Равномерная плотность выборки: nw/nh ≈ merc_w/merc_h; поднимаем меньшую сторону сетки.
    if aspect >= 1.0:
        nh = max(h0, min_grid_short_side)
        nw = max(w0, int(round(nh * aspect)))
    else:
        nw = max(w0, min_grid_short_side)
        nh = max(h0, int(round(nw / aspect)))
    xi = np.linspace(x_min, x_max, nw, dtype=np.float64)
    yi = np.linspace(y_max, y_min, nh, dtype=np.float64)
    X, Y = np.meshgrid(xi, yi)
    lon, lat = web_mercator_to_lon_lat(X.ravel(), Y.ravel())
    lon = lon.reshape(nh, nw)
    lat = lat.reshape(nh, nw)
    u = (lon - lon_min) / dlon * (w0 - 1)
    v = (lat_max - lat) / dlat * (h0 - 1)
    nch = img.shape[2]
    out = np.empty((nh, nw, nch), dtype=np.float32)
    for c in range(nch):
        out[..., c] = map_coordinates(
            np.asarray(img[..., c], dtype=np.float64),
            [v, u],
            order=1,
            mode="constant",
            cval=0.0,
        ).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def _fetch_google_static_hybrid_png(
    api_key: str,
    visible_lat_lon: str,
    *,
    width: int = 640,
    height: int = 640,
) -> bytes:
    """visible: 'lat1,lon1|lat2,lon2|...' per Static Maps API."""
    w = max(1, min(int(width), 640))
    h = max(1, min(int(height), 640))
    params = {
        "visible": visible_lat_lon,
        "size": f"{w}x{h}",
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
    mercator_force_square: bool = True,
    mercator_span_scale_x: float = 1.0,
    mercator_span_scale_y: float = 1.0,
    basemap_offset_east_m: float = 0.0,
    basemap_offset_north_m: float = 0.0,
) -> None:
    """Hybrid (satellite + labels) via official Static Maps API; axes in EPSG:3857."""
    inner_xlim, inner_ylim = compute_mercator_axis_extent(
        x,
        y,
        axis_margin,
        scale_x=mercator_span_scale_x,
        scale_y=mercator_span_scale_y,
        force_square=mercator_force_square,
    )
    fetch_xlim, fetch_ylim = expand_fetch_mercator_for_view_rotation_and_offset(
        inner_xlim,
        inner_ylim,
        view_rotation_deg,
        basemap_offset_east_m,
        basemap_offset_north_m,
    )

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

    lon_min, lon_max, lat_min, lat_max = _lon_lat_bbox_from_mercator_square(fetch_xlim, fetch_ylim)
    rx = (fetch_xlim[1] - fetch_xlim[0]) / max(inner_xlim[1] - inner_xlim[0], 1e-9)
    ry = (fetch_ylim[1] - fetch_ylim[0]) / max(inner_ylim[1] - inner_ylim[0], 1e-9)
    k_pix = float(max(rx, ry, 1.0))
    gw, gh = _static_map_pixel_size_for_mercator_extent(
        fetch_xlim, fetch_ylim, max_w=640, max_h=640, k_pix=k_pix
    )
    png = _fetch_google_static_hybrid_png(api_key, visible, width=gw, height=gh)
    img = _imread_image_bytes(png)
    min_grid = 560
    if mercator_span_scale_y < 0.66:
        min_grid = min(900, int(min_grid * max(1.0, 0.66 / max(float(mercator_span_scale_y), 0.18))))
    img = _warp_static_rgba_lonlat_linear_to_mercator_extent(
        img,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        mercator_xlim=fetch_xlim,
        mercator_ylim=fetch_ylim,
        min_grid_short_side=min_grid,
    )
    tfm = display_transform if display_transform is not None else ax.transData
    ext = _extent_with_basemap_offset_m(
        (fetch_xlim[0], fetch_xlim[1], fetch_ylim[0], fetch_ylim[1]),
        basemap_offset_east_m,
        basemap_offset_north_m,
    )
    im = ax.imshow(
        img,
        extent=ext,
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
    mercator_force_square: bool = True,
    mercator_span_scale_x: float = 1.0,
    mercator_span_scale_y: float = 1.0,
    basemap_offset_east_m: float = 0.0,
    basemap_offset_north_m: float = 0.0,
) -> None:
    """Подложка через Yandex Static API (схема ``map`` или гибрид ``sat,skl``), оси EPSG:3857."""
    inner_xlim, inner_ylim = compute_mercator_axis_extent(
        x,
        y,
        axis_margin,
        scale_x=mercator_span_scale_x,
        scale_y=mercator_span_scale_y,
        force_square=mercator_force_square,
    )
    fetch_xlim, fetch_ylim = expand_fetch_mercator_for_view_rotation_and_offset(
        inner_xlim,
        inner_ylim,
        view_rotation_deg,
        basemap_offset_east_m,
        basemap_offset_north_m,
    )

    ax.set_xlim(*inner_xlim)
    ax.set_ylim(*inner_ylim)
    ax.set_aspect("auto")

    lon_min, lon_max, lat_min, lat_max = _lon_lat_bbox_from_mercator_square(fetch_xlim, fetch_ylim)
    lon_span = max(lon_max - lon_min, 1e-5)
    lat_span = max(lat_max - lat_min, 1e-5)
    lon_c = 0.5 * (lon_min + lon_max)
    lat_c = 0.5 * (lat_min + lat_max)

    rx = (fetch_xlim[1] - fetch_xlim[0]) / max(inner_xlim[1] - inner_xlim[0], 1e-9)
    ry = (fetch_ylim[1] - fetch_ylim[0]) / max(inner_ylim[1] - inner_ylim[0], 1e-9)
    k_pix = float(max(rx, ry, 1.0))
    w, h = _static_map_pixel_size_for_mercator_extent(
        fetch_xlim, fetch_ylim, max_w=650, max_h=450, k_pix=k_pix
    )

    raw = _fetch_yandex_static_png(lon_c, lat_c, lon_span, lat_span, layer=layer, width=w, height=h)
    img = _imread_image_bytes(raw)
    min_grid = 560
    if mercator_span_scale_y < 0.66:
        min_grid = min(900, int(min_grid * max(1.0, 0.66 / max(float(mercator_span_scale_y), 0.18))))
    img = _warp_static_rgba_lonlat_linear_to_mercator_extent(
        img,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        mercator_xlim=fetch_xlim,
        mercator_ylim=fetch_ylim,
        min_grid_short_side=min_grid,
    )
    tfm = display_transform if display_transform is not None else ax.transData
    ext = _extent_with_basemap_offset_m(
        (fetch_xlim[0], fetch_xlim[1], fetch_ylim[0], fetch_ylim[1]),
        basemap_offset_east_m,
        basemap_offset_north_m,
    )
    im = ax.imshow(
        img,
        extent=ext,
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
    mercator_force_square: bool = True,
    mercator_span_scale_x: float = 1.0,
    mercator_span_scale_y: float = 1.0,
    basemap_offset_east_m: float = 0.0,
    basemap_offset_north_m: float = 0.0,
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
            mercator_force_square=mercator_force_square,
            mercator_span_scale_x=mercator_span_scale_x,
            mercator_span_scale_y=mercator_span_scale_y,
            basemap_offset_east_m=basemap_offset_east_m,
            basemap_offset_north_m=basemap_offset_north_m,
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
            mercator_force_square=mercator_force_square,
            mercator_span_scale_x=mercator_span_scale_x,
            mercator_span_scale_y=mercator_span_scale_y,
            basemap_offset_east_m=basemap_offset_east_m,
            basemap_offset_north_m=basemap_offset_north_m,
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
            mercator_force_square=mercator_force_square,
            mercator_span_scale_x=mercator_span_scale_x,
            mercator_span_scale_y=mercator_span_scale_y,
            basemap_offset_east_m=basemap_offset_east_m,
            basemap_offset_north_m=basemap_offset_north_m,
        )
        return

    if source is None:
        source = resolve_basemap_source(basemap_source_key)

    inner_xlim, inner_ylim = compute_mercator_axis_extent(
        x,
        y,
        axis_margin,
        scale_x=mercator_span_scale_x,
        scale_y=mercator_span_scale_y,
        force_square=mercator_force_square,
    )
    fetch_xlim, fetch_ylim = expand_fetch_mercator_for_view_rotation_and_offset(
        inner_xlim,
        inner_ylim,
        view_rotation_deg,
        basemap_offset_east_m,
        basemap_offset_north_m,
    )

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

    ox, oy = float(basemap_offset_east_m), float(basemap_offset_north_m)
    if ox != 0.0 or oy != 0.0:
        for im in ax.images:
            l, r, b, t = im.get_extent()
            im.set_extent(_extent_with_basemap_offset_m((l, r, b, t), ox, oy))

    ax.set_xlim(*inner_xlim)
    ax.set_ylim(*inner_ylim)
    ax.set_aspect("auto")
