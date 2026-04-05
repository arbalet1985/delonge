from __future__ import annotations

from pathlib import Path
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QColorDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.data_loader import ColumnMap, auto_detect_columns, load_points, read_excel_headers
from core.basemap import BasemapError, lon_lat_to_web_mercator, looks_like_wgs84_degrees
from core.config import load_app_config
from core.interpolation import build_triangulation
from core.plotting import render_dual_maps, render_overlay_map
from core.ui_state import (
    default_ui_config_path,
    default_ui_state_dict,
    load_ui_state_from_file,
    save_ui_state_to_file,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Delaunay Maps: Ap / Ac")
        self.resize(1360, 860)

        self.file_path: str | None = None
        self.data = None
        self.triangulation = None

        self.cmap_start = "#ffffff"
        self.cmap_end = "#000000"

        self.main_canvas = FigureCanvas(Figure(figsize=(12, 5)))
        self.overlay_canvas = FigureCanvas(Figure(figsize=(6, 5.5)))

        self.column_combos: dict[str, QComboBox] = {}
        self.file_label = QLabel("Файл: не выбран")
        self.file_label.setWordWrap(True)

        central = QWidget()
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(220)
        controls_scroll.setWidget(self._build_controls_panel())

        charts_panel = self._build_charts_panel()
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.setObjectName("mainSplitter")
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(controls_scroll)
        self._main_splitter.addWidget(charts_panel)
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes([380, 980])

        root_layout.addWidget(self._main_splitter, 1)

        self.setCentralWidget(central)
        self._loading_ui_config = False
        self._setup_menu_bar()
        self._apply_styles()
        self._redraw_debounce_timer = QTimer(self)
        self._redraw_debounce_timer.setSingleShot(True)
        self._redraw_debounce_timer.setInterval(320)
        self._redraw_debounce_timer.timeout.connect(self._flush_debounced_redraw)
        self._connect_live_redraw_signals()
        self._using_web_mercator = False
        self._label_lon_lat_deg: tuple[np.ndarray, np.ndarray] | None = None
        self._sync_map_opacity_text()
        self._update_basemap_availability()
        self._load_ui_config_auto()
        self._sync_placeholder_canvas_pixel_sizes()

    def _canvas_size_px_for_tab(self, tab_index: int) -> tuple[int, int]:
        if tab_index == 0:
            return (int(self.dual_canvas_width_spin.value()), int(self.dual_canvas_height_spin.value()))
        return (int(self.overlay_canvas_width_spin.value()), int(self.overlay_canvas_height_spin.value()))

    def _apply_pixel_size_to_figure_canvas(self, canvas: FigureCanvas, tab_index: int) -> None:
        w, h = self._canvas_size_px_for_tab(tab_index)
        w = int(max(120, min(6000, w)))
        h = int(max(120, min(6000, h)))
        fig = canvas.figure
        dpi = float(fig.dpi)
        fig.set_size_inches(w / dpi, h / dpi)
        canvas.setFixedSize(w, h)

    def _sync_placeholder_canvas_pixel_sizes(self) -> None:
        self._apply_pixel_size_to_figure_canvas(self.main_canvas, 0)
        self._apply_pixel_size_to_figure_canvas(self.overlay_canvas, 1)
        self.main_canvas.draw_idle()
        self.overlay_canvas.draw_idle()

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        file_group = QGroupBox("Данные")
        file_layout = QVBoxLayout(file_group)
        load_btn = QPushButton("Загрузить Excel")
        load_btn.clicked.connect(self.on_load_excel)
        file_layout.addWidget(load_btn)
        file_layout.addWidget(self.file_label)
        layout.addWidget(file_group)

        map_group = QGroupBox("Сопоставление колонок")
        form = QFormLayout(map_group)
        for key, label in [("rn", "rn"), ("x", "x"), ("y", "y"), ("ap", "Ap"), ("ac", "Ac")]:
            combo = QComboBox()
            combo.setMinimumWidth(160)
            combo.currentIndexChanged.connect(self._update_basemap_availability)
            self.column_combos[key] = combo
            form.addRow(f"{label}:", combo)
        layout.addWidget(map_group)

        settings_wrap = QWidget()
        settings_layout = QVBoxLayout(settings_wrap)
        settings_layout.setSpacing(8)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_group = settings_wrap

        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(5, 40)
        self.levels_spin.setValue(20)

        self.use_levels_step_checkbox = QCheckBox("Изолинии по шагу")
        self.use_levels_step_checkbox.setChecked(False)

        self.levels_step_spin = QDoubleSpinBox()
        self.levels_step_spin.setRange(0.1, 1_000_000.0)
        self.levels_step_spin.setDecimals(4)
        self.levels_step_spin.setSingleStep(0.1)
        self.levels_step_spin.setValue(1.0)

        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(5, 80)
        self.point_size_spin.setValue(18)

        self.annotation_font_spin = QSpinBox()
        self.annotation_font_spin.setRange(5, 24)
        self.annotation_font_spin.setValue(7)

        self.axis_tick_font_x_spin = QSpinBox()
        self.axis_tick_font_x_spin.setRange(5, 18)
        self.axis_tick_font_x_spin.setValue(9)
        self.axis_tick_font_x_spin.setToolTip("Размер шрифта делений по оси X (градусы/метры).")
        self.axis_tick_font_y_spin = QSpinBox()
        self.axis_tick_font_y_spin.setRange(5, 18)
        self.axis_tick_font_y_spin.setValue(9)
        self.axis_tick_font_y_spin.setToolTip("Размер шрифта делений по оси Y.")

        self.axis_margin_spin = QSpinBox()
        self.axis_margin_spin.setRange(0, 20)
        self.axis_margin_spin.setValue(5)
        self.axis_margin_spin.setSuffix(" %")
        self.axis_margin_spin.setToolTip(
            "Запас между точками данных и рамкой осей (в долях размаха). Влияет на изолинии "
            "и в обычных координатах, и в Web Mercator."
        )

        self.map_extent_zoom_spin = QSpinBox()
        self.map_extent_zoom_spin.setRange(25, 500)
        self.map_extent_zoom_spin.setValue(100)
        self.map_extent_zoom_spin.setSuffix(" %")
        self.map_extent_zoom_spin.setToolTip(
            "Общий множитель охвата карты изолиний в Web Mercator (умножает «По осям X/Y» ниже). "
            "Больше % — шире охват (больше территории в кадре); меньше % — крупнее детали."
        )

        self.mercator_square_extent_checkbox = QCheckBox("Квадратный охват (max X/Y)")
        self.mercator_square_extent_checkbox.setChecked(False)
        self.mercator_square_extent_checkbox.setToolTip(
            "Вкл: одинаковый масштаб по X и Y (квадратная область, как раньше). "
            "Выкл: охват по каждой оси по размаху данных — меньше пустого места у вытянутых полос. "
            "Для точек в градусах WGS84 карта строится в Web Mercator (с подложкой или без)."
        )
        self.mercator_span_x_spin = QSpinBox()
        self.mercator_span_x_spin.setRange(25, 400)
        self.mercator_span_x_spin.setValue(100)
        self.mercator_span_x_spin.setSuffix(" %")
        self.mercator_span_x_spin.setToolTip(
            "Множитель полуразмаха по оси X (после отступа рамки), вместе с «Общий %» задаёт масштаб осей."
        )
        self.mercator_span_y_spin = QSpinBox()
        self.mercator_span_y_spin.setRange(25, 400)
        self.mercator_span_y_spin.setValue(100)
        self.mercator_span_y_spin.setSuffix(" %")
        self.mercator_span_y_spin.setToolTip(
            "Множитель полуразмаха по оси Y; при разных X/Y можно поджать или вытянуть кадр по вертикали."
        )

        self.basemap_offset_e_spin = QDoubleSpinBox()
        self.basemap_offset_e_spin.setRange(-20000.0, 20000.0)
        self.basemap_offset_e_spin.setDecimals(1)
        self.basemap_offset_e_spin.setSingleStep(5.0)
        self.basemap_offset_e_spin.setSuffix(" м")
        self.basemap_offset_e_spin.setValue(0.0)
        self.basemap_offset_e_spin.setToolTip(
            "Сдвиг только слоя подложки по оси X в метрах Web Mercator (восток +). "
            "Точки и изолинии не смещаются — подстройка к Яндекс/Google и т.д."
        )
        self.basemap_offset_n_spin = QDoubleSpinBox()
        self.basemap_offset_n_spin.setRange(-20000.0, 20000.0)
        self.basemap_offset_n_spin.setDecimals(1)
        self.basemap_offset_n_spin.setSingleStep(5.0)
        self.basemap_offset_n_spin.setSuffix(" м")
        self.basemap_offset_n_spin.setValue(0.0)
        self.basemap_offset_n_spin.setToolTip(
            "Сдвиг подложки по оси Y в метрах Mercator (север +). Данные остаются на месте."
        )

        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(0, 40)
        self.smoothing_spin.setValue(20)
        self.smoothing_spin.setSuffix(" %")

        self.map_view_rotation_spin = QDoubleSpinBox()
        self.map_view_rotation_spin.setRange(-180.0, 180.0)
        self.map_view_rotation_spin.setDecimals(1)
        self.map_view_rotation_spin.setSingleStep(0.5)
        self.map_view_rotation_spin.setSuffix("°")
        self.map_view_rotation_spin.setValue(0.0)
        self.map_view_rotation_spin.setToolTip(
            "Единый поворот: изолинии, точки и подложка вместе вокруг центра охвата (Web Mercator). "
            "При ненулевом угле охват осей слегка расширяется (AABB повёрнутого прямоугольника), чтобы изолинии "
            "не обрезались и заполняли область; числа на осях (м / градусы) при этом меняются. "
            "Тайлы подложки запрашиваются с запасом. Масштаб 1 м = 1 м по осям."
        )

        self.dual_canvas_width_spin = QSpinBox()
        self.dual_canvas_width_spin.setRange(200, 6000)
        self.dual_canvas_width_spin.setSingleStep(10)
        self.dual_canvas_width_spin.setValue(1200)
        self.dual_canvas_width_spin.setSuffix(" px")
        self.dual_canvas_width_spin.setToolTip(
            "Ширина виджета отрисовки «Отдельные карты» в пикселях (логические px экрана)."
        )
        self.dual_canvas_height_spin = QSpinBox()
        self.dual_canvas_height_spin.setRange(200, 6000)
        self.dual_canvas_height_spin.setSingleStep(10)
        self.dual_canvas_height_spin.setValue(550)
        self.dual_canvas_height_spin.setSuffix(" px")
        self.dual_canvas_height_spin.setToolTip("Высота области карты «Отдельные карты» в пикселях.")
        self.overlay_canvas_width_spin = QSpinBox()
        self.overlay_canvas_width_spin.setRange(200, 6000)
        self.overlay_canvas_width_spin.setSingleStep(10)
        self.overlay_canvas_width_spin.setValue(600)
        self.overlay_canvas_width_spin.setSuffix(" px")
        self.overlay_canvas_width_spin.setToolTip("Ширина области отрисовки вкладки «Overlay» в пикселях.")
        self.overlay_canvas_height_spin = QSpinBox()
        self.overlay_canvas_height_spin.setRange(200, 6000)
        self.overlay_canvas_height_spin.setSingleStep(10)
        self.overlay_canvas_height_spin.setValue(550)
        self.overlay_canvas_height_spin.setSuffix(" px")
        self.overlay_canvas_height_spin.setToolTip("Высота области «Overlay» в пикселях.")

        self.basemap_checkbox = QCheckBox("Спутниковая подложка")
        self.basemap_checkbox.setChecked(False)
        self.basemap_checkbox.setToolTip(
            "Показать спутник/схему под изолиниями. Координаты в градусах WGS84 всегда переводятся "
            "в Web Mercator для карты; подложка только добавляет растр. Нужен интернет."
        )

        self.basemap_source_combo = QComboBox()
        self.basemap_source_combo.addItem("Esri World Imagery (тайлы)", "esri")
        self.basemap_source_combo.addItem("Yandex схема (Static API)", "yandex")
        self.basemap_source_combo.addItem("Yandex спутник+подписи (Static API)", "yandex_hybrid")
        self.basemap_source_combo.addItem("Google спутник (неофиц. тайлы)", "google")
        self.basemap_source_combo.addItem("Google Hybrid (Static Maps API)", "google_hybrid")
        self.basemap_source_combo.setToolTip(
            "Как в weather-bot: Yandex — static-maps.yandex.ru (ll+spn), без ключа; "
            "Google Hybrid — maps.googleapis.com/staticmap (нужен ключ в config.local.yaml или "
            "GOOGLE_MAPS_API_KEY). Esri/Google тайлы — contextily."
        )
        yh_idx = self.basemap_source_combo.findData("yandex_hybrid")
        if yh_idx >= 0:
            self.basemap_source_combo.setCurrentIndex(yh_idx)

        self.map_opacity_slider = QSlider(Qt.Horizontal)
        self.map_opacity_slider.setRange(10, 100)
        self.map_opacity_slider.setValue(75)
        self.map_opacity_label = QLabel("0.75")

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(10, 90)
        self.alpha_slider.setValue(50)
        self.alpha_label = QLabel("0.50")
        self.alpha_slider.valueChanged.connect(self._sync_alpha_text)
        self.map_opacity_slider.valueChanged.connect(self._sync_map_opacity_text)

        self.enforce_mirror_checkbox = QCheckBox("Принудительно зеркалить Ap/Ac")
        self.enforce_mirror_checkbox.setChecked(True)
        self.show_points_checkbox = QCheckBox("Показывать точки")
        self.show_points_checkbox.setChecked(True)
        self.show_coordinates_checkbox = QCheckBox("Показывать координаты точек")
        self.show_coordinates_checkbox.setChecked(False)
        self.show_rn_checkbox = QCheckBox("Показывать названия точек (rn)")
        self.show_rn_checkbox.setChecked(False)
        self.show_contour_lines_checkbox = QCheckBox("Показывать изолинии")
        self.show_contour_lines_checkbox.setChecked(True)

        self.show_contour_labels_checkbox = QCheckBox("Подписывать изолинии")
        self.show_contour_labels_checkbox.setChecked(False)

        self.contour_label_font_spin = QSpinBox()
        self.contour_label_font_spin.setRange(6, 20)
        self.contour_label_font_spin.setValue(8)

        self.contour_line_width_spin = QDoubleSpinBox()
        self.contour_line_width_spin.setRange(0.1, 6.0)
        self.contour_line_width_spin.setDecimals(1)
        self.contour_line_width_spin.setSingleStep(0.1)
        self.contour_line_width_spin.setValue(1.0)

        self.cmap_start_btn = QPushButton("Цвет 1")
        self.cmap_end_btn = QPushButton("Цвет 2")
        self._sync_cmap_button_styles()
        self.cmap_start_btn.clicked.connect(lambda: self._pick_cmap_color(which="start"))
        self.cmap_end_btn.clicked.connect(lambda: self._pick_cmap_color(which="end"))
        self.invert_x_checkbox = QCheckBox("Инвертировать ось X")
        self.invert_x_checkbox.setChecked(False)
        self.invert_y_checkbox = QCheckBox("Инвертировать ось Y")
        self.invert_y_checkbox.setChecked(False)
        self.swap_xy_checkbox = QCheckBox("Поменять X и Y местами")
        self.swap_xy_checkbox.setChecked(False)
        self.smooth_contours_checkbox = QCheckBox("Плавные изолинии")
        self.smooth_contours_checkbox.setChecked(True)

        self.use_custom_gradient_checkbox = QCheckBox("Свои цвета по градациям")
        self.use_custom_gradient_checkbox.setChecked(False)
        self.use_custom_gradient_checkbox.setToolTip(
            "Дискретная расцветка: число градаций и отдельный цвет на каждую. "
            "Иначе — плавный градиент между «Цвет 1» и «Цвет 2»."
        )
        self.gradient_steps_spin = QSpinBox()
        self.gradient_steps_spin.setRange(2, 30)
        self.gradient_steps_spin.setValue(8)
        self.gradient_steps_spin.setToolTip("Сколько цветовых градаций (полос) на карте.")
        self._gradient_color_buttons_per_row = 4
        self.gradient_colors_widget = QWidget()
        self.gradient_colors_layout = QGridLayout(self.gradient_colors_widget)
        self.gradient_colors_layout.setContentsMargins(0, 0, 0, 0)
        self.gradient_colors_layout.setHorizontalSpacing(4)
        self.gradient_colors_layout.setVerticalSpacing(4)
        self.custom_gradient_colors: list[str] = []
        self._gradient_user_edited: set[int] = set()
        self._init_default_gradient_colors(8)
        self._rebuild_gradient_color_buttons()

        grad_row = QWidget()
        grad_row_layout = QHBoxLayout(grad_row)
        grad_row_layout.setContentsMargins(0, 0, 0, 0)
        grad_row_layout.setSpacing(8)
        grad_row_layout.addWidget(self.use_custom_gradient_checkbox)
        grad_row_layout.addWidget(QLabel("Градаций:"))
        grad_row_layout.addWidget(self.gradient_steps_spin)
        self.gradient_blend_btn = QPushButton("Градиент 1→N")
        self.gradient_blend_btn.setToolTip(
            "Равномерно в RGB между якорными плитками: первая и последняя градация и все плитки, "
            "где вы уже выбирали цвет вручную — по очереди между соседними якорями."
        )
        self.gradient_blend_btn.clicked.connect(self._blend_gradient_colors_uniform)
        grad_row_layout.addWidget(self.gradient_blend_btn)
        grad_row_layout.addStretch(1)
        self.use_custom_gradient_checkbox.toggled.connect(self._on_custom_gradient_toggled)

        self.show_coordinate_grid_checkbox = QCheckBox("Координатная сетка")
        self.show_coordinate_grid_checkbox.setChecked(True)
        self.show_scale_bar_x_checkbox = QCheckBox("Шкала масштаба по X (горизонтально на экране)")
        self.show_scale_bar_x_checkbox.setChecked(True)
        self.show_scale_bar_y_checkbox = QCheckBox("Шкала масштаба по Y (вертикально на экране)")
        self.show_scale_bar_y_checkbox.setChecked(False)

        levels_row = QWidget()
        levels_row_layout = QHBoxLayout(levels_row)
        levels_row_layout.setContentsMargins(0, 0, 0, 0)
        levels_row_layout.setSpacing(6)
        levels_row_layout.addWidget(self.levels_spin)
        levels_row_layout.addWidget(self.use_levels_step_checkbox)
        levels_row_layout.addWidget(self.levels_step_spin)
        levels_row_layout.addStretch(1)

        axis_tick_row = QWidget()
        axis_tick_row_layout = QHBoxLayout(axis_tick_row)
        axis_tick_row_layout.setContentsMargins(0, 0, 0, 0)
        axis_tick_row_layout.setSpacing(6)
        axis_tick_row_layout.addWidget(QLabel("X"))
        axis_tick_row_layout.addWidget(self.axis_tick_font_x_spin)
        axis_tick_row_layout.addWidget(QLabel("Y"))
        axis_tick_row_layout.addWidget(self.axis_tick_font_y_spin)
        axis_tick_row_layout.addStretch(1)

        mercator_span_row = QWidget()
        mercator_span_row_layout = QHBoxLayout(mercator_span_row)
        mercator_span_row_layout.setContentsMargins(0, 0, 0, 0)
        mercator_span_row_layout.setSpacing(6)
        mercator_span_row_layout.addWidget(QLabel("X"))
        mercator_span_row_layout.addWidget(self.mercator_span_x_spin)
        mercator_span_row_layout.addWidget(QLabel("Y"))
        mercator_span_row_layout.addWidget(self.mercator_span_y_spin)
        mercator_span_row_layout.addStretch(1)

        map_extent_zoom_row = QWidget()
        map_extent_zoom_row_layout = QHBoxLayout(map_extent_zoom_row)
        map_extent_zoom_row_layout.setContentsMargins(0, 0, 0, 0)
        map_extent_zoom_row_layout.setSpacing(6)
        map_extent_zoom_row_layout.addWidget(QLabel("Общий"))
        map_extent_zoom_row_layout.addWidget(self.map_extent_zoom_spin)
        map_extent_zoom_row_layout.addStretch(1)

        mercator_extent_col = QWidget()
        mercator_extent_col_layout = QVBoxLayout(mercator_extent_col)
        mercator_extent_col_layout.setContentsMargins(0, 0, 0, 0)
        mercator_extent_col_layout.setSpacing(4)
        mercator_extent_col_layout.addWidget(map_extent_zoom_row)
        mercator_extent_col_layout.addWidget(mercator_span_row)

        dual_fig_size_row = QWidget()
        dual_fig_size_layout = QHBoxLayout(dual_fig_size_row)
        dual_fig_size_layout.setContentsMargins(0, 0, 0, 0)
        dual_fig_size_layout.setSpacing(6)
        dual_fig_size_layout.addWidget(QLabel("Шир."))
        dual_fig_size_layout.addWidget(self.dual_canvas_width_spin)
        dual_fig_size_layout.addWidget(QLabel("Выс."))
        dual_fig_size_layout.addWidget(self.dual_canvas_height_spin)

        overlay_fig_size_row = QWidget()
        overlay_fig_size_layout = QHBoxLayout(overlay_fig_size_row)
        overlay_fig_size_layout.setContentsMargins(0, 0, 0, 0)
        overlay_fig_size_layout.setSpacing(6)
        overlay_fig_size_layout.addWidget(QLabel("Шир."))
        overlay_fig_size_layout.addWidget(self.overlay_canvas_width_spin)
        overlay_fig_size_layout.addWidget(QLabel("Выс."))
        overlay_fig_size_layout.addWidget(self.overlay_canvas_height_spin)

        basemap_off_row = QWidget()
        basemap_off_row_layout = QHBoxLayout(basemap_off_row)
        basemap_off_row_layout.setContentsMargins(0, 0, 0, 0)
        basemap_off_row_layout.setSpacing(6)
        basemap_off_row_layout.addWidget(QLabel("E"))
        basemap_off_row_layout.addWidget(self.basemap_offset_e_spin)
        basemap_off_row_layout.addWidget(QLabel("N"))
        basemap_off_row_layout.addWidget(self.basemap_offset_n_spin)
        basemap_off_row_layout.addStretch(1)

        cmap_row = QWidget()
        cmap_row_layout = QHBoxLayout(cmap_row)
        cmap_row_layout.setContentsMargins(0, 0, 0, 0)
        cmap_row_layout.setSpacing(6)
        cmap_row_layout.addWidget(self.cmap_start_btn)
        cmap_row_layout.addWidget(self.cmap_end_btn)
        cmap_row_layout.addStretch(1)

        map_opacity_row = QWidget()
        map_opacity_row_layout = QHBoxLayout(map_opacity_row)
        map_opacity_row_layout.setContentsMargins(0, 0, 0, 0)
        map_opacity_row_layout.setSpacing(6)
        map_opacity_row_layout.addWidget(self.map_opacity_slider, 1)
        map_opacity_row_layout.addWidget(self.map_opacity_label)

        alpha_row = QWidget()
        alpha_row_layout = QHBoxLayout(alpha_row)
        alpha_row_layout.setContentsMargins(0, 0, 0, 0)
        alpha_row_layout.setSpacing(6)
        alpha_row_layout.addWidget(self.alpha_slider, 1)
        alpha_row_layout.addWidget(self.alpha_label)

        self.basemap_source_combo.setMinimumWidth(120)

        def _compact_form(g: QGroupBox) -> QFormLayout:
            fl = QFormLayout(g)
            fl.setSpacing(6)
            fl.setContentsMargins(8, 10, 8, 8)
            fl.setHorizontalSpacing(6)
            fl.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            return fl

        g_iso = QGroupBox("Изолинии")
        fl_iso = _compact_form(g_iso)
        fl_iso.addRow("Число уровней:", levels_row)
        fl_iso.addRow("Сглаживание:", self.smoothing_spin)
        fl_iso.addRow(self.smooth_contours_checkbox)
        settings_layout.addWidget(g_iso)

        g_view = QGroupBox("Охват и размер")
        fl_v = _compact_form(g_view)
        fl_v.addRow("Поворот:", self.map_view_rotation_spin)
        fl_v.addRow("Отступ рамки:", self.axis_margin_spin)
        fl_v.addRow(self.mercator_square_extent_checkbox)
        fl_v.addRow("Mercator:", mercator_extent_col)
        fl_v.addRow("«Отдельные карты»:", dual_fig_size_row)
        fl_v.addRow("«Overlay»:", overlay_fig_size_row)
        settings_layout.addWidget(g_view)

        g_bm = QGroupBox("Подложка")
        fl_bm = _compact_form(g_bm)
        fl_bm.addRow(self.basemap_checkbox)
        fl_bm.addRow("Источник:", self.basemap_source_combo)
        fl_bm.addRow("Сдвиг:", basemap_off_row)
        fl_bm.addRow("Слой / подложка:", map_opacity_row)
        fl_bm.addRow("Overlay Ap/Ac:", alpha_row)
        settings_layout.addWidget(g_bm)

        g_col = QGroupBox("Цвет")
        fl_c = _compact_form(g_col)
        fl_c.addRow("Градиент:", cmap_row)
        fl_c.addRow(grad_row)
        fl_c.addRow("Плитки:", self.gradient_colors_widget)
        settings_layout.addWidget(g_col)

        g_font = QGroupBox("Шрифты")
        fl_f = _compact_form(g_font)
        fl_f.addRow("Точки:", self.point_size_spin)
        fl_f.addRow("Подписи точек:", self.annotation_font_spin)
        fl_f.addRow("Шрифт по осям:", axis_tick_row)
        fl_f.addRow("Подписи изолиний:", self.contour_label_font_spin)
        fl_f.addRow("Толщина линий:", self.contour_line_width_spin)
        settings_layout.addWidget(g_font)

        g_disp = QGroupBox("Что показывать")
        vl_disp = QVBoxLayout(g_disp)
        vl_disp.setSpacing(4)
        vl_disp.setContentsMargins(8, 10, 8, 8)
        for w in (
            self.show_points_checkbox,
            self.show_coordinates_checkbox,
            self.show_rn_checkbox,
            self.show_coordinate_grid_checkbox,
            self.show_scale_bar_x_checkbox,
            self.show_scale_bar_y_checkbox,
            self.show_contour_lines_checkbox,
            self.show_contour_labels_checkbox,
        ):
            vl_disp.addWidget(w)
        settings_layout.addWidget(g_disp)

        g_axes = QGroupBox("Оси и данные")
        vl_ax = QVBoxLayout(g_axes)
        vl_ax.setSpacing(4)
        vl_ax.setContentsMargins(8, 10, 8, 8)
        for w in (
            self.invert_x_checkbox,
            self.invert_y_checkbox,
            self.swap_xy_checkbox,
            self.enforce_mirror_checkbox,
        ):
            vl_ax.addWidget(w)
        settings_layout.addWidget(g_axes)

        self.toggle_settings_btn = QToolButton()
        self.toggle_settings_btn.setText("Свернуть параметры")
        self.toggle_settings_btn.setCheckable(True)
        self.toggle_settings_btn.setChecked(True)
        self.toggle_settings_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.toggle_settings_btn.toggled.connect(self._toggle_settings_visibility)
        layout.addWidget(self.toggle_settings_btn)
        layout.addWidget(settings_wrap)

        actions = QGroupBox("Действия")
        actions_layout = QVBoxLayout(actions)
        self.horizontal_align_btn = QPushButton("Карты: слева направо")
        self.horizontal_align_btn.setCheckable(True)
        self.horizontal_align_btn.setChecked(False)
        self.horizontal_align_btn.setToolTip(
            "Выкл: Ap и Ac рядом (слева направо). Вкл: Ap сверху, Ac снизу. "
            "На данные и подложку не влияет (только расположение панелей)."
        )
        self.horizontal_align_btn.toggled.connect(self._sync_horizontal_align_btn_text)
        build_btn = QPushButton("Построить карты Ap / Ac")
        overlay_btn = QPushButton("Проверить наложение (overlay)")
        save_btn = QPushButton("Сохранить график PNG")
        export_corel_btn = QPushButton("Экспорт Corel")
        actions_layout.addWidget(self.horizontal_align_btn)
        build_btn.clicked.connect(self.on_build_maps)
        overlay_btn.clicked.connect(self.on_build_overlay)
        save_btn.clicked.connect(self.on_save_plot)
        export_corel_btn.clicked.connect(self.on_export_corel)
        actions_layout.addWidget(build_btn)
        actions_layout.addWidget(overlay_btn)
        export_row = QWidget()
        export_row_layout = QHBoxLayout(export_row)
        export_row_layout.setContentsMargins(0, 0, 0, 0)
        export_row_layout.setSpacing(8)
        export_row_layout.addWidget(save_btn)
        export_row_layout.addWidget(export_corel_btn)
        actions_layout.addWidget(export_row)
        layout.addWidget(actions)

        layout.addStretch(1)
        self._on_custom_gradient_toggled(self.use_custom_gradient_checkbox.isChecked())
        return panel

    def _build_charts_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.main_canvas, "Отдельные карты")
        self.tabs.addTab(self.overlay_canvas, "Overlay")
        layout.addWidget(self.tabs, 1)
        return panel

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget { font-size: 13px; }
            QGroupBox {
                border: 1px solid #d0d7de;
                border-radius: 8px;
                margin-top: 10px;
                padding: 8px;
                background: #f7f9fb;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QPushButton {
                background: #2f6feb;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 10px;
            }
            QPushButton:hover { background: #1f5ed6; }
            """
        )

    def _sync_cmap_button_styles(self) -> None:
        self.cmap_start_btn.setStyleSheet(f"background: {self.cmap_start}; color: #000; border-radius: 6px; padding: 8px 10px;")
        self.cmap_end_btn.setStyleSheet(f"background: {self.cmap_end}; color: #000; border-radius: 6px; padding: 8px 10px;")

    def _pick_cmap_color(self, which: str) -> None:
        initial = self.cmap_start if which == "start" else self.cmap_end
        color = QColorDialog.getColor(initial=Qt.white, parent=self, title="Выберите цвет")
        if not color.isValid():
            return
        if which == "start":
            self.cmap_start = color.name()
        else:
            self.cmap_end = color.name()
        self._sync_cmap_button_styles()
        self._schedule_debounced_redraw()

    def _init_default_gradient_colors(self, n: int) -> None:
        """Равномерно от белого к чёрному (для стартового набора)."""
        n = max(2, int(n))
        self.custom_gradient_colors = []
        self._gradient_user_edited = set()
        for i in range(n):
            t = i / max(n - 1, 1)
            v = int(round(255 * (1.0 - t)))
            self.custom_gradient_colors.append(f"#{v:02x}{v:02x}{v:02x}")

    def _rebuild_gradient_color_buttons(self) -> None:
        layout = self.gradient_colors_layout
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        n = self.gradient_steps_spin.value()
        while len(self.custom_gradient_colors) < n:
            self.custom_gradient_colors.append("#808080")
        self.custom_gradient_colors = self.custom_gradient_colors[:n]
        cols = self._gradient_color_buttons_per_row
        for i in range(n):
            btn = QPushButton(str(i + 1))
            btn.setFixedWidth(34)
            c = self.custom_gradient_colors[i]
            btn.setStyleSheet(
                f"background: {c}; color: #111; border-radius: 4px; padding: 3px; font-size: 11px;"
            )
            btn.clicked.connect(lambda checked=False, idx=i: self._pick_gradient_color(idx))
            row, col = i // cols, i % cols
            layout.addWidget(btn, row, col)

    def _on_gradient_steps_changed(self) -> None:
        n = self.gradient_steps_spin.value()
        while len(self.custom_gradient_colors) < n:
            self.custom_gradient_colors.append("#808080")
        self.custom_gradient_colors = self.custom_gradient_colors[:n]
        self._gradient_user_edited = {i for i in self._gradient_user_edited if i < n}
        self._rebuild_gradient_color_buttons()
        self._schedule_debounced_redraw()

    def _on_custom_gradient_toggled(self, checked: bool) -> None:
        self.cmap_start_btn.setEnabled(not checked)
        self.cmap_end_btn.setEnabled(not checked)
        self.gradient_steps_spin.setEnabled(checked)
        self.gradient_colors_widget.setEnabled(checked)
        self.gradient_blend_btn.setEnabled(checked)

    def _pick_gradient_color(self, idx: int) -> None:
        if not (0 <= idx < len(self.custom_gradient_colors)):
            return
        initial = QColor(self.custom_gradient_colors[idx])
        color = QColorDialog.getColor(initial=initial, parent=self, title=f"Градация {idx + 1}")
        if not color.isValid():
            return
        self.custom_gradient_colors[idx] = color.name()
        self._gradient_user_edited.add(idx)
        self._rebuild_gradient_color_buttons()
        self._schedule_debounced_redraw()

    def _custom_gradient_colors_for_render(self) -> list[str] | None:
        if not self.use_custom_gradient_checkbox.isChecked():
            return None
        n = self.gradient_steps_spin.value()
        return list(self.custom_gradient_colors[:n])

    def _blend_gradient_colors_uniform(self) -> None:
        """Линейная интерполяция RGB по сегментам между якорями: 1, N и вручную выбранные плитки."""
        n = self.gradient_steps_spin.value()
        if n < 2:
            return
        while len(self.custom_gradient_colors) < n:
            self.custom_gradient_colors.append("#808080")
        self.custom_gradient_colors = self.custom_gradient_colors[:n]
        rng = set(range(n))
        anchors = sorted({0, n - 1} | (self._gradient_user_edited & rng))
        if len(anchors) < 2:
            anchors = [0, n - 1]
        for si in range(len(anchors) - 1):
            ia, ib = anchors[si], anchors[si + 1]
            ca = QColor(self.custom_gradient_colors[ia])
            cb = QColor(self.custom_gradient_colors[ib])
            span = ib - ia
            if span <= 0:
                continue
            for j in range(ia, ib + 1):
                t = (j - ia) / span
                r = int(round(ca.red() + (cb.red() - ca.red()) * t))
                g = int(round(ca.green() + (cb.green() - ca.green()) * t))
                b = int(round(ca.blue() + (cb.blue() - ca.blue()) * t))
                self.custom_gradient_colors[j] = f"#{r:02x}{g:02x}{b:02x}"
        self._rebuild_gradient_color_buttons()
        self._schedule_debounced_redraw()

    def _setup_menu_bar(self) -> None:
        menu = self.menuBar().addMenu("Настройки")
        act_save = QAction("Сохранить конфигурацию в файл…", self)
        act_save.triggered.connect(self._on_save_ui_config_as)
        menu.addAction(act_save)
        act_load = QAction("Загрузить конфигурацию из файла…", self)
        act_load.triggered.connect(self._on_load_ui_config_from)
        menu.addAction(act_load)
        menu.addSeparator()
        act_reset = QAction("Сбросить к значениям по умолчанию…", self)
        act_reset.triggered.connect(self._on_reset_ui_config)
        menu.addAction(act_reset)

    def _collect_ui_state(self) -> dict:
        src_key = self.basemap_source_combo.currentData()
        src_key = str(src_key) if src_key is not None else "esri"
        return {
            **default_ui_state_dict(),
            "cmap_start": self.cmap_start,
            "cmap_end": self.cmap_end,
            "levels_spin": self.levels_spin.value(),
            "use_levels_step": self.use_levels_step_checkbox.isChecked(),
            "levels_step": self.levels_step_spin.value(),
            "point_size": self.point_size_spin.value(),
            "annotation_font": self.annotation_font_spin.value(),
            "axis_tick_font_x": self.axis_tick_font_x_spin.value(),
            "axis_tick_font_y": self.axis_tick_font_y_spin.value(),
            "axis_margin_pct": self.axis_margin_spin.value(),
            "mercator_square": self.mercator_square_extent_checkbox.isChecked(),
            "mercator_span_x_pct": self.mercator_span_x_spin.value(),
            "mercator_span_y_pct": self.mercator_span_y_spin.value(),
            "map_extent_zoom_pct": self.map_extent_zoom_spin.value(),
            "dual_canvas_width_px": self.dual_canvas_width_spin.value(),
            "dual_canvas_height_px": self.dual_canvas_height_spin.value(),
            "overlay_canvas_width_px": self.overlay_canvas_width_spin.value(),
            "overlay_canvas_height_px": self.overlay_canvas_height_spin.value(),
            "basemap_offset_e": self.basemap_offset_e_spin.value(),
            "basemap_offset_n": self.basemap_offset_n_spin.value(),
            "smoothing_pct": self.smoothing_spin.value(),
            "map_view_rotation": self.map_view_rotation_spin.value(),
            "basemap_enabled": self.basemap_checkbox.isChecked(),
            "basemap_source_key": src_key,
            "map_opacity_pct": self.map_opacity_slider.value(),
            "overlay_alpha_pct": self.alpha_slider.value(),
            "use_custom_gradient": self.use_custom_gradient_checkbox.isChecked(),
            "gradient_steps": self.gradient_steps_spin.value(),
            "custom_gradient_colors": list(self.custom_gradient_colors),
            "gradient_user_edited_indices": sorted(self._gradient_user_edited),
            "smooth_contours": self.smooth_contours_checkbox.isChecked(),
            "show_points": self.show_points_checkbox.isChecked(),
            "show_coordinates": self.show_coordinates_checkbox.isChecked(),
            "show_rn": self.show_rn_checkbox.isChecked(),
            "show_coordinate_grid": self.show_coordinate_grid_checkbox.isChecked(),
            "show_scale_bar_x": self.show_scale_bar_x_checkbox.isChecked(),
            "show_scale_bar_y": self.show_scale_bar_y_checkbox.isChecked(),
            "show_contour_lines": self.show_contour_lines_checkbox.isChecked(),
            "show_contour_labels": self.show_contour_labels_checkbox.isChecked(),
            "contour_label_font": self.contour_label_font_spin.value(),
            "contour_line_width": self.contour_line_width_spin.value(),
            "invert_x": self.invert_x_checkbox.isChecked(),
            "invert_y": self.invert_y_checkbox.isChecked(),
            "swap_xy": self.swap_xy_checkbox.isChecked(),
            "enforce_mirror": self.enforce_mirror_checkbox.isChecked(),
            "horizontal_align_vertical": self.horizontal_align_btn.isChecked(),
            "tabs_index": self.tabs.currentIndex(),
            "settings_panel_width_px": int(self._main_splitter.sizes()[0])
            if getattr(self, "_main_splitter", None) is not None
            else 380,
            "settings_panel_expanded": self.toggle_settings_btn.isChecked(),
            "last_excel_path": self.file_path,
            "column_selections": {k: self.column_combos[k].currentText() for k in self.column_combos},
        }

    def _merge_ui_state(self, loaded: dict) -> dict:
        base = default_ui_state_dict()
        for k, v in loaded.items():
            if k == "version":
                continue
            base[k] = v
        return base

    def _apply_ui_state(self, state: dict) -> None:
        d = self._merge_ui_state(state)
        self._loading_ui_config = True
        try:
            self.cmap_start = str(d["cmap_start"])
            self.cmap_end = str(d["cmap_end"])
            self.levels_spin.setValue(int(d["levels_spin"]))
            self.use_levels_step_checkbox.setChecked(bool(d["use_levels_step"]))
            self.levels_step_spin.setValue(float(d["levels_step"]))
            self.point_size_spin.setValue(int(d["point_size"]))
            self.annotation_font_spin.setValue(int(d["annotation_font"]))
            self.axis_tick_font_x_spin.setValue(int(d["axis_tick_font_x"]))
            self.axis_tick_font_y_spin.setValue(int(d["axis_tick_font_y"]))
            self.axis_margin_spin.setValue(int(d["axis_margin_pct"]))
            self.mercator_square_extent_checkbox.setChecked(bool(d["mercator_square"]))
            self.mercator_span_x_spin.setValue(int(d["mercator_span_x_pct"]))
            self.mercator_span_y_spin.setValue(int(d["mercator_span_y_pct"]))
            self.map_extent_zoom_spin.setValue(int(d.get("map_extent_zoom_pct", 100)))
            if "dual_canvas_width_px" not in d and "dual_fig_width_in" in d:
                self.dual_canvas_width_spin.setValue(
                    max(200, min(6000, int(float(d["dual_fig_width_in"]) * 100)))
                )
                self.dual_canvas_height_spin.setValue(
                    max(200, min(6000, int(float(d.get("dual_fig_height_in", 5.5)) * 100)))
                )
            else:
                self.dual_canvas_width_spin.setValue(int(d.get("dual_canvas_width_px", 1200)))
                self.dual_canvas_height_spin.setValue(int(d.get("dual_canvas_height_px", 550)))
            if "overlay_canvas_width_px" not in d and "overlay_fig_width_in" in d:
                self.overlay_canvas_width_spin.setValue(
                    max(200, min(6000, int(float(d["overlay_fig_width_in"]) * 100)))
                )
                self.overlay_canvas_height_spin.setValue(
                    max(200, min(6000, int(float(d.get("overlay_fig_height_in", 5.5)) * 100)))
                )
            else:
                self.overlay_canvas_width_spin.setValue(int(d.get("overlay_canvas_width_px", 600)))
                self.overlay_canvas_height_spin.setValue(int(d.get("overlay_canvas_height_px", 550)))
            self.basemap_offset_e_spin.setValue(float(d["basemap_offset_e"]))
            self.basemap_offset_n_spin.setValue(float(d["basemap_offset_n"]))
            self.smoothing_spin.setValue(int(d["smoothing_pct"]))
            self.map_view_rotation_spin.setValue(float(d["map_view_rotation"]))
            self.basemap_checkbox.setChecked(bool(d["basemap_enabled"]))
            key = str(d.get("basemap_source_key") or "esri")
            idx = self.basemap_source_combo.findData(key)
            if idx < 0:
                idx = 0
            self.basemap_source_combo.setCurrentIndex(idx)
            self.map_opacity_slider.setValue(int(d["map_opacity_pct"]))
            self.alpha_slider.setValue(int(d["overlay_alpha_pct"]))
            self.use_custom_gradient_checkbox.setChecked(bool(d["use_custom_gradient"]))
            self.gradient_steps_spin.setValue(int(d["gradient_steps"]))
            colors = d.get("custom_gradient_colors")
            if isinstance(colors, list) and all(isinstance(c, str) for c in colors):
                self.custom_gradient_colors = list(colors)
            else:
                self._init_default_gradient_colors(int(d["gradient_steps"]))
            n = self.gradient_steps_spin.value()
            while len(self.custom_gradient_colors) < n:
                self.custom_gradient_colors.append("#808080")
            self.custom_gradient_colors = self.custom_gradient_colors[:n]
            raw_edited = d.get("gradient_user_edited_indices")
            if isinstance(raw_edited, list) and all(isinstance(x, int) for x in raw_edited):
                self._gradient_user_edited = {i for i in raw_edited if 0 <= i < n}
            else:
                self._gradient_user_edited = set()
            self._rebuild_gradient_color_buttons()
            self.smooth_contours_checkbox.setChecked(bool(d["smooth_contours"]))
            self.show_points_checkbox.setChecked(bool(d["show_points"]))
            self.show_coordinates_checkbox.setChecked(bool(d["show_coordinates"]))
            self.show_rn_checkbox.setChecked(bool(d["show_rn"]))
            self.show_coordinate_grid_checkbox.setChecked(bool(d["show_coordinate_grid"]))
            self.show_scale_bar_x_checkbox.setChecked(bool(d["show_scale_bar_x"]))
            self.show_scale_bar_y_checkbox.setChecked(bool(d["show_scale_bar_y"]))
            self.show_contour_lines_checkbox.setChecked(bool(d["show_contour_lines"]))
            self.show_contour_labels_checkbox.setChecked(bool(d["show_contour_labels"]))
            self.contour_label_font_spin.setValue(int(d["contour_label_font"]))
            self.contour_line_width_spin.setValue(float(d["contour_line_width"]))
            self.invert_x_checkbox.setChecked(bool(d["invert_x"]))
            self.invert_y_checkbox.setChecked(bool(d["invert_y"]))
            self.swap_xy_checkbox.setChecked(bool(d["swap_xy"]))
            self.enforce_mirror_checkbox.setChecked(bool(d["enforce_mirror"]))
            self.horizontal_align_btn.setChecked(bool(d["horizontal_align_vertical"]))
            self._sync_horizontal_align_btn_text(self.horizontal_align_btn.isChecked())
            self.tabs.setCurrentIndex(int(d.get("tabs_index", 0)))
            exp = bool(d.get("settings_panel_expanded", True))
            self.toggle_settings_btn.setChecked(exp)
            self.settings_group.setVisible(exp)
            self.toggle_settings_btn.setText("Свернуть параметры" if exp else "Развернуть параметры")
            self._sync_cmap_button_styles()
            self._on_custom_gradient_toggled(self.use_custom_gradient_checkbox.isChecked())
            self._sync_map_opacity_text()
            self._sync_alpha_text()
            self._restore_excel_session_from_state(d)
            self._update_basemap_availability()
        finally:
            self._loading_ui_config = False
        sw = int(d.get("settings_panel_width_px", default_ui_state_dict()["settings_panel_width_px"]))
        QTimer.singleShot(0, lambda: self._apply_settings_panel_splitter_width(sw))

    def _apply_settings_panel_splitter_width(self, target_left: int) -> None:
        """Восстанавливает ширину левой колонки после загрузки конфига (когда splitter уже имеет размер)."""
        sp = getattr(self, "_main_splitter", None)
        if sp is None:
            return
        total = sp.width()
        if total <= 0:
            QTimer.singleShot(50, lambda t=target_left: self._apply_settings_panel_splitter_width(t))
            return
        left = max(220, min(int(target_left), total - 250))
        sp.setSizes([left, total - left])

    def _restore_excel_session_from_state(self, d: dict) -> None:
        path = d.get("last_excel_path")
        if not path or not isinstance(path, str):
            self.file_path = None
            self.file_label.setText("Файл: не выбран")
            return
        p = Path(path)
        if not p.is_file():
            self.file_path = None
            self.file_label.setText("Файл: не выбран")
            return
        try:
            headers = read_excel_headers(str(p))
        except OSError:
            self.file_path = None
            self.file_label.setText("Файл: не выбран")
            return
        self.file_path = str(p)
        self.file_label.setText(f"Файл: {p}")
        self._fill_column_combos(headers)
        cols = d.get("column_selections") or {}
        if isinstance(cols, dict):
            for key in ("rn", "x", "y", "ap", "ac"):
                name = str(cols.get(key, "")).strip()
                if name and self.column_combos[key].findText(name) >= 0:
                    self.column_combos[key].setCurrentText(name)
        self._prepare_data_silently()

    def _load_ui_config_auto(self) -> None:
        path = default_ui_config_path()
        raw = load_ui_state_from_file(path)
        if raw is None:
            return
        self._apply_ui_state(raw)
        self._schedule_debounced_redraw()

    def _on_save_ui_config_as(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить конфигурацию",
            str(Path.cwd() / "pasha_ui_config.json"),
            "JSON (*.json)",
        )
        if not path_str:
            return
        try:
            save_ui_state_to_file(Path(path_str), self._collect_ui_state())
        except OSError as exc:
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить: {exc}")
            return
        QMessageBox.information(self, "Готово", f"Конфигурация сохранена:\n{path_str}")

    def _on_load_ui_config_from(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить конфигурацию",
            str(Path.cwd()),
            "JSON (*.json)",
        )
        if not path_str:
            return
        raw = load_ui_state_from_file(Path(path_str))
        if raw is None:
            QMessageBox.warning(self, "Ошибка", "Не удалось прочитать файл.")
            return
        self._apply_ui_state(raw)
        self._schedule_debounced_redraw()
        QMessageBox.information(self, "Готово", "Конфигурация загружена.")

    def _on_reset_ui_config(self) -> None:
        reply = QMessageBox.question(
            self,
            "Сброс",
            "Сбросить все параметры интерфейса к значениям по умолчанию?\n"
            "Текущий файл данных будет отвязан, если не указан в сохранённой сессии.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._apply_ui_state(default_ui_state_dict())
        try:
            save_ui_state_to_file(default_ui_config_path(), self._collect_ui_state())
        except OSError:
            pass
        self._schedule_debounced_redraw()
        QMessageBox.information(self, "Готово", "Параметры сброшены. Автосохранение записано в файл по умолчанию.")

    def closeEvent(self, event) -> None:
        try:
            save_ui_state_to_file(default_ui_config_path(), self._collect_ui_state())
        except OSError:
            pass
        super().closeEvent(event)

    def _sync_alpha_text(self) -> None:
        self.alpha_label.setText(f"{self.alpha_slider.value() / 100:.2f}")

    def _sync_map_opacity_text(self) -> None:
        self.map_opacity_label.setText(f"{self.map_opacity_slider.value() / 100:.2f}")

    def _basemap_allowed(self) -> bool:
        if not self.file_path:
            return False
        try:
            _ = self._selected_map()
        except Exception:
            return False
        try:
            if self.data is not None:
                data = self.data
            else:
                mapping = self._selected_map()
                data = load_points(self.file_path, mapping)
            if self.swap_xy_checkbox.isChecked():
                x = data["y"].to_numpy(dtype=float)
                y = data["x"].to_numpy(dtype=float)
            else:
                x = data["x"].to_numpy(dtype=float)
                y = data["y"].to_numpy(dtype=float)
            return bool(looks_like_wgs84_degrees(x, y))
        except Exception:
            return False

    def _update_basemap_opacity_enabled(self) -> None:
        on = self.basemap_checkbox.isChecked() and self.basemap_checkbox.isEnabled()
        self.map_opacity_slider.setEnabled(on)
        self.map_opacity_label.setEnabled(on)
        self.basemap_source_combo.setEnabled(on)

    def _selected_basemap_source_key(self) -> str:
        data = self.basemap_source_combo.currentData()
        return str(data) if data is not None else "esri"

    def _update_basemap_availability(self) -> None:
        allowed = self._basemap_allowed()
        self.basemap_checkbox.setEnabled(allowed)
        if not allowed and self.basemap_checkbox.isChecked():
            self.basemap_checkbox.blockSignals(True)
            self.basemap_checkbox.setChecked(False)
            self.basemap_checkbox.blockSignals(False)
        self._update_basemap_opacity_enabled()
        self.basemap_checkbox.setToolTip(
            "Спутник по координатам WGS84 (нужен интернет)."
            if allowed
            else "Недоступно: координаты должны быть WGS84 в градусах."
        )

    def _connect_live_redraw_signals(self) -> None:
        live_checkboxes = [
            self.enforce_mirror_checkbox,
            self.show_points_checkbox,
            self.show_coordinates_checkbox,
            self.show_rn_checkbox,
            self.show_coordinate_grid_checkbox,
            self.show_scale_bar_x_checkbox,
            self.show_scale_bar_y_checkbox,
            self.use_custom_gradient_checkbox,
            self.show_contour_lines_checkbox,
            self.show_contour_labels_checkbox,
            self.use_levels_step_checkbox,
            self.invert_x_checkbox,
            self.invert_y_checkbox,
            self.swap_xy_checkbox,
            self.smooth_contours_checkbox,
            self.basemap_checkbox,
            self.mercator_square_extent_checkbox,
        ]
        for checkbox in live_checkboxes:
            checkbox.toggled.connect(self._schedule_debounced_redraw)

        self.basemap_checkbox.toggled.connect(self._update_basemap_opacity_enabled)

        self.horizontal_align_btn.toggled.connect(self._schedule_debounced_redraw)
        self.alpha_slider.valueChanged.connect(self._schedule_debounced_redraw)
        self.map_opacity_slider.valueChanged.connect(self._schedule_debounced_redraw)
        self.basemap_source_combo.currentIndexChanged.connect(self._schedule_debounced_redraw)
        self.levels_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.levels_step_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.point_size_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.annotation_font_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.axis_tick_font_x_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.axis_tick_font_y_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.contour_label_font_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.contour_line_width_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.map_view_rotation_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.axis_margin_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.mercator_span_x_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.mercator_span_y_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.map_extent_zoom_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.dual_canvas_width_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.dual_canvas_height_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.overlay_canvas_width_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.overlay_canvas_height_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.basemap_offset_e_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.basemap_offset_n_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.gradient_steps_spin.valueChanged.connect(self._on_gradient_steps_changed)
        self.smoothing_spin.valueChanged.connect(self._schedule_debounced_redraw)

    def _can_prepare_data_silently(self) -> bool:
        if not self.file_path:
            return False
        try:
            _ = self._selected_map()
        except Exception:
            return False
        return True

    def _prepare_data_silently(self) -> bool:
        if not self._can_prepare_data_silently():
            return False
        try:
            mapping = self._selected_map()
            self.data = load_points(self.file_path, mapping)
            if self.swap_xy_checkbox.isChecked():
                x = self.data["y"].to_numpy(dtype=float)
                y = self.data["x"].to_numpy(dtype=float)
            else:
                x = self.data["x"].to_numpy(dtype=float)
                y = self.data["y"].to_numpy(dtype=float)
            if looks_like_wgs84_degrees(x, y):
                self._label_lon_lat_deg = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
                xm, ym = lon_lat_to_web_mercator(x, y)
                self.triangulation = build_triangulation(xm, ym)
                self._using_web_mercator = True
            else:
                self._label_lon_lat_deg = None
                self.triangulation = build_triangulation(x, y)
                self._using_web_mercator = False
            return True
        except Exception:
            return False

    def _schedule_debounced_redraw(self) -> None:
        """Перерисовка с задержкой, чтобы бегунки и спины не вызывали тяжёлый рендер на каждый шаг."""
        if getattr(self, "_loading_ui_config", False):
            return
        self._redraw_debounce_timer.stop()
        self._redraw_debounce_timer.start()

    def _flush_debounced_redraw(self) -> None:
        if getattr(self, "_loading_ui_config", False):
            return
        self._update_basemap_availability()
        if not self._prepare_data_silently():
            return
        if self.tabs.currentIndex() == 1:
            self.on_build_overlay()
        else:
            self.on_build_maps()

    def _toggle_settings_visibility(self, checked: bool) -> None:
        self.settings_group.setVisible(checked)
        self.toggle_settings_btn.setText("Свернуть параметры" if checked else "Развернуть параметры")

    def _sync_horizontal_align_btn_text(self, checked: bool) -> None:
        self.horizontal_align_btn.setText("Карты: сверху вниз" if checked else "Карты: слева направо")

    def on_load_excel(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть Excel файл",
            str(Path.cwd()),
            "Excel Files (*.xlsx *.xls)",
        )
        if not file_path:
            return

        try:
            headers = read_excel_headers(file_path)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Не удалось прочитать Excel: {exc}")
            return

        self.file_path = file_path
        self.file_label.setText(f"Файл: {file_path}")
        self._fill_column_combos(headers)
        self._apply_expected_defaults(headers)
        detected = auto_detect_columns(headers)
        if detected:
            self._apply_detected_columns(detected)
        self._update_basemap_availability()

    def _fill_column_combos(self, headers: list[str]) -> None:
        options = [""] + headers
        for combo in self.column_combos.values():
            combo.clear()
            combo.addItems(options)

    def _apply_expected_defaults(self, headers: list[str]) -> None:
        expected = {"rn": "rn", "x": "x", "y": "y", "ap": "Ap", "ac": "Ac"}
        lower_map = {h.lower(): h for h in headers}
        for key, preferred in expected.items():
            matched = lower_map.get(preferred.lower())
            if not matched:
                continue
            combo = self.column_combos[key]
            idx = combo.findText(matched)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _apply_detected_columns(self, columns: ColumnMap) -> None:
        mapping = {
            "rn": columns.rn or "",
            "x": columns.x,
            "y": columns.y,
            "ap": columns.ap,
            "ac": columns.ac,
        }
        for key, value in mapping.items():
            combo = self.column_combos[key]
            idx = combo.findText(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _selected_map(self) -> ColumnMap:
        rn = self.column_combos["rn"].currentText().strip() or None
        x = self.column_combos["x"].currentText().strip()
        y = self.column_combos["y"].currentText().strip()
        ap = self.column_combos["ap"].currentText().strip()
        ac = self.column_combos["ac"].currentText().strip()

        if not (x and y and ap and ac):
            raise ValueError("Выберите колонки x, y, Ap и Ac.")
        return ColumnMap(rn=rn, x=x, y=y, ap=ap, ac=ac)

    def _ensure_data_loaded(self) -> bool:
        self._update_basemap_availability()
        if not self.file_path:
            self._show_error("Сначала загрузите Excel файл.")
            return False
        try:
            mapping = self._selected_map()
            self.data = load_points(self.file_path, mapping)
            if self.swap_xy_checkbox.isChecked():
                x = self.data["y"].to_numpy(dtype=float)
                y = self.data["x"].to_numpy(dtype=float)
            else:
                x = self.data["x"].to_numpy(dtype=float)
                y = self.data["y"].to_numpy(dtype=float)

            if looks_like_wgs84_degrees(x, y):
                self._label_lon_lat_deg = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
                xm, ym = lon_lat_to_web_mercator(x, y)
                self.triangulation = build_triangulation(xm, ym)
                self._using_web_mercator = True
            else:
                self._label_lon_lat_deg = None
                self.triangulation = build_triangulation(x, y)
                self._using_web_mercator = False
            return True
        except Exception as exc:  # noqa: BLE001
            self._show_error(str(exc))
            return False

    def on_build_maps(self) -> None:
        if not self._ensure_data_loaded():
            return
        self._app_config = load_app_config()

        ap = self.data["ap"].to_numpy(dtype=float)
        ac = self.data["ac"].to_numpy(dtype=float)
        rn_labels = None
        if "rn" in self.data.columns:
            rn_labels = [str(v) for v in self.data["rn"].tolist()]
        axis_x_label = "Y" if self.swap_xy_checkbox.isChecked() else "X"
        axis_y_label = "X" if self.swap_xy_checkbox.isChecked() else "Y"
        if self._using_web_mercator:
            axis_x_label = f"{axis_x_label} (м)"
            axis_y_label = f"{axis_y_label} (м)"

        levels_step = self.levels_step_spin.value() if self.use_levels_step_checkbox.isChecked() else None
        basemap_on = self.basemap_checkbox.isChecked() and self._basemap_allowed()
        _extent_z = self.map_extent_zoom_spin.value() / 100.0
        try:
            fig = render_dual_maps(
                triangulation=self.triangulation,
                ap=ap,
                ac=ac,
                levels_count=self.levels_spin.value(),
                levels_step=levels_step,
                point_size=self.point_size_spin.value(),
                enforce_mirror=self.enforce_mirror_checkbox.isChecked(),
                smooth_contours=self.smooth_contours_checkbox.isChecked(),
                smooth_sigma=self.smoothing_spin.value() / 10.0,
                show_points=self.show_points_checkbox.isChecked(),
                show_coordinates=self.show_coordinates_checkbox.isChecked(),
                show_rn_labels=self.show_rn_checkbox.isChecked(),
                rn_labels=rn_labels,
                show_coordinate_grid=self.show_coordinate_grid_checkbox.isChecked(),
                show_scale_bar_x=self.show_scale_bar_x_checkbox.isChecked(),
                show_scale_bar_y=self.show_scale_bar_y_checkbox.isChecked(),
                invert_x=self.invert_x_checkbox.isChecked(),
                invert_y=self.invert_y_checkbox.isChecked(),
                x_label=axis_x_label,
                y_label=axis_y_label,
                show_contour_lines=self.show_contour_lines_checkbox.isChecked(),
                contour_line_width=self.contour_line_width_spin.value(),
                show_contour_labels=self.show_contour_labels_checkbox.isChecked(),
                contour_label_font_size=self.contour_label_font_spin.value(),
                vertical_layout=self.horizontal_align_btn.isChecked(),
                annotation_font_size=self.annotation_font_spin.value(),
                axis_margin=self.axis_margin_spin.value() / 100.0,
                cmap_start=self.cmap_start,
                cmap_end=self.cmap_end,
                custom_gradient_colors=self._custom_gradient_colors_for_render(),
                basemap_enabled=basemap_on,
                map_layer_alpha=self.map_opacity_slider.value() / 100.0,
                web_mercator=self._using_web_mercator,
                basemap_source=self._selected_basemap_source_key(),
                coordinate_degrees_lon_lat=self._label_lon_lat_deg,
                google_maps_api_key=self._app_config.google_maps_api_key,
                view_rotation_deg=self.map_view_rotation_spin.value(),
                axis_tick_fontsize_x=float(self.axis_tick_font_x_spin.value()),
                axis_tick_fontsize_y=float(self.axis_tick_font_y_spin.value()),
                mercator_force_square=self.mercator_square_extent_checkbox.isChecked(),
                mercator_span_scale_x=self.mercator_span_x_spin.value() / 100.0 * _extent_z,
                mercator_span_scale_y=self.mercator_span_y_spin.value() / 100.0 * _extent_z,
                basemap_offset_east_m=self.basemap_offset_e_spin.value(),
                basemap_offset_north_m=self.basemap_offset_n_spin.value(),
            )
        except BasemapError as exc:
            self._show_error(str(exc))
            return
        self.main_canvas = self._replace_canvas(self.main_canvas, fig, "Отдельные карты", 0)
        self.tabs.setCurrentIndex(0)

    def on_build_overlay(self) -> None:
        if not self._ensure_data_loaded():
            return
        self._app_config = load_app_config()

        ap = self.data["ap"].to_numpy(dtype=float)
        ac = self.data["ac"].to_numpy(dtype=float)
        rn_labels = None
        if "rn" in self.data.columns:
            rn_labels = [str(v) for v in self.data["rn"].tolist()]
        alpha = self.alpha_slider.value() / 100.0
        axis_x_label = "Y" if self.swap_xy_checkbox.isChecked() else "X"
        axis_y_label = "X" if self.swap_xy_checkbox.isChecked() else "Y"
        if self._using_web_mercator:
            axis_x_label = f"{axis_x_label} (м)"
            axis_y_label = f"{axis_y_label} (м)"
        levels_step = self.levels_step_spin.value() if self.use_levels_step_checkbox.isChecked() else None
        basemap_on = self.basemap_checkbox.isChecked() and self._basemap_allowed()
        _extent_z = self.map_extent_zoom_spin.value() / 100.0
        try:
            fig = render_overlay_map(
                triangulation=self.triangulation,
                ap=ap,
                ac=ac,
                alpha=alpha,
                levels_count=self.levels_spin.value(),
                levels_step=levels_step,
                enforce_mirror=self.enforce_mirror_checkbox.isChecked(),
                smooth_contours=self.smooth_contours_checkbox.isChecked(),
                smooth_sigma=self.smoothing_spin.value() / 10.0,
                show_points=self.show_points_checkbox.isChecked(),
                show_coordinates=self.show_coordinates_checkbox.isChecked(),
                show_rn_labels=self.show_rn_checkbox.isChecked(),
                rn_labels=rn_labels,
                show_coordinate_grid=self.show_coordinate_grid_checkbox.isChecked(),
                show_scale_bar_x=self.show_scale_bar_x_checkbox.isChecked(),
                show_scale_bar_y=self.show_scale_bar_y_checkbox.isChecked(),
                invert_x=self.invert_x_checkbox.isChecked(),
                invert_y=self.invert_y_checkbox.isChecked(),
                x_label=axis_x_label,
                y_label=axis_y_label,
                show_contour_lines=self.show_contour_lines_checkbox.isChecked(),
                contour_line_width=self.contour_line_width_spin.value(),
                show_contour_labels=self.show_contour_labels_checkbox.isChecked(),
                contour_label_font_size=self.contour_label_font_spin.value(),
                annotation_font_size=self.annotation_font_spin.value(),
                axis_margin=self.axis_margin_spin.value() / 100.0,
                cmap_start=self.cmap_start,
                cmap_end=self.cmap_end,
                custom_gradient_colors=self._custom_gradient_colors_for_render(),
                basemap_enabled=basemap_on,
                map_layer_alpha=self.map_opacity_slider.value() / 100.0,
                web_mercator=self._using_web_mercator,
                basemap_source=self._selected_basemap_source_key(),
                coordinate_degrees_lon_lat=self._label_lon_lat_deg,
                google_maps_api_key=self._app_config.google_maps_api_key,
                view_rotation_deg=self.map_view_rotation_spin.value(),
                axis_tick_fontsize_x=float(self.axis_tick_font_x_spin.value()),
                axis_tick_fontsize_y=float(self.axis_tick_font_y_spin.value()),
                mercator_force_square=self.mercator_square_extent_checkbox.isChecked(),
                mercator_span_scale_x=self.mercator_span_x_spin.value() / 100.0 * _extent_z,
                mercator_span_scale_y=self.mercator_span_y_spin.value() / 100.0 * _extent_z,
                basemap_offset_east_m=self.basemap_offset_e_spin.value(),
                basemap_offset_north_m=self.basemap_offset_n_spin.value(),
            )
        except BasemapError as exc:
            self._show_error(str(exc))
            return
        self.overlay_canvas = self._replace_canvas(self.overlay_canvas, fig, "Overlay", 1)
        self.tabs.setCurrentIndex(1)

    def on_save_plot(self) -> None:
        canvas = self.main_canvas if self.tabs.currentIndex() == 0 else self.overlay_canvas
        if canvas.figure is None or len(canvas.figure.axes) == 0:
            self._show_error("Сначала постройте график.")
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить PNG",
            str(Path.cwd() / "map.png"),
            "PNG Image (*.png)",
        )
        if not output_path:
            return
        try:
            canvas.figure.savefig(output_path, dpi=220)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Не удалось сохранить файл: {exc}")
            return
        QMessageBox.information(self, "Готово", f"Сохранено: {output_path}")

    def on_export_corel(self) -> None:
        canvas = self.main_canvas if self.tabs.currentIndex() == 0 else self.overlay_canvas
        if canvas.figure is None or len(canvas.figure.axes) == 0:
            self._show_error("Сначала постройте график.")
            return

        output_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Экспорт для CorelDRAW",
            str(Path.cwd() / "map_for_corel.svg"),
            "SVG (*.svg);;PDF (*.pdf);;EPS (*.eps);;CorelDRAW (*.cdr)",
        )
        if not output_path:
            return

        try:
            ext = Path(output_path).suffix.lower()
            if selected_filter.startswith("CorelDRAW") or ext == ".cdr":
                # Direct CDR export is not supported by matplotlib.
                # Save SVG as Corel-friendly vector instead.
                svg_path = str(Path(output_path).with_suffix(".svg"))
                canvas.figure.savefig(svg_path, format="svg")
                QMessageBox.information(
                    self,
                    "Экспорт Corel",
                    (
                        "Прямой экспорт в .cdr недоступен.\n"
                        f"Сохранен векторный файл SVG для открытия в CorelDRAW:\n{svg_path}"
                    ),
                )
                return

            canvas.figure.savefig(output_path)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Не удалось экспортировать файл: {exc}")
            return

        QMessageBox.information(self, "Готово", f"Экспортировано: {output_path}")

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", message)

    def _replace_canvas(self, old_canvas: FigureCanvas, figure: Figure, tab_title: str, tab_index: int) -> FigureCanvas:
        new_canvas = FigureCanvas(figure)
        self._apply_pixel_size_to_figure_canvas(new_canvas, tab_index)
        self.tabs.removeTab(tab_index)
        self.tabs.insertTab(tab_index, new_canvas, tab_title)
        old_canvas.deleteLater()
        new_canvas.draw()
        return new_canvas
