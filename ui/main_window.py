from __future__ import annotations

from pathlib import Path
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
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
        controls_scroll.setMinimumWidth(360)
        controls_scroll.setWidget(self._build_controls_panel())
        root_layout.addWidget(controls_scroll, 0)
        root_layout.addWidget(self._build_charts_panel(), 1)

        self.setCentralWidget(central)
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
            combo.setMinimumWidth(220)
            combo.currentIndexChanged.connect(self._update_basemap_availability)
            self.column_combos[key] = combo
            form.addRow(f"{label}:", combo)
        layout.addWidget(map_group)

        settings = QGroupBox("Параметры")
        settings_layout = QGridLayout(settings)
        self.settings_group = settings

        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(5, 40)
        self.levels_spin.setValue(20)

        self.use_levels_step_checkbox = QCheckBox("Изолинии по шагу")
        self.use_levels_step_checkbox.setChecked(False)

        self.levels_step_spin = QDoubleSpinBox()
        self.levels_step_spin.setRange(0.0001, 1_000_000.0)
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
        self.axis_tick_font_x_spin.setToolTip("Размер шрифта подписей делений по оси X (градусы/метры).")
        self.axis_tick_font_y_spin = QSpinBox()
        self.axis_tick_font_y_spin.setRange(5, 18)
        self.axis_tick_font_y_spin.setValue(9)
        self.axis_tick_font_y_spin.setToolTip("Размер шрифта подписей делений по оси Y.")

        self.rotation_deg_spin = QDoubleSpinBox()
        self.rotation_deg_spin.setRange(-180.0, 180.0)
        self.rotation_deg_spin.setDecimals(1)
        self.rotation_deg_spin.setSingleStep(0.2)
        self.rotation_deg_spin.setSuffix("°")
        self.rotation_deg_spin.setValue(0.0)

        self.axis_margin_spin = QSpinBox()
        self.axis_margin_spin.setRange(0, 20)
        self.axis_margin_spin.setValue(5)
        self.axis_margin_spin.setSuffix(" %")

        self.mercator_square_extent_checkbox = QCheckBox("Квадратный охват (max X/Y)")
        self.mercator_square_extent_checkbox.setChecked(False)
        self.mercator_square_extent_checkbox.setToolTip(
            "Вкл: одинаковый масштаб по X и Y (квадратная область, как раньше). "
            "Выкл: охват по каждой оси по размаху данных — меньше пустого места у вытянутых полос. "
            "Только для Web Mercator и подложки."
        )
        self.mercator_span_x_spin = QSpinBox()
        self.mercator_span_x_spin.setRange(25, 400)
        self.mercator_span_x_spin.setValue(100)
        self.mercator_span_x_spin.setSuffix(" %")
        self.mercator_span_x_spin.setToolTip(
            "Дополнительный множитель охвата по оси X относительно размаха данных (после отступа рамки)."
        )
        self.mercator_span_y_spin = QSpinBox()
        self.mercator_span_y_spin.setRange(25, 400)
        self.mercator_span_y_spin.setValue(100)
        self.mercator_span_y_spin.setSuffix(" %")
        self.mercator_span_y_spin.setToolTip(
            "Дополнительный множитель охвата по оси Y (растянуть/сжать вертикально)."
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
            "Поворот подложки и данных в плоскости карты (Web Mercator), вокруг центра области. "
            "Оси остаются в исходном квадратном масштабе; подложка подгружается с запасом по углу поворота, "
            "чтобы не было белых углов у рамки графика."
        )

        self.basemap_checkbox = QCheckBox("Спутниковая подложка")
        self.basemap_checkbox.setChecked(False)
        self.basemap_checkbox.setToolTip(
            "WGS84 в градусах; поворот данных (поле «Поворот карты») должен быть 0°. "
            "Раскладка карт сверху вниз на подложку не влияет. Нужен интернет."
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
        self.show_scale_bar_checkbox = QCheckBox("Показывать шкалу масштаба")
        self.show_scale_bar_checkbox.setChecked(True)
        self.show_contour_lines_checkbox = QCheckBox("Показывать изолинии")
        self.show_contour_lines_checkbox.setChecked(True)

        self.show_contour_labels_checkbox = QCheckBox("Подписывать изолинии")
        self.show_contour_labels_checkbox.setChecked(False)

        self.contour_label_font_spin = QSpinBox()
        self.contour_label_font_spin.setRange(6, 20)
        self.contour_label_font_spin.setValue(8)

        self.contour_line_width_spin = QDoubleSpinBox()
        self.contour_line_width_spin.setRange(0.3, 6.0)
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

        settings_layout.addWidget(QLabel("Уровни изолиний:"), 0, 0)
        settings_layout.addWidget(self.levels_spin, 0, 1)
        settings_layout.addWidget(self.use_levels_step_checkbox, 0, 2)
        settings_layout.addWidget(self.levels_step_spin, 1, 2)
        settings_layout.addWidget(QLabel("Размер точек:"), 1, 0)
        settings_layout.addWidget(self.point_size_spin, 1, 1)
        settings_layout.addWidget(QLabel("Шрифт подписей (rn/коорд.):"), 2, 0)
        settings_layout.addWidget(self.annotation_font_spin, 2, 1)
        settings_layout.addWidget(QLabel("Деления осей X / Y (пт):"), 3, 0)
        axis_tick_row = QWidget()
        axis_tick_row_layout = QHBoxLayout(axis_tick_row)
        axis_tick_row_layout.setContentsMargins(0, 0, 0, 0)
        axis_tick_row_layout.setSpacing(8)
        axis_tick_row_layout.addWidget(QLabel("X"))
        axis_tick_row_layout.addWidget(self.axis_tick_font_x_spin)
        axis_tick_row_layout.addWidget(QLabel("Y"))
        axis_tick_row_layout.addWidget(self.axis_tick_font_y_spin)
        settings_layout.addWidget(axis_tick_row, 3, 1, 1, 2)
        settings_layout.addWidget(QLabel("Поворот карты:"), 4, 0)
        settings_layout.addWidget(self.rotation_deg_spin, 4, 1)
        settings_layout.addWidget(QLabel("Отступ от рамки карты:"), 5, 0)
        settings_layout.addWidget(self.axis_margin_spin, 5, 1)
        settings_layout.addWidget(self.mercator_square_extent_checkbox, 6, 0, 1, 3)
        settings_layout.addWidget(QLabel("Масштаб охвата X / Y (%):"), 7, 0)
        mercator_span_row = QWidget()
        mercator_span_row_layout = QHBoxLayout(mercator_span_row)
        mercator_span_row_layout.setContentsMargins(0, 0, 0, 0)
        mercator_span_row_layout.setSpacing(8)
        mercator_span_row_layout.addWidget(QLabel("X"))
        mercator_span_row_layout.addWidget(self.mercator_span_x_spin)
        mercator_span_row_layout.addWidget(QLabel("Y"))
        mercator_span_row_layout.addWidget(self.mercator_span_y_spin)
        settings_layout.addWidget(mercator_span_row, 7, 1, 1, 2)
        settings_layout.addWidget(QLabel("Сглаживание:"), 8, 0)
        settings_layout.addWidget(self.smoothing_spin, 8, 1)
        settings_layout.addWidget(QLabel("Поворот вида (карта+подложка):"), 9, 0)
        settings_layout.addWidget(self.map_view_rotation_spin, 9, 1)
        settings_layout.addWidget(self.basemap_checkbox, 10, 0, 1, 3)
        settings_layout.addWidget(QLabel("Источник подложки:"), 11, 0)
        settings_layout.addWidget(self.basemap_source_combo, 11, 1, 1, 2)
        settings_layout.addWidget(QLabel("Сдвиг подложки E / N (м):"), 12, 0)
        basemap_off_row = QWidget()
        basemap_off_row_layout = QHBoxLayout(basemap_off_row)
        basemap_off_row_layout.setContentsMargins(0, 0, 0, 0)
        basemap_off_row_layout.setSpacing(8)
        basemap_off_row_layout.addWidget(QLabel("E"))
        basemap_off_row_layout.addWidget(self.basemap_offset_e_spin)
        basemap_off_row_layout.addWidget(QLabel("N"))
        basemap_off_row_layout.addWidget(self.basemap_offset_n_spin)
        settings_layout.addWidget(basemap_off_row, 12, 1, 1, 2)
        settings_layout.addWidget(QLabel("Прозрачность слоя над подложкой:"), 13, 0)
        settings_layout.addWidget(self.map_opacity_slider, 13, 1)
        settings_layout.addWidget(self.map_opacity_label, 13, 2)
        settings_layout.addWidget(QLabel("Прозрачность overlay:"), 14, 0)
        settings_layout.addWidget(self.alpha_slider, 14, 1)
        settings_layout.addWidget(self.alpha_label, 14, 2)
        settings_layout.addWidget(QLabel("Градиент:"), 15, 0)
        cmap_row = QWidget()
        cmap_row_layout = QHBoxLayout(cmap_row)
        cmap_row_layout.setContentsMargins(0, 0, 0, 0)
        cmap_row_layout.setSpacing(8)
        cmap_row_layout.addWidget(self.cmap_start_btn)
        cmap_row_layout.addWidget(self.cmap_end_btn)
        settings_layout.addWidget(cmap_row, 15, 1, 1, 2)
        settings_layout.addWidget(self.smooth_contours_checkbox, 16, 0, 1, 3)
        settings_layout.addWidget(self.show_points_checkbox, 17, 0, 1, 3)
        settings_layout.addWidget(self.show_coordinates_checkbox, 18, 0, 1, 3)
        settings_layout.addWidget(self.show_rn_checkbox, 19, 0, 1, 3)
        settings_layout.addWidget(self.show_scale_bar_checkbox, 20, 0, 1, 3)
        settings_layout.addWidget(self.show_contour_lines_checkbox, 21, 0, 1, 3)
        settings_layout.addWidget(self.show_contour_labels_checkbox, 22, 0, 1, 3)
        settings_layout.addWidget(QLabel("Шрифт подписей изолиний:"), 23, 0)
        settings_layout.addWidget(self.contour_label_font_spin, 23, 1)
        settings_layout.addWidget(QLabel("Толщина изолиний:"), 24, 0)
        settings_layout.addWidget(self.contour_line_width_spin, 24, 1)
        settings_layout.addWidget(self.invert_x_checkbox, 25, 0, 1, 3)
        settings_layout.addWidget(self.invert_y_checkbox, 26, 0, 1, 3)
        settings_layout.addWidget(self.swap_xy_checkbox, 27, 0, 1, 3)
        settings_layout.addWidget(self.enforce_mirror_checkbox, 28, 0, 1, 3)

        self.toggle_settings_btn = QToolButton()
        self.toggle_settings_btn.setText("Свернуть параметры")
        self.toggle_settings_btn.setCheckable(True)
        self.toggle_settings_btn.setChecked(True)
        self.toggle_settings_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.toggle_settings_btn.toggled.connect(self._toggle_settings_visibility)
        layout.addWidget(self.toggle_settings_btn)
        layout.addWidget(settings)

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
        return panel

    def _build_charts_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.main_canvas, "Отдельные карты")
        self.tabs.addTab(self.overlay_canvas, "Overlay")
        layout.addWidget(self.tabs)
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

    def _sync_alpha_text(self) -> None:
        self.alpha_label.setText(f"{self.alpha_slider.value() / 100:.2f}")

    def _sync_map_opacity_text(self) -> None:
        self.map_opacity_label.setText(f"{self.map_opacity_slider.value() / 100:.2f}")

    def _basemap_allowed(self) -> bool:
        if abs(self.rotation_deg_spin.value()) > 1e-6:
            return False
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
            else "Недоступно: WGS84 в градусах и поворот данных 0°."
        )

    def _connect_live_redraw_signals(self) -> None:
        live_checkboxes = [
            self.enforce_mirror_checkbox,
            self.show_points_checkbox,
            self.show_coordinates_checkbox,
            self.show_rn_checkbox,
            self.show_scale_bar_checkbox,
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
        self.rotation_deg_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.map_view_rotation_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.axis_margin_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.mercator_span_x_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.mercator_span_y_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.basemap_offset_e_spin.valueChanged.connect(self._schedule_debounced_redraw)
        self.basemap_offset_n_spin.valueChanged.connect(self._schedule_debounced_redraw)
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
            x, y = self._rotate_points_by_degrees(x, y, self.rotation_deg_spin.value())
            use_basemap = self.basemap_checkbox.isChecked() and self._basemap_allowed()
            if use_basemap:
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
        self._redraw_debounce_timer.stop()
        self._redraw_debounce_timer.start()

    def _flush_debounced_redraw(self) -> None:
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

    def _rotate_points_by_degrees(self, x: np.ndarray, y: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
        if abs(degrees) < 1e-9:
            return x, y
        points = np.column_stack((x, y))
        center = points.mean(axis=0)
        centered = points - center
        theta = float(np.deg2rad(degrees))
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated = centered @ rotation_matrix.T + center
        return rotated[:, 0], rotated[:, 1]

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

            x, y = self._rotate_points_by_degrees(x, y, self.rotation_deg_spin.value())
            use_basemap = self.basemap_checkbox.isChecked() and self._basemap_allowed()
            if use_basemap:
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
                show_scale_bar=self.show_scale_bar_checkbox.isChecked(),
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
                mercator_span_scale_x=self.mercator_span_x_spin.value() / 100.0,
                mercator_span_scale_y=self.mercator_span_y_spin.value() / 100.0,
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
                show_scale_bar=self.show_scale_bar_checkbox.isChecked(),
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
                mercator_span_scale_x=self.mercator_span_x_spin.value() / 100.0,
                mercator_span_scale_y=self.mercator_span_y_spin.value() / 100.0,
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
        self.tabs.removeTab(tab_index)
        self.tabs.insertTab(tab_index, new_canvas, tab_title)
        old_canvas.deleteLater()
        new_canvas.draw()
        return new_canvas
