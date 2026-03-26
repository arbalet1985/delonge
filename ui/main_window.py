from __future__ import annotations

from pathlib import Path
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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

        self.main_canvas = FigureCanvas(Figure(figsize=(12, 5)))
        self.overlay_canvas = FigureCanvas(Figure(figsize=(7, 5)))

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

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        file_group = QGroupBox("Данные")
        file_layout = QVBoxLayout(file_group)
        load_btn = QPushButton("Загрузить Excel (111.xlsx)")
        load_btn.clicked.connect(self.on_load_excel)
        file_layout.addWidget(load_btn)
        file_layout.addWidget(self.file_label)
        layout.addWidget(file_group)

        map_group = QGroupBox("Сопоставление колонок")
        form = QFormLayout(map_group)
        for key, label in [("rn", "rn"), ("x", "x"), ("y", "y"), ("ap", "Ap"), ("ac", "Ac")]:
            combo = QComboBox()
            combo.setMinimumWidth(220)
            self.column_combos[key] = combo
            form.addRow(f"{label}:", combo)
        layout.addWidget(map_group)

        settings = QGroupBox("Параметры")
        settings_layout = QGridLayout(settings)
        self.settings_group = settings

        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(5, 40)
        self.levels_spin.setValue(10)

        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(5, 80)
        self.point_size_spin.setValue(18)

        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(0, 40)
        self.smoothing_spin.setValue(6)
        self.smoothing_spin.setSuffix(" %")

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(10, 90)
        self.alpha_slider.setValue(50)
        self.alpha_label = QLabel("0.50")
        self.alpha_slider.valueChanged.connect(self._sync_alpha_text)

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
        settings_layout.addWidget(QLabel("Размер точек:"), 1, 0)
        settings_layout.addWidget(self.point_size_spin, 1, 1)
        settings_layout.addWidget(QLabel("Сглаживание:"), 2, 0)
        settings_layout.addWidget(self.smoothing_spin, 2, 1)
        settings_layout.addWidget(QLabel("Прозрачность overlay:"), 3, 0)
        settings_layout.addWidget(self.alpha_slider, 3, 1)
        settings_layout.addWidget(self.alpha_label, 3, 2)
        settings_layout.addWidget(self.smooth_contours_checkbox, 4, 0, 1, 3)
        settings_layout.addWidget(self.show_points_checkbox, 5, 0, 1, 3)
        settings_layout.addWidget(self.show_coordinates_checkbox, 6, 0, 1, 3)
        settings_layout.addWidget(self.show_rn_checkbox, 7, 0, 1, 3)
        settings_layout.addWidget(self.show_scale_bar_checkbox, 8, 0, 1, 3)
        settings_layout.addWidget(self.invert_x_checkbox, 9, 0, 1, 3)
        settings_layout.addWidget(self.invert_y_checkbox, 10, 0, 1, 3)
        settings_layout.addWidget(self.swap_xy_checkbox, 11, 0, 1, 3)
        settings_layout.addWidget(self.enforce_mirror_checkbox, 12, 0, 1, 3)

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
        build_btn = QPushButton("Построить карты Ap / Ac")
        overlay_btn = QPushButton("Проверить наложение (overlay)")
        save_btn = QPushButton("Сохранить текущий график PNG")
        build_btn.clicked.connect(self.on_build_maps)
        overlay_btn.clicked.connect(self.on_build_overlay)
        save_btn.clicked.connect(self.on_save_plot)
        actions_layout.addWidget(build_btn)
        actions_layout.addWidget(overlay_btn)
        actions_layout.addWidget(save_btn)
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

    def _sync_alpha_text(self) -> None:
        self.alpha_label.setText(f"{self.alpha_slider.value() / 100:.2f}")

    def _toggle_settings_visibility(self, checked: bool) -> None:
        self.settings_group.setVisible(checked)
        self.toggle_settings_btn.setText("Свернуть параметры" if checked else "Развернуть параметры")

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
            self.triangulation = build_triangulation(x, y)
            return True
        except Exception as exc:  # noqa: BLE001
            self._show_error(str(exc))
            return False

    def on_build_maps(self) -> None:
        if not self._ensure_data_loaded():
            return

        ap = self.data["ap"].to_numpy(dtype=float)
        ac = self.data["ac"].to_numpy(dtype=float)
        rn_labels = None
        if "rn" in self.data.columns:
            rn_labels = [str(v) for v in self.data["rn"].tolist()]
        axis_x_label = "Y" if self.swap_xy_checkbox.isChecked() else "X"
        axis_y_label = "X" if self.swap_xy_checkbox.isChecked() else "Y"

        fig = render_dual_maps(
            triangulation=self.triangulation,
            ap=ap,
            ac=ac,
            levels_count=self.levels_spin.value(),
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
        )
        self.main_canvas = self._replace_canvas(self.main_canvas, fig, "Отдельные карты", 0)
        self.tabs.setCurrentIndex(0)

    def on_build_overlay(self) -> None:
        if not self._ensure_data_loaded():
            return

        ap = self.data["ap"].to_numpy(dtype=float)
        ac = self.data["ac"].to_numpy(dtype=float)
        rn_labels = None
        if "rn" in self.data.columns:
            rn_labels = [str(v) for v in self.data["rn"].tolist()]
        alpha = self.alpha_slider.value() / 100.0
        axis_x_label = "Y" if self.swap_xy_checkbox.isChecked() else "X"
        axis_y_label = "X" if self.swap_xy_checkbox.isChecked() else "Y"
        fig = render_overlay_map(
            triangulation=self.triangulation,
            ap=ap,
            ac=ac,
            alpha=alpha,
            levels_count=self.levels_spin.value(),
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
        )
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

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", message)

    def _replace_canvas(self, old_canvas: FigureCanvas, figure: Figure, tab_title: str, tab_index: int) -> FigureCanvas:
        new_canvas = FigureCanvas(figure)
        self.tabs.removeTab(tab_index)
        self.tabs.insertTab(tab_index, new_canvas, tab_title)
        old_canvas.deleteLater()
        new_canvas.draw()
        return new_canvas
