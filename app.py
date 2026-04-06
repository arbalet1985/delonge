from __future__ import annotations

import os
import sys

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication


def main() -> int:
    # QtWebEngine can fail silently on some Windows setups (policies/AV/sandbox/GPU).
    # These defaults make WebEngine more robust for local HTML rendering.
    os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
    os.environ.setdefault("QT_OPENGL", "software")
    # Use a conservative software-friendly Chromium config.
    # Avoid conflicting swiftshader flags; rely on software OpenGL + GPU disabled.
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = " ".join(
        [
            "--disable-gpu",
            "--disable-gpu-compositing",
            "--disable-gpu-sandbox",
            "--ignore-gpu-blocklist",
            "--disable-features=VizDisplayCompositor",
        ]
    )

    # Force software rendering path for Qt/QtWebEngine (helps on systems where GPU context creation fails).
    QCoreApplication.setAttribute(Qt.AA_UseSoftwareOpenGL, True)

    # Import after env + attributes are set, so QtWebEngine picks them up.
    from ui.main_window import MainWindow  # noqa: WPS433

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
