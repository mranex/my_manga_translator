"""Entry point for the PyQt6 desktop shell."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from . import APP_NAME
from .app_settings import APP_SETTINGS_APPLICATION, APP_SETTINGS_ORGANIZATION
from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_SETTINGS_ORGANIZATION)
    app.setOrganizationDomain("local.mmt")
    app.setApplicationDisplayName(APP_SETTINGS_APPLICATION)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
