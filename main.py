"""Perspective Combination - Standalone Application Entry Point.

Usage:
    python main.py
"""
import sys

from PySide6 import QtWidgets, QtCore


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Perspective Combination")
    app.setApplicationVersion("1.0.0")

    # High-DPI support
    app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    from perscomb.ui.dialog import PerspectiveCombinationDialog
    dialog = PerspectiveCombinationDialog()
    dialog.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
