"""Fusi³ — SEM Image Fusion & Defect Analysis

Usage:
    python main.py
"""
import sys

from PySide6 import QtWidgets, QtCore, QtGui


def _create_splash(app: QtWidgets.QApplication) -> QtWidgets.QSplashScreen:
    """Create a branded splash screen."""
    from perscomb.ui.design_tokens import Colors, Typography

    W, H = 480, 280
    pixmap = QtGui.QPixmap(W, H)
    pixmap.fill(QtGui.QColor("#FFFFFF"))

    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)

    # Orange accent bar at top
    painter.fillRect(0, 0, W, 4, QtGui.QColor(Colors.BRAND_PRIMARY))

    # App icon placeholder (orange circle with F)
    painter.setBrush(QtGui.QBrush(QtGui.QColor(Colors.BRAND_PRIMARY)))
    painter.setPen(QtCore.Qt.NoPen)
    painter.drawEllipse(W // 2 - 28, 50, 56, 56)
    icon_font = QtGui.QFont("Liberation Sans", 26, QtGui.QFont.Bold)
    painter.setFont(icon_font)
    painter.setPen(QtGui.QColor("#FFFFFF"))
    painter.drawText(QtCore.QRect(W // 2 - 28, 50, 56, 56),
                     QtCore.Qt.AlignCenter, "F")

    # Title
    title_font = QtGui.QFont("Liberation Sans", 22, QtGui.QFont.Bold)
    painter.setFont(title_font)
    painter.setPen(QtGui.QColor(Colors.TEXT_PRIMARY))
    painter.drawText(QtCore.QRect(0, 120, W, 36),
                     QtCore.Qt.AlignCenter, "Fusi\u00b3")

    # Subtitle
    sub_font = QtGui.QFont("Liberation Sans", 11)
    painter.setFont(sub_font)
    painter.setPen(QtGui.QColor(Colors.TEXT_SECONDARY))
    painter.drawText(QtCore.QRect(0, 158, W, 22),
                     QtCore.Qt.AlignCenter,
                     "SEM Image Fusion & Defect Analysis")

    # Version
    ver_font = QtGui.QFont("Liberation Sans", 9)
    painter.setFont(ver_font)
    painter.setPen(QtGui.QColor(Colors.TEXT_MUTED))
    painter.drawText(QtCore.QRect(0, 190, W, 18),
                     QtCore.Qt.AlignCenter, "v1.1.0")

    # Loading status area
    painter.drawText(QtCore.QRect(0, H - 34, W, 18),
                     QtCore.Qt.AlignCenter, "Loading...")

    # Bottom orange line
    painter.fillRect(0, H - 4, W, 4, QtGui.QColor(Colors.BRAND_PRIMARY))

    painter.end()

    splash = QtWidgets.QSplashScreen(pixmap)
    splash.setWindowFlags(
        QtCore.Qt.SplashScreen | QtCore.Qt.FramelessWindowHint
    )
    return splash


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Fusi\u00b3")
    app.setApplicationVersion("1.1.0")
    app.setOrganizationName("Fusi3")

    # High-DPI support
    app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    # Show splash screen
    splash = _create_splash(app)
    splash.show()
    app.processEvents()

    splash.showMessage(
        "Initializing UI...",
        QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter,
        QtGui.QColor("#9CA3AF"),
    )
    app.processEvents()

    from perscomb.ui.dialog import PerspectiveCombinationDialog
    dialog = PerspectiveCombinationDialog()

    splash.showMessage(
        "Ready",
        QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter,
        QtGui.QColor("#9CA3AF"),
    )
    app.processEvents()

    dialog.show()
    splash.finish(dialog)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
