"""Application icon loading helpers for Fusi³."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def icon_candidates() -> list[Path]:
    root = _project_root()
    return [
        root / "assets" / "fusi3_icon.ico",
        root / "assets" / "fusi3_icon.png",
    ]


def get_icon_path() -> Path | None:
    for candidate in icon_candidates():
        if candidate.exists():
            return candidate
    return None


def load_app_icon() -> QtGui.QIcon:
    path = get_icon_path()
    if path is not None:
        return QtGui.QIcon(str(path))
    return QtGui.QIcon()


def load_toolbar_icon(size: int = 16) -> QtGui.QIcon:
    path = get_icon_path()
    if path is not None:
        pix = QtGui.QPixmap(str(path))
        if not pix.isNull():
            scaled = pix.scaled(
                size,
                size,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            return QtGui.QIcon(scaled)
    return QtGui.QIcon()

