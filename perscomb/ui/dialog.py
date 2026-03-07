"""Perspective Combination Dialog - Multi-image comparison for defect detection.

This dialog allows users to:
1. Select a base image and multiple compare images
2. Align and blend compare images
3. Subtract from base to reveal defect hints
4. View SNR map highlighting strong signal areas
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pathlib import Path

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, Signal

import numpy as np
import cv2

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..core.ebeam_snr import EbeamCondition, load_image_gray
from ..core.perspective_combine import (
    align_and_subtract,
    compute_single_pair,
    compute_multi_pairs,
    compute_quadrant_fusion,
    colorize_snr_map,
    CombineResult,
    SinglePairResult,
    QuadrantFusionResult,
)

# 使用 design_tokens 統一配色（遵循 AGENTS.md 規範）
from .design_tokens import Colors, Typography, Spacing, BorderRadius

UI_PRIMARY = Colors.BRAND_PRIMARY
UI_PRIMARY_HOVER = Colors.BRAND_PRIMARY_HOVER
UI_PRIMARY_PRESSED = Colors.BRAND_PRIMARY_PRESSED
UI_PRIMARY_SOFT = Colors.BRAND_PRIMARY_SOFT

UI_BG_WINDOW = Colors.BG_WINDOW
UI_BG_PANEL = Colors.BG_PANEL
UI_BG_CARD = Colors.BG_CARD
UI_BG_SUBTLE = Colors.BG_SUBTLE
UI_BG_INPUT = Colors.BG_INPUT
UI_BG_VIEWER = Colors.BG_VIEWER

UI_TEXT = Colors.TEXT_PRIMARY
UI_TEXT_SECONDARY = Colors.TEXT_SECONDARY
UI_TEXT_MUTED = Colors.TEXT_MUTED
UI_TEXT_INVERSE = Colors.TEXT_INVERSE
UI_TEXT_ON_PRIMARY = Colors.TEXT_ON_PRIMARY

UI_BORDER = Colors.BORDER_DEFAULT
UI_BORDER_HOVER = Colors.BORDER_HOVER
UI_BORDER_ACTIVE = Colors.BORDER_ACTIVE

UI_SUCCESS = Colors.SUCCESS
UI_WARNING = Colors.WARNING
UI_INFO = Colors.INFO


def _hex_to_bgr(color: str) -> Tuple[int, int, int]:
    """Convert #RRGGBB color to BGR tuple for OpenCV drawing."""
    c = color.lstrip("#")
    return (int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16))


DIALOG_STYLE = f"""
    QDialog {{
        background-color: {UI_BG_WINDOW};
        color: {UI_TEXT};
        font-family: {Typography.FONT_FAMILY};
        font-size: {Typography.FONT_SIZE_BODY};
    }}

    QGroupBox {{
        font-weight: {Typography.FONT_WEIGHT_SEMIBOLD};
        font-size: {Typography.FONT_SIZE_BODY};
        border: 1px solid {UI_BORDER};
        border-radius: {BorderRadius.MD};
        background-color: {UI_BG_PANEL};
        margin-top: 14px;
        padding: 16px 12px 12px 12px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        top: 2px;
        padding: 0 6px;
        color: {UI_TEXT};
        background-color: transparent;
    }}
    QGroupBox#AdvancedSettings {{
        background-color: {UI_BG_CARD};
    }}
    QGroupBox#AdvancedSettings::title {{
        color: {UI_TEXT_SECONDARY};
    }}
    QFrame#TopToolbar {{
        background-color: {UI_BG_PANEL};
        border: 1px solid {UI_BORDER};
        border-radius: {BorderRadius.MD};
    }}
    QFrame#MainToolbar {{
        background-color: {UI_BG_PANEL};
        border: 1px solid {UI_BORDER};
        border-radius: {BorderRadius.MD};
        padding: 4px 8px;
    }}
    QFrame#BottomCard {{
        background-color: {UI_BG_PANEL};
        border: 1px solid {UI_BORDER};
        border-radius: {BorderRadius.MD};
    }}
    QLabel#SectionTitle {{
        color: {UI_TEXT};
        font-size: {Typography.FONT_SIZE_H3};
        font-weight: {Typography.FONT_WEIGHT_BOLD};
        padding: 0px;
        border: none;
        background: transparent;
    }}
    QLabel#SectionSeparator {{
        background-color: {UI_PRIMARY};
        min-height: 2px;
        max-height: 2px;
        border: none;
    }}
    QPushButton#ToolbarAction {{
        background-color: {UI_BG_PANEL};
        color: {UI_TEXT};
        border: 1px solid {UI_BORDER};
        border-radius: {BorderRadius.SM};
        padding: 6px 14px;
        min-height: 32px;
        min-width: 80px;
        font-weight: {Typography.FONT_WEIGHT_MEDIUM};
        font-size: {Typography.FONT_SIZE_SMALL};
    }}
    QPushButton#ToolbarAction:hover {{
        background-color: #FFF8ED;
        border-color: {UI_PRIMARY};
    }}
    QPushButton#ToolbarPrimary {{
        background-color: {UI_PRIMARY};
        color: {UI_TEXT_ON_PRIMARY};
        border: 1px solid {UI_PRIMARY};
        border-radius: {BorderRadius.SM};
        padding: 6px 18px;
        min-height: 32px;
        font-weight: {Typography.FONT_WEIGHT_BOLD};
        font-size: {Typography.FONT_SIZE_SMALL};
    }}
    QPushButton#ToolbarPrimary:hover {{
        background-color: {UI_PRIMARY_HOVER};
        border-color: {UI_PRIMARY_HOVER};
    }}
    QPushButton[viewerMode="true"] {{
        background-color: {UI_BG_PANEL};
        color: {UI_TEXT_SECONDARY};
        border: 1px solid {UI_BORDER};
        border-radius: 0px;
        padding: 6px 16px;
        min-height: 30px;
        min-width: 72px;
        font-size: {Typography.FONT_SIZE_SMALL};
        font-weight: {Typography.FONT_WEIGHT_MEDIUM};
    }}
    QPushButton[viewerMode="true"]:hover {{
        background-color: #FFF8ED;
        color: {UI_TEXT};
    }}
    QPushButton[viewerMode="true"]:checked {{
        background-color: {UI_PRIMARY};
        color: {UI_TEXT_ON_PRIMARY};
        border-color: {UI_PRIMARY};
        font-weight: {Typography.FONT_WEIGHT_BOLD};
    }}
    QPushButton#ViewerModeFirst {{
        border-top-left-radius: {BorderRadius.SM};
        border-bottom-left-radius: {BorderRadius.SM};
    }}
    QPushButton#ViewerModeLast {{
        border-top-right-radius: {BorderRadius.SM};
        border-bottom-right-radius: {BorderRadius.SM};
    }}

    QLabel {{
        color: {UI_TEXT};
        font-size: {Typography.FONT_SIZE_SMALL};
    }}
    QLabel[toolbarLabel="true"] {{
        color: {UI_TEXT_SECONDARY};
        font-size: {Typography.FONT_SIZE_SMALL};
        font-weight: {Typography.FONT_WEIGHT_MEDIUM};
    }}
    QLabel[secondary="true"] {{
        color: {UI_TEXT_SECONDARY};
        font-size: {Typography.FONT_SIZE_CAPTION};
    }}
    QLabel[statLabel="true"] {{
        color: {UI_TEXT_SECONDARY};
        font-size: {Typography.FONT_SIZE_SMALL};
    }}
    QLabel[statValue="true"] {{
        color: {UI_TEXT};
        font-size: {Typography.FONT_SIZE_BODY};
        font-weight: {Typography.FONT_WEIGHT_BOLD};
    }}
    QLabel[statValueLong="true"] {{
        color: {UI_TEXT_SECONDARY};
        font-size: {Typography.FONT_SIZE_SMALL};
        font-weight: {Typography.FONT_WEIGHT_SEMIBOLD};
    }}

    QPushButton {{
        background-color: {UI_BG_PANEL};
        color: {UI_TEXT};
        border: 1px solid {UI_BORDER};
        border-radius: {BorderRadius.MD};
        padding: 8px 14px;
        min-height: 34px;
        min-width: 92px;
        font-weight: {Typography.FONT_WEIGHT_MEDIUM};
        font-size: {Typography.FONT_SIZE_SMALL};
    }}
    QPushButton:hover {{
        background-color: #FFF8ED;
        border-color: {UI_PRIMARY};
    }}
    QPushButton:pressed {{
        background-color: #FDE7C2;
        border-color: {UI_PRIMARY_PRESSED};
    }}
    QPushButton:disabled {{
        background-color: {UI_BG_SUBTLE};
        color: {UI_TEXT_MUTED};
        border-color: {UI_BORDER};
    }}
    QPushButton[variant="primary"] {{
        background-color: {UI_PRIMARY};
        color: {UI_TEXT_ON_PRIMARY};
        border: 1px solid {UI_PRIMARY};
        font-weight: {Typography.FONT_WEIGHT_BOLD};
    }}
    QPushButton[variant="primary"]:hover {{
        background-color: {UI_PRIMARY_HOVER};
        border-color: {UI_PRIMARY_HOVER};
    }}
    QPushButton[variant="primary"]:pressed {{
        background-color: {UI_PRIMARY_PRESSED};
        border-color: {UI_PRIMARY_PRESSED};
    }}
    QPushButton[variant="ghost"] {{
        background-color: transparent;
        border-color: transparent;
    }}
    QPushButton[variant="ghost"]:hover {{
        background-color: #FFF8ED;
        border-color: {UI_BORDER};
    }}
    QPushButton#ViewerToolToggle {{
        background-color: {UI_BG_PANEL};
        color: {UI_TEXT_SECONDARY};
        border: 1px solid {UI_BORDER};
        border-radius: {BorderRadius.SM};
        padding: 6px 12px;
        min-height: 30px;
        min-width: 96px;
        font-size: {Typography.FONT_SIZE_SMALL};
        font-weight: {Typography.FONT_WEIGHT_MEDIUM};
    }}
    QPushButton#ViewerToolToggle:hover {{
        background-color: #FFF8ED;
        border-color: {UI_PRIMARY};
    }}
    QPushButton#ViewerToolToggle:checked {{
        background-color: {UI_PRIMARY};
        color: {UI_TEXT_ON_PRIMARY};
        border-color: {UI_PRIMARY};
        font-weight: {Typography.FONT_WEIGHT_BOLD};
    }}
    QLabel[toolbarGroupLabel="true"] {{
        color: {UI_TEXT_MUTED};
        font-size: {Typography.FONT_SIZE_CAPTION};
        font-weight: {Typography.FONT_WEIGHT_BOLD};
        letter-spacing: 0.35px;
    }}
    QPushButton[toolbarToggle="true"] {{
        background-color: {UI_BG_SUBTLE};
        border: 1px solid {UI_BORDER};
        border-radius: {BorderRadius.SM};
        padding: 6px 12px;
        min-height: 32px;
        min-width: 88px;
        font-size: {Typography.FONT_SIZE_SMALL};
    }}
    QPushButton[toolbarToggle="true"]:hover {{
        background-color: #FFF8ED;
        border-color: {UI_PRIMARY};
    }}
    QPushButton[toolbarToggle="true"]:checked {{
        background-color: {UI_PRIMARY};
        color: {UI_TEXT_ON_PRIMARY};
        border-color: {UI_PRIMARY};
        font-weight: {Typography.FONT_WEIGHT_BOLD};
    }}

    QComboBox {{
        background-color: {UI_BG_INPUT};
        color: {UI_TEXT};
        border: 1px solid {UI_BORDER};
        border-radius: {BorderRadius.SM};
        padding: {Spacing.INPUT_PADDING};
        font-size: {Typography.FONT_SIZE_SMALL};
        min-height: 28px;
    }}
    QComboBox:hover, QComboBox:focus {{
        border-color: {UI_PRIMARY};
    }}
    QComboBox::drop-down {{
        border: none;
        padding-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {UI_BG_PANEL};
        color: {UI_TEXT};
        selection-background-color: {UI_PRIMARY};
        selection-color: {UI_TEXT_ON_PRIMARY};
        border: 1px solid {UI_BORDER};
    }}

    QCheckBox {{
        color: {UI_TEXT};
        spacing: {Spacing.SM};
        font-size: {Typography.FONT_SIZE_SMALL};
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid #F7C267;
        border-radius: 4px;
        background-color: #FFF8ED;
    }}
    QCheckBox::indicator:checked {{
        background-color: {UI_PRIMARY};
        border-color: {UI_PRIMARY};
    }}

    QSpinBox, QDoubleSpinBox, QLineEdit {{
        background-color: {UI_BG_INPUT};
        color: {UI_TEXT};
        border: 1px solid {UI_BORDER};
        border-radius: {BorderRadius.SM};
        padding: 5px 26px 5px 10px;
        min-height: 24px;
        font-size: {Typography.FONT_SIZE_SMALL};
    }}
    QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover,
    QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
        border-color: {UI_PRIMARY};
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        width: 14px;
        border-left: 1px solid {UI_BORDER};
        background-color: {UI_BG_CARD};
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-position: top right;
        border-top-right-radius: {BorderRadius.SM};
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-position: bottom right;
        border-bottom-right-radius: {BorderRadius.SM};
        border-top: 1px solid {UI_BORDER};
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: #FFF8ED;
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        width: 7px;
        height: 7px;
    }}

    QSlider::groove:horizontal {{
        border: none;
        height: 8px;
        background: {UI_BG_SUBTLE};
        border-radius: 4px;
    }}
    QSlider::handle:horizontal {{
        background: {UI_PRIMARY};
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {UI_PRIMARY_HOVER};
    }}
    QSlider::sub-page:horizontal {{
        background: {UI_PRIMARY};
        border-radius: 4px;
    }}

    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QScrollBar:vertical {{
        background-color: {UI_BG_SUBTLE};
        width: 8px;
        margin: 0;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background-color: #C7D0DC;
        min-height: 30px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {UI_PRIMARY};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    QFrame {{
        border: none;
    }}
"""


class SyncZoomImageWidget(QtWidgets.QWidget):
    """Image widget with synchronized zoom on hover.

    Signals:
        cursor_moved: Emitted when cursor moves, sends (norm_x, norm_y)
        cursor_left: Emitted when cursor leaves the widget
        roi_selected: Emitted when user finishes drawing ROI (norm_x, norm_y, norm_w, norm_h)
    """

    cursor_moved = Signal(float, float)  # Normalized position (0-1)
    cursor_left = Signal()
    roi_selected = Signal(float, float, float, float)  # norm x, y, w, h
    clicked = Signal()

    ZOOM_FACTOR = 2.0  # Magnification factor (lower = more FOV)
    ZOOM_SIZE = 220  # Larger circular FOV for easier inspection

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self._image: Optional[np.ndarray] = None
        self._hint_rect: Optional[tuple] = None
        self._hint_info: str = ""
        self._show_hint: bool = False
        self._cursor_pos: Optional[tuple] = None  # (norm_x, norm_y)
        self._show_zoom: bool = False
        self._partner: Optional['SyncZoomImageWidget'] = None  # Linked partner
        # ROI drawing state
        self._roi_mode: bool = False
        self._roi_start: Optional[tuple] = None  # (norm_x, norm_y)
        self._roi_current: Optional[tuple] = None  # (norm_x, norm_y)
        self._roi_dragging: bool = False
        # Active ROI overlay (drawn permanently until cleared)
        self._active_roi: Optional[tuple] = None  # (norm_x, norm_y, norm_w, norm_h)

        self.setMinimumSize(350, 350)  # Larger minimum for better visibility
        self.setMouseTracking(True)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {UI_BG_PANEL};
                border: 1px solid {UI_BORDER};
                border-radius: 12px;
            }}
            QLabel {{
                background-color: {UI_BG_VIEWER};
                border: none;
                border-radius: 8px;
                color: #D1D5DB;
            }}
        """)

        # Image label
        self._label = QtWidgets.QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(f"border: none; background: {UI_BG_VIEWER}; border-radius: 8px;")
        self._label.setMouseTracking(True)  # Enable hover without clicking

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._label)

    def linkPartner(self, partner: 'SyncZoomImageWidget'):
        """Link to another widget for synchronized cursor display."""
        self._partner = partner

    def setImage(self, image: np.ndarray):
        """Set image to display."""
        self._image = image
        self._update_display()

    def setHint(self, rect: tuple = None, info: str = "", show: bool = True):
        """Set hint rectangle and info."""
        self._hint_rect = rect
        self._hint_info = info
        self._show_hint = show
        self._update_display()

    def setShowHint(self, show: bool):
        """Toggle hint visibility."""
        self._show_hint = show
        self._update_display()

    def set_roi_mode(self, enabled: bool):
        """Enter or exit ROI drawing mode."""
        self._roi_mode = enabled
        self._roi_start = None
        self._roi_current = None
        self._roi_dragging = False
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self._update_display()

    def set_active_roi(self, norm_rect: Optional[tuple]):
        """Set the persistent ROI overlay (norm_x, norm_y, norm_w, norm_h), or None to clear."""
        self._active_roi = norm_rect
        self._update_display()

    def _widget_to_norm(self, pos) -> Optional[tuple]:
        """Convert widget-space QPoint to normalized image coordinates (0-1)."""
        label_rect = self._label.geometry()
        p = pos - label_rect.topLeft()
        pixmap = self._label.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        px_w, px_h = pixmap.width(), pixmap.height()
        lbl_w, lbl_h = label_rect.width(), label_rect.height()
        offset_x = (lbl_w - px_w) // 2
        offset_y = (lbl_h - px_h) // 2
        img_x = p.x() - offset_x
        img_y = p.y() - offset_y
        if 0 <= img_x < px_w and 0 <= img_y < px_h:
            return float(img_x) / px_w, float(img_y) / px_h
        return None

    def setCursorPos(self, norm_x: float, norm_y: float):
        """Set cursor position from external source (partner sync)."""
        self._cursor_pos = (norm_x, norm_y)
        self._show_zoom = True
        self._update_display()

    def clearCursor(self):
        """Clear cursor when partner cursor leaves."""
        self._cursor_pos = None
        self._show_zoom = False
        self._update_display()

    def mousePressEvent(self, event):
        """Start ROI drawing on left click when in roi_mode."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        if self._roi_mode and event.button() == Qt.LeftButton:
            norm = self._widget_to_norm(event.pos())
            if norm:
                self._roi_start = norm
                self._roi_current = norm
                self._roi_dragging = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Finish ROI drawing on left release."""
        if self._roi_mode and self._roi_dragging and event.button() == Qt.LeftButton:
            norm = self._widget_to_norm(event.pos())
            if norm:
                self._roi_current = norm
            if self._roi_start and self._roi_current:
                x0, y0 = self._roi_start
                x1, y1 = self._roi_current
                nx, ny = min(x0, x1), min(y0, y1)
                nw, nh = abs(x1 - x0), abs(y1 - y0)
                if nw > 0.01 and nh > 0.01:
                    self._active_roi = (nx, ny, nw, nh)
                    self.roi_selected.emit(nx, ny, nw, nh)
            self._roi_dragging = False
            self._roi_mode = False
            self.setCursor(Qt.ArrowCursor)
            self._update_display()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        """Track mouse position and emit signal."""
        # ROI drag mode: update rubber band
        if self._roi_mode and self._roi_dragging:
            norm = self._widget_to_norm(event.pos())
            if norm:
                self._roi_current = norm
            self._update_display()
            event.accept()
            return

        if self._image is None:
            return

        # Get label geometry
        label_rect = self._label.geometry()
        pos = event.pos() - label_rect.topLeft()

        # Calculate normalized position within the displayed image
        pixmap = self._label.pixmap()
        if pixmap is None or pixmap.isNull():
            return

        # Image is centered in label
        px_w, px_h = pixmap.width(), pixmap.height()
        lbl_w, lbl_h = label_rect.width(), label_rect.height()
        offset_x = (lbl_w - px_w) // 2
        offset_y = (lbl_h - px_h) // 2

        img_x = pos.x() - offset_x
        img_y = pos.y() - offset_y

        if 0 <= img_x < px_w and 0 <= img_y < px_h:
            norm_x = img_x / px_w
            norm_y = img_y / px_h
            self._cursor_pos = (norm_x, norm_y)
            self._show_zoom = True
            self.cursor_moved.emit(norm_x, norm_y)
        else:
            self._cursor_pos = None
            self._show_zoom = False

        self._update_display()

    def leaveEvent(self, event):
        """Handle mouse leaving widget."""
        self._cursor_pos = None
        self._show_zoom = False
        self.cursor_left.emit()
        self._update_display()

    def _update_display(self):
        """Update the displayed pixmap."""
        if self._image is None:
            self._label.setText(
                f"{self._title}\n\nNo image loaded\n"
                f"Load a folder and select images to begin."
            )
            self._label.setStyleSheet(
                f"border: none; background: {UI_BG_VIEWER}; border-radius: 8px;"
                f" color: {UI_TEXT_MUTED}; font-size: {Typography.FONT_SIZE_SMALL};"
            )
            return

        h, w = self._image.shape[:2]

        # Convert to RGB
        if len(self._image.shape) == 2:
            img_display = self._image
            if img_display.dtype != np.uint8:
                img_display = np.clip(img_display, 0, 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
        else:
            img_rgb = self._image.copy()

        # Draw hint box if enabled
        if self._show_hint and self._hint_rect is not None:
            x, y, rw, rh = self._hint_rect
            hint_color = _hex_to_bgr(UI_SUCCESS)
            cv2.rectangle(img_rgb, (x, y), (x + rw, y + rh), hint_color, 2)
            cx, cy = x + rw // 2, y + rh // 2
            mark_len = 10
            cv2.line(img_rgb, (cx - mark_len, cy), (cx + mark_len, cy), hint_color, 1)
            cv2.line(img_rgb, (cx, cy - mark_len), (cx, cy + mark_len), hint_color, 1)
            if self._hint_info:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_y = max(y - 8, 18)
                cv2.putText(img_rgb, self._hint_info, (x + 1, text_y + 1),
                            font, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img_rgb, self._hint_info, (x, text_y),
                            font, 0.45, hint_color, 1, cv2.LINE_AA)

        # Draw active ROI (persistent cyan box)
        if self._active_roi is not None:
            nx, ny, nw, nh = self._active_roi
            rx, ry = int(nx * w), int(ny * h)
            rw2, rh2 = max(1, int(nw * w)), max(1, int(nh * h))
            cv2.rectangle(img_rgb, (rx, ry), (rx + rw2, ry + rh2), (0, 255, 255), 2)
            cv2.rectangle(img_rgb, (rx + 1, ry + 1), (rx + rw2 - 1, ry + rh2 - 1), (0, 0, 0), 1)
            cv2.putText(img_rgb, "ROI", (rx + 3, ry + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw rubber band while dragging ROI
        if self._roi_mode and self._roi_start and self._roi_current:
            x0n, y0n = self._roi_start
            x1n, y1n = self._roi_current
            rx = int(min(x0n, x1n) * w)
            ry = int(min(y0n, y1n) * h)
            rw3 = max(1, int(abs(x1n - x0n) * w))
            rh3 = max(1, int(abs(y1n - y0n) * h))
            cv2.rectangle(img_rgb, (rx, ry), (rx + rw3, ry + rh3), (0, 255, 255), 2)

        # Draw zoom overlay if cursor is on image
        if self._show_zoom and self._cursor_pos is not None:
            norm_x, norm_y = self._cursor_pos
            src_x, src_y = int(norm_x * w), int(norm_y * h)

            # No cursor marker - just magnifier
            cross_color = (0, 200, 255)  # Cyan in BGR

            # Extract zoom region
            half_zoom = int(self.ZOOM_SIZE / self.ZOOM_FACTOR / 2)
            x1 = max(0, src_x - half_zoom)
            y1 = max(0, src_y - half_zoom)
            x2 = min(w, src_x + half_zoom)
            y2 = min(h, src_y + half_zoom)

            if x2 > x1 and y2 > y1:
                zoom_region = img_rgb[y1:y2, x1:x2]
                # Resize to zoom window size
                zoomed = cv2.resize(zoom_region, (self.ZOOM_SIZE, self.ZOOM_SIZE),
                                    interpolation=cv2.INTER_LINEAR)

                # Place magnifier centered on cursor for direct observation.
                zx = src_x - self.ZOOM_SIZE // 2
                zy = src_y - self.ZOOM_SIZE // 2

                # Keep zoom window inside image bounds
                zx = max(0, min(zx, w - self.ZOOM_SIZE))
                zy = max(0, min(zy, h - self.ZOOM_SIZE))

                if zx >= 0 and zy >= 0:
                    # Apply circular mask to zoomed region
                    mask = np.zeros((self.ZOOM_SIZE, self.ZOOM_SIZE), dtype=np.uint8)
                    center = self.ZOOM_SIZE // 2
                    cv2.circle(mask, (center, center), center - 2, 255, -1)

                    # Get background region
                    bg_region = img_rgb[zy:zy + self.ZOOM_SIZE, zx:zx + self.ZOOM_SIZE].copy()

                    # Blend zoomed with background using mask
                    mask_3ch = cv2.merge([mask, mask, mask])
                    result = np.where(mask_3ch == 255, zoomed, bg_region)
                    img_rgb[zy:zy + self.ZOOM_SIZE, zx:zx + self.ZOOM_SIZE] = result

                    # Draw circular border
                    cv2.circle(img_rgb, (zx + center, zy + center), center - 1, cross_color, 2)

        # Convert BGR to RGB for Qt
        img_qt = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_qt = np.ascontiguousarray(img_qt)
        qimg = QtGui.QImage(img_qt.data, w, h, w * 3, QtGui.QImage.Format_RGB888).copy()

        pixmap = QtGui.QPixmap.fromImage(qimg)

        # Scale to fit
        scaled = pixmap.scaled(
            self._label.size() - QtCore.QSize(8, 8),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self._label.setPixmap(scaled)


class SplitViewWidget(QtWidgets.QWidget):
    """Split-view image comparison widget with draggable vertical divider.

    Left side shows the Base image, right side shows the Compare image.
    The vertical divider can be dragged to reveal more/less of each side.
    The slider_blend value (0-100) also controls the divider position.
    """

    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_image: Optional[np.ndarray] = None
        self._compare_image: Optional[np.ndarray] = None
        self._divider_ratio: float = 0.5  # 0.0 = all compare, 1.0 = all base
        self._dragging: bool = False
        self._base_pix: Optional[QtGui.QPixmap] = None
        self._comp_pix: Optional[QtGui.QPixmap] = None

        self.setMinimumSize(350, 350)
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {UI_BG_PANEL};
                border: 1px solid {UI_BORDER};
                border-radius: 12px;
            }}
            QLabel {{
                background-color: {UI_BG_VIEWER};
                border: none;
                border-radius: 8px;
                color: #D1D5DB;
            }}
        """)

    # ── Public API ─────────────────────────────────────────────────────────
    def set_images(self, base: Optional[np.ndarray], compare: Optional[np.ndarray]):
        """Set the two images to compare."""
        self._base_image = base
        self._compare_image = compare
        self._base_pix = self._to_pixmap(base)
        self._comp_pix = self._to_pixmap(compare)
        self.update()

    def set_divider(self, ratio: float):
        """Set divider position: 0.0 = all compare at left, 1.0 = all base at right."""
        self._divider_ratio = max(0.0, min(1.0, ratio))
        self.update()

    # ── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _to_pixmap(img: Optional[np.ndarray]) -> Optional[QtGui.QPixmap]:
        if img is None:
            return None
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_c = np.ascontiguousarray(img_rgb)
        h, w = img_c.shape[:2]
        qimg = QtGui.QImage(img_c.data, w, h, w * 3,
                            QtGui.QImage.Format_RGB888).copy()
        return QtGui.QPixmap.fromImage(qimg)

    def _content_rect(self) -> QtCore.QRect:
        """Return the inner viewer content rect, matching SyncZoomImageWidget margins."""
        return self.rect().adjusted(8, 8, -8, -8)

    def _fitted_target_rect(self) -> QtCore.QRect:
        """Return centered fit-rect using the same KeepAspectRatio logic as standard viewer."""
        content = self._content_rect()
        pix = self._base_pix if self._base_pix is not None else self._comp_pix
        if pix is None or pix.isNull() or content.width() <= 0 or content.height() <= 0:
            return content

        scaled = pix.size().scaled(content.size(), Qt.KeepAspectRatio)
        x = content.x() + (content.width() - scaled.width()) // 2
        y = content.y() + (content.height() - scaled.height()) // 2
        return QtCore.QRect(x, y, scaled.width(), scaled.height())

    # ── Painting ───────────────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        target_rect = self._fitted_target_rect()
        x0, y0, iw, ih = target_rect.x(), target_rect.y(), target_rect.width(), target_rect.height()
        div_x = x0 + int(self._divider_ratio * iw)

        # Background fill
        p.fillRect(self.rect(), QtGui.QColor(UI_BG_CARD))

        if self._base_pix is None and self._comp_pix is None:
            p.setPen(QtGui.QColor(UI_TEXT_MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "No images loaded")
            p.end()
            return

        # Left half → Base
        left_w = max(0, div_x - x0)
        if self._base_pix and left_w > 0:
            p.save()
            p.setClipRect(x0, y0, left_w, ih)
            p.drawPixmap(target_rect, self._base_pix)
            p.restore()

        # Right half → Compare
        right_w = max(0, iw - left_w)
        if self._comp_pix and right_w > 0:
            p.save()
            p.setClipRect(div_x, y0, right_w, ih)
            p.drawPixmap(target_rect, self._comp_pix)
            p.restore()

        # Divider line
        p.setPen(QtGui.QPen(QtGui.QColor(UI_PRIMARY), 2, Qt.SolidLine))
        p.drawLine(div_x, y0, div_x, y0 + ih)

        # Diamond handle at mid-height
        mid_y = y0 + ih // 2
        s = 9
        diamond = QtGui.QPolygon([
            QtCore.QPoint(div_x, mid_y - s),
            QtCore.QPoint(div_x + s, mid_y),
            QtCore.QPoint(div_x, mid_y + s),
            QtCore.QPoint(div_x - s, mid_y),
        ])
        p.setBrush(QtGui.QBrush(QtGui.QColor(UI_PRIMARY)))
        p.setPen(Qt.NoPen)
        p.drawPolygon(diamond)

        # Side labels
        font = p.font()
        font.setPointSize(8)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QtGui.QColor(UI_TEXT))
        if left_w > 40:
            p.drawText(x0 + 6, y0 + 18, "Base")
        if right_w > 60:
            p.drawText(div_x + 6, y0 + 18, "Compare")

        p.end()

    # ── Mouse interaction ──────────────────────────────────────────────────
    def _near_divider(self, pos_x: int) -> bool:
        target_rect = self._fitted_target_rect()
        x0, iw = target_rect.x(), target_rect.width()
        div_x = x0 + int(self._divider_ratio * iw)
        return abs(pos_x - div_x) <= 14

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        if event.button() == Qt.LeftButton and self._near_divider(event.pos().x()):
            self._dragging = True
            self.setCursor(Qt.SplitHCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        target_rect = self._fitted_target_rect()
        x0, iw = target_rect.x(), target_rect.width()
        if self._dragging:
            ratio = (event.pos().x() - x0) / max(iw, 1)
            self._divider_ratio = max(0.0, min(1.0, ratio))
            self.update()
            event.accept()
        else:
            self.setCursor(Qt.SplitHCursor if self._near_divider(event.pos().x())
                           else Qt.ArrowCursor)
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._dragging = False
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def resizeEvent(self, event):
        self.update()
        super().resizeEvent(event)


class ImageDisplayWidget(QtWidgets.QLabel):
    """Widget for displaying images with modern styling."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self._image: Optional[np.ndarray] = None
        self._hint_rect: Optional[tuple] = None  # (x, y, w, h) for hint box
        self._hint_info: str = ""  # Info text for hint
        self._show_hint: bool = False
        self.setMinimumSize(280, 280)  # Square display
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {UI_BG_CARD};
                border: 1px solid {UI_BORDER};
                border-radius: 12px;
            }}
        """)
        self._update_display()

    def setImage(self, image: np.ndarray):
        """Set image to display."""
        self._image = image
        self._update_display()

    def setHint(self, rect: tuple = None, info: str = "", show: bool = True):
        """Set hint rectangle and info.

        Args:
            rect: (x, y, w, h) in image coordinates
            info: Text to display near the hint
            show: Whether to show the hint
        """
        self._hint_rect = rect
        self._hint_info = info
        self._show_hint = show
        self._update_display()

    def setShowHint(self, show: bool):
        """Toggle hint visibility."""
        self._show_hint = show
        self._update_display()

    def _update_display(self):
        """Update the displayed pixmap."""
        if self._image is None:
            self.setText(f"{self._title}\n\nNo image")
            return

        h, w = self._image.shape[:2]

        # Convert to QImage
        if len(self._image.shape) == 2:
            # Grayscale -> Convert to RGB for drawing
            img_display = self._image
            if img_display.dtype != np.uint8:
                img_display = np.clip(img_display, 0, 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
        else:
            # Already color
            img_rgb = self._image.copy()

        # Draw hint box if enabled
        if self._show_hint and self._hint_rect is not None:
            x, y, rw, rh = self._hint_rect
            # Draw rectangle (teal accent color - BGR format)
            hint_color = _hex_to_bgr(UI_SUCCESS)
            cv2.rectangle(img_rgb, (x, y), (x + rw, y + rh), hint_color, 2)
            # Draw corner marks instead of full crosshair
            cx, cy = x + rw // 2, y + rh // 2
            mark_len = 10
            cv2.line(img_rgb, (cx - mark_len, cy), (cx + mark_len, cy), hint_color, 1)
            cv2.line(img_rgb, (cx, cy - mark_len), (cx, cy + mark_len), hint_color, 1)
            # Draw info text with shadow for readability
            if self._hint_info:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_y = max(y - 8, 18)
                # Shadow
                cv2.putText(img_rgb, self._hint_info, (x + 1, text_y + 1),
                            font, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                # Main text
                cv2.putText(img_rgb, self._hint_info, (x, text_y),
                            font, 0.45, hint_color, 1, cv2.LINE_AA)

        # Convert BGR to RGB for Qt
        img_qt = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        # Make a contiguous copy to ensure data lifetime
        img_qt = np.ascontiguousarray(img_qt)
        qimg = QtGui.QImage(img_qt.data, w, h, w * 3, QtGui.QImage.Format_RGB888).copy()

        pixmap = QtGui.QPixmap.fromImage(qimg)

        # Scale to fit
        scaled = pixmap.scaled(
            self.size() - QtCore.QSize(20, 20),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class HistogramCanvas(FigureCanvas):
    """Interactive matplotlib canvas for histogram display.

    Click once → set low bound, click again → set high bound and emit range_changed.
    Right-click or clicking a third time resets the selection.
    """

    range_changed = Signal(int, int)  # (lo, hi) gray-level values

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 2.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self._style()
        # Range selection state
        self._counts = None
        self._edges = None
        self._lo: Optional[int] = None
        self._hi: Optional[int] = None
        self._click_step: int = 0  # 0=idle, 1=waiting for hi
        self.mpl_connect('button_press_event', self._on_click)

    def _style(self):
        """Apply modern dark theme."""
        self.fig.patch.set_facecolor(UI_BG_WINDOW)
        self.ax.set_facecolor(UI_BG_CARD)
        self.ax.tick_params(colors=UI_TEXT_SECONDARY, labelsize=8, length=0)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.yaxis.grid(True, color=UI_BORDER, linestyle='-', linewidth=0.5, alpha=0.5)
        self.ax.set_axisbelow(True)

    def clear_range(self):
        """Reset range selection without re-plotting data."""
        self._lo = None
        self._hi = None
        self._click_step = 0
        if self._counts is not None:
            self._draw(self._counts, self._edges)

    def plot_histogram(self, counts: np.ndarray, edges: np.ndarray):
        """Plot histogram data, preserving any active range selection."""
        self._counts = counts
        self._edges = edges
        self._draw(counts, edges)

    def _draw(self, counts, edges):
        """Internal render with range overlay."""
        self.ax.clear()
        self._style()

        if counts is None or len(counts) == 0:
            self.ax.text(0.5, 0.5, "No data", ha='center', va='center',
                         color=UI_TEXT_SECONDARY, transform=self.ax.transAxes, fontsize=11)
            self.draw()
            return

        # Base bars
        lo_set = self._lo is not None
        hi_set = self._hi is not None and hi_set if False else (self._hi is not None)
        lo_v = self._lo if lo_set else -1
        hi_v = self._hi if hi_set else 256

        bar_colors = []
        for edge in edges[:-1]:
            if lo_set and hi_set and lo_v <= edge <= hi_v:
                bar_colors.append(UI_SUCCESS)  # success highlight
            elif lo_set and not hi_set and edge >= lo_v:
                bar_colors.append(UI_SUCCESS)  # pending second click
            else:
                bar_colors.append(UI_PRIMARY)

        self.ax.bar(edges[:-1], counts, width=1, color=bar_colors, alpha=0.85,
                    edgecolor='none', linewidth=0)

        # Range shading
        if lo_set and hi_set:
            self.ax.axvspan(lo_v, hi_v, alpha=0.12, color=UI_SUCCESS, zorder=0)

        # Vertical markers
        if lo_set:
            self.ax.axvline(lo_v, color=UI_SUCCESS, linewidth=1.5, linestyle='--')
        if hi_set:
            self.ax.axvline(hi_v, color=UI_SUCCESS, linewidth=1.5, linestyle='--')

        # Annotation
        if lo_set and not hi_set:
            self.ax.text(0.02, 0.96, f"lo={lo_v}  ← click to set hi",
                         transform=self.ax.transAxes, fontsize=7.5,
                         color=UI_SUCCESS, va='top')
        elif lo_set and hi_set:
            pct = np.sum((edges[:-1] >= lo_v) & (edges[:-1] <= hi_v) * counts)
            total = counts.sum() or 1
            self.ax.text(0.02, 0.96, f"GL {lo_v}–{hi_v}  ({pct / total * 100:.1f}%)",
                         transform=self.ax.transAxes, fontsize=7.5,
                         color=UI_SUCCESS, va='top')

        # Y-axis formatter
        from matplotlib.ticker import FuncFormatter
        self.ax.yaxis.set_major_formatter(FuncFormatter(
            lambda x, _: f'{x / 1000:.0f}K' if x >= 1000 else f'{x:.0f}'
        ))
        self.ax.set_xlabel("Gray Level", color=UI_TEXT_SECONDARY, fontsize=9)
        self.ax.set_ylabel("Count", color=UI_TEXT_SECONDARY, fontsize=9)
        self.ax.set_xlim(0, 255)

        try:
            self.fig.tight_layout(pad=1.2)
        except Exception:
            pass
        self.draw()

    def _on_click(self, event):
        """Handle mouse click: set lo then hi, right-click resets."""
        if event.inaxes != self.ax or event.xdata is None:
            return
        x = int(round(max(0, min(255, event.xdata))))

        if event.button == 3:  # Right-click → reset
            self._lo = None
            self._hi = None
            self._click_step = 0
            self._draw(self._counts, self._edges)
            return

        if self._click_step == 0:
            # First click: set lo
            self._lo = x
            self._hi = None
            self._click_step = 1
        elif self._click_step == 1:
            # Second click: set hi, emit signal
            lo, hi = sorted([self._lo, x])
            self._lo, self._hi = lo, hi
            self._click_step = 0
            self._draw(self._counts, self._edges)
            self.range_changed.emit(lo, hi)
            return

        self._draw(self._counts, self._edges)


class _Worker(QtCore.QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            result = self._fn()
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.finished.emit(result)


class AlignmentScoreWidget(QtWidgets.QFrame):
    """Alignment summary card styled to match the design mockup."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AlignCard")
        self.setMinimumWidth(248)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet(f"""
            QFrame#AlignCard {{
                background-color: {UI_BG_CARD};
                border: 1px solid {UI_BORDER};
                border-radius: {BorderRadius.MD};
            }}
            QFrame#AlignCard QLabel {{
                border: none;
                background: transparent;
            }}
            QFrame#AlignCard QLabel[cardTitle="true"] {{
                color: {UI_TEXT};
                font-size: {Typography.FONT_SIZE_BODY};
                font-weight: {Typography.FONT_WEIGHT_BOLD};
            }}
            QFrame#AlignCard QLabel[cardSubTitle="true"] {{
                color: {UI_TEXT_MUTED};
                font-size: {Typography.FONT_SIZE_CAPTION};
            }}
            QFrame#AlignCard QLabel[statLabel="true"] {{
                color: {UI_TEXT_MUTED};
                font-size: {Typography.FONT_SIZE_SMALL};
            }}
            QFrame#AlignCard QLabel[statValue="true"] {{
                color: {UI_TEXT};
                font-size: {Typography.FONT_SIZE_BODY};
                font-weight: {Typography.FONT_WEIGHT_BOLD};
                font-family: {Typography.FONT_FAMILY_MONO};
            }}
            QFrame#AlignCard QLabel[statValueLong="true"] {{
                color: {UI_TEXT_SECONDARY};
                font-size: {Typography.FONT_SIZE_SMALL};
                font-weight: {Typography.FONT_WEIGHT_SEMIBOLD};
                font-family: {Typography.FONT_FAMILY_MONO};
            }}
            QFrame#AlignCard QFrame[divider="true"] {{
                background-color: {UI_BORDER};
                min-height: 1px;
                max-height: 1px;
                border: none;
            }}
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        header = QtWidgets.QVBoxLayout()
        header.setSpacing(4)
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)
        dot = QtWidgets.QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background-color: {UI_PRIMARY}; border-radius: 4px;")
        title = QtWidgets.QLabel("Alignment Quality")
        title.setProperty("cardTitle", True)
        top_row.addWidget(dot, 0, Qt.AlignVCenter)
        top_row.addWidget(title)
        top_row.addStretch()
        header.addLayout(top_row)
        sub = QtWidgets.QLabel("Single-pair alignment confidence")
        sub.setProperty("cardSubTitle", True)
        header.addWidget(sub)
        layout.addLayout(header)

        divider = QtWidgets.QFrame()
        divider.setProperty("divider", True)
        layout.addWidget(divider)

        self.rows: Dict[str, QtWidgets.QLabel] = {}
        for key, label in [
            ("phase", "Phase Shift"),
            ("ncc", "NCC"),
            ("residual", "Residual"),
            ("final", "Final Score"),
            ("shift", "Shift"),
        ]:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(8)
            lbl_name = QtWidgets.QLabel(label)
            lbl_name.setProperty("statLabel", True)
            lbl_value = QtWidgets.QLabel("--")
            lbl_value.setProperty("statValue", True)
            lbl_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if key == "shift":
                lbl_value.setProperty("statValueLong", True)
            row.addWidget(lbl_name)
            row.addStretch()
            row.addWidget(lbl_value)
            layout.addLayout(row)
            self.rows[key] = lbl_value

        divider2 = QtWidgets.QFrame()
        divider2.setProperty("divider", True)
        layout.addWidget(divider2)

        # Backward-compatible aliases used by the existing update logic.
        self.lbl_phase = self.rows["phase"]
        self.lbl_ncc = self.rows["ncc"]
        self.lbl_residual = self.rows["residual"]
        self.lbl_final = self.rows["final"]
        self.lbl_shift = self.rows["shift"]

        status_row = QtWidgets.QHBoxLayout()
        self.lbl_status_text = QtWidgets.QLabel("Status")
        self.lbl_status_text.setProperty("statLabel", True)
        self.lbl_status = QtWidgets.QLabel("--")
        self.lbl_status.setProperty("statValue", True)
        self.lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        status_row.addWidget(self.lbl_status_text)
        status_row.addStretch()
        status_row.addWidget(self.lbl_status)
        layout.addLayout(status_row)
        layout.addStretch(1)

    def update_scores(self, result: CombineResult):
        if not result or not result.alignments:
            self._set_empty()
            return

        n = len(result.alignments)
        avg_phase = sum(a.score_phase for a in result.alignments) / n
        avg_ncc = sum(a.score_ncc for a in result.alignments) / n
        avg_residual = sum(a.score_residual for a in result.alignments) / n
        avg_final = sum(a.final_score for a in result.alignments) / n
        avg_dx = sum(a.dx for a in result.alignments) / n
        avg_dy = sum(a.dy for a in result.alignments) / n

        self.rows["phase"].setText(f"{avg_phase:.3f}")
        self.rows["ncc"].setText(f"{avg_ncc:.3f}")
        self.rows["residual"].setText(f"{avg_residual:.3f}")
        self.rows["final"].setText(f"{avg_final:.1f}")
        self.rows["shift"].setText(f"({avg_dx:+.1f}, {avg_dy:+.1f})")

        worst = result.worst_alignment_score
        if worst >= 75:
            status_text = "OK"
            status_color = UI_SUCCESS
        elif worst >= 55:
            status_text = "WARN"
            status_color = UI_PRIMARY_HOVER
        else:
            status_text = "FAIL"
            status_color = UI_WARNING
        self.lbl_status.setText(status_text)
        self.lbl_status.setStyleSheet(
            f"color: {status_color}; font-size: {Typography.FONT_SIZE_BODY}; font-weight: {Typography.FONT_WEIGHT_BOLD}; border: none;"
        )

    def _set_empty(self):
        for key, lbl in self.rows.items():
            lbl.setText("--")
            prop = "true" if key == "shift" else None
            if prop:
                lbl.setProperty("statValueLong", True)
                lbl.style().unpolish(lbl)
                lbl.style().polish(lbl)
        self.lbl_status.setText("--")
        self.lbl_status.setStyleSheet(
            f"color: {UI_TEXT}; font-size: {Typography.FONT_SIZE_BODY}; font-weight: {Typography.FONT_WEIGHT_BOLD}; border: none;"
        )


class StatisticsWidget(QtWidgets.QFrame):
    """Merged Analysis card — shows Alignment and Difference metrics in one panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatsCard")
        self.setMinimumWidth(248)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet(f"""
            QFrame#StatsCard {{
                background-color: {UI_BG_CARD};
                border: 1px solid {UI_BORDER};
                border-radius: {BorderRadius.MD};
            }}
            QFrame#StatsCard QLabel {{
                border: none;
                background: transparent;
            }}
            QFrame#StatsCard QLabel[cardTitle="true"] {{
                color: {UI_TEXT};
                font-size: {Typography.FONT_SIZE_BODY};
                font-weight: {Typography.FONT_WEIGHT_BOLD};
            }}
            QFrame#StatsCard QLabel[sectionHeader="true"] {{
                color: {UI_TEXT_SECONDARY};
                font-size: {Typography.FONT_SIZE_CAPTION};
                font-weight: {Typography.FONT_WEIGHT_BOLD};
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            QFrame#StatsCard QLabel[statLabel="true"] {{
                color: {UI_TEXT_MUTED};
                font-size: {Typography.FONT_SIZE_SMALL};
            }}
            QFrame#StatsCard QLabel[statValue="true"] {{
                color: {UI_TEXT};
                font-size: {Typography.FONT_SIZE_BODY};
                font-weight: {Typography.FONT_WEIGHT_BOLD};
                font-family: {Typography.FONT_FAMILY_MONO};
            }}
            QFrame#StatsCard QLabel[statValueLong="true"] {{
                color: {UI_TEXT_SECONDARY};
                font-size: {Typography.FONT_SIZE_SMALL};
                font-weight: {Typography.FONT_WEIGHT_SEMIBOLD};
                font-family: {Typography.FONT_FAMILY_MONO};
            }}
            QFrame#StatsCard QFrame[divider="true"] {{
                background-color: {UI_BORDER};
                min-height: 1px;
                max-height: 1px;
                border: none;
            }}
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        # Card title
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)
        dot = QtWidgets.QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background-color: {UI_PRIMARY}; border-radius: 4px;")
        title = QtWidgets.QLabel("Analysis")
        title.setProperty("cardTitle", True)
        top_row.addWidget(dot, 0, Qt.AlignVCenter)
        top_row.addWidget(title)
        top_row.addStretch()
        layout.addLayout(top_row)

        divider0 = QtWidgets.QFrame()
        divider0.setProperty("divider", True)
        layout.addWidget(divider0)

        # ── ALIGNMENT section ─────────────────────────────────────────────────
        lbl_align_hdr = QtWidgets.QLabel("Alignment")
        lbl_align_hdr.setProperty("sectionHeader", True)
        layout.addWidget(lbl_align_hdr)

        self.align_labels: Dict[str, QtWidgets.QLabel] = {}
        for key, label in [
            ("phase", "Phase Shift"),
            ("ncc", "NCC"),
            ("residual", "Residual"),
            ("final", "Final Score"),
            ("shift", "Shift"),
        ]:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(8)
            lbl_name = QtWidgets.QLabel(label)
            lbl_name.setProperty("statLabel", True)
            lbl_value = QtWidgets.QLabel("--")
            lbl_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if key == "shift":
                lbl_value.setProperty("statValueLong", True)
            else:
                lbl_value.setProperty("statValue", True)
            row.addWidget(lbl_name)
            row.addStretch()
            row.addWidget(lbl_value)
            layout.addLayout(row)
            self.align_labels[key] = lbl_value

        # Status row (align)
        status_row = QtWidgets.QHBoxLayout()
        lbl_status_name = QtWidgets.QLabel("Status")
        lbl_status_name.setProperty("statLabel", True)
        self.lbl_align_status = QtWidgets.QLabel("--")
        self.lbl_align_status.setProperty("statValue", True)
        self.lbl_align_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        status_row.addWidget(lbl_status_name)
        status_row.addStretch()
        status_row.addWidget(self.lbl_align_status)
        layout.addLayout(status_row)

        divider1 = QtWidgets.QFrame()
        divider1.setProperty("divider", True)
        layout.addWidget(divider1)

        # ── DIFFERENCE section ────────────────────────────────────────────────
        lbl_diff_hdr = QtWidgets.QLabel("Difference")
        lbl_diff_hdr.setProperty("sectionHeader", True)
        layout.addWidget(lbl_diff_hdr)

        self.stats_labels: Dict[str, QtWidgets.QLabel] = {}
        for key, label in [
            ("diff_mean", "Diff Mean"),
            ("diff_std", "Diff Std"),
            ("hot_pixels", "Hot Pixels"),
            ("norm_coeff", "Normalize"),
            ("subpixel_shift", "Sub-px Shift"),
        ]:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(8)
            lbl_name = QtWidgets.QLabel(label)
            lbl_name.setProperty("statLabel", True)
            lbl_value = QtWidgets.QLabel("--")
            lbl_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if key in {"norm_coeff", "subpixel_shift"}:
                lbl_value.setProperty("statValueLong", True)
            else:
                lbl_value.setProperty("statValue", True)
            row.addWidget(lbl_name)
            row.addStretch()
            row.addWidget(lbl_value)
            layout.addLayout(row)
            self.stats_labels[key] = lbl_value

        layout.addStretch(1)
        self.reset()

    def update_stats(self, result: CombineResult):
        if not result or not result.stats:
            for lbl in self.stats_labels.values():
                lbl.setText("--")
            return

        s = result.stats
        self.stats_labels["diff_mean"].setText(f"{s.get('diff_mean', 0):.4f}")
        self.stats_labels["diff_std"].setText(f"{s.get('diff_std', 0):.4f}")
        self.stats_labels["hot_pixels"].setText(f"{s.get('hot_pixels', 0)}")
        self.stats_labels["norm_coeff"].setText(f"a={s.get('norm_a', 0):.4f}, b={s.get('norm_b', 0):.2f}")
        self.stats_labels["subpixel_shift"].setText(
            f"({s.get('subpixel_dx', 0):+.2f}, {s.get('subpixel_dy', 0):+.2f})"
        )

    def update_alignment(self, phase: str, ncc: str, residual: str,
                         final: str, shift: str, status: str, status_color: str):
        """Update the alignment section of the merged analysis card."""
        self.align_labels["phase"].setText(phase)
        self.align_labels["ncc"].setText(ncc)
        self.align_labels["residual"].setText(residual)
        self.align_labels["final"].setText(final)
        self.align_labels["shift"].setText(shift)
        self.lbl_align_status.setText(f" {status} ")
        self.lbl_align_status.setStyleSheet(
            f"color: #FFFFFF; background-color: {status_color}; border: 1px solid {status_color};"
            f" border-radius: {BorderRadius.SM}; padding: 1px 8px;"
            f" font-size: {Typography.FONT_SIZE_SMALL}; font-weight: {Typography.FONT_WEIGHT_BOLD};"
        )

    def reset(self):
        """Reset Analysis card to default empty-state values."""
        for lbl in self.align_labels.values():
            lbl.setText("--")
        for lbl in self.stats_labels.values():
            lbl.setText("--")
        self.lbl_align_status.setText(" -- ")
        self.lbl_align_status.setStyleSheet(
            f"color: {UI_TEXT_MUTED}; background-color: {UI_BG_SUBTLE}; border: 1px solid {UI_BORDER};"
            f" border-radius: {BorderRadius.SM}; padding: 1px 8px;"
            f" font-size: {Typography.FONT_SIZE_SMALL}; font-weight: {Typography.FONT_WEIGHT_BOLD};"
        )


class GLVMaskPreviewDialog(QtWidgets.QDialog):
    """Interactive dialog that visualises the GLV-Mask on the Base image.

    The user can adjust GLV Low / High directly inside the dialog and see the
    mask update in real-time.  Clicking 'Apply to Main' syncs the values back
    to the parent dialog's spinboxes.

    Visualisation:
      • In-mask pixels  : green-tinted highlight (R×0.65, G×1.0, B×0.65)
      • Out-of-mask pixels: dimmed to 8 % so the selected band stands out sharply
    """

    def __init__(
            self,
            base_img: np.ndarray,
            glv_low: int,
            glv_high: int,
            parent=None,
            apply_callback=None,  # callable(low: int, high: int) → None
    ):
        super().__init__(parent)
        self._base_img = base_img
        self._apply_callback = apply_callback
        self.setWindowTitle("GLV Mask Preview")
        self.setMinimumSize(540, 600)
        self.setStyleSheet(DIALOG_STYLE)
        self._setup_ui(glv_low, glv_high)
        self._update_preview()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self, glv_low: int, glv_high: int):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # ── GLV range controls ────────────────────────────────────────────────
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.addWidget(QtWidgets.QLabel("GLV range:"))

        self.spn_low = QtWidgets.QSpinBox()
        self.spn_low.setRange(0, 254)
        self.spn_low.setValue(glv_low)
        self.spn_low.setFixedWidth(65)
        self.spn_low.setToolTip("Lower bound (inclusive, 0–255)")
        ctrl_row.addWidget(self.spn_low)

        ctrl_row.addWidget(QtWidgets.QLabel("–"))

        self.spn_high = QtWidgets.QSpinBox()
        self.spn_high.setRange(1, 255)
        self.spn_high.setValue(glv_high)
        self.spn_high.setFixedWidth(65)
        self.spn_high.setToolTip("Upper bound (inclusive, 0–255)")
        ctrl_row.addWidget(self.spn_high)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # ── Image display ─────────────────────────────────────────────────────
        self._lbl_image = QtWidgets.QLabel()
        self._lbl_image.setAlignment(Qt.AlignCenter)
        self._lbl_image.setMinimumSize(480, 420)
        layout.addWidget(self._lbl_image, stretch=1)

        # ── Info bar (pixel count / %) ────────────────────────────────────────
        self._lbl_info = QtWidgets.QLabel()
        self._lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lbl_info)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        if self._apply_callback is not None:
            self._btn_apply = QtWidgets.QPushButton("Apply to Main")
            self._btn_apply.setToolTip("Copy the current GLV range back to the main dialog")
            self._btn_apply.clicked.connect(self._on_apply)
            btn_row.addWidget(self._btn_apply)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        # Live update on spinbox change
        self.spn_low.valueChanged.connect(self._update_preview)
        self.spn_high.valueChanged.connect(self._update_preview)

    # ── Preview rendering ─────────────────────────────────────────────────────

    def _update_preview(self):
        """Recompute mask and refresh the image label."""
        glv_low = self.spn_low.value()
        glv_high = self.spn_high.value()

        img_f = self._base_img.astype(np.float32)
        mask = (img_f >= glv_low) & (img_f <= glv_high)

        # Stats
        n_in = int(mask.sum())
        pct = 100.0 * n_in / max(mask.size, 1)
        status = "too few pixels — will fall back to full-image Percentile" if n_in < 50 else f"{n_in:,} px in mask  ({pct:.1f} %)"
        self._lbl_info.setText(f"GLV {glv_low} – {glv_high}   |   {status}")

        # Build RGB visualisation
        rgb = self._build_rgb_preview(self._base_img, mask)

        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QtGui.QImage(rgb.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)

        avail = self._lbl_image.size()
        scaled = pixmap.scaled(
            avail.width() - 4, avail.height() - 4,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._lbl_image.setPixmap(scaled)

        self.setWindowTitle(f"GLV Mask Preview  [{glv_low} – {glv_high}]")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview()

    @staticmethod
    def _build_rgb_preview(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return uint8 RGB array.

        In-mask  : green highlight (R×0.65, G×1.0, B×0.65)
        Out-mask : dimmed to 8 % as neutral grey
        """
        img_f = img.astype(np.float32)
        h, w = img_f.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Out-of-mask: dim to 8 %
        dim = np.clip(img_f * 0.08, 0, 255).astype(np.uint8)
        rgb[..., 0] = dim
        rgb[..., 1] = dim
        rgb[..., 2] = dim

        # In-mask: green-tinted highlight
        v = img_f[mask]
        rgb[mask, 0] = np.clip(v * 0.65, 0, 255).astype(np.uint8)
        rgb[mask, 1] = np.clip(v, 0, 255).astype(np.uint8)
        rgb[mask, 2] = np.clip(v * 0.65, 0, 255).astype(np.uint8)

        return rgb

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_apply(self):
        """Sync current spinbox values back to the parent dialog."""
        self._apply_callback(self.spn_low.value(), self.spn_high.value())


_NORMALIZE_METHOD_LABELS = {
    'percentile': 'Percentile (P2–P98)',
    'glv_mask': 'GLV-Mask',
    'heq': 'Histogram EQ (HEQ)',
    'clahe': 'CLAHE',
    'skip': 'Skip (raw ÷ 255)',
    'roi_match': 'ROI-Match (EPI Nulling)',
}


class NormalizedCompareDialog(QtWidgets.QDialog):
    """Dialog to preview normalization effect: shows raw compare, normalized compare, and base."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Normalize Preview")
        self.resize(900, 800)
        self.setStyleSheet(DIALOG_STYLE)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet(f"color: {UI_TEXT_SECONDARY}; font-size: {Typography.FONT_SIZE_SMALL};")
        layout.addWidget(self.lbl_info)

        label_row = QtWidgets.QHBoxLayout()
        self.lbl_compare_title = QtWidgets.QLabel("Aligned Compare (Before)")
        self.lbl_norm_title = QtWidgets.QLabel("After Normalize")
        self.lbl_base_title = QtWidgets.QLabel("Base (Target)")
        for _lbl in (self.lbl_compare_title, self.lbl_norm_title, self.lbl_base_title):
            _lbl.setAlignment(Qt.AlignCenter)
            _lbl.setStyleSheet(
                f"font-size: {Typography.FONT_SIZE_SMALL}; font-weight: {Typography.FONT_WEIGHT_BOLD};"
                f" color: {UI_TEXT};"
            )
            label_row.addWidget(_lbl, 1)
        layout.addLayout(label_row)

        image_row = QtWidgets.QHBoxLayout()
        self.img_compare = SyncZoomImageWidget("Aligned Compare (Before)")
        self.img_norm_compare = SyncZoomImageWidget("After Normalize")
        self.img_base = SyncZoomImageWidget("Base (Target)")
        image_row.addWidget(self.img_compare, 1)
        image_row.addWidget(self.img_norm_compare, 1)
        image_row.addWidget(self.img_base, 1)
        layout.addLayout(image_row, 1)

        hist_row = QtWidgets.QHBoxLayout()
        self.hist_compare = HistogramCanvas()
        self.hist_compare.setFixedHeight(140)
        self.hist_norm_compare = HistogramCanvas()
        self.hist_norm_compare.setFixedHeight(140)
        self.hist_base = HistogramCanvas()
        self.hist_base.setFixedHeight(140)
        hist_row.addWidget(self.hist_compare, 1)
        hist_row.addWidget(self.hist_norm_compare, 1)
        hist_row.addWidget(self.hist_base, 1)
        layout.addLayout(hist_row)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        button_row.addWidget(btn_close)
        layout.addLayout(button_row)

    def set_images(
            self,
            compare_image: np.ndarray,
            normalized_compare: np.ndarray,
            base_image: np.ndarray,
            base_label: str,
            compare_label: str,
            norm_a: float,
            norm_b: float,
            method_name: str = 'percentile',
    ):
        method_display = _NORMALIZE_METHOD_LABELS.get(method_name, method_name)
        is_linear = method_name in ('percentile', 'glv_mask')
        if is_linear:
            coeff_str = f"  |  a={norm_a:.4f}, b={norm_b:.1f}"
        else:
            coeff_str = "  |  (non-linear)"
        self.lbl_info.setText(
            f"Method: {method_display}  |  Compare: {compare_label}  →  Base: {base_label}{coeff_str}"
        )
        self.lbl_compare_title.setText(f"Aligned Compare — Before\n({compare_label})")
        self.lbl_norm_title.setText(f"After Normalize\n[{method_display}]")
        self.lbl_base_title.setText(f"Base — Target\n({base_label})")
        self.setWindowTitle(f"Normalize Preview  [{method_display}]  —  {compare_label}")
        if compare_image is not None:
            self.img_compare.setImage(compare_image)
        if normalized_compare is not None:
            self.img_norm_compare.setImage(normalized_compare)
        if base_image is not None:
            self.img_base.setImage(base_image)

        self._update_histogram(self.hist_compare, compare_image)
        self._update_histogram(self.hist_norm_compare, normalized_compare)
        self._update_histogram(self.hist_base, base_image)

    @staticmethod
    def _update_histogram(canvas: HistogramCanvas, image: Optional[np.ndarray]):
        if image is None or image.size == 0:
            canvas.plot_histogram(np.zeros(256), np.arange(257))
            return
        img_uint8 = image.astype(np.uint8)
        counts, edges = np.histogram(img_uint8.flatten(), bins=256, range=(0, 256))
        canvas.plot_histogram(counts, edges)


class PerspectiveCombinationDialog(QtWidgets.QDialog):
    """Dialog for multi-image perspective combination and defect detection."""

    def __init__(self, parent=None, conditions: List[EbeamCondition] = None):
        super().__init__(parent)
        self.setWindowTitle("Fusi\u00b3 \u2014 SEM Perspective Combination Tool")
        self.setMinimumSize(1500, 900)
        self.resize(1600, 950)

        # Apply a clean sans-serif font (Arial on Windows/macOS,
        # Liberation Sans on Linux — metrically identical to Arial).
        _ui_font = QtGui.QFont()
        for candidate in ("Liberation Sans", "Arial", "Helvetica Neue", "Segoe UI"):
            _ui_font.setFamily(candidate)
            _info = QtGui.QFontInfo(_ui_font)
            if _info.family().lower().replace(" ", "") == candidate.lower().replace(" ", ""):
                break  # found a font that is actually installed
        _ui_font.setPointSize(10)
        self.setFont(_ui_font)

        self.setStyleSheet(DIALOG_STYLE)

        self._conditions = conditions or []
        self._images: Dict[str, np.ndarray] = {}
        self._results: List[SinglePairResult] = []  # Multiple results
        self._current_result_idx: int = 0  # Current result index
        self._result: Optional[CombineResult] = None  # Legacy
        self._last_settings: Dict[str, object] = {}
        self._compute_thread: Optional[QtCore.QThread] = None
        self._compute_worker: Optional[_Worker] = None
        self._display_mode = 'diff'  # right viewer: 'diff' or 'zmap'
        self._left_non_split_mode = 'base'
        self._left_view_mode = 'base'
        self._norm_compare_dialog: Optional[NormalizedCompareDialog] = None
        self._hist_range: Optional[tuple] = None  # (lo, hi) gray-level range for highlight, or None

        self._setup_ui()
        self._apply_toolbar_icons()
        self._connect_signals()
        self._load_images()

    def _set_button_icon(self, button: QtWidgets.QPushButton, pixmap_enum, text: str = None, size: int = 16):
        if text is not None:
            button.setText(text)
        icon = self.style().standardIcon(pixmap_enum)
        button.setIcon(icon)
        button.setIconSize(QtCore.QSize(size, size))

    def _polish_button(self, button: QtWidgets.QPushButton, variant: str | None = None):
        if variant is not None:
            button.setProperty("variant", variant)
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    def _apply_toolbar_icons(self):
        self._set_button_icon(self.btn_load_folder, QtWidgets.QStyle.SP_DirOpenIcon, "Load Folder")
        self._set_button_icon(self.btn_compute, QtWidgets.QStyle.SP_MediaPlay, "Compute")
        self._set_button_icon(self.btn_export, QtWidgets.QStyle.SP_DialogSaveButton, "Export")
        self._set_button_icon(self.btn_clear_hist_range, QtWidgets.QStyle.SP_DialogResetButton, "Clear Range")
        self._set_button_icon(self.btn_show_norm_compare, QtWidgets.QStyle.SP_FileDialogContentsView, "Normalize")
        self._set_button_icon(self.btn_qf_auto_detect, QtWidgets.QStyle.SP_FileDialogDetailedView, "Auto Detect")
        self._set_button_icon(self.btn_qf_pick_roi, QtWidgets.QStyle.SP_DialogOpenButton, "Pick ROI")
        self._set_button_icon(self.btn_qf_clear_roi, QtWidgets.QStyle.SP_DialogResetButton, "Clear ROI")
        self._set_button_icon(self.btn_pick_roi, QtWidgets.QStyle.SP_DialogOpenButton, "Pick ROI")
        self._set_button_icon(self.btn_clear_roi, QtWidgets.QStyle.SP_DialogResetButton, "Clear ROI")
        self._set_button_icon(self.btn_preview_glv_mask, QtWidgets.QStyle.SP_DesktopIcon, "Preview Mask")

    def closeEvent(self, event):
        """Handle dialog close - ensure compute thread is stopped safely."""
        if self._compute_thread is not None:
            self._compute_thread.requestInterruption()
            self._compute_thread.quit()
            self._compute_worker = None
        event.accept()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for navigation."""
        key = event.key()

        # ← Arrow: Previous result
        if key == Qt.Key_Left:
            self._on_prev_result()
            event.accept()
            return

        # → Arrow: Next result
        if key == Qt.Key_Right:
            self._on_next_result()
            event.accept()
            return

        # Space: Compute
        if key == Qt.Key_Space and self.btn_compute.isEnabled():
            self._on_compute()
            event.accept()
            return

        # Escape: Close dialog
        if key == Qt.Key_Escape:
            self.close()
            event.accept()
            return

        super().keyPressEvent(event)

    def _setup_ui(self):
        """Build the dialog UI — target-style layout with toolbar, compact sidebar, viewer, bottom cards."""
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.setSpacing(8)

        # ================================================================
        # TOP TOOLBAR
        # ================================================================
        toolbar = QtWidgets.QFrame()
        toolbar.setObjectName("MainToolbar")
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(6)

        # Load Folder
        self.btn_load_folder = QtWidgets.QPushButton("Load Folder")
        self.btn_load_folder.setObjectName("ToolbarAction")
        toolbar_layout.addWidget(self.btn_load_folder)

        # Navigation arrows (prev/next folder image set)
        self.btn_prev_result = QtWidgets.QPushButton("\u25C0")
        self.btn_prev_result.setFixedWidth(32)
        self.btn_prev_result.setObjectName("ToolbarAction")
        self.btn_prev_result.setEnabled(False)
        self.btn_next_result = QtWidgets.QPushButton("\u25B6")
        self.btn_next_result.setFixedWidth(32)
        self.btn_next_result.setObjectName("ToolbarAction")
        self.btn_next_result.setEnabled(False)
        toolbar_layout.addWidget(self.btn_prev_result)
        toolbar_layout.addWidget(self.btn_next_result)

        toolbar_layout.addStretch(1)

        # Center: main actions
        self.btn_auto_pair_toolbar = QtWidgets.QPushButton("Auto Pair")
        self.btn_auto_pair_toolbar.setObjectName("ToolbarAction")
        self.btn_auto_pair_toolbar.setCheckable(True)
        toolbar_layout.addWidget(self.btn_auto_pair_toolbar)

        self.btn_compute = QtWidgets.QPushButton("Compute")
        self.btn_compute.setObjectName("ToolbarPrimary")
        toolbar_layout.addWidget(self.btn_compute)

        self.btn_export = QtWidgets.QPushButton("Export")
        self.btn_export.setObjectName("ToolbarAction")
        self.btn_export.setEnabled(False)
        toolbar_layout.addWidget(self.btn_export)

        toolbar_layout.addStretch(1)

        self.btn_about = QtWidgets.QPushButton("\u2026")
        self.btn_about.setObjectName("ToolbarAction")
        self.btn_about.setFixedWidth(36)
        self.btn_about.setToolTip("About Fusi\u00b3")
        toolbar_layout.addWidget(self.btn_about)

        outer_layout.addWidget(toolbar)

        # Result info label (spans full width, below toolbar)
        self.lbl_result_info = QtWidgets.QLabel("")
        self.lbl_result_info.setAlignment(Qt.AlignCenter)
        self.lbl_result_info.setStyleSheet(f"""
            QLabel {{
                color: {UI_TEXT};
                background-color: {UI_BG_CARD};
                border: 1px solid {UI_BORDER};
                border-radius: {BorderRadius.SM};
                padding: 4px 12px;
                font-size: {Typography.FONT_SIZE_SMALL};
            }}
        """)
        self.lbl_result_info.setFixedHeight(28)
        self.lbl_result_info.setVisible(False)
        outer_layout.addWidget(self.lbl_result_info)

        # ================================================================
        # CONTENT AREA: sidebar + viewer
        # ================================================================
        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(8)

        # === LEFT SIDEBAR ===
        left_panel = QtWidgets.QWidget()
        left_panel.setObjectName("LeftPanel")
        left_panel.setFixedWidth(260)
        left_panel.setStyleSheet(f"""
            QWidget#LeftPanel {{
                background-color: {UI_BG_PANEL};
                border: 1px solid {UI_BORDER};
                border-radius: {BorderRadius.MD};
            }}
            QWidget#LeftPanel > QWidget,
            QWidget#LeftPanel QWidget {{
                background-color: transparent;
                border: none;
            }}
            QWidget#LeftPanel QComboBox {{
                background-color: {UI_BG_INPUT};
                border: 1px solid {UI_BORDER};
                border-radius: {BorderRadius.SM};
                padding: {Spacing.INPUT_PADDING};
                font-size: {Typography.FONT_SIZE_SMALL};
                min-height: 28px;
            }}
            QWidget#LeftPanel QComboBox:hover,
            QWidget#LeftPanel QComboBox:focus {{
                border-color: {UI_PRIMARY};
            }}
            QWidget#LeftPanel QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
            QWidget#LeftPanel QComboBox QAbstractItemView {{
                background-color: {UI_BG_PANEL};
                color: {UI_TEXT};
                selection-background-color: {UI_PRIMARY};
                selection-color: {UI_TEXT_ON_PRIMARY};
                border: 1px solid {UI_BORDER};
            }}
            QWidget#LeftPanel QCheckBox {{
                color: {UI_TEXT};
                spacing: {Spacing.SM};
                font-size: {Typography.FONT_SIZE_SMALL};
                background: transparent;
                border: none;
            }}
            QWidget#LeftPanel QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 2px solid #D1D5DB;
                border-radius: 3px;
                background-color: {UI_BG_INPUT};
            }}
            QWidget#LeftPanel QCheckBox::indicator:checked {{
                background-color: {UI_PRIMARY};
                border-color: {UI_PRIMARY};
            }}
            QWidget#LeftPanel QSpinBox,
            QWidget#LeftPanel QDoubleSpinBox {{
                background-color: {UI_BG_INPUT};
                color: {UI_TEXT};
                border: 1px solid {UI_BORDER};
                border-radius: {BorderRadius.SM};
                padding: 4px 22px 4px 8px;
                min-height: 24px;
                font-size: {Typography.FONT_SIZE_SMALL};
            }}
            QWidget#LeftPanel QSpinBox:hover,
            QWidget#LeftPanel QDoubleSpinBox:hover {{
                border-color: {UI_PRIMARY};
            }}
            QWidget#LeftPanel QSpinBox::up-button,
            QWidget#LeftPanel QDoubleSpinBox::up-button,
            QWidget#LeftPanel QSpinBox::down-button,
            QWidget#LeftPanel QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                width: 14px;
                border-left: 1px solid {UI_BORDER};
                background-color: {UI_BG_CARD};
            }}
            QWidget#LeftPanel QSpinBox::up-button,
            QWidget#LeftPanel QDoubleSpinBox::up-button {{
                subcontrol-position: top right;
                border-top-right-radius: {BorderRadius.SM};
            }}
            QWidget#LeftPanel QSpinBox::down-button,
            QWidget#LeftPanel QDoubleSpinBox::down-button {{
                subcontrol-position: bottom right;
                border-bottom-right-radius: {BorderRadius.SM};
                border-top: 1px solid {UI_BORDER};
            }}
            QWidget#LeftPanel QPushButton {{
                background-color: {UI_BG_PANEL};
                color: {UI_TEXT};
                border: 1px solid {UI_BORDER};
                border-radius: {BorderRadius.SM};
                padding: 4px 10px;
                min-height: 26px;
                font-size: {Typography.FONT_SIZE_SMALL};
                font-weight: {Typography.FONT_WEIGHT_MEDIUM};
            }}
            QWidget#LeftPanel QPushButton:hover {{
                background-color: #FFF8ED;
                border-color: {UI_PRIMARY};
            }}
            QWidget#LeftPanel QPushButton:pressed {{
                background-color: #FDE7C2;
            }}
            QWidget#LeftPanel QLabel {{
                color: {UI_TEXT};
                font-size: {Typography.FONT_SIZE_SMALL};
                background: transparent;
                border: none;
            }}
            QWidget#LeftPanel QLabel#SectionTitle {{
                color: {UI_TEXT};
                font-size: {Typography.FONT_SIZE_BODY};
                font-weight: {Typography.FONT_WEIGHT_SEMIBOLD};
                text-transform: uppercase;
                padding: 2px 0px;
                letter-spacing: 0.6px;
            }}
            QWidget#LeftPanel QLabel#SectionSeparator {{
                background-color: {UI_BORDER};
                min-height: 1px;
                max-height: 1px;
                border: none;
                margin-top: 2px;
                margin-bottom: 2px;
            }}
            QWidget#LeftPanel QScrollArea {{
                border: none;
                background: transparent;
            }}
        """)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(5)

        # --- INPUT section ---
        lbl_img_mgr = QtWidgets.QLabel("INPUT")
        lbl_img_mgr.setObjectName("SectionTitle")
        left_layout.addWidget(lbl_img_mgr)
        sep1 = QtWidgets.QLabel()
        sep1.setObjectName("SectionSeparator")
        left_layout.addWidget(sep1)
        left_layout.addSpacing(6)

        self.lbl_input_empty_hint = QtWidgets.QLabel("No images loaded.\nLoad a folder to begin.")
        self.lbl_input_empty_hint.setProperty("secondary", True)
        self.lbl_input_empty_hint.setWordWrap(True)
        left_layout.addWidget(self.lbl_input_empty_hint)

        # Auto Pair dropdown (input mode)
        self.cmb_input_mode = QtWidgets.QComboBox()
        self.cmb_input_mode.addItems([
            "Standard (Base/Compare)",
            "Quadrant Fusion (Illum + T/B/L/R)",
        ])
        self.cmb_input_mode.setToolTip(
            "Standard: classic Base vs Compare subtract/blend workflow.\n"
            "Quadrant Fusion: combine Illuminator + 4 Quadrant detector images\n"
            "to produce BSE-clean, Topography, or Composite maps."
        )
        left_layout.addWidget(self.cmb_input_mode)

        # ── Standard mode widgets (Base/Compare) ─────────────────────────────
        self.wgt_standard_select = QtWidgets.QWidget()
        std_layout = QtWidgets.QVBoxLayout(self.wgt_standard_select)
        std_layout.setContentsMargins(0, 4, 0, 0)
        std_layout.setSpacing(4)

        # Base Image label
        lbl_base = QtWidgets.QLabel("Base Image")
        lbl_base.setStyleSheet(f"font-weight: {Typography.FONT_WEIGHT_SEMIBOLD}; font-size: {Typography.FONT_SIZE_SMALL}; color: {UI_TEXT};")
        std_layout.addWidget(lbl_base)

        self.cmb_base = QtWidgets.QComboBox()
        std_layout.addWidget(self.cmb_base)

        self.lbl_base_empty_hint = QtWidgets.QLabel("Select a base image after loading a folder.")
        self.lbl_base_empty_hint.setProperty("secondary", True)
        self.lbl_base_empty_hint.setWordWrap(True)
        std_layout.addWidget(self.lbl_base_empty_hint)

        # Compare images (scrollable checkbox list)
        self.scroll_compare = QtWidgets.QScrollArea()
        self.scroll_compare.setWidgetResizable(True)
        self.scroll_compare.setMaximumHeight(140)
        self.scroll_compare.setStyleSheet("border: none; background: transparent;")
        self.compare_container = QtWidgets.QWidget()
        self.compare_layout = QtWidgets.QVBoxLayout(self.compare_container)
        self.compare_layout.setContentsMargins(2, 2, 2, 2)
        self.compare_layout.setSpacing(2)
        self.scroll_compare.setWidget(self.compare_container)
        std_layout.addWidget(self.scroll_compare)

        # Quick select buttons
        quick_row = QtWidgets.QHBoxLayout()
        self.btn_select_all = QtWidgets.QPushButton("All")
        self.btn_select_all.setFixedWidth(50)
        self.btn_select_all.setFixedHeight(24)
        self.btn_select_none = QtWidgets.QPushButton("None")
        self.btn_select_none.setFixedWidth(50)
        self.btn_select_none.setFixedHeight(24)
        quick_row.addWidget(self.btn_select_all)
        quick_row.addWidget(self.btn_select_none)
        quick_row.addStretch()
        std_layout.addLayout(quick_row)

        # Auto pair checkbox (hidden, controlled by toolbar button)
        self.chk_auto_pair = QtWidgets.QCheckBox("Auto Pair")
        self.chk_auto_pair.setToolTip("Generate all unique pairs from selected images")
        self.chk_auto_pair.setVisible(False)  # controlled by toolbar
        std_layout.addWidget(self.chk_auto_pair)

        # Swap Base button
        self.btn_swap_base = QtWidgets.QPushButton("\u25b6 Swap Base")
        self.btn_swap_base.setFixedHeight(26)
        self.btn_swap_base.setToolTip("Swap base image with compare image selection")
        std_layout.addWidget(self.btn_swap_base)

        left_layout.addWidget(self.wgt_standard_select)

        # ── Quadrant Fusion mode widgets ─────────────────────────────────────
        self.wgt_quadrant_select = QtWidgets.QWidget()
        qf_layout = QtWidgets.QVBoxLayout(self.wgt_quadrant_select)
        qf_layout.setContentsMargins(0, 0, 0, 0)
        qf_layout.setSpacing(4)

        # Five detector dropdowns
        self._qf_combos: Dict[str, QtWidgets.QComboBox] = {}
        for det_name in ("Illuminator", "Top", "Bottom", "Left", "Right"):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(f"{det_name}:"))
            cmb = QtWidgets.QComboBox()
            cmb.setMinimumWidth(140)
            row.addWidget(cmb, 1)
            qf_layout.addLayout(row)
            self._qf_combos[det_name] = cmb

        # Auto-detect button
        self.btn_qf_auto_detect = QtWidgets.QPushButton("Auto-detect by filename")
        self.btn_qf_auto_detect.setToolTip(
            "Match files by keywords: illum/central → Illuminator,\n"
            "top/bottom/left/right → corresponding quadrant."
        )
        qf_layout.addWidget(self.btn_qf_auto_detect)

        # Output type
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(QtWidgets.QLabel("Output:"))
        self.cmb_qf_output = QtWidgets.QComboBox()
        self.cmb_qf_output.addItems(["BSE Enhanced", "Topography", "Composite"])
        out_row.addWidget(self.cmb_qf_output, 1)
        qf_layout.addLayout(out_row)

        # Alpha mode
        alpha_row = QtWidgets.QHBoxLayout()
        alpha_row.addWidget(QtWidgets.QLabel("Alpha:"))
        self.cmb_qf_alpha_mode = QtWidgets.QComboBox()
        self.cmb_qf_alpha_mode.addItems(["Auto", "Manual"])
        self.cmb_qf_alpha_mode.setFixedWidth(100)
        alpha_row.addWidget(self.cmb_qf_alpha_mode)
        self.spn_qf_alpha = QtWidgets.QDoubleSpinBox()
        self.spn_qf_alpha.setRange(0.0, 5.0)
        self.spn_qf_alpha.setSingleStep(0.1)
        self.spn_qf_alpha.setValue(1.0)
        self.spn_qf_alpha.setFixedWidth(76)
        self.spn_qf_alpha.setEnabled(False)
        alpha_row.addWidget(self.spn_qf_alpha)
        alpha_row.addStretch()
        qf_layout.addLayout(alpha_row)

        self.cmb_qf_alpha_mode.currentIndexChanged.connect(
            lambda idx: self.spn_qf_alpha.setEnabled(idx == 1)
        )

        # Beta (composite weight)
        beta_row = QtWidgets.QHBoxLayout()
        beta_row.addWidget(QtWidgets.QLabel("Beta (composite):"))
        self.spn_qf_beta = QtWidgets.QDoubleSpinBox()
        self.spn_qf_beta.setRange(0.0, 1.0)
        self.spn_qf_beta.setSingleStep(0.05)
        self.spn_qf_beta.setValue(0.3)
        self.spn_qf_beta.setFixedWidth(76)
        beta_row.addWidget(self.spn_qf_beta)
        beta_row.addStretch()
        qf_layout.addLayout(beta_row)

        # Gaussian sigma
        sigma_row = QtWidgets.QHBoxLayout()
        sigma_row.addWidget(QtWidgets.QLabel("Topo smoothing:"))
        self.spn_qf_sigma = QtWidgets.QDoubleSpinBox()
        self.spn_qf_sigma.setRange(0.0, 3.0)
        self.spn_qf_sigma.setSingleStep(0.5)
        self.spn_qf_sigma.setValue(0.0)
        self.spn_qf_sigma.setFixedWidth(76)
        self.spn_qf_sigma.setToolTip("Gaussian sigma for topo map smoothing (0 = off)")
        sigma_row.addWidget(self.spn_qf_sigma)
        sigma_row.addStretch()
        qf_layout.addLayout(sigma_row)

        # ROI for alpha fit (reuse existing Pick/Clear ROI buttons)
        qf_roi_row = QtWidgets.QHBoxLayout()
        self.btn_qf_pick_roi = QtWidgets.QPushButton("Pick ROI")
        self.btn_qf_pick_roi.setToolTip("Draw ROI on Base magnifier for alpha auto-fit")
        self.btn_qf_pick_roi.setFixedWidth(96)
        self.btn_qf_clear_roi = QtWidgets.QPushButton("Clear ROI")
        self.btn_qf_clear_roi.setFixedWidth(96)
        self.btn_qf_clear_roi.setEnabled(False)
        qf_roi_row.addWidget(self.btn_qf_pick_roi)
        qf_roi_row.addWidget(self.btn_qf_clear_roi)
        qf_roi_row.addStretch()
        qf_layout.addLayout(qf_roi_row)
        self.lbl_qf_roi_info = QtWidgets.QLabel("ROI: not set")
        self.lbl_qf_roi_info.setStyleSheet(
            f"color: {UI_TEXT_SECONDARY}; font-family: {Typography.FONT_FAMILY_MONO};"
            f" font-size: {Typography.FONT_SIZE_SMALL};"
        )
        qf_layout.addWidget(self.lbl_qf_roi_info)

        self.wgt_quadrant_select.setVisible(False)
        left_layout.addWidget(self.wgt_quadrant_select)

        # Internal state for Quadrant Fusion ROI
        self._qf_roi_rect: Optional[tuple] = None  # (norm_x, norm_y, norm_w, norm_h)
        self._qf_last_result: Optional[QuadrantFusionResult] = None

        # --- OPERATION section ---
        left_layout.addSpacing(8)
        lbl_params = QtWidgets.QLabel("OPERATION")
        lbl_params.setObjectName("SectionTitle")
        left_layout.addWidget(lbl_params)
        sep2 = QtWidgets.QLabel()
        sep2.setObjectName("SectionSeparator")
        left_layout.addWidget(sep2)
        left_layout.addSpacing(4)

        # Operation Settings (Standard mode only)
        self.grp_op = QtWidgets.QWidget()
        op_layout = QtWidgets.QVBoxLayout(self.grp_op)
        op_layout.setContentsMargins(0, 4, 0, 0)
        op_layout.setSpacing(4)

        lbl_op = QtWidgets.QLabel("Operation")
        lbl_op.setStyleSheet(f"font-weight: {Typography.FONT_WEIGHT_SEMIBOLD}; font-size: {Typography.FONT_SIZE_SMALL}; color: {UI_TEXT};")
        op_layout.addWidget(lbl_op)

        # Operation as radio-button-style checkboxes
        self.cmb_operation = QtWidgets.QComboBox()
        self.cmb_operation.addItems(["Subtract (|Base \u2212 Compare|)", "Blend (\u03b1\u00d7Base + \u03b2\u00d7Compare)"])
        # Combobox styled by sidebar stylesheet
        op_layout.addWidget(self.cmb_operation)

        # Blend coefficients (only visible in Blend mode)
        self.grp_blend_coef = QtWidgets.QWidget()
        blend_layout = QtWidgets.QHBoxLayout(self.grp_blend_coef)
        blend_layout.setContentsMargins(0, 0, 0, 0)
        blend_layout.addWidget(QtWidgets.QLabel("α (Base):"))
        self.spin_alpha = QtWidgets.QDoubleSpinBox()
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setSingleStep(0.1)
        self.spin_alpha.setValue(0.5)
        self.spin_alpha.setFixedWidth(74)
        blend_layout.addWidget(self.spin_alpha)
        blend_layout.addWidget(QtWidgets.QLabel("β (Cmp):"))
        self.spin_beta = QtWidgets.QDoubleSpinBox()
        self.spin_beta.setRange(0.0, 1.0)
        self.spin_beta.setSingleStep(0.1)
        self.spin_beta.setValue(0.5)
        self.spin_beta.setFixedWidth(74)
        blend_layout.addWidget(self.spin_beta)
        self.grp_blend_coef.setVisible(False)
        op_layout.addWidget(self.grp_blend_coef)

        # ── Advanced options (collapsed by default) ─────────────────────────
        self.btn_adv_toggle = QtWidgets.QPushButton("Advanced \u25b6")
        self.btn_adv_toggle.setCheckable(True)
        self.btn_adv_toggle.setChecked(False)
        self.btn_adv_toggle.setProperty("variant", "ghost")
        op_layout.addWidget(self.btn_adv_toggle)

        self.grp_advanced = QtWidgets.QWidget()
        adv_layout = QtWidgets.QVBoxLayout(self.grp_advanced)
        adv_layout.setContentsMargins(8, 0, 0, 0)
        adv_layout.setSpacing(4)
        # ── Subtract mode drop-down (Subtract operation only) ─────────────────
        sub_mode_row = QtWidgets.QHBoxLayout()
        sub_mode_row.addWidget(QtWidgets.QLabel("Subtract mode:"))
        self.cmb_subtract_mode = QtWidgets.QComboBox()
        self.cmb_subtract_mode.addItems([
            "|diff| × 2  (default)",  # index 0
            "|diff|  (abs, no gain)",  # index 1
            "clip ≥ 0  (keep direction)",  # index 2
        ])
        self.cmb_subtract_mode.setToolTip(
            "|diff| × 2   : |Base−Compare|, then ×2 to enhance small differences (default)\n"
            "|diff|        : |Base−Compare|, no gain — preserves true magnitude\n"
            "clip ≥ 0      : Base−Compare, keep direction, clamp negatives to 0"
        )
        sub_mode_row.addWidget(self.cmb_subtract_mode, 1)
        adv_layout.addLayout(sub_mode_row)
        invert_row = QtWidgets.QHBoxLayout()
        self.chk_invert_base = QtWidgets.QCheckBox("Inv Base")
        self.chk_invert_base.setToolTip("Apply 255−X to Base before operation")
        self.chk_invert_compare = QtWidgets.QCheckBox("Inv Cmp")
        self.chk_invert_compare.setToolTip("Apply 255−X to Compare before operation")
        self.chk_invert_result = QtWidgets.QCheckBox("Inv Result")
        self.chk_invert_result.setToolTip("Apply 255−X to result after operation")
        invert_row.addWidget(self.chk_invert_base)
        invert_row.addWidget(self.chk_invert_compare)
        invert_row.addWidget(self.chk_invert_result)
        adv_layout.addLayout(invert_row)

        # ── Normalize mode drop-down ──────────────────────────────────────────
        norm_mode_row = QtWidgets.QHBoxLayout()
        norm_mode_row.addWidget(QtWidgets.QLabel("Normalize:"))
        self.cmb_normalize_mode = QtWidgets.QComboBox()
        self.cmb_normalize_mode.addItems([
            "Percentile (P2–P98)",  # index 0 – default
            "GLV-Mask",  # index 1
            "Histogram EQ (HEQ)",  # index 2
            "CLAHE",  # index 3
            "Skip (raw ÷ 255)",  # index 4
            "ROI-Match (EPI Nulling)",  # index 5
        ])
        self.cmb_normalize_mode.setToolTip(
            "Percentile: each image independently mapped to [0,1] via its P2–P98 range.\n"
            "GLV-Mask:   P2/P98 computed only from pixels inside the specified GLV range\n"
            "            (e.g. MG 110–145, EPI 200–255); map applied to full image.\n"
            "HEQ:        Histogram Equalization — non-linear, enhances visual contrast.\n"
            "            NOTE: non-linear; subtraction result loses quantitative meaning.\n"
            "CLAHE:      Contrast-Limited Adaptive HEQ — non-linear, local enhancement.\n"
            "            NOTE: non-linear; subtraction result loses quantitative meaning.\n"
            "Skip:       bypass normalization; divide pixels by 255 directly.\n"
            "ROI-Match:  Calibrate a scale factor from a user-drawn ROI (typically EPI)\n"
            "            so that the EPI region cancels in subtraction, leaving residual\n"
            "            HK/Hf defect signals near inner spacer more visible."
        )
        norm_mode_row.addWidget(self.cmb_normalize_mode, 1)
        adv_layout.addLayout(norm_mode_row)

        # GLV-Mask controls (shown only when GLV-Mask mode is selected)
        self.wgt_glv_controls = QtWidgets.QWidget()
        glv_ctrl_layout = QtWidgets.QHBoxLayout(self.wgt_glv_controls)
        glv_ctrl_layout.setContentsMargins(0, 0, 0, 0)
        glv_ctrl_layout.addWidget(QtWidgets.QLabel("GLV range:"))
        self.spn_glv_low = QtWidgets.QSpinBox()
        self.spn_glv_low.setRange(0, 254)
        self.spn_glv_low.setValue(100)
        self.spn_glv_low.setFixedWidth(68)
        self.spn_glv_low.setToolTip("Lower bound of GLV mask (inclusive, 0–255)")
        glv_ctrl_layout.addWidget(self.spn_glv_low)
        glv_ctrl_layout.addWidget(QtWidgets.QLabel("–"))
        self.spn_glv_high = QtWidgets.QSpinBox()
        self.spn_glv_high.setRange(1, 255)
        self.spn_glv_high.setValue(160)
        self.spn_glv_high.setFixedWidth(68)
        self.spn_glv_high.setToolTip("Upper bound of GLV mask (inclusive, 0–255)")
        glv_ctrl_layout.addWidget(self.spn_glv_high)
        self.btn_preview_glv_mask = QtWidgets.QPushButton("Preview Mask")
        self.btn_preview_glv_mask.setToolTip(
            "Show a preview window highlighting which pixels of the Base image\n"
            "fall within the GLV range and will be used for normalization."
        )
        self.btn_preview_glv_mask.clicked.connect(self._on_preview_glv_mask)
        glv_ctrl_layout.addWidget(self.btn_preview_glv_mask)
        glv_ctrl_layout.addStretch()
        adv_layout.addWidget(self.wgt_glv_controls)

        # CLAHE controls (shown only when CLAHE mode is selected)
        self.wgt_clahe_controls = QtWidgets.QWidget()
        clahe_ctrl_layout = QtWidgets.QHBoxLayout(self.wgt_clahe_controls)
        clahe_ctrl_layout.setContentsMargins(0, 0, 0, 0)
        clahe_ctrl_layout.addWidget(QtWidgets.QLabel("Clip Limit:"))
        self.spn_clahe_clip = QtWidgets.QDoubleSpinBox()
        self.spn_clahe_clip.setRange(0.5, 10.0)
        self.spn_clahe_clip.setSingleStep(0.5)
        self.spn_clahe_clip.setValue(2.0)
        self.spn_clahe_clip.setFixedWidth(78)
        self.spn_clahe_clip.setToolTip(
            "CLAHE contrast clip limit (higher = stronger local contrast).\n"
            "Typical values: 1.0–4.0. Default: 2.0."
        )
        clahe_ctrl_layout.addWidget(self.spn_clahe_clip)
        clahe_ctrl_layout.addStretch()
        adv_layout.addWidget(self.wgt_clahe_controls)

        # ROI-Match controls (shown only when ROI-Match mode is selected)
        self.wgt_roi_match_controls = QtWidgets.QWidget()
        roi_match_layout = QtWidgets.QVBoxLayout(self.wgt_roi_match_controls)
        roi_match_layout.setContentsMargins(0, 0, 0, 0)
        roi_match_layout.setSpacing(4)

        roi_btn_row = QtWidgets.QHBoxLayout()
        self.btn_pick_roi = QtWidgets.QPushButton("Pick ROI")
        self.btn_pick_roi.setToolTip(
            "Draw a rectangular ROI on the Base image (Comparison View).\n"
            "Choose a uniform EPI-dominated region for best results."
        )
        self.btn_pick_roi.setFixedWidth(96)
        roi_btn_row.addWidget(self.btn_pick_roi)
        self.btn_clear_roi = QtWidgets.QPushButton("Clear ROI")
        self.btn_clear_roi.setToolTip("Remove the current ROI selection")
        self.btn_clear_roi.setFixedWidth(96)
        self.btn_clear_roi.setEnabled(False)
        roi_btn_row.addWidget(self.btn_clear_roi)
        roi_btn_row.addStretch()
        roi_match_layout.addLayout(roi_btn_row)

        self.lbl_roi_info = QtWidgets.QLabel("ROI: not set")
        self.lbl_roi_info.setStyleSheet(
            f"color: {UI_TEXT_SECONDARY}; font-family: {Typography.FONT_FAMILY_MONO};"
            f" font-size: {Typography.FONT_SIZE_SMALL};"
        )
        roi_match_layout.addWidget(self.lbl_roi_info)

        self.lbl_roi_alpha = QtWidgets.QLabel("")
        self.lbl_roi_alpha.setStyleSheet(
            f"color: {UI_SUCCESS}; font-family: {Typography.FONT_FAMILY_MONO};"
            f" font-size: {Typography.FONT_SIZE_SMALL}; font-weight: {Typography.FONT_WEIGHT_SEMIBOLD};"
        )
        roi_match_layout.addWidget(self.lbl_roi_alpha)

        self.chk_roi_abs_diff = QtWidgets.QCheckBox("Use |diff| instead of keep-direction")
        self.chk_roi_abs_diff.setChecked(False)
        self.chk_roi_abs_diff.setToolTip(
            "Default (unchecked): Enhanced = clip(Base - α·Compare, 0) — highlights\n"
            "where Base is brighter than scaled Compare (defect path).\n"
            "Checked: Enhanced = |Base - α·Compare| — symmetric edges, legacy behavior."
        )
        roi_match_layout.addWidget(self.chk_roi_abs_diff)

        adv_layout.addWidget(self.wgt_roi_match_controls)

        # Internal state for ROI-Match
        self._roi_match_rect: Optional[tuple] = None  # (norm_x, norm_y, norm_w, norm_h)

        self.cmb_normalize_mode.currentIndexChanged.connect(self._on_normalize_mode_changed)
        self._on_normalize_mode_changed()  # set initial visibility

        self.grp_advanced.setVisible(False)
        op_layout.addWidget(self.grp_advanced)

        left_layout.addWidget(self.grp_op)

        # --- ALIGNMENT section ---
        left_layout.addSpacing(10)
        lbl_align_section = QtWidgets.QLabel("ALIGNMENT")
        lbl_align_section.setObjectName("SectionTitle")
        left_layout.addWidget(lbl_align_section)
        sep_align = QtWidgets.QLabel()
        sep_align.setObjectName("SectionSeparator")
        left_layout.addWidget(sep_align)
        left_layout.addSpacing(6)

        self.grp_align = QtWidgets.QWidget()
        align_layout = QtWidgets.QVBoxLayout(self.grp_align)
        align_layout.setContentsMargins(0, 4, 0, 0)
        align_layout.setSpacing(4)

        align_method_row = QtWidgets.QHBoxLayout()
        align_method_row.addWidget(QtWidgets.QLabel("Align:"))
        self.cmb_align_method = QtWidgets.QComboBox()
        self.cmb_align_method.addItems(["Phase (robust)", "NCC (brute force)"])
        align_method_row.addWidget(self.cmb_align_method, 1)
        align_layout.addLayout(align_method_row)

        snr_win_row = QtWidgets.QHBoxLayout()
        snr_win_row.addWidget(QtWidgets.QLabel("SNR Win:"))
        self.spn_snr_window = QtWidgets.QSpinBox()
        self.spn_snr_window.setRange(7, 127)
        self.spn_snr_window.setSingleStep(2)
        self.spn_snr_window.setValue(31)
        self.spn_snr_window.setFixedWidth(60)
        self.spn_snr_window.setToolTip(
            "Box-filter window size for Z-Map SNR calculation (odd, \u22657).\n"
            "Larger values \u2192 smoother map.\n"
            "Recommended: 15 for ~512 px, 31 for ~1000 px, 63 for >2000 px."
        )
        snr_win_row.addWidget(self.spn_snr_window)
        snr_win_row.addStretch()
        align_layout.addLayout(snr_win_row)
        left_layout.addWidget(self.grp_align)

        left_layout.addStretch()
        content_layout.addWidget(left_panel)

        # ================================================================
        # RIGHT PANEL: Viewer + Controls
        # ================================================================
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # ── LEFT SECTION (input preview): control bar + stacked viewer ───────
        _viewer_ctrl_style = f"""
            QFrame#ViewerControlBar {{
                background-color: {UI_BG_PANEL};
                border: 1px solid {UI_BORDER};
                border-radius: {BorderRadius.SM};
            }}
        """

        left_section = QtWidgets.QVBoxLayout()
        left_section.setSpacing(4)

        # Left control bar: Base | Compare  +  Split View toggle + slider
        left_ctrl_bar = QtWidgets.QFrame()
        left_ctrl_bar.setObjectName("ViewerControlBar")
        left_ctrl_bar.setStyleSheet(_viewer_ctrl_style)
        left_ctrl_layout = QtWidgets.QHBoxLayout(left_ctrl_bar)
        left_ctrl_layout.setContentsMargins(8, 4, 8, 4)
        left_ctrl_layout.setSpacing(0)

        self.btn_mode_base = QtWidgets.QPushButton("Base")
        self.btn_mode_base.setCheckable(True)
        self.btn_mode_base.setProperty("viewerMode", True)
        self.btn_mode_base.setObjectName("ViewerModeFirst")
        self.btn_mode_compare = QtWidgets.QPushButton("Compare")
        self.btn_mode_compare.setCheckable(True)
        self.btn_mode_compare.setProperty("viewerMode", True)
        self.btn_mode_compare.setObjectName("ViewerModeLast")

        for btn in (self.btn_mode_base, self.btn_mode_compare):
            btn.setFixedHeight(30)
            left_ctrl_layout.addWidget(btn)

        left_ctrl_layout.addSpacing(12)

        # Split View toggle placed in left control bar (only affects left viewer)
        self.btn_split_view = QtWidgets.QPushButton("Split View")
        self.btn_split_view.setObjectName("ViewerToolToggle")
        self.btn_split_view.setCheckable(True)
        self.btn_split_view.setChecked(False)
        self.btn_split_view.setToolTip(
            "OFF: Magnifier mode \u2014 synchronized with Difference Map\n"
            "ON:  Split-view Base vs Aligned Compare (drag divider or use slider)"
        )
        left_ctrl_layout.addWidget(self.btn_split_view)

        # Slider for split view divider
        self.blend_slider_widget = QtWidgets.QWidget()
        blend_slider_layout = QtWidgets.QHBoxLayout(self.blend_slider_widget)
        blend_slider_layout.setContentsMargins(4, 0, 0, 0)
        self.slider_blend = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_blend.setRange(0, 100)
        self.slider_blend.setValue(50)
        blend_slider_layout.addWidget(self.slider_blend, 1)
        self.lbl_blend_value = QtWidgets.QLabel("50%")
        self.lbl_blend_value.setStyleSheet(
            f"font-family: {Typography.FONT_FAMILY_MONO}; font-size: {Typography.FONT_SIZE_SMALL};")
        blend_slider_layout.addWidget(self.lbl_blend_value)
        self.blend_slider_widget.setVisible(False)
        left_ctrl_layout.addWidget(self.blend_slider_widget, 1)

        left_ctrl_layout.addStretch(1)

        self.chk_sync_zoom = QtWidgets.QCheckBox("Sync Zoom")
        self.chk_sync_zoom.setChecked(True)
        self.chk_sync_zoom.setToolTip("Synchronize zoom between left and right viewers")
        left_ctrl_layout.addWidget(self.chk_sync_zoom)

        left_section.addWidget(left_ctrl_bar)

        # Left stacked viewer: magnifier (page 0) + split view (page 1)
        self.stk_blend = QtWidgets.QStackedWidget()
        self.stk_blend.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.img_base_mag = SyncZoomImageWidget("Base")
        self.img_base_mag.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.stk_blend.addWidget(self.img_base_mag)  # index 0 magnifier

        self.img_blend = SplitViewWidget()
        self.img_blend.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.stk_blend.addWidget(self.img_blend)  # index 1 split view
        self.stk_blend.setCurrentIndex(0)
        left_section.addWidget(self.stk_blend, stretch=1)

        # ── RIGHT SECTION (result view): control bar + difference map ─────────
        right_section = QtWidgets.QVBoxLayout()
        right_section.setSpacing(4)

        # Right control bar: Diff | Z-Map  +  Range  Normalize  Colormap
        right_ctrl_bar = QtWidgets.QFrame()
        right_ctrl_bar.setObjectName("ViewerControlBar")
        right_ctrl_bar.setStyleSheet(_viewer_ctrl_style)
        right_ctrl_layout = QtWidgets.QHBoxLayout(right_ctrl_bar)
        right_ctrl_layout.setContentsMargins(8, 4, 8, 4)
        right_ctrl_layout.setSpacing(0)

        self.btn_mode_diff = QtWidgets.QPushButton("Diff")
        self.btn_mode_diff.setCheckable(True)
        self.btn_mode_diff.setChecked(True)
        self.btn_mode_diff.setProperty("viewerMode", True)
        self.btn_mode_diff.setObjectName("ViewerModeFirst")
        self.btn_mode_topo = QtWidgets.QPushButton("Z-Map")
        self.btn_mode_topo.setCheckable(True)
        self.btn_mode_topo.setProperty("viewerMode", True)
        self.btn_mode_topo.setObjectName("ViewerModeLast")
        # Backward-compat alias
        self.btn_mode_zmap = self.btn_mode_topo

        for btn in (self.btn_mode_diff, self.btn_mode_topo):
            btn.setFixedHeight(30)
            right_ctrl_layout.addWidget(btn)

        right_ctrl_layout.addSpacing(10)

        _sep_v = QtWidgets.QFrame()
        _sep_v.setFrameShape(QtWidgets.QFrame.VLine)
        _sep_v.setStyleSheet(f"color: {UI_BORDER}; max-width: 1px; margin: 2px 8px;")
        right_ctrl_layout.addWidget(_sep_v)

        lbl_display_group = QtWidgets.QLabel("DISPLAY")
        lbl_display_group.setProperty("toolbarGroupLabel", True)
        right_ctrl_layout.addWidget(lbl_display_group)
        right_ctrl_layout.addSpacing(8)

        self.btn_show_norm_compare = QtWidgets.QPushButton("Normalize")
        self.btn_show_norm_compare.setFixedWidth(100)
        self.btn_show_norm_compare.setToolTip("Preview normalization effect")
        right_ctrl_layout.addWidget(self.btn_show_norm_compare)

        right_ctrl_layout.addSpacing(8)
        lbl_colormap = QtWidgets.QLabel("Colormap")
        lbl_colormap.setProperty("toolbarLabel", True)
        right_ctrl_layout.addWidget(lbl_colormap)
        right_ctrl_layout.addSpacing(4)
        self.cmb_colormap = QtWidgets.QComboBox()
        self.cmb_colormap.addItems(["Grayscale", "JET", "Hot", "Inferno", "Viridis"])
        self.cmb_colormap.setFixedWidth(104)
        right_ctrl_layout.addWidget(self.cmb_colormap)

        right_ctrl_layout.addSpacing(8)
        lbl_range = QtWidgets.QLabel("Range")
        lbl_range.setProperty("toolbarLabel", True)
        right_ctrl_layout.addWidget(lbl_range)
        right_ctrl_layout.addSpacing(4)
        self.cmb_range = QtWidgets.QComboBox()
        self.cmb_range.addItems(["Auto", "Zero-centered", "P1-P99", "P0.5-P99.5"])
        self.cmb_range.setFixedWidth(128)
        self.cmb_range.setToolTip("Control how difference values are scaled for display")
        right_ctrl_layout.addWidget(self.cmb_range)

        right_ctrl_layout.addStretch(1)

        right_section.addWidget(right_ctrl_bar)

        # Right: Difference Map
        self.img_diff = SyncZoomImageWidget("Difference Map")
        self.img_diff.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        right_section.addWidget(self.img_diff, stretch=1)

        # ── IMAGE AREA: left section + right section side by side ─────────────
        image_row = QtWidgets.QHBoxLayout()
        image_row.setSpacing(8)
        image_row.addLayout(left_section, stretch=1)
        image_row.addLayout(right_section, stretch=1)
        right_layout.addLayout(image_row, stretch=4)

        # Hidden backward-compat widgets (not shown)
        self.lbl_blend_info = QtWidgets.QLabel("")
        self.lbl_blend_info.setVisible(False)
        self.btn_mode_blend = QtWidgets.QPushButton("Blend")
        self.btn_mode_blend.setCheckable(True)
        self.btn_mode_blend.setVisible(False)

        right_layout.addSpacing(4)

        # === BOTTOM: Analysis (merged) | Histogram (under Difference Map) ===
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(8)

        # Card 1: Analysis — combined alignment + difference metrics
        self.stats_widget = StatisticsWidget()
        bottom_row.addWidget(self.stats_widget, 2)

        # Card 2: Histogram (wider, positioned under the Difference Map)
        hist_card = QtWidgets.QFrame()
        hist_card.setObjectName("BottomCard")
        hist_layout = QtWidgets.QVBoxLayout(hist_card)
        hist_layout.setContentsMargins(12, 10, 12, 8)
        hist_layout.setSpacing(3)
        lbl_hist_title = QtWidgets.QLabel("Histogram")
        lbl_hist_title.setStyleSheet(
            f"font-weight: {Typography.FONT_WEIGHT_BOLD}; font-size: {Typography.FONT_SIZE_BODY};"
            f" color: {UI_TEXT}; border: none; background: transparent;"
        )
        hist_layout.addWidget(lbl_hist_title)
        hist_hint = QtWidgets.QLabel("Click once to set low, click again to set high")
        hist_hint.setStyleSheet(
            f"color: {UI_TEXT_MUTED}; font-size: {Typography.FONT_SIZE_CAPTION};"
            f" border: none; background: transparent;"
        )
        hist_layout.addWidget(hist_hint)
        self.histogram_canvas = HistogramCanvas()
        self.histogram_canvas.setFixedHeight(186)
        hist_layout.addWidget(self.histogram_canvas)
        hist_ctrl_row = QtWidgets.QHBoxLayout()
        self.lbl_hist_range = QtWidgets.QLabel("Range: \u2014")
        self.lbl_hist_range.setStyleSheet(
            f"color: {UI_TEXT_SECONDARY}; font-family: {Typography.FONT_FAMILY_MONO};"
            f" font-size: {Typography.FONT_SIZE_SMALL}; border: none;"
        )
        hist_ctrl_row.addWidget(self.lbl_hist_range, 1)
        self.btn_clear_hist_range = QtWidgets.QPushButton("Clear Range")
        self.btn_clear_hist_range.setFixedWidth(96)
        self.btn_clear_hist_range.setFixedHeight(24)
        self.btn_clear_hist_range.setEnabled(False)
        hist_ctrl_row.addWidget(self.btn_clear_hist_range)
        hist_layout.addLayout(hist_ctrl_row)
        bottom_row.addWidget(hist_card, 3)

        # Hidden AlignmentScoreWidget — kept for backward-compat with update logic
        self.align_score_widget = AlignmentScoreWidget()
        self.align_score_widget.setVisible(False)

        right_layout.addLayout(bottom_row)

        # === QUADRANT FUSION RIGHT PANEL (Page 1) ===
        qf_right_panel = QtWidgets.QWidget()
        qf_right_layout = QtWidgets.QVBoxLayout(qf_right_panel)
        qf_right_layout.setContentsMargins(0, 0, 0, 0)
        qf_right_layout.setSpacing(8)

        # Top: output type toggle buttons
        qf_top_toolbar = QtWidgets.QFrame()
        qf_top_toolbar.setObjectName("TopToolbar")
        qf_top_row = QtWidgets.QHBoxLayout(qf_top_toolbar)
        qf_top_row.setContentsMargins(10, 8, 10, 8)
        qf_top_row.setSpacing(8)
        lbl_output_view = QtWidgets.QLabel("Output View")
        lbl_output_view.setProperty("toolbarLabel", True)
        qf_top_row.addWidget(lbl_output_view)
        self.btn_qf_show_bse = QtWidgets.QPushButton("BSE Enhanced")
        self.btn_qf_show_bse.setCheckable(True)
        self.btn_qf_show_bse.setChecked(True)
        self.btn_qf_show_bse.setFixedWidth(128)
        self.btn_qf_show_bse.setProperty("toolbarToggle", True)

        self.btn_qf_show_topo = QtWidgets.QPushButton("Topography")
        self.btn_qf_show_topo.setCheckable(True)
        self.btn_qf_show_topo.setFixedWidth(128)
        self.btn_qf_show_topo.setProperty("toolbarToggle", True)

        self.btn_qf_show_comp = QtWidgets.QPushButton("Composite")
        self.btn_qf_show_comp.setCheckable(True)
        self.btn_qf_show_comp.setFixedWidth(128)
        self.btn_qf_show_comp.setProperty("toolbarToggle", True)

        qf_top_row.addWidget(self.btn_qf_show_bse)
        qf_top_row.addWidget(self.btn_qf_show_topo)
        qf_top_row.addWidget(self.btn_qf_show_comp)

        qf_top_row.addSpacing(20)
        lbl_qf_colormap = QtWidgets.QLabel("Colormap")
        lbl_qf_colormap.setProperty("toolbarLabel", True)
        qf_top_row.addWidget(lbl_qf_colormap)
        self.cmb_qf_colormap = QtWidgets.QComboBox()
        self.cmb_qf_colormap.addItems(["Grayscale", "JET", "Hot", "Inferno", "Viridis"])
        self.cmb_qf_colormap.setFixedWidth(106)
        qf_top_row.addWidget(self.cmb_qf_colormap)
        qf_top_row.addStretch()
        qf_right_layout.addWidget(qf_top_toolbar)

        # 3-column image viewers: Illuminator | Main Output | Topo Reference
        qf_image_row = QtWidgets.QHBoxLayout()

        # Left: Illuminator (original reference)
        qf_illum_grp = QtWidgets.QGroupBox("Illuminator (Original)")
        qf_illum_lay = QtWidgets.QVBoxLayout(qf_illum_grp)
        qf_illum_lay.setContentsMargins(6, 12, 6, 6)
        self.img_qf_illum = SyncZoomImageWidget("Illuminator")
        self.img_qf_illum.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        qf_illum_lay.addWidget(self.img_qf_illum)
        qf_image_row.addWidget(qf_illum_grp, 1)

        # Center: Main output (BSE / Topo / Composite, switched by toggle)
        self.qf_main_grp = QtWidgets.QGroupBox("BSE Enhanced (Illum − α·Topo)")
        qf_main_lay = QtWidgets.QVBoxLayout(self.qf_main_grp)
        qf_main_lay.setContentsMargins(6, 12, 6, 6)
        self.img_qf_main = SyncZoomImageWidget("Fusion Output")
        self.img_qf_main.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        qf_main_lay.addWidget(self.img_qf_main)
        qf_image_row.addWidget(self.qf_main_grp, 1)

        # Right: Secondary reference (Topo when main=BSE, BSE when main=Topo, etc.)
        self.qf_ref_grp = QtWidgets.QGroupBox("Topography")
        qf_ref_lay = QtWidgets.QVBoxLayout(self.qf_ref_grp)
        qf_ref_lay.setContentsMargins(6, 12, 6, 6)
        self.img_qf_ref = SyncZoomImageWidget("Reference")
        self.img_qf_ref.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        qf_ref_lay.addWidget(self.img_qf_ref)
        qf_image_row.addWidget(self.qf_ref_grp, 1)

        qf_right_layout.addLayout(qf_image_row, stretch=4)

        # Link magnifiers for synchronized zoom
        self.img_qf_illum.linkPartner(self.img_qf_main)
        self.img_qf_main.linkPartner(self.img_qf_illum)

        # Sync cursors between QF viewers
        self.img_qf_illum.cursor_moved.connect(
            lambda nx, ny: (self.img_qf_main.setCursorPos(nx, ny),
                            self.img_qf_ref.setCursorPos(nx, ny))
        )
        self.img_qf_illum.cursor_left.connect(
            lambda: (self.img_qf_main.clearCursor(), self.img_qf_ref.clearCursor())
        )
        self.img_qf_main.cursor_moved.connect(
            lambda nx, ny: (self.img_qf_illum.setCursorPos(nx, ny),
                            self.img_qf_ref.setCursorPos(nx, ny))
        )
        self.img_qf_main.cursor_left.connect(
            lambda: (self.img_qf_illum.clearCursor(), self.img_qf_ref.clearCursor())
        )
        self.img_qf_ref.cursor_moved.connect(
            lambda nx, ny: (self.img_qf_illum.setCursorPos(nx, ny),
                            self.img_qf_main.setCursorPos(nx, ny))
        )
        self.img_qf_ref.cursor_left.connect(
            lambda: (self.img_qf_illum.clearCursor(), self.img_qf_main.clearCursor())
        )

        # QF Fusion Info bar
        self.lbl_qf_result_info = QtWidgets.QLabel("No fusion result yet")
        self.lbl_qf_result_info.setAlignment(Qt.AlignCenter)
        self.lbl_qf_result_info.setStyleSheet(f"""
            QLabel {{
                color: {UI_TEXT};
                background-color: {UI_BG_CARD};
                border: 1px solid {UI_BORDER};
                border-radius: {BorderRadius.LG};
                padding: {Spacing.BUTTON_PADDING};
                font-size: {Typography.FONT_SIZE_BODY};
            }}
        """)
        qf_right_layout.addWidget(self.lbl_qf_result_info)

        # Bottom row: Histogram + Fusion Stats
        qf_bottom_row = QtWidgets.QHBoxLayout()

        # Histogram
        qf_hist_grp = QtWidgets.QGroupBox("Fusion Histogram")
        qf_hist_lay = QtWidgets.QVBoxLayout(qf_hist_grp)
        qf_hist_lay.setSpacing(4)
        self.qf_histogram_canvas = HistogramCanvas()
        self.qf_histogram_canvas.setFixedHeight(160)
        qf_hist_lay.addWidget(self.qf_histogram_canvas)
        qf_bottom_row.addWidget(qf_hist_grp, 1)

        # Fusion Stats
        qf_stats_grp = QtWidgets.QGroupBox("Fusion Parameters")
        qf_stats_grp.setStyleSheet(f"""
            QGroupBox {{
                background-color: {UI_BG_CARD};
                border: 1px solid {UI_BORDER};
                border-radius: {BorderRadius.MD};
            }}
        """)
        qf_stats_lay = QtWidgets.QGridLayout(qf_stats_grp)
        qf_stats_lay.setContentsMargins(12, 16, 12, 12)
        qf_stats_lay.setSpacing(6)

        _mono_style = (
            f"font-family: {Typography.FONT_FAMILY_MONO};"
            f" font-size: {Typography.FONT_SIZE_SMALL};"
            f" color: {UI_TEXT}; border: none;"
        )
        _label_style = (
            f"color: {UI_TEXT_SECONDARY}; font-size: {Typography.FONT_SIZE_CAPTION}; border: none;"
        )

        self._qf_stat_labels: Dict[str, QtWidgets.QLabel] = {}
        for i, (key, label_text) in enumerate([
            ("alpha", "Alpha (α):"),
            ("beta", "Beta (β):"),
            ("sigma", "Topo Sigma:"),
            ("output", "Output Type:"),
            ("mean", "Mean:"),
            ("std", "Std Dev:"),
            ("dim", "Dimensions:"),
        ]):
            lbl = QtWidgets.QLabel(label_text)
            lbl.setStyleSheet(_label_style)
            val = QtWidgets.QLabel("--")
            val.setStyleSheet(_mono_style)
            qf_stats_lay.addWidget(lbl, i, 0)
            qf_stats_lay.addWidget(val, i, 1)
            self._qf_stat_labels[key] = val

        qf_bottom_row.addWidget(qf_stats_grp, 1)
        qf_right_layout.addLayout(qf_bottom_row)

        # === RIGHT PANEL STACKED WIDGET ===
        self.stk_right_panel = QtWidgets.QStackedWidget()
        self.stk_right_panel.addWidget(right_panel)  # index 0 = Standard
        self.stk_right_panel.addWidget(qf_right_panel)  # index 1 = Quadrant Fusion
        self.stk_right_panel.setCurrentIndex(0)

        content_layout.addWidget(self.stk_right_panel, stretch=1)
        outer_layout.addLayout(content_layout, stretch=1)

    def _connect_signals(self):
        """Connect UI signals."""
        self.btn_load_folder.clicked.connect(self._on_load_image_folder)
        self.btn_compute.clicked.connect(self._on_compute)
        self.btn_export.clicked.connect(self._on_export)
        self.btn_select_all.clicked.connect(self._select_all_compare)
        self.btn_select_none.clicked.connect(self._select_none_compare)
        self.cmb_base.currentIndexChanged.connect(self._on_base_changed)
        self.chk_auto_pair.stateChanged.connect(self._on_auto_pair_toggle)
        self.btn_adv_toggle.toggled.connect(self._on_adv_toggle)
        self.slider_blend.valueChanged.connect(self._on_blend_change)
        self.histogram_canvas.range_changed.connect(self._on_hist_range_changed)
        self.btn_clear_hist_range.clicked.connect(self._on_clear_hist_range)

        # Toolbar Auto Pair button -> sync with hidden checkbox
        self.btn_auto_pair_toolbar.toggled.connect(self.chk_auto_pair.setChecked)
        self.btn_auto_pair_toolbar.toggled.connect(self._on_auto_pair_toggle)

        # About button
        self.btn_about.clicked.connect(self._show_about_dialog)

        # Input Mode selector
        self.cmb_input_mode.currentIndexChanged.connect(self._on_input_mode_changed)

        # Quadrant Fusion ROI, auto-detect, and view toggles
        self.btn_qf_auto_detect.clicked.connect(self._on_qf_auto_detect)
        self.btn_qf_pick_roi.clicked.connect(self._on_qf_pick_roi)
        self.btn_qf_clear_roi.clicked.connect(self._on_qf_clear_roi)
        self.img_base_mag.roi_selected.connect(self._on_roi_selected)
        self.img_qf_illum.roi_selected.connect(self._on_roi_selected)
        self.btn_qf_show_bse.clicked.connect(lambda: self._on_qf_view_toggle("bse_clean"))
        self.btn_qf_show_topo.clicked.connect(lambda: self._on_qf_view_toggle("topo"))
        self.btn_qf_show_comp.clicked.connect(lambda: self._on_qf_view_toggle("composite"))
        self.cmb_qf_colormap.currentIndexChanged.connect(lambda _: self._refresh_qf_display())

        # Operation / navigation
        self.cmb_operation.currentIndexChanged.connect(self._on_operation_changed)
        self.btn_prev_result.clicked.connect(self._on_prev_result)
        self.btn_next_result.clicked.connect(self._on_next_result)

        # Segmented display mode buttons
        self.btn_mode_base.clicked.connect(lambda: self._on_display_mode('base'))
        self.btn_mode_compare.clicked.connect(lambda: self._on_display_mode('compare'))
        self.btn_mode_diff.clicked.connect(lambda: self._on_display_mode('diff'))
        self.btn_mode_topo.clicked.connect(lambda: self._on_display_mode('zmap'))

        # Dynamic range control
        self.cmb_range.currentIndexChanged.connect(self._on_range_changed)
        self.btn_show_norm_compare.clicked.connect(self._on_show_normalized_compare)

        # Colormap selector
        self.cmb_colormap.currentIndexChanged.connect(lambda _: self._refresh_diff_display())

        # Swap Base button
        self.btn_swap_base.clicked.connect(self._on_swap_base)

        # ROI-Match: pick / clear
        self.btn_pick_roi.clicked.connect(self._on_pick_roi)
        self.btn_clear_roi.clicked.connect(self._on_clear_roi)

        # Split View toggle + bidirectional cursor sync
        self.btn_split_view.toggled.connect(self._on_split_view_toggle)
        # Diff map → left magnifier
        self.img_diff.cursor_moved.connect(self._on_diff_cursor_moved)
        self.img_diff.cursor_left.connect(self._on_diff_cursor_left)
        # Left magnifier → diff map (bidirectional)
        self.img_base_mag.cursor_moved.connect(self._on_base_mag_cursor_moved)
        self.img_base_mag.cursor_left.connect(self._on_base_mag_cursor_left)

        self.img_base_mag.clicked.connect(self._on_left_viewer_clicked)
        self.img_blend.clicked.connect(self._on_split_viewer_clicked)
        self.img_diff.clicked.connect(self._on_right_viewer_clicked)

        self._update_sidebar_empty_state(False)
        self._sync_viewer_mode_buttons()

    def _on_load_image_folder(self):
        """Select an image folder and load all supported images."""
        import os

        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder for Perspective Combination",
            str(Path.home())
        )
        if not folder:
            return

        extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
        image_files = [
            os.path.join(folder, fn)
            for fn in sorted(os.listdir(folder))
            if fn.lower().endswith(extensions)
        ]

        if not image_files:
            QtWidgets.QMessageBox.warning(
                self,
                "No Images Found",
                f"No image files found in the selected folder.\n\nSupported formats: {', '.join(extensions)}"
            )
            return

        if len(image_files) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "Not Enough Images",
                "Perspective Combination requires at least 2 images.\n"
                f"Found only {len(image_files)} image(s)."
            )
            return

        self._conditions = [
            EbeamCondition(
                image_path=img_path,
                kev=0.0,
                current_value=0.0,
                current_unit="nA",
                label=os.path.basename(img_path),
                defect_id="",
            )
            for img_path in image_files
        ]
        self._load_images()

    def _on_auto_pair_toggle(self, state):
        """Enable/disable base selection when auto pairing."""
        auto_pair = bool(state)
        self.cmb_base.setEnabled(not auto_pair)
        if auto_pair:
            for chk in self._compare_checkboxes:
                chk.setEnabled(True)
            # Auto pair should evaluate all directional pairs by default.
            self._select_all_compare()
        else:
            self._on_base_changed()

    # ── Input Mode switching ─────────────────────────────────────────────

    def _on_input_mode_changed(self, index: int = None):
        """Toggle between Standard (0) and Quadrant Fusion (1) UI panels."""
        is_qf = self.cmb_input_mode.currentIndex() == 1
        # Left panel: image selection
        self.wgt_standard_select.setVisible(not is_qf)
        self.wgt_quadrant_select.setVisible(is_qf)
        # Left panel: Standard-only groups
        self.grp_op.setVisible(not is_qf)
        self.grp_align.setVisible(not is_qf)
        # Keep using the existing embedded pages; never create/detach viewers here.
        self.stk_right_panel.setCurrentIndex(1 if is_qf else 0)

        if is_qf:
            self.setWindowTitle("Fusi³ — Quadrant Fusion")
        else:
            self.setWindowTitle("Fusi³ — SEM Perspective Combination Tool")
            self._refresh_standard_mode_views()

    def _refresh_standard_mode_views(self):
        """Refresh existing embedded Standard-mode viewers after mode switching."""
        if self._results:
            self._update_current_result()
        else:
            self._show_standard_empty_state()

    def _show_standard_empty_state(self):
        """Show an embedded empty state for Standard mode without extra windows."""
        self._display_mode = 'diff'
        self._left_non_split_mode = 'base'
        self._left_view_mode = 'base'
        if self.btn_split_view.isChecked():
            self.btn_split_view.setChecked(False)
        self.img_base_mag.setImage(None)
        self.img_blend.set_images(None, None)
        self.img_diff.setImage(None)
        self.histogram_canvas.plot_histogram(None, None)
        self._hist_range = None
        self.lbl_hist_range.setText("Range: —")
        self.btn_clear_hist_range.setEnabled(False)
        self.stats_widget.reset()
        self._sync_viewer_mode_buttons()

    def _on_qf_auto_detect(self):
        """Auto-detect Quadrant Fusion detector assignments from filenames."""
        file_labels = list(self._images.keys())
        if not file_labels:
            return
        patterns = {
            "Illuminator": ("illum", "central"),
            "Top": ("top",),
            "Bottom": ("bottom",),
            "Left": ("left",),
            "Right": ("right",),
        }
        for det_name, keywords in patterns.items():
            cmb = self._qf_combos[det_name]
            for i in range(cmb.count()):
                text_lower = cmb.itemText(i).lower()
                if any(kw in text_lower for kw in keywords):
                    cmb.setCurrentIndex(i)
                    break

    def _on_qf_pick_roi(self):
        """Enter ROI drawing mode for Quadrant Fusion alpha fit."""
        if not self._images:
            QtWidgets.QMessageBox.information(
                self, "Quadrant Fusion", "Please load images first."
            )
            return
        # Show Illuminator in QF left viewer and enable ROI drawing on it
        illum_label = self._qf_combos["Illuminator"].currentText()
        if illum_label and illum_label in self._images:
            self.img_qf_illum.setImage(self._images[illum_label])
        self.img_qf_illum.set_roi_mode(True)
        self.lbl_qf_roi_info.setText("ROI: draw on Illuminator image...")

    def _on_qf_clear_roi(self):
        """Clear the Quadrant Fusion ROI."""
        self._qf_roi_rect = None
        self.img_qf_illum.set_active_roi(None)
        self.img_qf_main.set_active_roi(None)
        self.btn_qf_clear_roi.setEnabled(False)
        self.lbl_qf_roi_info.setText("ROI: not set")

    def _on_blend_change(self, value):
        self.lbl_blend_value.setText(f"{value}%")
        self._update_blend_preview()

    def _on_split_view_toggle(self, checked: bool):
        """Switch left panel between magnifier (off) and split-view (on)."""
        if checked:
            self._left_view_mode = 'split'
            self.stk_blend.setCurrentIndex(1)  # show SplitViewWidget
            self.blend_slider_widget.setVisible(True)
        else:
            self._left_view_mode = self._left_non_split_mode
            self.stk_blend.setCurrentIndex(0)  # show magnifier
            self.blend_slider_widget.setVisible(False)
            self._refresh_base_compare_display(self._left_non_split_mode)
        self._sync_viewer_mode_buttons()
        self._update_blend_preview()

    def _on_diff_cursor_moved(self, norm_x: float, norm_y: float):
        """Relay Difference Map cursor position to left-panel magnifier."""
        if not self.btn_split_view.isChecked():
            self.img_base_mag.setCursorPos(norm_x, norm_y)

    def _on_diff_cursor_left(self):
        """Clear left-panel magnifier when cursor leaves Difference Map."""
        if not self.btn_split_view.isChecked():
            self.img_base_mag.clearCursor()

    def _on_operation_changed(self, index):
        """Show/hide blend coefficients based on operation mode."""
        is_blend = (index == 1)
        self.grp_blend_coef.setVisible(is_blend)

    def _load_images(self):
        """Load images from conditions and populate UI."""
        self._compare_checkboxes: List[QtWidgets.QCheckBox] = []
        self._images.clear()

        # Clear existing
        while self.compare_layout.count():
            item = self.compare_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.cmb_base.clear()
        # Clear Quadrant Fusion combos
        for cmb in self._qf_combos.values():
            cmb.clear()

        if not self._conditions:
            self._update_sidebar_empty_state(False)
            return

        # Load each image
        for cond in self._conditions:
            # Always use filename instead of label (which may have keV/nA format)
            import os
            filename = os.path.basename(cond.image_path)
            label = filename  # Use pure filename

            # Load image
            img = load_image_gray(cond.image_path)
            if img is not None:
                self._images[label] = img

                # Add to base dropdown
                self.cmb_base.addItem(label)

                # Add checkbox for compare
                chk = QtWidgets.QCheckBox(label)
                chk.setChecked(False)
                self.compare_layout.addWidget(chk)
                self._compare_checkboxes.append(chk)

                # Populate Quadrant Fusion combos
                for cmb in self._qf_combos.values():
                    cmb.addItem(label)

        self.compare_layout.addStretch()

        has_images = self.cmb_base.count() > 0
        self._update_sidebar_empty_state(has_images)

        # Select first image as base
        if has_images:
            self.cmb_base.setCurrentIndex(0)
            self._on_base_changed()

    def _update_sidebar_empty_state(self, has_images: bool):
        """Update sidebar placeholders and enablement when image set is empty/non-empty."""
        self.lbl_input_empty_hint.setVisible(not has_images)
        self.lbl_base_empty_hint.setVisible(not has_images)

        self.cmb_base.setEnabled(has_images)
        self.scroll_compare.setEnabled(has_images)
        self.btn_swap_base.setEnabled(has_images)
        self.btn_select_all.setEnabled(has_images)
        self.btn_select_none.setEnabled(has_images)

    def _on_base_changed(self):
        """Update compare checkboxes when base changes."""
        base_label = self.cmb_base.currentText()

        for chk in self._compare_checkboxes:
            # Disable checkbox for the base image
            is_base = (chk.text() == base_label)
            chk.setEnabled(not is_base)
            if is_base:
                chk.setChecked(False)

    def _select_all_compare(self):
        """Select all compare images."""
        for chk in self._compare_checkboxes:
            if chk.isEnabled():
                chk.setChecked(True)

    def _select_none_compare(self):
        """Deselect all compare images."""
        for chk in self._compare_checkboxes:
            chk.setChecked(False)

    def _on_display_mode(self, mode: str):
        """Switch left or right viewer display mode and keep button highlights in sync."""
        if mode in ('base', 'compare'):
            self._left_non_split_mode = mode
            if self.btn_split_view.isChecked():
                self.btn_split_view.setChecked(False)
            else:
                self._left_view_mode = mode
                self._refresh_base_compare_display(mode)
            self._sync_viewer_mode_buttons()
            return

        if mode in ('diff', 'zmap'):
            self._display_mode = mode
            self._sync_viewer_mode_buttons()
            self._refresh_diff_display()

    def _sync_viewer_mode_buttons(self):
        """Synchronize viewer mode button highlight states with active view."""
        is_split = self._left_view_mode == 'split'
        self.btn_split_view.setChecked(is_split)
        self.btn_mode_base.setChecked((not is_split) and self._left_non_split_mode == 'base')
        self.btn_mode_compare.setChecked((not is_split) and self._left_non_split_mode == 'compare')
        self.btn_mode_diff.setChecked(self._display_mode == 'diff')
        self.btn_mode_topo.setChecked(self._display_mode == 'zmap')

    def _on_left_viewer_clicked(self):
        """Keep left mode button highlight synced when clicking in the left magnifier."""
        if self._left_view_mode != 'split':
            self._left_view_mode = self._left_non_split_mode
        self._sync_viewer_mode_buttons()

    def _on_split_viewer_clicked(self):
        """Keep split button highlighted when user interacts in split view."""
        self._left_view_mode = 'split'
        self._sync_viewer_mode_buttons()

    def _on_right_viewer_clicked(self):
        """Keep right mode button highlight synced when clicking in the right viewer."""
        self._sync_viewer_mode_buttons()

    def _refresh_base_compare_display(self, mode: str):
        """Show base or compare image in the magnifier view."""
        if not self._results:
            return
        result = self._results[self._current_result_idx]
        if mode == 'base':
            base_img = self._images.get(result.base_label)
            if base_img is not None:
                self.img_base_mag.setImage(base_img)
        elif mode == 'compare':
            compare_img = (result.aligned_compare
                           if result.aligned_compare is not None
                           else self._images.get(result.compare_label))
            if compare_img is not None:
                self.img_base_mag.setImage(compare_img)

    def _on_range_changed(self, index: int):
        """Handle range control change."""
        self._refresh_diff_display()

    def _apply_colormap(self, gray_img: np.ndarray) -> np.ndarray:
        """Apply the selected colormap to a grayscale uint8 image → BGR uint8."""
        name = self.cmb_colormap.currentText()
        if name == "Grayscale":
            return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        cv2_maps = {
            "JET": cv2.COLORMAP_JET,
            "Hot": cv2.COLORMAP_HOT,
            "Inferno": cv2.COLORMAP_INFERNO,
            "Viridis": cv2.COLORMAP_VIRIDIS,
        }
        if name in cv2_maps:
            return cv2.applyColorMap(gray_img, cv2_maps[name])
        return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    def _refresh_diff_display(self):
        """Refresh the difference map display with current mode, threshold, and ROI settings."""
        if not self._results:
            return

        result = self._results[self._current_result_idx]

        if self._display_mode == 'zmap':
            display_img = colorize_snr_map(result.snr_map)
        else:
            scaled_img = self._apply_range_scaling(result.result_image)
            if scaled_img.ndim == 2:
                display_img = self._apply_colormap(scaled_img)
            else:
                display_img = scaled_img.copy()

        # ── Gray-level range highlight ─────────────────────────────────────
        if self._hist_range is not None:
            lo, hi = self._hist_range
            # Get source gray values (use original result_image, not scaled display)
            src_gray = result.result_image  # uint8
            if src_gray is not None:
                if display_img.ndim == 2:
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                in_range = ((src_gray >= lo) & (src_gray <= hi))
                mask3 = np.stack([in_range, in_range, in_range], axis=2)
                # Pixels outside range: dimmed to 15%
                dimmed = (display_img.astype(np.float32) * 0.15).astype(np.uint8)
                display_img = np.where(mask3, display_img, dimmed)
                # Teal tint on in-range pixels
                tint = display_img.copy()
                tint[:, :, 1] = np.where(in_range,
                                         np.clip(display_img[:, :, 1].astype(np.int16) + 30, 0, 255),
                                         display_img[:, :, 1])
                display_img = tint

        # ── Alignment failure visual overlay ──────────────────────────────
        align_score = result.alignment.final_score
        if align_score < 55:
            if display_img.ndim == 2:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            overlay = display_img.copy()
            h_img, w_img = display_img.shape[:2]
            # Red semi-transparent banner at top
            cv2.rectangle(overlay, (0, 0), (w_img, 36), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.65, display_img, 0.35, 0, display_img)
            warn_txt = f"\u26a0 ALIGN FAIL  score={align_score:.0f}  diff may be unreliable"
            cv2.putText(display_img, warn_txt, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        self.img_diff.setImage(display_img)

    def _apply_range_scaling(self, img: np.ndarray) -> np.ndarray:
        """Apply dynamic range scaling based on selected mode.

        Modes:
        - Auto: Use full min-max range
        - Zero-centered: Center at 128, scale symmetric
        - P1-P99: Clip to 1st-99th percentile
        - P0.5-P99.5: Clip to 0.5th-99.5th percentile
        """
        if img is None or img.size == 0:
            return img

        range_mode = self.cmb_range.currentText()
        img_f = img.astype(np.float32)

        if range_mode == "Zero-centered":
            # Center at 128, use symmetric range
            center = 128.0
            max_dev = max(abs(img_f.max() - center), abs(img_f.min() - center))
            if max_dev > 0:
                scaled = (img_f - center) / max_dev * 127 + 128
            else:
                scaled = np.full_like(img_f, 128)
            return np.clip(scaled, 0, 255).astype(np.uint8)

        elif range_mode == "P1-P99":
            p_low, p_high = np.percentile(img_f, [1, 99])
            clipped = np.clip(img_f, p_low, p_high)
            if p_high > p_low:
                scaled = (clipped - p_low) / (p_high - p_low) * 255
            else:
                scaled = np.full_like(img_f, 128)
            return scaled.astype(np.uint8)

        elif range_mode == "P0.5-P99.5":
            p_low, p_high = np.percentile(img_f, [0.5, 99.5])
            clipped = np.clip(img_f, p_low, p_high)
            if p_high > p_low:
                scaled = (clipped - p_low) / (p_high - p_low) * 255
            else:
                scaled = np.full_like(img_f, 128)
            return scaled.astype(np.uint8)

        else:  # Auto - use full range
            return img

    def _on_compute(self):
        """Run the combination computation for all selected pairs."""
        # ── Quadrant Fusion mode ──────────────────────────────────────────
        if self.cmb_input_mode.currentIndex() == 1:
            self._on_compute_quadrant_fusion()
            return

        base_label = self.cmb_base.currentText()
        is_auto_pair = self.chk_auto_pair.isChecked()

        # Get selected compare images first
        compare_labels = [
            chk.text() for chk in self._compare_checkboxes
            if chk.isChecked() and chk.isEnabled()
        ]

        # Validation
        if is_auto_pair:
            # Auto-pair mode requires at least 2 images
            if len(compare_labels) < 2:
                QtWidgets.QMessageBox.warning(self, "Error", "Please select at least two images for auto pairing.")
                return
        else:
            # Normal mode requires valid base + at least 1 compare
            if not base_label or base_label not in self._images:
                QtWidgets.QMessageBox.warning(self, "Error", "Please select a valid base image.")
                return
            if not compare_labels:
                QtWidgets.QMessageBox.warning(self, "Error", "Please select at least one compare image.")
                return

        # Get operation settings
        operation = 'subtract' if self.cmb_operation.currentIndex() == 0 else 'blend'
        alpha = self.spin_alpha.value()
        beta = self.spin_beta.value()
        invert_base = self.chk_invert_base.isChecked()
        invert_compare = self.chk_invert_compare.isChecked()
        invert_result = self.chk_invert_result.isChecked()
        alignment_method = "phase" if self.cmb_align_method.currentIndex() == 0 else "ncc"
        snr_window_size = self.spn_snr_window.value()
        # Ensure window_size is odd
        if snr_window_size % 2 == 0:
            snr_window_size += 1
        norm_mode = self.cmb_normalize_mode.currentIndex()
        # 0 = Percentile, 1 = GLV-Mask, 2 = HEQ, 3 = CLAHE, 4 = Skip, 5 = ROI-Match
        _method_map = {0: 'percentile', 1: 'glv_mask', 2: 'heq', 3: 'clahe', 4: 'skip', 5: 'roi_match'}
        normalize_method = _method_map.get(norm_mode, 'percentile')
        normalize = (normalize_method not in ('skip', 'roi_match'))
        glv_range = None
        if norm_mode == 1:
            glv_low = self.spn_glv_low.value()
            glv_high = self.spn_glv_high.value()
            if glv_low < glv_high:
                glv_range = (glv_low, glv_high)
        clahe_clip_limit = self.spn_clahe_clip.value() if norm_mode == 3 else 2.0

        # ROI-Match (EPI Nulling) parameters
        use_roi_match = (norm_mode == 5)
        roi_rect_px = None
        if use_roi_match:
            if self._roi_match_rect is None:
                QtWidgets.QMessageBox.warning(
                    self, "ROI-Match",
                    "Please draw an ROI on the Base image first.\n"
                    "Use the 'Pick ROI' button and select a uniform EPI-dominated region."
                )
                return
            # Convert normalized ROI to pixel coordinates using base image dimensions
            base_img_check = self._images.get(base_label)
            if base_img_check is not None:
                img_h, img_w = base_img_check.shape[:2]
                nx, ny, nw, nh = self._roi_match_rect
                roi_rect_px = (
                    int(nx * img_w), int(ny * img_h),
                    max(1, int(nw * img_w)), max(1, int(nh * img_h)),
                )

        sub_mode = self.cmb_subtract_mode.currentIndex()
        # 0 = |diff|×2 (default), 1 = |diff| no gain, 2 = clip≥0 (preserve direction)
        preserve_positive_diff = (sub_mode == 2)
        abs_no_gain = (sub_mode == 1)

        # ROI-Match override: if user wants |diff| with ROI-Match, honour the checkbox
        if use_roi_match and self.chk_roi_abs_diff.isChecked():
            abs_no_gain = True
            preserve_positive_diff = False

        self._last_settings = {
            "operation": operation,
            "alpha": alpha,
            "beta": beta,
            "invert_base": invert_base,
            "invert_compare": invert_compare,
            "invert_result": invert_result,
            "alignment_method": alignment_method,
            "subtract_mode": sub_mode,
            "preserve_positive_diff": preserve_positive_diff,
            "abs_no_gain": abs_no_gain,
            "snr_window_size": snr_window_size,
            "normalize": normalize,
            "normalize_method": normalize_method,
            "glv_range": glv_range,
            "clahe_clip_limit": clahe_clip_limit,
            "roi_match": use_roi_match,
            "roi_rect_px": roi_rect_px,
        }

        # Get images
        base_img = self._images[base_label] if base_label in self._images else None
        compare_imgs = {lbl: self._images[lbl] for lbl in compare_labels if lbl in self._images}

        self._start_compute_worker(
            base_label, base_img, compare_imgs,
            operation, alpha, beta,
            invert_base, invert_compare, invert_result,
            alignment_method,
            preserve_positive_diff,
            abs_no_gain=abs_no_gain,
            normalize=normalize,
            normalize_method=normalize_method,
            glv_range=glv_range,
            clahe_clip_limit=clahe_clip_limit,
            snr_window_size=snr_window_size,
            is_auto_pair=is_auto_pair,  # captured on main thread — do NOT read inside thread
            roi_rect=roi_rect_px,
            roi_match=use_roi_match,
        )

    def _start_compute_worker(
            self,
            base_label: str,
            base_img: Optional[np.ndarray],
            compare_imgs: Dict[str, np.ndarray],
            operation: str,
            alpha: float,
            beta: float,
            invert_base: bool,
            invert_compare: bool,
            invert_result: bool,
            alignment_method: str,
            preserve_positive_diff: bool,
            abs_no_gain: bool = False,
            normalize: bool = True,
            normalize_method: str = 'percentile',
            glv_range: Optional[Tuple[int, int]] = None,
            clahe_clip_limit: float = 2.0,
            snr_window_size: int = 31,
            is_auto_pair: bool = False,
            roi_rect: Optional[tuple] = None,
            roi_match: bool = False,
    ):
        if self._compute_thread is not None:
            return

        if is_auto_pair:
            pair_count = len(compare_imgs) * (len(compare_imgs) - 1)
        else:
            pair_count = len(compare_imgs)
        self.btn_compute.setEnabled(False)
        self.btn_compute.setText("Computing…")
        self.btn_export.setEnabled(False)
        self.btn_prev_result.setEnabled(False)
        self.btn_next_result.setEnabled(False)

        # ── Indeterminate progress dialog so the UI never looks frozen ─────
        self._compute_progress = QtWidgets.QProgressDialog(
            f"Computing {pair_count} pair(s)…\n"
            "Aligning and processing — please wait.",
            None,  # no cancel button while running
            0, 0,  # min == max == 0 → indeterminate spinning bar
            self,
        )
        self._compute_progress.setWindowTitle("Fusi\u00b3 \u2014 Computing")
        # NonModal: parent window stays interactive; Compute button is already
        # disabled so users cannot double-submit.
        self._compute_progress.setWindowModality(Qt.NonModal)
        self._compute_progress.setMinimumDuration(0)
        self._compute_progress.setStyleSheet(DIALOG_STYLE)
        self._compute_progress.setValue(0)
        self._compute_progress.show()
        QtWidgets.QApplication.processEvents()

        def _run_compute():
            # IMPORTANT: do NOT access any Qt widget here — this runs on a
            # background thread.  All widget state must be captured before
            # _start_compute_worker is called (see is_auto_pair below).
            if is_auto_pair:
                results: List[SinglePairResult] = []
                labels = list(compare_imgs.keys())
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        if i == j:
                            continue
                        base_lbl = labels[i]
                        cmp_lbl = labels[j]
                        result = compute_single_pair(
                            base=compare_imgs[base_lbl],
                            compare=compare_imgs[cmp_lbl],
                            base_label=base_lbl,
                            compare_label=cmp_lbl,
                            operation=operation,
                            alpha=alpha,
                            beta=beta,
                            invert_base=invert_base,
                            invert_compare=invert_compare,
                            invert_result=invert_result,
                            normalize=normalize,
                            normalize_method=normalize_method,
                            glv_range=glv_range,
                            clahe_clip_limit=clahe_clip_limit,
                            search_radius=50,
                            alignment_method=alignment_method,
                            preserve_positive_diff=preserve_positive_diff,
                            abs_no_gain=abs_no_gain,
                            snr_window_size=snr_window_size,
                            roi_rect=roi_rect,
                            roi_match=roi_match,
                        )
                        results.append(result)
                return results
            return compute_multi_pairs(
                base=base_img,
                base_label=base_label,
                compare_images=compare_imgs,
                operation=operation,
                alpha=alpha,
                beta=beta,
                invert_base=invert_base,
                invert_compare=invert_compare,
                invert_result=invert_result,
                normalize=normalize,
                normalize_method=normalize_method,
                glv_range=glv_range,
                clahe_clip_limit=clahe_clip_limit,
                search_radius=50,
                alignment_method=alignment_method,
                preserve_positive_diff=preserve_positive_diff,
                abs_no_gain=abs_no_gain,
                snr_window_size=snr_window_size,
                roi_rect=roi_rect,
                roi_match=roi_match,
            )

        self._compute_thread, self._compute_worker = self._start_worker(
            _run_compute,
            on_success=self._on_compute_finished,
            on_error=self._on_compute_error,
            on_done=self._on_compute_done
        )

    def _start_worker(self, fn, on_success, on_error=None, on_done=None):
        """Run *fn* on a QThread; all callbacks execute on the main thread.

        Signal routing:
          worker.finished  ──QueuedConn──▶  on_success  (main thread)
          worker.error     ──QueuedConn──▶  on_error    (main thread)
          worker.finished/error ──Direct──▶ thread.quit()  (safe, thread-safe call)
          thread.finished  ──QueuedConn──▶  on_done     (main thread)
          thread.finished  ──QueuedConn──▶  deleteLater (deferred, safe)
        """
        thread = QtCore.QThread()  # no parent → avoids thread-affinity issues
        worker = _Worker(fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        # on_success / on_error: worker is a QObject living in worker-thread;
        # self._on_compute_* are methods of a QObject living in main-thread
        # → AutoConnection auto-promotes to QueuedConnection cross-thread ✓
        worker.finished.connect(on_success)
        if on_error is not None:
            worker.error.connect(on_error)

        # Tell the thread's event loop to exit once work is done.
        # These lambdas discard the argument and call thread.quit(),
        # which is thread-safe from any thread.
        worker.finished.connect(lambda _: thread.quit())
        worker.error.connect(lambda _: thread.quit())

        # After the event loop exits, thread.finished fires on the main thread.
        # Wire housekeeping and on_done here so they always run on main thread.
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        if on_done is not None:
            thread.finished.connect(on_done)  # QueuedConnection → main thread ✓

        thread.start()
        return thread, worker

    def _on_compute_finished(self, results: List[SinglePairResult]):
        self._results = results
        self._current_result_idx = 0
        self._update_current_result()
        self._update_navigation()
        self.btn_export.setEnabled(bool(self._results))

    def _on_compute_error(self, message: str):
        QtWidgets.QMessageBox.critical(self, "Error", f"Computation failed:\n{message}")

    def _on_compute_done(self):
        self._compute_thread = None
        self._compute_worker = None
        # Close indeterminate progress dialog
        if hasattr(self, '_compute_progress') and self._compute_progress is not None:
            self._compute_progress.close()
            self._compute_progress = None
        self.btn_compute.setEnabled(True)
        self.btn_compute.setText("Compute")
        self._set_button_icon(self.btn_compute, QtWidgets.QStyle.SP_MediaPlay, "Compute")

    # ── Quadrant Fusion compute ─────────────────────────────────────────

    def _on_compute_quadrant_fusion(self):
        """Run Quadrant Fusion computation."""
        # Validate: all 5 detectors must be assigned
        det_labels = {}
        for det_name, cmb in self._qf_combos.items():
            label = cmb.currentText()
            if not label or label not in self._images:
                QtWidgets.QMessageBox.warning(
                    self, "Quadrant Fusion",
                    f"Please assign a valid image for '{det_name}'."
                )
                return
            det_labels[det_name] = label

        # Check all images are same shape
        shapes = set()
        imgs = {}
        for det_name, label in det_labels.items():
            img = self._images[label]
            shapes.add(img.shape[:2])
            imgs[det_name] = img

        if len(shapes) > 1:
            QtWidgets.QMessageBox.warning(
                self, "Quadrant Fusion",
                "All 5 images must have the same dimensions.\n"
                f"Found different shapes: {shapes}"
            )
            return

        # Read parameters
        output_idx = self.cmb_qf_output.currentIndex()
        output_type = ["bse_clean", "topo", "composite"][output_idx]
        alpha_mode = "auto" if self.cmb_qf_alpha_mode.currentIndex() == 0 else "manual"
        alpha_manual = self.spn_qf_alpha.value()
        beta = self.spn_qf_beta.value()
        sigma = self.spn_qf_sigma.value()

        # ROI for alpha fit
        roi_rect_px = None
        if self._qf_roi_rect is not None:
            h, w = imgs["Illuminator"].shape[:2]
            nx, ny, nw, nh = self._qf_roi_rect
            roi_rect_px = (
                int(nx * w), int(ny * h),
                max(1, int(nw * w)), max(1, int(nh * h)),
            )

        # Capture images for thread
        illum = imgs["Illuminator"]
        top = imgs["Top"]
        bottom = imgs["Bottom"]
        left = imgs["Left"]
        right = imgs["Right"]

        self.btn_compute.setEnabled(False)
        self.btn_compute.setText("Computing…")
        self.btn_export.setEnabled(False)

        self._compute_progress = QtWidgets.QProgressDialog(
            "Computing Quadrant Fusion…\nProcessing 5 detector images.",
            None, 0, 0, self,
        )
        self._compute_progress.setWindowTitle("Quadrant Fusion")
        self._compute_progress.setWindowModality(Qt.NonModal)
        self._compute_progress.setMinimumDuration(0)
        self._compute_progress.setStyleSheet(DIALOG_STYLE)
        self._compute_progress.setValue(0)
        self._compute_progress.show()
        QtWidgets.QApplication.processEvents()

        def _run():
            return compute_quadrant_fusion(
                illum=illum, top=top, bottom=bottom, left=left, right=right,
                output_type=output_type,
                alpha_mode=alpha_mode, alpha_manual=alpha_manual,
                beta=beta, gaussian_sigma=sigma,
                roi_rect=roi_rect_px,
            )

        self._compute_thread, self._compute_worker = self._start_worker(
            _run,
            on_success=self._on_qf_compute_finished,
            on_error=self._on_compute_error,
            on_done=self._on_compute_done,
        )

    def _on_qf_compute_finished(self, result: QuadrantFusionResult):
        """Handle Quadrant Fusion result — display in QF-specific viewers."""
        self._qf_last_result = result
        self._result = None
        self._qf_current_view = "bse_clean"  # default main view

        # Display Illuminator in left viewer
        illum_label = self._qf_combos["Illuminator"].currentText()
        if illum_label in self._images:
            self.img_qf_illum.setImage(self._images[illum_label])

        # Display main output + reference
        self._refresh_qf_display()

        # Update toggle button states
        self.btn_qf_show_bse.setChecked(True)
        self.btn_qf_show_topo.setChecked(False)
        self.btn_qf_show_comp.setChecked(False)

        # Histogram of primary output
        counts, edges = np.histogram(result.fused_image_uint8.ravel(), bins=256, range=(0, 255))
        self.qf_histogram_canvas.plot_histogram(counts, edges)

        # Output type name
        output_names = {"bse_clean": "BSE Enhanced", "topo": "Topography", "composite": "Composite"}
        out_name = output_names.get(
            ["bse_clean", "topo", "composite"][self.cmb_qf_output.currentIndex()], "Fusion"
        )

        # Info bar
        self.lbl_qf_result_info.setText(
            f"Quadrant Fusion complete  |  Output: {out_name}  |  "
            f"alpha={result.alpha_used:.4f}  beta={result.beta_used:.2f}"
        )

        # Stats panel
        fused_f = result.fused_image_uint8.astype(np.float32)
        h, w = result.fused_image_uint8.shape[:2]
        self._qf_stat_labels["alpha"].setText(f"{result.alpha_used:.6f}")
        self._qf_stat_labels["beta"].setText(f"{result.beta_used:.4f}")
        self._qf_stat_labels["sigma"].setText(f"{self.spn_qf_sigma.value():.1f}")
        self._qf_stat_labels["output"].setText(out_name)
        self._qf_stat_labels["mean"].setText(f"{fused_f.mean():.2f}")
        self._qf_stat_labels["std"].setText(f"{fused_f.std():.2f}")
        self._qf_stat_labels["dim"].setText(f"{w} × {h}")

        self.btn_export.setEnabled(True)

    def _on_qf_view_toggle(self, view: str):
        """Switch the main QF output view (bse_clean / topo / composite)."""
        self._qf_current_view = view
        # Exclusive toggle
        self.btn_qf_show_bse.setChecked(view == "bse_clean")
        self.btn_qf_show_topo.setChecked(view == "topo")
        self.btn_qf_show_comp.setChecked(view == "composite")
        self._refresh_qf_display()

    def _refresh_qf_display(self):
        """Refresh QF image viewers based on current view selection."""
        result = self._qf_last_result
        if result is None:
            return

        view = getattr(self, '_qf_current_view', 'bse_clean')

        # Choose main + reference images and group titles
        if view == "bse_clean":
            main_img = result.bse_clean_uint8
            ref_img = result.topo_uint8
            main_title = "BSE Enhanced (Illum − α·Topo)"
            ref_title = "Topography"
        elif view == "topo":
            main_img = result.topo_uint8
            ref_img = result.bse_clean_uint8
            main_title = "Topography"
            ref_title = "BSE Enhanced"
        else:  # composite
            main_img = result.composite_uint8
            ref_img = result.topo_uint8
            main_title = "Composite (BSE + β·Topo)"
            ref_title = "Topography"

        # Apply colormap if not grayscale
        cmap_idx = self.cmb_qf_colormap.currentIndex()
        main_display = self._apply_qf_colormap(main_img, cmap_idx)
        ref_display = self._apply_qf_colormap(ref_img, cmap_idx)

        self.qf_main_grp.setTitle(main_title)
        self.qf_ref_grp.setTitle(ref_title)
        self.img_qf_main.setImage(main_display)
        self.img_qf_ref.setImage(ref_display)

        # Update histogram for main view
        counts, edges = np.histogram(main_img.ravel(), bins=256, range=(0, 255))
        self.qf_histogram_canvas.plot_histogram(counts, edges)

    @staticmethod
    def _apply_qf_colormap(gray_img: np.ndarray, cmap_idx: int) -> np.ndarray:
        """Apply colormap to a grayscale image. 0=Grayscale, 1=JET, 2=Hot, 3=Inferno, 4=Viridis."""
        if cmap_idx == 0:
            return gray_img
        cmap_map = {1: cv2.COLORMAP_JET, 2: cv2.COLORMAP_HOT,
                    3: cv2.COLORMAP_INFERNO, 4: cv2.COLORMAP_VIRIDIS}
        cm = cmap_map.get(cmap_idx, cv2.COLORMAP_JET)
        return cv2.applyColorMap(gray_img, cm)

    def _on_adv_toggle(self, checked: bool):
        """Show/hide Advanced operation options."""
        self.grp_advanced.setVisible(checked)
        self.btn_adv_toggle.setText("Advanced \u25bc" if checked else "Advanced \u25b6")

    def _show_about_dialog(self):
        """Show the About Fusi\u00b3 dialog."""
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("About Fusi\u00b3")
        dlg.setIcon(QtWidgets.QMessageBox.Information)
        dlg.setText(
            "<h2>Fusi\u00b3</h2>"
            "<p><b>SEM Image Fusion & Defect Analysis</b></p>"
        )
        dlg.setInformativeText(
            "Fusi\u00b3 is an advanced SEM image analysis tool for alignment, "
            "fusion, and nanoscale defect detection across multi-detector SEM images.\n\n"
            "Version 1.1.0"
        )
        dlg.setStyleSheet(DIALOG_STYLE)
        dlg.exec()

    def _on_swap_base(self):
        """Swap base image: cycle to next image as base."""
        if self.cmb_base.count() <= 1:
            return
        next_idx = (self.cmb_base.currentIndex() + 1) % self.cmb_base.count()
        self.cmb_base.setCurrentIndex(next_idx)

    def _on_hist_range_changed(self, lo: int, hi: int):
        """Called when user finishes selecting a gray-level range on histogram."""
        self._hist_range = (lo, hi)
        self.lbl_hist_range.setText(f"Range: GL {lo} - {hi}")
        self.btn_clear_hist_range.setEnabled(True)
        self._refresh_diff_display()

    def _on_clear_hist_range(self):
        """Clear histogram range filter."""
        self._hist_range = None
        self.lbl_hist_range.setText("Range: -")
        self.btn_clear_hist_range.setEnabled(False)
        self.histogram_canvas.clear_range()
        self._refresh_diff_display()

    def _update_current_result(self):
        """Update displays for the current result."""
        if not self._results:
            return

        result = self._results[self._current_result_idx]

        # Result image - use centralized refresh method
        self._refresh_diff_display()

        # Blend preview
        self._update_blend_preview()
        if self._left_view_mode != 'split':
            self._refresh_base_compare_display(self._left_non_split_mode)
        self._sync_viewer_mode_buttons()

        # Histogram
        counts, edges = result.histogram
        self.histogram_canvas.plot_histogram(counts, edges)

        # Alignment scores - adapt SinglePairResult to CombineResult format
        self._update_alignment_display(result)

        # Statistics
        self._update_stats_display(result)
        self._refresh_normalized_compare_dialog(result)

    def _update_blend_preview(self):
        if not self._results:
            return
        result = self._results[self._current_result_idx]
        base_img = self._images.get(result.base_label)
        # Use aligned compare so split view spatial coordinates match Diff/SNR maps
        compare_img = (result.aligned_compare
                       if result.aligned_compare is not None
                       else self._images.get(result.compare_label))

        # Page 0 – Magnifier: show Base image
        if base_img is not None:
            self.img_base_mag.setImage(base_img)

        # Page 1 – Split View: Base left, Aligned Compare right
        self.img_blend.set_images(base_img, compare_img)
        # slider 0 → Base only (divider at far right → ratio=1.0)
        # slider 100 → Compare only (divider at far left → ratio=0.0)
        ratio = 1.0 - self.slider_blend.value() / 100.0
        self.img_blend.set_divider(ratio)

    def _update_alignment_display(self, result: SinglePairResult):
        """Update alignment metrics in the merged Analysis card."""
        a = result.alignment

        if a.final_score >= 75:
            status_text = "\u2713 OK"
            status_color = UI_SUCCESS
        elif a.final_score >= 55:
            status_text = "\u26a0 WARN"
            status_color = UI_PRIMARY_HOVER
        else:
            status_text = "\u2717 FAIL"
            status_color = UI_WARNING

        self.stats_widget.update_alignment(
            phase=f"{a.score_phase:.3f}",
            ncc=f"{a.score_ncc:.3f}",
            residual=f"{a.score_residual:.3f}",
            final=f"{a.final_score:.1f}",
            shift=f"({a.dx:+d}, {a.dy:+d})",
            status=status_text,
            status_color=status_color,
        )

    def _update_stats_display(self, result: SinglePairResult):
        """Update statistics widget for single result."""
        s = result.stats
        self.stats_widget.stats_labels["diff_mean"].setText(f"{s.get('diff_mean', 0):.4f}")
        self.stats_widget.stats_labels["diff_std"].setText(f"{s.get('diff_std', 0):.4f}")
        self.stats_widget.stats_labels["hot_pixels"].setText(f"{s.get('hot_pixels', 0)}")
        # Display normalize coefficients: a*I + b
        method_display = _NORMALIZE_METHOD_LABELS.get(result.normalize_method, result.normalize_method)
        is_linear = result.normalize_method in ('percentile', 'glv_mask')
        if is_linear:
            self.stats_widget.stats_labels["norm_coeff"].setText(
                f"{method_display}  a={result.norm_a:.4f}"
            )
        else:
            self.stats_widget.stats_labels["norm_coeff"].setText(method_display)
        # Sub-pixel shift
        dx_f = s.get('dx_subpixel', float(result.alignment.dx))
        dy_f = s.get('dy_subpixel', float(result.alignment.dy))
        self.stats_widget.stats_labels["subpixel_shift"].setText(f"({dx_f:+.2f}, {dy_f:+.2f})")

        # ROI-Match alpha readout
        if result.roi_match_alpha is not None:
            self.lbl_roi_alpha.setText(f"ROI-match \u03b1 = {result.roi_match_alpha:.4f}")
        else:
            self.lbl_roi_alpha.setText("")

    def _update_navigation(self):
        """Update navigation buttons and label."""
        n = len(self._results)
        if n == 0:
            self.lbl_result_info.setText("")
            self.lbl_result_info.setVisible(False)
            self.btn_prev_result.setEnabled(False)
            self.btn_next_result.setEnabled(False)
            return

        idx = self._current_result_idx
        result = self._results[idx]

        # Update label with pair info
        op_name = "Subtract" if result.operation == 'subtract' else "Blend"
        self.lbl_result_info.setText(
            f"{idx + 1}/{n}: {result.base_label} \u2192 {result.compare_label} ({op_name})"
        )
        self.lbl_result_info.setVisible(True)
        self.setWindowTitle(
            f"Fusi\u00b3 \u2014 {result.base_label} \u2192 {result.compare_label}"
        )

        # Enable/disable buttons
        self.btn_prev_result.setEnabled(idx > 0)
        self.btn_next_result.setEnabled(idx < n - 1)

    def _reset_hist_range(self):
        """Clear histogram range filter when switching to a different pair."""
        self._hist_range = None
        self.lbl_hist_range.setText("Range: —")
        self.btn_clear_hist_range.setEnabled(False)
        self.histogram_canvas.clear_range()

    def _on_prev_result(self):
        """Go to previous result."""
        if self._current_result_idx > 0:
            self._reset_hist_range()
            self._current_result_idx -= 1
            self._update_current_result()
            self._update_navigation()

    def _on_next_result(self):
        """Go to next result."""
        if self._current_result_idx < len(self._results) - 1:
            self._reset_hist_range()
            self._current_result_idx += 1
            self._update_current_result()
            self._update_navigation()

    # Legacy methods for backward compatibility
    def _update_results(self):
        """Legacy: Update result displays."""
        if self._result is None:
            return
        self.img_diff.setImage(self._result.diff_image)
        counts, edges = self._result.histogram
        self.histogram_canvas.plot_histogram(counts, edges)
        self.align_score_widget.update_scores(self._result)
        self.stats_widget.update_stats(self._result)

    def _update_hint_overlay(self):
        """Legacy: no-op (hint overlay removed)."""
        pass

    def _on_base_mag_cursor_moved(self, norm_x: float, norm_y: float):
        """Relay left-panel magnifier cursor position to Difference Map (bidirectional zoom)."""
        if not self.btn_split_view.isChecked():
            self.img_diff.setCursorPos(norm_x, norm_y)

    def _on_base_mag_cursor_left(self):
        """Clear Difference Map magnifier when cursor leaves left-panel magnifier."""
        if not self.btn_split_view.isChecked():
            self.img_diff.clearCursor()

    # ── ROI-Match (EPI Nulling) handlers ─────────────────────────────────

    def _on_pick_roi(self):
        """Enter ROI drawing mode on the Base magnifier widget."""
        if not self._images:
            QtWidgets.QMessageBox.information(
                self, "ROI-Match", "Please load images first."
            )
            return
        # Ensure magnifier page is visible so the user can draw on Base
        if self.btn_split_view.isChecked():
            self.btn_split_view.setChecked(False)
        self.img_base_mag.set_roi_mode(True)
        self.lbl_roi_info.setText("ROI: draw on Base image...")

    def _on_clear_roi(self):
        """Clear the current ROI selection."""
        self._roi_match_rect = None
        self.img_base_mag.set_active_roi(None)
        self.img_diff.set_active_roi(None)
        self.btn_clear_roi.setEnabled(False)
        self.lbl_roi_info.setText("ROI: not set")
        self.lbl_roi_alpha.setText("")

    def _on_roi_selected(self, norm_x: float, norm_y: float, norm_w: float, norm_h: float):
        """Handle ROI drawn on an image widget (shared by Standard & QF modes)."""
        roi = (norm_x, norm_y, norm_w, norm_h)

        if self.cmb_input_mode.currentIndex() == 1:
            # Quadrant Fusion mode — ROI on QF viewers
            self._qf_roi_rect = roi
            self.img_qf_illum.set_active_roi(roi)
            self.img_qf_main.set_active_roi(roi)
            self.btn_qf_clear_roi.setEnabled(True)
            self.lbl_qf_roi_info.setText(
                f"ROI: ({norm_x:.2f}, {norm_y:.2f}) {norm_w:.2f}×{norm_h:.2f}"
            )
        else:
            # Standard mode — ROI on Standard viewers
            self._roi_match_rect = roi
            self.img_base_mag.set_active_roi(roi)
            self.img_diff.set_active_roi(roi)
            self.btn_clear_roi.setEnabled(True)
            self.lbl_roi_info.setText(
                f"ROI: ({norm_x:.2f}, {norm_y:.2f}) {norm_w:.2f}×{norm_h:.2f}"
            )
            self.lbl_roi_alpha.setText("")  # clear previous alpha

    def _on_normalize_mode_changed(self):
        """Show/hide GLV-Mask / CLAHE / ROI-Match controls depending on the selected normalize mode."""
        idx = self.cmb_normalize_mode.currentIndex()
        self.wgt_glv_controls.setVisible(idx == 1)  # GLV-Mask
        self.wgt_clahe_controls.setVisible(idx == 3)  # CLAHE
        self.wgt_roi_match_controls.setVisible(idx == 5)  # ROI-Match (EPI Nulling)

    def _on_preview_glv_mask(self):
        """Open the interactive GLV Mask Preview dialog."""
        base_label = self.cmb_base.currentText() if hasattr(self, "cmb_base") else ""
        base_img = self._images.get(base_label) if base_label else None
        if base_img is None:
            QtWidgets.QMessageBox.information(
                self, "GLV Mask Preview", "Please load a Base image first."
            )
            return
        glv_low = self.spn_glv_low.value()
        glv_high = self.spn_glv_high.value()

        def _apply_to_main(low: int, high: int):
            """Sync values from the preview dialog back to the main dialog spinboxes."""
            self.spn_glv_low.setValue(low)
            self.spn_glv_high.setValue(high)

        dlg = GLVMaskPreviewDialog(
            base_img, glv_low, glv_high,
            parent=self,
            apply_callback=_apply_to_main,
        )
        dlg.exec()

    def _on_show_normalized_compare(self):
        """Show normalized compare image dialog."""
        if not self._results:
            QtWidgets.QMessageBox.information(self, "Info", "Please run Compute first.")
            return
        result = self._results[self._current_result_idx]
        if result.normalized_compare is None or result.aligned_compare is None:
            QtWidgets.QMessageBox.information(self, "Info", "Normalized compare image is not available.")
            return
        base_img = self._images.get(result.base_label)
        if self._norm_compare_dialog is None:
            self._norm_compare_dialog = NormalizedCompareDialog(self)
        self._norm_compare_dialog.set_images(
            result.aligned_compare,
            result.normalized_compare,
            base_img,
            result.base_label,
            result.compare_label,
            result.norm_a,
            result.norm_b,
            method_name=result.normalize_method,
        )
        self._norm_compare_dialog.show()
        self._norm_compare_dialog.raise_()
        self._norm_compare_dialog.activateWindow()

    def _refresh_normalized_compare_dialog(self, result: SinglePairResult):
        if self._norm_compare_dialog is None:
            return
        if not self._norm_compare_dialog.isVisible():
            return
        if result.normalized_compare is None or result.aligned_compare is None:
            return
        base_img = self._images.get(result.base_label)
        self._norm_compare_dialog.set_images(
            result.aligned_compare,
            result.normalized_compare,
            base_img,
            result.base_label,
            result.compare_label,
            result.norm_a,
            result.norm_b,
            method_name=result.normalize_method,
        )

    def _center_crop_image(self, image: Optional[np.ndarray], crop_size: int) -> Optional[np.ndarray]:
        """Center-crop an image to square crop_size; if too small, crop to max possible square."""
        if image is None or image.size == 0:
            return image
        h, w = image.shape[:2]
        side = max(1, min(int(crop_size), h, w))
        cy, cx = h // 2, w // 2
        half = side // 2
        y1 = max(0, cy - half)
        x1 = max(0, cx - half)
        y2 = y1 + side
        x2 = x1 + side
        if y2 > h:
            y2 = h
            y1 = h - side
        if x2 > w:
            x2 = w
            x1 = w - side
        return image[y1:y2, x1:x2]

    def _save_export_images_for_result(
            self,
            result: SinglePairResult,
            export_dirs: Dict[str, str],
            prefix: str,
            center_crop: bool = False,
            crop_size: int = 512,
    ) -> Dict[str, str]:
        """Save per-result images and return saved path mapping."""
        import os

        def _maybe_crop(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if not center_crop:
                return img
            return self._center_crop_image(img, crop_size)

        paths: Dict[str, str] = {}

        base_img = _maybe_crop(self._images.get(result.base_label))
        # Use aligned compare so it shares the same coordinate system as diff / SNR maps
        compare_img = _maybe_crop(
            result.aligned_compare
            if result.aligned_compare is not None
            else self._images.get(result.compare_label)
        )
        diff_img = _maybe_crop(result.result_image)
        # snr_map is computed from result_float (same spatial shape as result_image),
        # so colorized SNR map is pixel-perfectly aligned with diff_img — same coord system.
        snr_img = _maybe_crop(colorize_snr_map(result.snr_map))
        norm_img = _maybe_crop(result.normalized_compare)
        aligned_img = _maybe_crop(result.aligned_compare)

        if base_img is not None:
            base_path = os.path.join(export_dirs["base"], f"{prefix}_base.png")
            cv2.imwrite(base_path, base_img)
            paths["base"] = base_path
        if compare_img is not None:
            compare_path = os.path.join(export_dirs["compare"], f"{prefix}_compare.png")
            cv2.imwrite(compare_path, compare_img)
            paths["compare"] = compare_path

        diff_path = os.path.join(export_dirs["diff"], f"{prefix}_diff.png")
        cv2.imwrite(diff_path, diff_img)
        paths["diff"] = diff_path

        snr_path = os.path.join(export_dirs["snr"], f"{prefix}_snr.png")
        cv2.imwrite(snr_path, snr_img)
        paths["snr"] = snr_path

        if norm_img is not None:
            norm_path = os.path.join(export_dirs["normalized"], f"{prefix}_normalized.png")
            cv2.imwrite(norm_path, norm_img)
            paths["normalized"] = norm_path

        if aligned_img is not None:
            aligned_path = os.path.join(export_dirs["aligned"], f"{prefix}_aligned.png")
            cv2.imwrite(aligned_path, aligned_img)
            paths["aligned"] = aligned_path

        return paths

    def _ask_export_options(self) -> tuple:
        """Show a small dialog asking for export options.
        Returns (do_center_crop: bool, crop_size: int, export_gif: bool).
        """
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Export Options")
        dlg.setStyleSheet(DIALOG_STYLE)
        dlg.resize(380, 190)
        layout = QtWidgets.QVBoxLayout(dlg)

        chk = QtWidgets.QCheckBox("Center-crop images (useful when defect is centered)")
        chk.setChecked(False)
        layout.addWidget(chk)

        crop_row = QtWidgets.QHBoxLayout()
        crop_row.addWidget(QtWidgets.QLabel("Crop size (px):"))
        spn = QtWidgets.QSpinBox()
        spn.setRange(64, 4096)
        spn.setSingleStep(64)
        spn.setValue(512)
        spn.setEnabled(False)
        crop_row.addWidget(spn)
        crop_row.addStretch()
        layout.addLayout(crop_row)
        chk.toggled.connect(spn.setEnabled)

        chk_gif = QtWidgets.QCheckBox(
            "Export animated GIF  (Base → Normalized Compare → Diff loop)"
        )
        chk_gif.setToolTip(
            "Generates a looping animated GIF for each pair — handy for slides/email.\n"
            "Requires Pillow (pip install Pillow)."
        )
        chk_gif.setChecked(False)
        layout.addWidget(chk_gif)

        btn_row = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_row.accepted.connect(dlg.accept)
        btn_row.rejected.connect(dlg.reject)
        layout.addWidget(btn_row)

        if dlg.exec() == QtWidgets.QDialog.Accepted:
            return chk.isChecked(), spn.value(), chk_gif.isChecked()
        return False, 512, False

    def _export_ppt_report(self, out_dir: str, result_rows: List[Dict[str, object]],
                           settings: Dict[str, object],
                           do_center_crop: bool = False,
                           crop_size: int = 512) -> Optional[str]:
        """Build a dark-themed PPT report for all computed image pairs."""
        try:
            from pptx import Presentation
            from pptx.util import Inches, Pt, Emu
            from pptx.enum.text import PP_ALIGN
            from pptx.dml.color import RGBColor
            from pptx.oxml.ns import qn
            from lxml import etree
        except Exception:
            return None

        import datetime

        # ── Color palette ────────────────────────────────────────────────────
        C_BG = RGBColor(0x0F, 0x14, 0x1A)  # #0F141A  almost-black bg
        C_CARD = RGBColor(0x16, 0x1E, 0x2C)  # #161E2C  card bg
        C_PRIMARY = RGBColor(0xF5, 0x9E, 0x0B)  # #F59E0B  amber accent
        C_TEXT = RGBColor(0xE2, 0xE8, 0xF0)  # #E2E8F0  light text
        C_TEXT_SEC = RGBColor(0x8A, 0x99, 0xAA)  # muted secondary
        C_SUCCESS = RGBColor(0x26, 0xD7, 0xAE)  # teal / highlight
        C_WARN = RGBColor(0xEF, 0x44, 0x44)  # red for bad score

        SLIDE_W = Inches(13.33)
        SLIDE_H = Inches(7.5)

        prs = Presentation()
        prs.slide_width = SLIDE_W
        prs.slide_height = SLIDE_H

        def _fill_bg(slide, color: RGBColor):
            """Fill slide background with a solid color via XML."""
            bg = slide.background
            fill = bg.fill
            fill.solid()
            fill.fore_color.rgb = color

        def _add_text(slide, text, left, top, w, h, size=11, bold=False,
                      color=None, align=PP_ALIGN.LEFT, italic=False):
            txb = slide.shapes.add_textbox(left, top, w, h)
            tf = txb.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = align
            run = p.add_run()
            run.text = text
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.italic = italic
            run.font.color.rgb = color or C_TEXT
            return txb

        def _score_color(score: float) -> RGBColor:
            if score >= 75:
                return C_SUCCESS
            if score >= 55:
                return C_PRIMARY
            return C_WARN

        # ── Title slide ──────────────────────────────────────────────────────
        title_slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
        _fill_bg(title_slide, C_BG)

        # Amber bar on left
        bar = title_slide.shapes.add_shape(
            1,  # MSO_SHAPE_TYPE.RECTANGLE
            Inches(0), Inches(0), Inches(0.18), SLIDE_H
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = C_PRIMARY
        bar.line.fill.background()

        _add_text(title_slide, "Perspective Combination",
                  Inches(0.4), Inches(2.6), Inches(9), Inches(0.9),
                  size=36, bold=True, color=C_TEXT)
        _add_text(title_slide, "Multi-Image Comparison Report",
                  Inches(0.4), Inches(3.5), Inches(9), Inches(0.55),
                  size=22, color=C_PRIMARY)
        _add_text(title_slide,
                  f"Generated: {datetime.date.today()}   |   "
                  f"Pairs: {len(result_rows)}   |   "
                  f"Mode: {settings.get('operation', '?').capitalize()}   |   "
                  f"Alignment: {settings.get('alignment_method', '?').capitalize()}",
                  Inches(0.4), Inches(4.3), Inches(12.5), Inches(0.4),
                  size=11, color=C_TEXT_SEC)

        # ── Overview table slide(s) ─────────────────────────────────────────
        ROWS_PER = 18
        for block_start in range(0, len(result_rows), ROWS_PER):
            block = result_rows[block_start:block_start + ROWS_PER]
            ov = prs.slides.add_slide(prs.slide_layouts[6])
            _fill_bg(ov, C_BG)

            page_n = block_start // ROWS_PER + 1
            total_pages = (len(result_rows) + ROWS_PER - 1) // ROWS_PER
            _add_text(ov, f"Overview  —  All Pairs  (page {page_n}/{total_pages})",
                      Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.45),
                      size=16, bold=True, color=C_PRIMARY)

            # Column headers
            cols = ["#", "Base → Compare", "Mode", "dx / dy", "Align", "Diff mean", "Diff std", "Hot px"]
            col_x = [0.4, 0.8, 4.5, 5.5, 6.5, 7.6, 9.0, 10.4]
            col_w = [0.38, 3.65, 0.95, 0.95, 1.05, 1.35, 1.35, 1.9]
            for ci, (ch, cx, cw) in enumerate(zip(cols, col_x, col_w)):
                _add_text(ov, ch, Inches(cx), Inches(0.72), Inches(cw), Inches(0.28),
                          size=9, bold=True, color=C_TEXT_SEC)

            # Separator line
            line = ov.shapes.add_shape(1, Inches(0.4), Inches(1.02), Inches(12.5), Inches(0.02))
            line.fill.solid()
            line.fill.fore_color.rgb = C_TEXT_SEC
            line.line.fill.background()

            row_h = Inches(0.33)
            for ri, row in enumerate(block):
                result: SinglePairResult = row["result"]
                s = result.stats
                y = Inches(1.1) + ri * row_h
                bg_col = C_CARD if ri % 2 == 0 else C_BG
                bg_rect = ov.shapes.add_shape(1, Inches(0.35), y, Inches(12.6), row_h)
                bg_rect.fill.solid()
                bg_rect.fill.fore_color.rgb = bg_col
                bg_rect.line.fill.background()

                global_idx = block_start + ri + 1
                score = result.stats.get('alignment_score', result.alignment.final_score)
                vals = [
                    f"{global_idx:02d}",
                    f"{result.base_label[:18]} → {result.compare_label[:18]}",
                    result.operation,
                    f"({result.alignment.dx:+d}, {result.alignment.dy:+d})",
                    f"{score:.1f}",
                    f"{s.get('diff_mean', 0):.5f}",
                    f"{s.get('diff_std', 0):.5f}",
                    f"{s.get('hot_pixels', 0)}",
                ]
                for ci, (val, cx, cw) in enumerate(zip(vals, col_x, col_w)):
                    col_color = _score_color(score) if ci == 4 else C_TEXT
                    _add_text(ov, val, Inches(cx), y + Inches(0.04), Inches(cw), row_h,
                              size=9, color=col_color)

        # ── Detail slides  ─  3 columns × 2 rows  ────────────────────────────
        #  Row 1: Base Image | Compare Image | Normalized Compare
        #  Row 2: Difference Map | JET Z-Map  | GIF Animation frame
        LABEL_H = Inches(0.26)
        HEADER_H = Inches(0.85)
        LEFT_M = Inches(0.22)
        COL_GAP = Inches(0.09)
        ROW_GAP = Inches(0.09)
        BOT_M = Inches(0.08)

        # 3-column widths (equal)
        IMG_W = (SLIDE_W - 2 * LEFT_M - 2 * COL_GAP) / 3  # ≈ 4.24″
        col_x = [
            LEFT_M,
            LEFT_M + IMG_W + COL_GAP,
            LEFT_M + 2 * (IMG_W + COL_GAP),
        ]

        # 2-row heights (equal)
        total_img_h = (SLIDE_H - HEADER_H - LABEL_H * 2 - ROW_GAP - BOT_M)
        IMG_H = total_img_h / 2
        row_y = [
            HEADER_H + Inches(0.04),
            HEADER_H + Inches(0.04) + LABEL_H + IMG_H + ROW_GAP,
        ]

        # Operation badge colors
        C_SUBTRACT = RGBColor(0x26, 0xD7, 0xAE)  # teal  → Subtract
        C_BLEND = RGBColor(0x60, 0x9C, 0xFF)  # blue  → Blend

        for idx, row in enumerate(result_rows, start=1):
            result: SinglePairResult = row["result"]
            paths: Dict[str, str] = row["paths"]
            s = result.stats
            score = s.get('alignment_score', result.alignment.final_score)

            slide = prs.slides.add_slide(prs.slide_layouts[6])
            _fill_bg(slide, C_BG)

            # ── Header bar ──────────────────────────────────────────────────
            hdr_bar = slide.shapes.add_shape(1, Inches(0), Inches(0), SLIDE_W, HEADER_H)
            hdr_bar.fill.solid()
            hdr_bar.fill.fore_color.rgb = C_CARD
            hdr_bar.line.fill.background()

            # Pair number badge (amber)
            badge = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(0.55), HEADER_H)
            badge.fill.solid()
            badge.fill.fore_color.rgb = C_PRIMARY
            badge.line.fill.background()
            _add_text(slide, f"{idx:02d}",
                      Inches(0.0), Inches(0.18), Inches(0.55), Inches(0.5),
                      size=18, bold=True, color=C_BG, align=PP_ALIGN.CENTER)

            _add_text(slide,
                      f"{result.base_label}  →  {result.compare_label}",
                      Inches(0.65), Inches(0.06), Inches(8.0), Inches(0.38),
                      size=14, bold=True, color=C_TEXT)

            crop_note = f"  crop={crop_size}×{crop_size}px" if do_center_crop else ""
            _add_text(slide,
                      f"align={result.alignment.method}  "
                      f"shift=({result.alignment.dx:+d},{result.alignment.dy:+d})  "
                      f"score={score:.1f}  "
                      f"norm a={result.norm_a:.3f} b={result.norm_b:.2f}  "
                      f"diff μ={s.get('diff_mean', 0):.4f} σ={s.get('diff_std', 0):.4f}  "
                      f"hot={s.get('hot_pixels', 0)}{crop_note}",
                      Inches(0.65), Inches(0.48), Inches(9.6), Inches(0.32),
                      size=8.5, color=C_TEXT_SEC)

            # ── Operation mode badge ─────────────────────────────────────────
            op_label = result.operation.capitalize()
            op_color = C_SUBTRACT if result.operation == 'subtract' else C_BLEND
            op_pill = slide.shapes.add_shape(1, Inches(10.35), Inches(0.08),
                                             Inches(1.25), Inches(0.36))
            op_pill.fill.solid()
            op_pill.fill.fore_color.rgb = op_color
            op_pill.line.fill.background()
            _add_text(slide, op_label,
                      Inches(10.35), Inches(0.10), Inches(1.25), Inches(0.32),
                      size=10, bold=True, color=C_BG, align=PP_ALIGN.CENTER)

            # ── Alignment score colored pill ─────────────────────────────────
            score_c = _score_color(score)
            spill = slide.shapes.add_shape(1, Inches(11.7), Inches(0.08),
                                           Inches(1.45), Inches(0.36))
            spill.fill.solid()
            spill.fill.fore_color.rgb = score_c
            spill.line.fill.background()
            align_status = ("OK" if score >= 75
                            else ("WARN" if score >= 55 else "FAIL"))
            _add_text(slide, f"Align {align_status}  {score:.0f}",
                      Inches(11.7), Inches(0.10), Inches(1.45), Inches(0.32),
                      size=9, bold=True, color=C_BG, align=PP_ALIGN.CENTER)

            # ── 6-image grid  (3 × 2) ────────────────────────────────────────
            gif_label = "GIF Animation (1st frame)" if "gif" in paths else "GIF (not exported)"
            layout_slots = [
                # Top row
                ("base", "Base Image", col_x[0], row_y[0]),
                ("compare", "Aligned Compare", col_x[1], row_y[0]),
                ("normalized", "Normalized Compare", col_x[2], row_y[0]),
                # Bottom row
                ("diff", "Difference Map", col_x[0], row_y[1]),
                ("snr", "JET Z-Map  (SNR)", col_x[1], row_y[1]),
                ("gif", gif_label, col_x[2], row_y[1]),
            ]

            def _add_picture_aspect(slide, path, lft, img_top, avail_w, avail_h):
                """Insert picture centered within (avail_w × avail_h), preserving aspect ratio."""
                probe = cv2.imread(path)
                if probe is None:
                    slide.shapes.add_picture(path, lft, img_top,
                                             width=avail_w, height=avail_h)
                    return
                nat_h, nat_w = probe.shape[:2]
                if nat_h == 0 or nat_w == 0:
                    slide.shapes.add_picture(path, lft, img_top,
                                             width=avail_w, height=avail_h)
                    return
                ar_img = nat_w / nat_h
                ar_box = avail_w / avail_h
                if ar_img >= ar_box:
                    disp_w = avail_w
                    disp_h = int(avail_w / ar_img)
                else:
                    disp_h = avail_h
                    disp_w = int(avail_h * ar_img)
                offset_x = (avail_w - disp_w) // 2
                offset_y = (avail_h - disp_h) // 2
                slide.shapes.add_picture(
                    path,
                    lft + offset_x,
                    img_top + offset_y,
                    width=disp_w,
                    height=disp_h,
                )

            for key, label, lft, top_y in layout_slots:
                if key not in paths:
                    # Empty placeholder with label
                    ph = slide.shapes.add_shape(1, lft, top_y, IMG_W, LABEL_H + IMG_H)
                    ph.fill.solid()
                    ph.fill.fore_color.rgb = C_CARD
                    ph.line.fill.background()
                    _add_text(slide, f"{label}  [n/a]",
                              lft + Inches(0.08), top_y + Inches(0.05),
                              IMG_W - Inches(0.1), LABEL_H,
                              size=9, italic=True, color=C_TEXT_SEC)
                    continue

                # Label bar
                lbl_bg = slide.shapes.add_shape(1, lft, top_y, IMG_W, LABEL_H)
                lbl_bg.fill.solid()
                lbl_bg.fill.fore_color.rgb = C_CARD
                lbl_bg.line.fill.background()
                _add_text(slide, label,
                          lft + Inches(0.08), top_y + Inches(0.01),
                          IMG_W - Inches(0.1), LABEL_H,
                          size=9, bold=True, color=C_PRIMARY)
                # Image — inserted with aspect-ratio preservation
                _add_picture_aspect(slide, paths[key], lft, top_y + LABEL_H, IMG_W, IMG_H)

        ppt_path = str(Path(out_dir) / "perspective_report.pptx")
        prs.save(ppt_path)
        return ppt_path

    def _on_export(self):
        """Export results to PNG files."""
        # ── Quadrant Fusion export ────────────────────────────────────────
        if self._qf_last_result is not None and self.cmb_input_mode.currentIndex() == 1:
            self._on_export_quadrant_fusion()
            return

        if not self._results and self._result is None:
            return

        import csv
        import os

        def _safe_name(text: str) -> str:
            return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)

        if self._results:
            out_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Export Results Folder"
            )
            if not out_dir:
                return
            try:
                summary_path = os.path.join(out_dir, "perspective_summary.csv")
                export_dirs = {
                    "base": os.path.join(out_dir, "base"),
                    "compare": os.path.join(out_dir, "compare"),
                    "diff": os.path.join(out_dir, "diff"),
                    "snr": os.path.join(out_dir, "snr"),
                    "normalized": os.path.join(out_dir, "normalized"),
                    "aligned": os.path.join(out_dir, "aligned"),
                }
                for folder_path in export_dirs.values():
                    os.makedirs(folder_path, exist_ok=True)

                settings = self._last_settings or {}
                # Read export options from UI at export time (not from stale _last_settings)
                do_center_crop, crop_size, export_gif = self._ask_export_options()

                csv_headers = [
                    "base", "compare", "operation", "align_method", "dx", "dy",
                    "align_score", "diff_mean", "diff_std", "hot_pixels",
                    "invert_base", "invert_compare", "invert_result",
                    "alpha", "beta", "normalize", "glv_range", "norm_a", "norm_b",
                    "subtract_mode", "preserve_positive_diff", "abs_no_gain",
                    "center_crop", "crop_size",
                ]
                result_rows: List[Dict[str, object]] = []
                with open(summary_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_headers)
                    for result in self._results:
                        base_name = _safe_name(result.base_label)
                        cmp_name = _safe_name(result.compare_label)
                        op_name = _safe_name(result.operation)
                        prefix = f"{base_name}__{op_name}__{cmp_name}"

                        saved_paths = self._save_export_images_for_result(
                            result, export_dirs, prefix,
                            center_crop=do_center_crop,
                            crop_size=crop_size,
                        )
                        result_rows.append({"result": result, "paths": saved_paths})

                        s = result.stats
                        writer.writerow([
                            result.base_label, result.compare_label, result.operation,
                            result.alignment.method, result.alignment.dx, result.alignment.dy,
                            f"{s.get('alignment_score', 0):.2f}",
                            f"{s.get('diff_mean', 0):.6f}", f"{s.get('diff_std', 0):.6f}",
                            s.get("hot_pixels", 0),
                            settings.get("invert_base", False),
                            settings.get("invert_compare", False),
                            settings.get("invert_result", False),
                            settings.get("alpha", result.blend_alpha),
                            settings.get("beta", result.blend_beta),
                            bool(settings.get("normalize", True)),
                            str(settings.get("glv_range", "")),
                            f"{result.norm_a:.6f}", f"{result.norm_b:.6f}",
                            settings.get("subtract_mode", 0),
                            bool(settings.get("preserve_positive_diff", False)),
                            bool(settings.get("abs_no_gain", False)),
                            do_center_crop, crop_size,
                        ])

                # ── Animated GIF export (before PPT so frames can be embedded) ─
                gif_note = ""
                gif_dir = os.path.join(out_dir, "gif")
                if export_gif:
                    os.makedirs(gif_dir, exist_ok=True)
                    gif_count = 0
                    try:
                        from PIL import Image as _PilImage

                        def _maybe_crop_gif(img):
                            if do_center_crop and img is not None:
                                return self._center_crop_image(img, crop_size)
                            return img

                        for row in result_rows:
                            result = row["result"]
                            base_img_g = self._images.get(result.base_label)
                            norm_img_g = result.normalized_compare
                            diff_img_g = result.result_image

                            frames_np = [
                                _maybe_crop_gif(base_img_g),
                                _maybe_crop_gif(norm_img_g),
                                _maybe_crop_gif(diff_img_g),
                            ]
                            pil_frames = []
                            for fr in frames_np:
                                if fr is None:
                                    continue
                                if fr.dtype != np.uint8:
                                    fr = np.clip(fr, 0, 255).astype(np.uint8)
                                if fr.ndim == 2:
                                    pil_frames.append(
                                        _PilImage.fromarray(fr, mode="L").convert("RGB"))
                                else:
                                    pil_frames.append(_PilImage.fromarray(
                                        cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))

                            if len(pil_frames) >= 2:
                                base_n = _safe_name(result.base_label)
                                cmp_n = _safe_name(result.compare_label)
                                gif_path = os.path.join(gif_dir, f"{base_n}__{cmp_n}.gif")
                                pil_frames[0].save(
                                    gif_path,
                                    save_all=True,
                                    append_images=pil_frames[1:],
                                    loop=0,
                                    duration=700,
                                    optimize=False,
                                )
                                # Store gif path so PPT can embed it
                                row["paths"]["gif"] = gif_path
                                gif_count += 1

                        gif_note = f"\n• {gif_count} GIF(s) → {gif_dir}"
                    except ImportError:
                        gif_note = "\n⚠ GIF skipped — Pillow not installed (pip install Pillow)"
                    except Exception as gif_err:
                        gif_note = f"\n⚠ GIF error: {gif_err}"

                ppt_path = self._export_ppt_report(out_dir, result_rows, settings,
                                                   do_center_crop=do_center_crop,
                                                   crop_size=crop_size)
                if ppt_path:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Export Complete",
                        f"Saved:\n• {out_dir}\n• {summary_path}\n• {ppt_path}{gif_note}",
                    )
                else:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Export Complete",
                        f"Saved:\n• {out_dir}\n• {summary_path}\n\n"
                        f"PPT report was skipped because python-pptx is not available.{gif_note}",
                    )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Error", str(e))
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Difference Map", "perspective_diff.png",
            "PNG (*.png);;TIFF (*.tif)"
        )

        if not path:
            return

        try:
            # Save difference map
            cv2.imwrite(path, self._result.diff_image)

            # Save SNR map
            snr_path = path.replace(".png", "_snr.png").replace(".tif", "_snr.tif")
            snr_color = colorize_snr_map(self._result.snr_map)
            cv2.imwrite(snr_path, snr_color)

            # Save normalized compare image
            norm_path = path.replace(".png", "_normalized.png").replace(".tif", "_normalized.tif")
            saved_paths = [path, snr_path]
            if self._result.normalized_compare is not None:
                cv2.imwrite(norm_path, self._result.normalized_compare)
                saved_paths.append(norm_path)

            # Save aligned compare image
            aligned_path = path.replace(".png", "_aligned.png").replace(".tif", "_aligned.tif")
            if self._result.aligned_compare is not None:
                cv2.imwrite(aligned_path, self._result.aligned_compare)
                saved_paths.append(aligned_path)

            QtWidgets.QMessageBox.information(
                self, "Export Complete",
                "Saved:\n" + "\n".join(f"• {saved}" for saved in saved_paths)
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))

    def _on_export_quadrant_fusion(self):
        """Export Quadrant Fusion result images."""
        import os

        result = self._qf_last_result
        if result is None:
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export Quadrant Fusion Results"
        )
        if not out_dir:
            return

        try:
            paths = []
            failed = []
            for name, img in [
                ("bse_enhanced", result.bse_clean_uint8),
                ("topography", result.topo_uint8),
                ("composite", result.composite_uint8),
                ("fused_output", result.fused_image_uint8),
            ]:
                if img is None or img.size == 0:
                    failed.append(f"{name}: image is empty")
                    continue
                # Ensure contiguous uint8
                out_img = np.ascontiguousarray(img)
                if out_img.dtype != np.uint8:
                    out_img = np.clip(out_img, 0, 255).astype(np.uint8)
                p = os.path.join(out_dir, f"quadrant_fusion_{name}.png")
                ok = cv2.imwrite(p, out_img)
                if ok:
                    paths.append(p)
                else:
                    failed.append(f"{name}: cv2.imwrite failed for {p}")

            # Write a small metadata text file
            meta_path = os.path.join(out_dir, "quadrant_fusion_info.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write("Quadrant Fusion Result\n")
                f.write(f"Alpha used: {result.alpha_used:.6f}\n")
                f.write(f"Beta used: {result.beta_used:.4f}\n")
                f.write(f"Notes: {result.notes}\n")
                for det_name, cmb in self._qf_combos.items():
                    f.write(f"{det_name}: {cmb.currentText()}\n")
            paths.append(meta_path)

            msg = "Saved:\n" + "\n".join(f"  {p}" for p in paths)
            if failed:
                msg += "\n\nWarnings:\n" + "\n".join(f"  {f}" for f in failed)
            QtWidgets.QMessageBox.information(self, "Export Complete", msg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))


__all__ = ['PerspectiveCombinationDialog']
