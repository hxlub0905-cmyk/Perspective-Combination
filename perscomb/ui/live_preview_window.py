"""Live Preview Window — independent floating panel.

Layout (horizontal splitter):
  [Settings Panel] | [BASE image] | [COMPARE image] | [DIFF image]

All three image panels support synchronized zoom-lens hover and independent
pan/zoom via scroll wheel + drag.

Settings in this window are bidirectionally synced with the main dialog via the
``settings_changed`` signal.  The main dialog calls ``apply_settings()`` to push
its current state here without triggering a re-emit loop.
"""
from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, Signal, QPoint, QPointF, QRectF, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush

from ..core.perspective_combine import SinglePairResult
from .design_tokens import Colors, Typography, Spacing, BorderRadius

# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------
C = Colors
T = Typography


# ---------------------------------------------------------------------------
# _PanZoomImageWidget — lightweight zoom/pan viewer for a numpy image
# ---------------------------------------------------------------------------

class _PanZoomImageWidget(QtWidgets.QWidget):
    """Standalone pan/zoom image panel with synchronized cursor overlay.

    Signals
    -------
    cursor_moved(norm_x, norm_y)  — mouse hover position in normalised coords
    cursor_left()                 — mouse left the widget
    """

    cursor_moved = Signal(float, float)
    cursor_left = Signal()

    _MIN_ZOOM = 0.05
    _MAX_ZOOM = 32.0
    _ZOOM_STEP = 1.18   # per wheel notch

    def __init__(self, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._image: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None

        # Zoom / pan state (image-space coordinates)
        self._zoom: float = 1.0
        self._pan: QPointF = QPointF(0.0, 0.0)   # top-left corner of viewport in image pixels
        self._drag_start: Optional[QPoint] = None
        self._pan_at_drag: Optional[QPointF] = None

        # Synced cursor (normalised 0-1)
        self._cursor_norm: Optional[Tuple[float, float]] = None
        self._local_cursor_norm: Optional[Tuple[float, float]] = None

        self.setMinimumSize(220, 220)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setImage(self, image: Optional[np.ndarray]) -> None:
        """Set the image to display (H×W uint8 grayscale or H×W×3 RGB)."""
        self._image = image
        if image is None:
            self._pixmap = None
        else:
            self._pixmap = self._ndarray_to_pixmap(image)
            self._fit_to_widget()
        self.update()

    def setCursorPos(self, norm_x: float, norm_y: float) -> None:
        """Set synced cursor from a partner widget (normalised coords)."""
        self._cursor_norm = (norm_x, norm_y)
        self.update()

    def clearCursor(self) -> None:
        """Hide the synced cursor."""
        self._cursor_norm = None
        self._local_cursor_norm = None
        self.update()

    def resetView(self) -> None:
        """Fit image to widget."""
        self._fit_to_widget()
        self.update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ndarray_to_pixmap(arr: np.ndarray) -> QPixmap:
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 2:
            h, w = arr.shape
            qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, c = arr.shape
            fmt = QImage.Format_RGB888 if c == 3 else QImage.Format_RGBA8888
            qimg = QImage(arr.data, w, h, w * c, fmt)
        return QPixmap.fromImage(qimg.copy())  # .copy() detaches from numpy buffer

    def _fit_to_widget(self) -> None:
        """Compute zoom/pan so image fills the widget while keeping aspect ratio."""
        if self._pixmap is None:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if pw <= 0 or ph <= 0 or ww <= 0 or wh <= 0:
            return
        self._zoom = min(ww / pw, wh / ph)
        # Centre the image
        visible_w = ww / self._zoom
        visible_h = wh / self._zoom
        self._pan = QPointF((pw - visible_w) / 2, (ph - visible_h) / 2)

    def _image_rect_in_widget(self) -> QRectF:
        """Return the QRectF of the (zoomed, panned) image in widget coordinates."""
        if self._pixmap is None:
            return QRectF()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        x = -self._pan.x() * self._zoom
        y = -self._pan.y() * self._zoom
        return QRectF(x, y, pw * self._zoom, ph * self._zoom)

    def _widget_to_norm(self, pos: QPoint) -> Optional[Tuple[float, float]]:
        """Convert widget-space QPoint to normalised image coords (0-1), or None if outside."""
        if self._pixmap is None:
            return None
        r = self._image_rect_in_widget()
        if r.width() <= 0 or r.height() <= 0:
            return None
        nx = (pos.x() - r.x()) / r.width()
        ny = (pos.y() - r.y()) / r.height()
        if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
            return nx, ny
        return None

    def _norm_to_widget(self, norm_x: float, norm_y: float) -> QPointF:
        """Convert normalised image coords to widget-space QPointF."""
        r = self._image_rect_in_widget()
        return QPointF(r.x() + norm_x * r.width(), r.y() + norm_y * r.height())

    def _clamp_pan(self) -> None:
        """Prevent panning beyond image boundaries."""
        if self._pixmap is None:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        max_pan_x = pw - ww / self._zoom
        max_pan_y = ph - wh / self._zoom
        px = max(0.0, min(self._pan.x(), max(0.0, max_pan_x)))
        py = max(0.0, min(self._pan.y(), max(0.0, max_pan_y)))
        self._pan = QPointF(px, py)

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_to_widget()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return

        factor = self._ZOOM_STEP if delta > 0 else 1.0 / self._ZOOM_STEP
        new_zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, self._zoom * factor))

        # Zoom toward the mouse position
        pos = event.position()
        # Image coordinate under cursor before zoom
        img_x = self._pan.x() + pos.x() / self._zoom
        img_y = self._pan.y() + pos.y() / self._zoom

        self._zoom = new_zoom

        # Adjust pan so the same image point stays under cursor
        self._pan = QPointF(img_x - pos.x() / self._zoom,
                            img_y - pos.y() / self._zoom)
        self._clamp_pan()
        self.update()
        event.accept()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
            self._pan_at_drag = QPointF(self._pan)
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_start is not None and event.buttons() & Qt.LeftButton:
            delta = event.pos() - self._drag_start
            self._pan = QPointF(
                self._pan_at_drag.x() - delta.x() / self._zoom,
                self._pan_at_drag.y() - delta.y() / self._zoom,
            )
            self._clamp_pan()
            self.update()
        else:
            norm = self._widget_to_norm(event.pos())
            if norm is not None:
                self._local_cursor_norm = norm
                self.cursor_moved.emit(*norm)
            else:
                self._local_cursor_norm = None
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_start = None
            self._pan_at_drag = None
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Double-click resets zoom to fit."""
        self._fit_to_widget()
        self.update()

    def leaveEvent(self, event) -> None:
        self._local_cursor_norm = None
        self.cursor_left.emit()
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, self._zoom < 1.0)

        # Background
        painter.fillRect(self.rect(), QColor(C.BG_VIEWER))

        # Draw image
        if self._pixmap is not None:
            r = self._image_rect_in_widget()
            painter.drawPixmap(r.toRect(), self._pixmap)

            # Crosshair: prefer local cursor, fall back to synced cursor
            active_norm = self._local_cursor_norm or self._cursor_norm
            if active_norm is not None:
                nx, ny = active_norm
                pt = self._norm_to_widget(nx, ny)
                px, py = pt.x(), pt.y()
                w, h = self.width(), self.height()

                # Crosshair lines
                pen = QPen(QColor(255, 220, 60, 200), 1.0)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(int(px), 0, int(px), h)
                painter.drawLine(0, int(py), w, int(py))

                # Centre dot
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(255, 220, 60, 230)))
                painter.drawEllipse(int(px) - 3, int(py) - 3, 6, 6)
        else:
            # Placeholder text
            painter.setPen(QColor(C.TEXT_MUTED))
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, f"— {self._title} —\n(awaiting preview)")

        painter.end()


# ---------------------------------------------------------------------------
# _ImageColumn — title label + _PanZoomImageWidget stacked vertically
# ---------------------------------------------------------------------------

class _ImageColumn(QtWidgets.QWidget):
    """Vertical stack: bold header label + pan/zoom image viewer."""

    cursor_moved = Signal(float, float)
    cursor_left = Signal()

    _HEADER_COLORS = {
        "BASE":    ("#1D4ED8", "#DBEAFE"),   # blue
        "COMPARE": ("#065F46", "#D1FAE5"),   # green
        "DIFF":    ("#7C2D12", "#FEF3C7"),   # amber/brown
    }

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self._title = title.upper()

        vlay = QtWidgets.QVBoxLayout(self)
        vlay.setContentsMargins(4, 4, 4, 4)
        vlay.setSpacing(4)

        # Header
        fg, bg = self._HEADER_COLORS.get(self._title, (C.TEXT_PRIMARY, C.BG_SUBTLE))
        self._header = QtWidgets.QLabel(self._title)
        self._header.setAlignment(Qt.AlignCenter)
        self._header.setFixedHeight(24)
        self._header.setStyleSheet(
            f"color: {fg}; background: {bg}; font-weight: 700; font-size: 11px;"
            f" border-radius: 4px; letter-spacing: 1px;"
        )
        vlay.addWidget(self._header)

        # Subtitle (pair labels)
        self._subtitle = QtWidgets.QLabel("")
        self._subtitle.setAlignment(Qt.AlignCenter)
        self._subtitle.setStyleSheet(
            f"color: {C.TEXT_MUTED}; font-size: 10px; background: transparent;"
        )
        self._subtitle.setFixedHeight(16)
        vlay.addWidget(self._subtitle)

        # Image viewer
        self.viewer = _PanZoomImageWidget(title, self)
        self.viewer.cursor_moved.connect(self.cursor_moved)
        self.viewer.cursor_left.connect(self.cursor_left)
        vlay.addWidget(self.viewer, stretch=1)

    def setSubtitle(self, text: str) -> None:
        self._subtitle.setText(text)

    def setImage(self, image: Optional[np.ndarray]) -> None:
        self.viewer.setImage(image)

    def setCursorPos(self, nx: float, ny: float) -> None:
        self.viewer.setCursorPos(nx, ny)

    def clearCursor(self) -> None:
        self.viewer.clearCursor()

    def resetView(self) -> None:
        self.viewer.resetView()


# ---------------------------------------------------------------------------
# LivePreviewWindow — the main floating window
# ---------------------------------------------------------------------------

class LivePreviewWindow(QtWidgets.QWidget):
    """Independent floating panel showing Base / Compare / Diff side-by-side.

    Emits
    -----
    settings_changed(dict)
        Fired whenever the user changes a parameter in this window.
        The dict has the same keys as ``_collect_preview_params()`` in the
        main dialog (operation, alpha, beta, invert_*, normalize_method, …).

    closed()
        Fired when the window is closed.
    """

    settings_changed = Signal(dict)
    closed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(
            parent,
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint,
        )
        self.setWindowTitle("Live Preview")
        self.setMinimumSize(900, 500)
        self.resize(1300, 720)

        # Suppress re-emit loops during programmatic sync
        self._syncing: bool = False

        self._build_ui()
        self._connect_internal_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Title bar ────────────────────────────────────────────────
        title_bar = self._make_title_bar()
        root.addWidget(title_bar)

        # ── Main splitter ────────────────────────────────────────────
        self._splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self._splitter.setHandleWidth(5)
        self._splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {C.BORDER_DEFAULT}; }}"
            f"QSplitter::handle:hover {{ background: {C.BRAND_PRIMARY}; }}"
        )

        # Settings panel (left)
        settings_scroll = self._build_settings_panel()
        self._splitter.addWidget(settings_scroll)

        # Image columns
        self._col_base = _ImageColumn("BASE")
        self._col_compare = _ImageColumn("COMPARE")
        self._col_diff = _ImageColumn("DIFF")
        self._splitter.addWidget(self._col_base)
        self._splitter.addWidget(self._col_compare)
        self._splitter.addWidget(self._col_diff)

        # Initial sizes: settings narrow, images equal
        self._splitter.setSizes([200, 350, 350, 350])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 1)
        self._splitter.setStretchFactor(3, 1)

        root.addWidget(self._splitter, stretch=1)

        # ── Status bar ───────────────────────────────────────────────
        status_bar = self._make_status_bar()
        root.addWidget(status_bar)

        # ── Background / stylesheet ───────────────────────────────────
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {C.BG_WINDOW};
                font-family: {T.FONT_FAMILY_SANS};
                font-size: {T.FONT_SIZE_BASE};
                color: {C.TEXT_PRIMARY};
            }}
            QLabel {{ background: transparent; border: none; }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {C.BG_INPUT};
                border: 1px solid {C.BORDER_DEFAULT};
                border-radius: 4px;
                padding: 2px 6px;
                min-height: 24px;
            }}
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {C.BORDER_ACTIVE};
            }}
            QCheckBox {{ spacing: 6px; }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border: 1px solid {C.BORDER_DEFAULT};
                border-radius: 3px;
                background: {C.BG_INPUT};
            }}
            QCheckBox::indicator:checked {{
                background: {C.BRAND_PRIMARY};
                border-color: {C.BRAND_PRIMARY};
            }}
            QScrollArea {{ border: none; }}
        """)

    def _make_title_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        bar.setFixedHeight(42)
        bar.setStyleSheet(
            f"background: {C.BG_PANEL}; border-bottom: 1px solid {C.BORDER_DEFAULT};"
        )
        hlay = QtWidgets.QHBoxLayout(bar)
        hlay.setContentsMargins(12, 0, 12, 0)
        hlay.setSpacing(10)

        icon = QtWidgets.QLabel("\u26a1")
        icon.setStyleSheet(f"font-size: 16px; color: {C.BRAND_PRIMARY};")
        hlay.addWidget(icon)

        title = QtWidgets.QLabel("Live Preview")
        title.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {C.TEXT_PRIMARY};"
        )
        hlay.addWidget(title)

        self._lbl_pair = QtWidgets.QLabel("— no preview yet —")
        self._lbl_pair.setStyleSheet(
            f"font-size: 11px; color: {C.TEXT_SECONDARY};"
        )
        hlay.addWidget(self._lbl_pair)

        hlay.addStretch()

        # Reset views button
        btn_reset = QtWidgets.QPushButton("Reset zoom")
        btn_reset.setFixedHeight(26)
        btn_reset.setStyleSheet(
            f"background: {C.BG_SUBTLE}; border: 1px solid {C.BORDER_DEFAULT};"
            f" border-radius: 4px; padding: 0 10px; font-size: 11px;"
            f" color: {C.TEXT_SECONDARY};"
        )
        btn_reset.clicked.connect(self._on_reset_zoom)
        hlay.addWidget(btn_reset)

        return bar

    def _make_status_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        bar.setFixedHeight(28)
        bar.setStyleSheet(
            f"background: {C.BG_PANEL}; border-top: 1px solid {C.BORDER_DEFAULT};"
        )
        hlay = QtWidgets.QHBoxLayout(bar)
        hlay.setContentsMargins(12, 0, 12, 0)
        hlay.setSpacing(18)

        def _stat_lbl() -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel("")
            lbl.setStyleSheet(
                f"font-size: 10px; color: {C.TEXT_SECONDARY};"
                f" font-family: {T.FONT_FAMILY_MONO};"
            )
            return lbl

        self._lbl_align = _stat_lbl()
        self._lbl_snr = _stat_lbl()
        self._lbl_dxdy = _stat_lbl()
        hlay.addWidget(self._lbl_align)
        hlay.addWidget(self._lbl_snr)
        hlay.addWidget(self._lbl_dxdy)
        hlay.addStretch()

        self._lbl_state = QtWidgets.QLabel("\u25cf  Ready")
        self._lbl_state.setStyleSheet(
            f"font-size: 10px; color: {C.SUCCESS};"
        )
        hlay.addWidget(self._lbl_state)
        return bar

    # ------------------------------------------------------------------
    # Settings panel
    # ------------------------------------------------------------------

    def _build_settings_panel(self) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumWidth(180)
        scroll.setMaximumWidth(260)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        container = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(container)
        vlay.setContentsMargins(10, 10, 10, 10)
        vlay.setSpacing(14)

        # Section header helper
        def _section(label: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet(
                f"color: {C.TEXT_SECONDARY}; font-size: 10px; font-weight: 700;"
                f" letter-spacing: 1px; padding-top: 4px;"
            )
            return lbl

        def _row_label(text: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(f"font-size: 11px; color: {C.TEXT_SECONDARY};")
            return lbl

        # ── OPERATION ────────────────────────────────────────────────
        vlay.addWidget(_section("OPERATION"))

        vlay.addWidget(_row_label("Mode"))
        self._cmb_operation = QtWidgets.QComboBox()
        self._cmb_operation.addItems(["Subtract", "Blend"])
        vlay.addWidget(self._cmb_operation)

        # Blend weights (only visible in Blend mode)
        self._wgt_blend = QtWidgets.QWidget()
        blend_lay = QtWidgets.QFormLayout(self._wgt_blend)
        blend_lay.setContentsMargins(0, 0, 0, 0)
        blend_lay.setSpacing(4)
        self._spn_alpha = QtWidgets.QDoubleSpinBox()
        self._spn_alpha.setRange(0.0, 1.0)
        self._spn_alpha.setSingleStep(0.05)
        self._spn_alpha.setDecimals(2)
        self._spn_alpha.setValue(0.5)
        self._spn_beta = QtWidgets.QDoubleSpinBox()
        self._spn_beta.setRange(0.0, 1.0)
        self._spn_beta.setSingleStep(0.05)
        self._spn_beta.setDecimals(2)
        self._spn_beta.setValue(0.5)
        blend_lay.addRow(_row_label("α (Base)"), self._spn_alpha)
        blend_lay.addRow(_row_label("β (Cmp)"), self._spn_beta)
        vlay.addWidget(self._wgt_blend)

        # Subtract mode (only visible in Subtract mode)
        self._wgt_sub_mode = QtWidgets.QWidget()
        sub_lay = QtWidgets.QVBoxLayout(self._wgt_sub_mode)
        sub_lay.setContentsMargins(0, 0, 0, 0)
        sub_lay.setSpacing(4)
        sub_lay.addWidget(_row_label("Subtract mode"))
        self._cmb_subtract_mode = QtWidgets.QComboBox()
        self._cmb_subtract_mode.addItems([
            "|diff| × 2  (default)",
            "|diff|  (abs, no gain)",
            "clip ≥ 0  (keep dir.)",
        ])
        sub_lay.addWidget(self._cmb_subtract_mode)
        vlay.addWidget(self._wgt_sub_mode)

        # ── INVERT ───────────────────────────────────────────────────
        vlay.addWidget(_section("INVERT"))
        self._chk_inv_base = QtWidgets.QCheckBox("Base")
        self._chk_inv_compare = QtWidgets.QCheckBox("Compare")
        self._chk_inv_result = QtWidgets.QCheckBox("Result")
        for chk in (self._chk_inv_base, self._chk_inv_compare, self._chk_inv_result):
            chk.setStyleSheet(f"font-size: 11px; color: {C.TEXT_PRIMARY};")
            vlay.addWidget(chk)

        # ── NORMALIZATION ─────────────────────────────────────────────
        vlay.addWidget(_section("NORMALIZATION"))
        vlay.addWidget(_row_label("Method"))
        self._cmb_norm_mode = QtWidgets.QComboBox()
        self._cmb_norm_mode.addItems([
            "Percentile (P2–P98)",
            "GLV-Mask",
            "Skip (raw ÷ 255)",
            "ROI-Match (EPI Nulling)",
        ])
        vlay.addWidget(self._cmb_norm_mode)

        # GLV controls (only visible for GLV-Mask mode)
        self._wgt_glv = QtWidgets.QWidget()
        glv_lay = QtWidgets.QFormLayout(self._wgt_glv)
        glv_lay.setContentsMargins(0, 0, 0, 0)
        glv_lay.setSpacing(4)
        self._spn_glv_low = QtWidgets.QSpinBox()
        self._spn_glv_low.setRange(0, 254)
        self._spn_glv_low.setValue(100)
        self._spn_glv_high = QtWidgets.QSpinBox()
        self._spn_glv_high.setRange(1, 255)
        self._spn_glv_high.setValue(160)
        glv_lay.addRow(_row_label("GLV Min"), self._spn_glv_low)
        glv_lay.addRow(_row_label("GLV Max"), self._spn_glv_high)
        vlay.addWidget(self._wgt_glv)

        # ── ALIGNMENT ─────────────────────────────────────────────────
        vlay.addWidget(_section("ALIGNMENT"))
        vlay.addWidget(_row_label("Method"))
        self._cmb_align_method = QtWidgets.QComboBox()
        self._cmb_align_method.addItems(["Phase (robust)", "NCC (brute force)"])
        vlay.addWidget(self._cmb_align_method)

        vlay.addStretch()

        scroll.setWidget(container)
        self._update_op_visibility()
        self._update_norm_visibility()
        return scroll

    # ------------------------------------------------------------------
    # Internal signal wiring
    # ------------------------------------------------------------------

    def _connect_internal_signals(self) -> None:
        """Connect settings widgets to the _on_setting_changed handler."""
        self._cmb_operation.currentIndexChanged.connect(self._on_op_mode_changed)

        _tier2 = [
            (self._cmb_operation, "currentIndexChanged"),
            (self._spn_alpha, "valueChanged"),
            (self._spn_beta, "valueChanged"),
            (self._chk_inv_base, "stateChanged"),
            (self._chk_inv_compare, "stateChanged"),
            (self._chk_inv_result, "stateChanged"),
            (self._cmb_norm_mode, "currentIndexChanged"),
            (self._spn_glv_low, "valueChanged"),
            (self._spn_glv_high, "valueChanged"),
            (self._cmb_subtract_mode, "currentIndexChanged"),
        ]
        for widget, sig in _tier2:
            getattr(widget, sig).connect(lambda *_: self._on_setting_changed(False))

        self._cmb_align_method.currentIndexChanged.connect(
            lambda *_: self._on_setting_changed(True)
        )
        self._cmb_norm_mode.currentIndexChanged.connect(self._update_norm_visibility)

        # Cross-panel cursor sync
        for src, others in [
            (self._col_base, [self._col_compare, self._col_diff]),
            (self._col_compare, [self._col_base, self._col_diff]),
            (self._col_diff, [self._col_base, self._col_compare]),
        ]:
            def _make_move(targets):
                def _on_move(nx, ny):
                    for t in targets:
                        t.setCursorPos(nx, ny)
                return _on_move

            def _make_leave(targets):
                def _on_leave():
                    for t in targets:
                        t.clearCursor()
                return _on_leave

            src.cursor_moved.connect(_make_move(others))
            src.cursor_left.connect(_make_leave(others))

    # ------------------------------------------------------------------
    # Visibility helpers
    # ------------------------------------------------------------------

    def _update_op_visibility(self) -> None:
        is_blend = self._cmb_operation.currentIndex() == 1
        self._wgt_blend.setVisible(is_blend)
        self._wgt_sub_mode.setVisible(not is_blend)

    def _update_norm_visibility(self) -> None:
        is_glv = self._cmb_norm_mode.currentIndex() == 1
        self._wgt_glv.setVisible(is_glv)

    def _on_op_mode_changed(self) -> None:
        self._update_op_visibility()

    # ------------------------------------------------------------------
    # Settings I/O
    # ------------------------------------------------------------------

    def _on_setting_changed(self, is_align_change: bool = False) -> None:
        """Emit settings_changed with current state unless we're in a sync loop."""
        if self._syncing:
            return
        d = self._collect_settings()
        d["_is_align_change"] = is_align_change
        self.settings_changed.emit(d)

    def _collect_settings(self) -> dict:
        """Return current settings as a dict (matches keys used by dialog)."""
        norm_idx = self._cmb_norm_mode.currentIndex()
        _method_map = {0: "percentile", 1: "glv_mask", 2: "skip", 3: "roi_match"}
        normalize_method = _method_map.get(norm_idx, "percentile")
        normalize = normalize_method not in ("skip", "roi_match")

        glv_range = None
        if norm_idx == 1:
            lo, hi = self._spn_glv_low.value(), self._spn_glv_high.value()
            if lo < hi:
                glv_range = (lo, hi)

        sub_idx = self._cmb_subtract_mode.currentIndex()
        return {
            "operation": "subtract" if self._cmb_operation.currentIndex() == 0 else "blend",
            "alpha": self._spn_alpha.value(),
            "beta": self._spn_beta.value(),
            "invert_base": self._chk_inv_base.isChecked(),
            "invert_compare": self._chk_inv_compare.isChecked(),
            "invert_result": self._chk_inv_result.isChecked(),
            "normalize_method": normalize_method,
            "normalize": normalize,
            "glv_range": glv_range,
            "norm_mode_index": norm_idx,
            "subtract_mode_index": sub_idx,
            "preserve_positive_diff": sub_idx == 2,
            "abs_no_gain": sub_idx == 1,
            "align_method": "phase" if self._cmb_align_method.currentIndex() == 0 else "ncc",
        }

    def apply_settings(self, s: dict) -> None:
        """Apply settings from the main dialog without triggering settings_changed."""
        self._syncing = True
        try:
            if "operation" in s:
                self._cmb_operation.setCurrentIndex(0 if s["operation"] == "subtract" else 1)
                self._update_op_visibility()
            if "alpha" in s:
                self._spn_alpha.setValue(s["alpha"])
            if "beta" in s:
                self._spn_beta.setValue(s["beta"])
            if "invert_base" in s:
                self._chk_inv_base.setChecked(s["invert_base"])
            if "invert_compare" in s:
                self._chk_inv_compare.setChecked(s["invert_compare"])
            if "invert_result" in s:
                self._chk_inv_result.setChecked(s["invert_result"])
            if "norm_mode_index" in s:
                self._cmb_norm_mode.setCurrentIndex(s["norm_mode_index"])
                self._update_norm_visibility()
            if "glv_range" in s and s["glv_range"]:
                self._spn_glv_low.setValue(s["glv_range"][0])
                self._spn_glv_high.setValue(s["glv_range"][1])
            if "subtract_mode_index" in s:
                self._cmb_subtract_mode.setCurrentIndex(s["subtract_mode_index"])
            if "align_method" in s:
                self._cmb_align_method.setCurrentIndex(
                    0 if s["align_method"] == "phase" else 1
                )
        finally:
            self._syncing = False

    # ------------------------------------------------------------------
    # Preview update
    # ------------------------------------------------------------------

    def update_preview(self, result: SinglePairResult) -> None:
        """Push a new SinglePairResult into all three image columns."""
        # Title bar pair label
        self._lbl_pair.setText(
            f"Base: {result.base_label}   \u2502   Compare: {result.compare_label}"
        )

        # Images
        base_img = result.normalized_base
        cmp_img = result.normalized_compare
        diff_img = result.result_image

        self._col_base.setImage(base_img)
        self._col_base.setSubtitle(result.base_label)

        self._col_compare.setImage(cmp_img)
        self._col_compare.setSubtitle(result.compare_label)

        self._col_diff.setImage(diff_img)
        self._col_diff.setSubtitle(result.operation)

        # Status bar
        aln = result.alignment
        score = getattr(aln, "final_score", 0.0)
        dx = getattr(aln, "dx_subpixel", getattr(aln, "dx", 0.0))
        dy = getattr(aln, "dy_subpixel", getattr(aln, "dy", 0.0))
        snr = result.stats.get("snr_peak", 0.0)
        self._lbl_align.setText(f"Align: {score:.3f}")
        self._lbl_snr.setText(f"SNR: {snr:.2f}")
        self._lbl_dxdy.setText(f"dx {dx:+.1f}  dy {dy:+.1f}")
        self._set_state_ready()

    def set_state_computing(self) -> None:
        self._lbl_state.setText("\u23f3  Computing…")
        self._lbl_state.setStyleSheet(
            f"font-size: 10px; color: {C.BRAND_PRIMARY};"
        )

    def _set_state_ready(self) -> None:
        self._lbl_state.setText("\u25cf  Preview ready")
        self._lbl_state.setStyleSheet(
            f"font-size: 10px; color: {C.SUCCESS};"
        )

    def set_state_error(self, msg: str) -> None:
        self._lbl_state.setText(f"\u26a0  {msg[:60]}")
        self._lbl_state.setStyleSheet(
            f"font-size: 10px; color: {C.WARNING};"
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _on_reset_zoom(self) -> None:
        for col in (self._col_base, self._col_compare, self._col_diff):
            col.resetView()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.closed.emit()
        super().closeEvent(event)
