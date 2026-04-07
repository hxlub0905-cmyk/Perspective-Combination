"""Histogram Matcher — Fusi³

Standalone tool for batch histogram matching of grayscale SEM images.
Select a base (reference) image and match the histogram of multiple
target images to it, producing outputs with similar brightness/contrast.

Usage:
    python histogram_match_tool.py
"""

import sys
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Design tokens (imported from main project)
# ---------------------------------------------------------------------------
from perscomb.ui.design_tokens import (
    Colors, Typography, Spacing, BorderRadius, Sizing, Shadows,
    create_gradient,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_FILTERS = "Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg);;All Files (*)"
DEFAULT_SUFFIX = "_matched"
THUMB_SIZE = 200  # preview thumbnail edge length


# ============================================================================
# Core: histogram matching (pure numpy, no extra dependency)
# ============================================================================

def match_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match the histogram of *source* to that of *reference*.

    Supports 8-bit and 16-bit grayscale images.  The output has the same
    dtype as the source.
    """
    src_dtype = source.dtype
    if src_dtype == np.uint16:
        max_val = 65536
    else:
        max_val = 256

    # Compute CDFs
    src_hist, _ = np.histogram(source.flatten(), bins=max_val, range=(0, max_val))
    ref_hist, _ = np.histogram(reference.flatten(), bins=max_val, range=(0, max_val))

    src_cdf = src_hist.cumsum().astype(np.float64)
    ref_cdf = ref_hist.cumsum().astype(np.float64)

    src_cdf /= src_cdf[-1]
    ref_cdf /= ref_cdf[-1]

    # Build look-up table: for each source intensity find closest ref intensity
    mapping = np.interp(src_cdf, ref_cdf, np.arange(max_val)).astype(src_dtype)

    return mapping[source]


def compute_histogram(image: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Return (counts, bin_edges) for a grayscale image."""
    max_val = 65536 if image.dtype == np.uint16 else 256
    counts, edges = np.histogram(image.flatten(), bins=bins, range=(0, max_val))
    return counts, edges


# ============================================================================
# Widgets
# ============================================================================

def _global_stylesheet() -> str:
    """Return application-wide QSS."""
    return f"""
        QWidget {{
            font-family: {Typography.FONT_FAMILY};
            font-size: {Typography.FONT_SIZE_BODY};
            color: {Colors.TEXT_PRIMARY};
        }}
        QMainWindow {{
            background: {Colors.BG_WINDOW};
        }}
        QGroupBox {{
            font-weight: {Typography.FONT_WEIGHT_SEMIBOLD};
            font-size: {Typography.FONT_SIZE_H3};
            border: 1px solid {Colors.BORDER_DEFAULT};
            border-radius: {BorderRadius.MD};
            margin-top: 14px;
            padding: {Spacing.GROUPBOX_PADDING};
            background: {Colors.BG_PANEL};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
        }}
        QPushButton {{
            background: {Colors.BG_PANEL};
            border: 1px solid {Colors.BORDER_DEFAULT};
            border-radius: {BorderRadius.SM};
            padding: {Spacing.BUTTON_PADDING};
            min-height: {Sizing.BUTTON_HEIGHT_MD};
            font-weight: {Typography.FONT_WEIGHT_MEDIUM};
        }}
        QPushButton:hover {{
            border-color: {Colors.BORDER_HOVER};
            background: {Colors.BG_SUBTLE};
        }}
        QPushButton:pressed {{
            background: {Colors.BRAND_PRIMARY_SOFT};
        }}
        QPushButton#primary {{
            background: {Colors.BRAND_PRIMARY};
            color: {Colors.TEXT_ON_PRIMARY};
            border: none;
            font-weight: {Typography.FONT_WEIGHT_BOLD};
            font-size: {Typography.FONT_SIZE_H3};
            min-height: {Sizing.BUTTON_HEIGHT_LG};
        }}
        QPushButton#primary:hover {{
            background: {Colors.BRAND_PRIMARY_HOVER};
        }}
        QPushButton#primary:pressed {{
            background: {Colors.BRAND_PRIMARY_PRESSED};
        }}
        QPushButton#primary:disabled {{
            background: {Colors.TEXT_MUTED};
        }}
        QLineEdit {{
            border: 1px solid {Colors.BORDER_DEFAULT};
            border-radius: {BorderRadius.SM};
            padding: {Spacing.INPUT_PADDING};
            min-height: {Sizing.INPUT_HEIGHT};
            background: {Colors.BG_INPUT};
        }}
        QLineEdit:focus {{
            border-color: {Colors.BORDER_FOCUS};
        }}
        QListWidget {{
            border: 1px solid {Colors.BORDER_DEFAULT};
            border-radius: {BorderRadius.SM};
            background: {Colors.BG_INPUT};
            padding: 4px;
        }}
        QListWidget::item {{
            padding: 3px 6px;
            border-radius: 4px;
        }}
        QListWidget::item:selected {{
            background: {Colors.BRAND_PRIMARY_SOFT};
            color: {Colors.TEXT_PRIMARY};
        }}
        QProgressBar {{
            border: 1px solid {Colors.BORDER_DEFAULT};
            border-radius: {BorderRadius.SM};
            text-align: center;
            height: 22px;
            background: {Colors.BG_SUBTLE};
        }}
        QProgressBar::chunk {{
            background: {Colors.BRAND_PRIMARY};
            border-radius: {BorderRadius.SM};
        }}
        QLabel#sectionTitle {{
            font-weight: {Typography.FONT_WEIGHT_SEMIBOLD};
            font-size: {Typography.FONT_SIZE_H3};
        }}
        QLabel#muted {{
            color: {Colors.TEXT_MUTED};
            font-size: {Typography.FONT_SIZE_SMALL};
        }}
    """


class HistogramCanvas(FigureCanvas):
    """Matplotlib canvas that draws before / after histograms."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        self._fig = Figure(figsize=(5, 2.4), dpi=100)
        self._fig.patch.set_facecolor(Colors.BG_PANEL)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setMinimumHeight(180)

    # -- public API ----------------------------------------------------------

    def plot_single(self, image: np.ndarray, title: str = "Histogram",
                    color: str = Colors.BRAND_PRIMARY) -> None:
        """Plot histogram of a single image."""
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        counts, edges = compute_histogram(image)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.fill_between(centers, counts, alpha=0.55, color=color)
        ax.plot(centers, counts, linewidth=0.8, color=color)
        ax.set_title(title, fontsize=10, fontweight="semibold",
                     color=Colors.TEXT_PRIMARY)
        ax.set_xlabel("Intensity", fontsize=9, color=Colors.TEXT_SECONDARY)
        ax.set_ylabel("Count", fontsize=9, color=Colors.TEXT_SECONDARY)
        ax.tick_params(labelsize=8, colors=Colors.TEXT_SECONDARY)
        self._fig.tight_layout()
        self.draw()

    def plot_comparison(self, before: np.ndarray, after: np.ndarray,
                        ref: np.ndarray, name: str = "") -> None:
        """Plot before / after / reference histograms side-by-side."""
        self._fig.clear()

        ax1 = self._fig.add_subplot(121)
        ax2 = self._fig.add_subplot(122)

        for ax, img, title, color in [
            (ax1, before, "Before", Colors.TEXT_MUTED),
            (ax2, after, "After matching", Colors.BRAND_PRIMARY),
        ]:
            counts, edges = compute_histogram(img)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.fill_between(centers, counts, alpha=0.45, color=color)
            ax.plot(centers, counts, linewidth=0.8, color=color)

            # overlay reference histogram as dashed line
            rc, re = compute_histogram(ref)
            rc_centers = 0.5 * (re[:-1] + re[1:])
            ax.plot(rc_centers, rc, linewidth=0.9, color=Colors.SUCCESS,
                    linestyle="--", alpha=0.7, label="Base")

            ax.set_title(title, fontsize=10, fontweight="semibold",
                         color=Colors.TEXT_PRIMARY)
            ax.tick_params(labelsize=7, colors=Colors.TEXT_SECONDARY)
            ax.legend(fontsize=7, loc="upper right")

        if name:
            self._fig.suptitle(name, fontsize=9, color=Colors.TEXT_SECONDARY,
                               y=0.01)
        self._fig.tight_layout()
        self.draw()


class ImageThumbLabel(QtWidgets.QLabel):
    """A small label that displays a grayscale thumbnail."""

    def __init__(self, parent=None, size: int = THUMB_SIZE):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet(
            f"border: 1px solid {Colors.BORDER_DEFAULT}; "
            f"border-radius: {BorderRadius.SM}; "
            f"background: {Colors.BG_VIEWER};"
        )
        self._set_placeholder()

    def _set_placeholder(self):
        self.setText("No image")
        self.setStyleSheet(self.styleSheet() + f" color: {Colors.TEXT_MUTED};")

    def set_image(self, img: np.ndarray) -> None:
        """Display a grayscale numpy array as a thumbnail."""
        h, w = img.shape[:2]
        scale = self._size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Normalize to 8-bit for display
        if resized.dtype == np.uint16:
            disp = (resized / 256).astype(np.uint8)
        else:
            disp = resized.astype(np.uint8)

        qimg = QtGui.QImage(disp.data, new_w, new_h, new_w,
                            QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(pixmap)


# ============================================================================
# Main Window
# ============================================================================

class HistogramMatchWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histogram Matcher \u2014 Fusi\u00b3")
        self.setMinimumSize(960, 720)

        self._base_path: Optional[str] = None
        self._base_image: Optional[np.ndarray] = None
        self._target_paths: list[str] = []
        self._output_dir: Optional[str] = None
        self._results: list[tuple[str, np.ndarray, np.ndarray]] = []  # (name, before, after)

        self._build_ui()

    # -- UI construction -----------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ---- Title bar ----
        title_bar = QtWidgets.QHBoxLayout()
        title_lbl = QtWidgets.QLabel("Histogram Matcher")
        title_lbl.setStyleSheet(
            f"font-size: {Typography.FONT_SIZE_H1}; "
            f"font-weight: {Typography.FONT_WEIGHT_BOLD}; "
            f"color: {Colors.BRAND_PRIMARY};"
        )
        subtitle_lbl = QtWidgets.QLabel("Match target image histograms to a base reference")
        subtitle_lbl.setObjectName("muted")
        title_bar.addWidget(title_lbl)
        title_bar.addSpacing(12)
        title_bar.addWidget(subtitle_lbl)
        title_bar.addStretch()
        root.addLayout(title_bar)

        # ---- Top accent line ----
        accent = QtWidgets.QFrame()
        accent.setFixedHeight(3)
        accent.setStyleSheet(f"background: {Colors.BRAND_PRIMARY}; border: none;")
        root.addWidget(accent)

        # ---- Content: left panel + right preview ----
        content = QtWidgets.QHBoxLayout()
        content.setSpacing(16)
        root.addLayout(content, stretch=1)

        # LEFT: controls
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)
        content.addLayout(left, stretch=0)

        # -- Base image group --
        base_grp = QtWidgets.QGroupBox("Base Image (Reference)")
        base_lay = QtWidgets.QHBoxLayout(base_grp)
        self._base_thumb = ImageThumbLabel(size=120)
        base_lay.addWidget(self._base_thumb)

        base_right = QtWidgets.QVBoxLayout()
        self._base_path_lbl = QtWidgets.QLabel("No file selected")
        self._base_path_lbl.setObjectName("muted")
        self._base_path_lbl.setWordWrap(True)
        base_right.addWidget(self._base_path_lbl)
        btn_base = QtWidgets.QPushButton("Browse...")
        btn_base.setFixedWidth(120)
        btn_base.clicked.connect(self._on_browse_base)
        base_right.addWidget(btn_base)
        base_right.addStretch()
        base_lay.addLayout(base_right, stretch=1)
        left.addWidget(base_grp)

        # -- Target images group --
        target_grp = QtWidgets.QGroupBox("Target Images")
        target_lay = QtWidgets.QVBoxLayout(target_grp)

        self._target_list = QtWidgets.QListWidget()
        self._target_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self._target_list.setMinimumHeight(120)
        target_lay.addWidget(self._target_list)

        tgt_btns = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("Add Files...")
        btn_add.clicked.connect(self._on_add_targets)
        btn_remove = QtWidgets.QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._on_remove_targets)
        btn_clear = QtWidgets.QPushButton("Clear All")
        btn_clear.clicked.connect(self._on_clear_targets)
        for b in (btn_add, btn_remove, btn_clear):
            tgt_btns.addWidget(b)
        target_lay.addLayout(tgt_btns)
        left.addWidget(target_grp)

        # -- Output settings group --
        out_grp = QtWidgets.QGroupBox("Output Settings")
        out_lay = QtWidgets.QFormLayout(out_grp)

        out_dir_row = QtWidgets.QHBoxLayout()
        self._out_dir_edit = QtWidgets.QLineEdit()
        self._out_dir_edit.setPlaceholderText("Same as base image folder")
        self._out_dir_edit.setReadOnly(True)
        btn_out = QtWidgets.QPushButton("Browse...")
        btn_out.setFixedWidth(90)
        btn_out.clicked.connect(self._on_browse_output)
        out_dir_row.addWidget(self._out_dir_edit, stretch=1)
        out_dir_row.addWidget(btn_out)
        out_lay.addRow("Output folder:", out_dir_row)

        self._suffix_edit = QtWidgets.QLineEdit(DEFAULT_SUFFIX)
        self._suffix_edit.setMaximumWidth(200)
        out_lay.addRow("Filename suffix:", self._suffix_edit)
        left.addWidget(out_grp)

        # -- Run button --
        self._btn_run = QtWidgets.QPushButton("\u25b6  Run Matching")
        self._btn_run.setObjectName("primary")
        self._btn_run.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_run.clicked.connect(self._on_run)
        left.addWidget(self._btn_run)

        # -- Progress --
        self._progress = QtWidgets.QProgressBar()
        self._progress.setValue(0)
        self._progress.setVisible(False)
        left.addWidget(self._progress)

        self._status_lbl = QtWidgets.QLabel("")
        self._status_lbl.setObjectName("muted")
        left.addWidget(self._status_lbl)

        left.addStretch()

        # RIGHT: preview area
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(8)
        content.addLayout(right, stretch=1)

        # Base histogram
        self._base_canvas = HistogramCanvas()
        right.addWidget(self._base_canvas, stretch=1)

        # Result comparison
        self._result_canvas = HistogramCanvas()
        right.addWidget(self._result_canvas, stretch=1)

        # Navigation for results
        nav = QtWidgets.QHBoxLayout()
        self._btn_prev = QtWidgets.QPushButton("\u25c0  Prev")
        self._btn_next = QtWidgets.QPushButton("Next  \u25b6")
        self._result_label = QtWidgets.QLabel("")
        self._result_label.setAlignment(QtCore.Qt.AlignCenter)
        self._result_label.setObjectName("sectionTitle")
        self._btn_prev.clicked.connect(self._show_prev_result)
        self._btn_next.clicked.connect(self._show_next_result)
        nav.addWidget(self._btn_prev)
        nav.addStretch()
        nav.addWidget(self._result_label)
        nav.addStretch()
        nav.addWidget(self._btn_next)
        right.addLayout(nav)

        self._result_idx = 0
        self._update_nav_state()

    # -- Slots: file selection ------------------------------------------------

    def _on_browse_base(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Base Image", "", IMAGE_FILTERS
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Error",
                                          f"Cannot read image:\n{path}")
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self._base_path = path
        self._base_image = img
        self._base_path_lbl.setText(Path(path).name)
        self._base_thumb.set_image(img)
        self._base_canvas.plot_single(img, title="Base Histogram",
                                       color=Colors.BRAND_PRIMARY)

        # default output dir
        if not self._output_dir:
            self._out_dir_edit.setPlaceholderText(str(Path(path).parent))

    def _on_add_targets(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Target Images", "", IMAGE_FILTERS
        )
        for p in paths:
            if p not in self._target_paths and p != self._base_path:
                self._target_paths.append(p)
                self._target_list.addItem(Path(p).name)

    def _on_remove_targets(self):
        for item in reversed(self._target_list.selectedItems()):
            idx = self._target_list.row(item)
            self._target_list.takeItem(idx)
            self._target_paths.pop(idx)

    def _on_clear_targets(self):
        self._target_list.clear()
        self._target_paths.clear()

    def _on_browse_output(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder"
        )
        if d:
            self._output_dir = d
            self._out_dir_edit.setText(d)

    # -- Run matching ---------------------------------------------------------

    def _on_run(self):
        if self._base_image is None:
            QtWidgets.QMessageBox.warning(self, "Error",
                                          "Please select a base image first.")
            return
        if not self._target_paths:
            QtWidgets.QMessageBox.warning(self, "Error",
                                          "Please add at least one target image.")
            return

        out_dir = self._output_dir or str(Path(self._base_path).parent)
        suffix = self._suffix_edit.text().strip() or DEFAULT_SUFFIX

        self._results.clear()
        self._result_idx = 0
        total = len(self._target_paths)
        self._progress.setMaximum(total)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._btn_run.setEnabled(False)
        self._status_lbl.setText("Processing...")
        QtWidgets.QApplication.processEvents()

        errors: list[str] = []

        for i, tgt_path in enumerate(self._target_paths):
            name = Path(tgt_path).name
            self._status_lbl.setText(f"Processing {name}  ({i + 1}/{total})")
            QtWidgets.QApplication.processEvents()

            img = cv2.imread(tgt_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                errors.append(name)
                self._progress.setValue(i + 1)
                continue

            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            matched = match_histogram(img, self._base_image)
            self._results.append((name, img, matched))

            # Save
            stem = Path(tgt_path).stem
            ext = Path(tgt_path).suffix or ".tif"
            out_path = os.path.join(out_dir, f"{stem}{suffix}{ext}")
            cv2.imwrite(out_path, matched)

            self._progress.setValue(i + 1)
            QtWidgets.QApplication.processEvents()

        self._btn_run.setEnabled(True)

        if errors:
            self._status_lbl.setText(
                f"Done. {len(errors)} file(s) could not be read."
            )
        else:
            self._status_lbl.setText(
                f"Done — {total} image(s) matched and saved to {out_dir}"
            )

        if self._results:
            self._result_idx = 0
            self._show_current_result()

    # -- Result navigation ----------------------------------------------------

    def _show_current_result(self):
        if not self._results:
            return
        name, before, after = self._results[self._result_idx]
        self._result_canvas.plot_comparison(before, after, self._base_image,
                                            name=name)
        self._update_nav_state()

    def _show_prev_result(self):
        if self._result_idx > 0:
            self._result_idx -= 1
            self._show_current_result()

    def _show_next_result(self):
        if self._result_idx < len(self._results) - 1:
            self._result_idx += 1
            self._show_current_result()

    def _update_nav_state(self):
        n = len(self._results)
        has = n > 0
        self._btn_prev.setEnabled(has and self._result_idx > 0)
        self._btn_next.setEnabled(has and self._result_idx < n - 1)
        if has:
            name = self._results[self._result_idx][0]
            self._result_label.setText(
                f"{name}  ({self._result_idx + 1}/{n})"
            )
        else:
            self._result_label.setText("")


# ============================================================================
# Entry point
# ============================================================================

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Histogram Matcher — Fusi\u00b3")
    app.setStyleSheet(_global_stylesheet())

    window = HistogramMatchWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
