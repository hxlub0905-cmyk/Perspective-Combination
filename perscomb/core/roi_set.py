"""ROI data model for multi-ROI analysis across Landing Energy images.

Classes:
    NamedROI        – A single named bounding-box ROI (normalized coordinates).
    MultiROISet     – Ordered collection of NamedROIs with management helpers.
    ROIStats        – Per-ROI pixel statistics extracted from one image.
    ROIImageLayer   – All ROI stats for one image (base / compare / diff).
    ROIFullResult   – Complete cross-layer statistics + derived SNR per LE.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Colour palette for auto-assignment (BGR tuples)
# ---------------------------------------------------------------------------
_REFERENCE_COLORS: List[Tuple[int, int, int]] = [
    (255, 255,   0),   # cyan
    (  0, 255, 165),   # orange
    (  0, 255,   0),   # green
    (255,   0, 255),   # magenta
    (128,   0, 128),   # purple
    (  0, 215, 255),   # gold
    (  0, 128, 255),   # sky-blue
    (  0, 165, 255),   # light-orange
]
_TARGET_COLOR: Tuple[int, int, int] = (0, 0, 255)   # red (BGR)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class NamedROI:
    """A single bounding-box ROI in normalized image coordinates.

    Attributes
    ----------
    id          : Unique identifier (auto-generated UUID4 short string).
    label       : Human-readable name shown in the UI (e.g. "ROI_001").
    roi_type    : 'target' (defect of interest) or 'reference' (background).
    color_bgr   : BGR colour used for drawing this ROI on images.
    norm_rect   : (nx, ny, nw, nh) all in [0, 1], measured on the BASE image.
    """
    id: str
    label: str
    roi_type: Literal['target', 'reference']
    color_bgr: Tuple[int, int, int]
    norm_rect: Tuple[float, float, float, float]   # (nx, ny, nw, nh)

    def to_pixel_rect(self, img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Convert norm_rect to (x, y, w, h) integer pixels for *img_shape* (H, W)."""
        h, w = img_shape[:2]
        nx, ny, nw, nh = self.norm_rect
        x = int(round(nx * w))
        y = int(round(ny * h))
        pw = max(1, int(round(nw * w)))
        ph = max(1, int(round(nh * h)))
        # Clamp to image bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        pw = min(pw, w - x)
        ph = min(ph, h - y)
        return x, y, pw, ph

    def crop(self, img: np.ndarray) -> np.ndarray:
        """Return the pixel region of *img* corresponding to this ROI."""
        x, y, pw, ph = self.to_pixel_rect(img.shape)
        return img[y:y + ph, x:x + pw]


@dataclass
class ROIStats:
    """Pixel statistics for one ROI extracted from one image."""
    mean: float
    std: float
    p2: float
    p98: float
    median: float
    pixel_count: int

    @classmethod
    def from_pixels(cls, pixels: np.ndarray) -> "ROIStats":
        f = pixels.astype(np.float32).ravel()
        if f.size == 0:
            return cls(mean=0.0, std=0.0, p2=0.0, p98=0.0, median=0.0, pixel_count=0)
        p2, p98 = float(np.percentile(f, 2)), float(np.percentile(f, 98))
        return cls(
            mean=float(np.mean(f)),
            std=float(np.std(f)),
            p2=p2,
            p98=p98,
            median=float(np.median(f)),
            pixel_count=int(f.size),
        )


@dataclass
class ROIImageLayer:
    """All ROI stats extracted from a single image.

    Attributes
    ----------
    image_label : Human-readable identifier, e.g. "base", "LE1_compare", "LE1_diff".
    layer_type  : Category of the image.
    roi_stats   : Mapping from roi.id → ROIStats.
    """
    image_label: str
    layer_type: Literal['base', 'compare', 'diff']
    roi_stats: Dict[str, ROIStats] = field(default_factory=dict)


@dataclass
class ROISNREntry:
    """SNR and supporting values for one diff image."""
    le_label: str
    snr: float
    mu_target: float
    mu_ref: float
    sigma_ref: float


@dataclass
class ROIFullResult:
    """Complete cross-layer ROI analysis result.

    Attributes
    ----------
    roi_set         : The MultiROISet used for this computation.
    layers          : [base layer] + [compare layers…] + [diff layers…]
    snr_per_diff    : SNR computed from diff layers, keyed by LE label.
    normalize_method: The normalization method that was applied.
    nonlinear_warning: True when HEQ/CLAHE was used (stats not cross-LE comparable).
    """
    roi_set: "MultiROISet"
    layers: List[ROIImageLayer] = field(default_factory=list)
    snr_per_diff: Dict[str, ROISNREntry] = field(default_factory=dict)
    normalize_method: str = 'percentile'
    nonlinear_warning: bool = False

    # Convenience accessors -------------------------------------------------

    def get_layer(self, image_label: str) -> Optional[ROIImageLayer]:
        for layer in self.layers:
            if layer.image_label == image_label:
                return layer
        return None

    def get_base_layer(self) -> Optional[ROIImageLayer]:
        for layer in self.layers:
            if layer.layer_type == 'base':
                return layer
        return None

    def compare_labels(self) -> List[str]:
        return [l.image_label for l in self.layers if l.layer_type == 'compare']

    def diff_labels(self) -> List[str]:
        return [l.image_label for l in self.layers if l.layer_type == 'diff']


# ---------------------------------------------------------------------------
# MultiROISet
# ---------------------------------------------------------------------------

class MultiROISet:
    """Ordered collection of NamedROIs drawn on the BASE image.

    Rules
    -----
    - At most ONE ROI may have roi_type == 'target' at any time.
    - set_target(id) demotes the previous target to 'reference' automatically.
    - Colours are auto-assigned from _REFERENCE_COLORS (cycling); the target
      always uses _TARGET_COLOR.
    """

    def __init__(self) -> None:
        self._rois: List[NamedROI] = []
        self._color_index: int = 0

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_roi(
        self,
        norm_rect: Tuple[float, float, float, float],
        label: Optional[str] = None,
        roi_type: Literal['target', 'reference'] = 'reference',
    ) -> str:
        """Add a new ROI and return its id.

        If *roi_type* is 'target', any existing target is demoted automatically.
        """
        roi_id = uuid.uuid4().hex[:8]
        if label is None:
            label = f"ROI_{len(self._rois) + 1:03d}"

        if roi_type == 'target':
            self._demote_current_target()
            color = _TARGET_COLOR
        else:
            color = _REFERENCE_COLORS[self._color_index % len(_REFERENCE_COLORS)]
            self._color_index += 1

        roi = NamedROI(
            id=roi_id,
            label=label,
            roi_type=roi_type,
            color_bgr=color,
            norm_rect=norm_rect,
        )
        self._rois.append(roi)
        return roi_id

    def remove_roi(self, roi_id: str) -> bool:
        """Remove ROI by id. Returns True if found and removed."""
        before = len(self._rois)
        self._rois = [r for r in self._rois if r.id != roi_id]
        return len(self._rois) < before

    def clear(self) -> None:
        self._rois.clear()
        self._color_index = 0

    def update_rect(self, roi_id: str, norm_rect: Tuple[float, float, float, float]) -> bool:
        """Update the norm_rect of an existing ROI. Returns True if found."""
        for roi in self._rois:
            if roi.id == roi_id:
                roi.norm_rect = norm_rect
                return True
        return False

    # ------------------------------------------------------------------
    # Target management
    # ------------------------------------------------------------------

    def set_target(self, roi_id: str) -> bool:
        """Promote *roi_id* to target; demote any existing target. Returns True if found."""
        roi = self._get_by_id(roi_id)
        if roi is None:
            return False
        self._demote_current_target()
        roi.roi_type = 'target'
        roi.color_bgr = _TARGET_COLOR
        return True

    def set_reference(self, roi_id: str) -> bool:
        """Force *roi_id* to reference type."""
        roi = self._get_by_id(roi_id)
        if roi is None:
            return False
        roi.roi_type = 'reference'
        roi.color_bgr = _REFERENCE_COLORS[self._color_index % len(_REFERENCE_COLORS)]
        self._color_index += 1
        return True

    def get_target(self) -> Optional[NamedROI]:
        for roi in self._rois:
            if roi.roi_type == 'target':
                return roi
        return None

    def get_references(self) -> List[NamedROI]:
        return [r for r in self._rois if r.roi_type == 'reference']

    # ------------------------------------------------------------------
    # Grid generation (Multi-Add)
    # ------------------------------------------------------------------

    def generate_grid(
        self,
        anchor_tl_norm: Tuple[float, float],
        anchor_br_norm: Tuple[float, float],
        period_x_px: int,
        period_y_px: int,
        roi_w_px: int,
        roi_h_px: int,
        img_shape: Tuple[int, int],
    ) -> List[Tuple[float, float, float, float]]:
        """Generate a regular grid of norm_rects between two anchor points.

        Anchors define the centers of the top-left and bottom-right ROIs.
        The grid fills the space between anchors using the specified periods.
        All returned rects are in normalized [0, 1] coordinates.
        The rois are NOT added to the set here — caller must call add_roi() for each.

        Parameters
        ----------
        anchor_tl_norm  : (norm_cx, norm_cy) center of top-left anchor ROI.
        anchor_br_norm  : (norm_cx, norm_cy) center of bottom-right anchor ROI.
        period_x_px     : Horizontal distance between ROI centers (pixels).
        period_y_px     : Vertical distance between ROI centers (pixels).
        roi_w_px        : Width of each ROI (pixels).
        roi_h_px        : Height of each ROI (pixels).
        img_shape       : (H, W) of the base image for pixel↔norm conversion.
        """
        h, w = img_shape[:2]
        if w == 0 or h == 0:
            return []

        # Convert anchor centers to pixel coordinates
        tl_cx = anchor_tl_norm[0] * w
        tl_cy = anchor_tl_norm[1] * h
        br_cx = anchor_br_norm[0] * w
        br_cy = anchor_br_norm[1] * h

        # Pixel-space half sizes for norm conversion
        nw = roi_w_px / w
        nh = roi_h_px / h

        rects: List[Tuple[float, float, float, float]] = []

        cy = tl_cy
        while cy <= br_cy + 1e-3:
            cx = tl_cx
            while cx <= br_cx + 1e-3:
                # Top-left corner of ROI in norm coords
                nx = (cx - roi_w_px / 2.0) / w
                ny = (cy - roi_h_px / 2.0) / h
                # Clamp to [0, 1]
                nx = max(0.0, min(nx, 1.0 - nw))
                ny = max(0.0, min(ny, 1.0 - nh))
                rects.append((nx, ny, nw, nh))
                cx += period_x_px
            cy += period_y_px

        return rects

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def rois(self) -> List[NamedROI]:
        return list(self._rois)

    def __len__(self) -> int:
        return len(self._rois)

    def __bool__(self) -> bool:
        return bool(self._rois)

    def get_by_id(self, roi_id: str) -> Optional[NamedROI]:
        return self._get_by_id(roi_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_by_id(self, roi_id: str) -> Optional[NamedROI]:
        for roi in self._rois:
            if roi.id == roi_id:
                return roi
        return None

    def _demote_current_target(self) -> None:
        for roi in self._rois:
            if roi.roi_type == 'target':
                roi.roi_type = 'reference'
                roi.color_bgr = _REFERENCE_COLORS[self._color_index % len(_REFERENCE_COLORS)]
                self._color_index += 1
