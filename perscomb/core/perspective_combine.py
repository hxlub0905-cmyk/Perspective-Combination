"""Perspective Combination - Multi-image alignment and subtraction for defect detection.

This module provides functionality for:
1. Aligning multiple SEM images (e.g., different Energy Filter conditions)
2. Single-pair operations: Subtract, Blend (Add), Invert
3. Computing local SNR maps to highlight strong signals
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .ebeam_snr import calculate_alignment_robust, AlignResult


class OperationType(Enum):
    """Types of image operations."""
    SUBTRACT = "subtract"         # Base - Compare
    BLEND = "blend"               # α×Base + β×Compare
    INVERT_BASE = "invert_base"   # 255 - Base
    INVERT_COMPARE = "invert_compare"  # 255 - Compare
    INVERT_RESULT = "invert_result"    # 255 - Result


@dataclass
class SinglePairResult:
    """Result of a single Base-Compare pair operation."""
    
    base_label: str                 # Label of base image
    compare_label: str              # Label of compare image
    operation: str                  # Operation type
    result_image: np.ndarray        # Operation result (0-255 uint8)
    snr_map: np.ndarray             # Local SNR map (0-255 uint8)
    histogram: Tuple[np.ndarray, np.ndarray]  # (counts, bin_edges)
    alignment: AlignResult          # Alignment result
    stats: Dict[str, float]         # Statistical metrics
    blend_alpha: float = 0.5        # Base weight for blend
    blend_beta: float = 0.5         # Compare weight for blend
    norm_a: float = 1.0             # Normalize scale coefficient (I' = a*I + b)
    norm_b: float = 0.0             # Normalize offset coefficient
    normalize_method: str = 'percentile'  # Method used for normalization
    normalized_compare: Optional[np.ndarray] = None  # Normalized compare image (uint8)
    aligned_compare: Optional[np.ndarray] = None  # Aligned compare image after inversion (uint8)
    defect_rois: List = field(default_factory=list)  # List[DefectROI] from segmentation
    roi_match_alpha: Optional[float] = None  # ROI-match scale coefficient (set when roi_match mode used)
    
    @property
    def alignment_ok(self) -> bool:
        """Check if alignment is acceptable."""
        return self.alignment.status in ('ok', 'warn')


@dataclass
class CombineResult:
    """Result of perspective combination operation."""
    
    diff_image: np.ndarray          # Normalized difference map (0-255 uint8)
    snr_map: np.ndarray             # Local SNR highlighting (0-255 uint8)
    histogram: Tuple[np.ndarray, np.ndarray]  # (counts, bin_edges)
    alignments: List[AlignResult]   # Alignment results for each compare image
    stats: Dict[str, float]         # Statistical metrics
    blended_image: np.ndarray       # Average of aligned compare images
    
    @property
    def overall_alignment_ok(self) -> bool:
        """Check if all alignments are acceptable."""
        return all(a.status in ('ok', 'warn') for a in self.alignments)
    
    @property
    def worst_alignment_score(self) -> float:
        """Return the lowest alignment score."""
        if not self.alignments:
            return 0.0
        return min(a.final_score for a in self.alignments)


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 float range using robust percentile scaling."""
    if img is None or img.size == 0:
        return img
    
    img_f = img.astype(np.float32)
    p2, p98 = np.percentile(img_f, [2, 98])
    rng = p98 - p2
    if rng < 1e-6:
        rng = 1.0
    
    normalized = (img_f - p2) / rng
    return np.clip(normalized, 0, 1)


def _percentile_range(img: np.ndarray, low: float = 2.0, high: float = 98.0) -> Tuple[float, float]:
    """Return percentile range for normalization."""
    if img is None or img.size == 0:
        return 0.0, 1.0
    img_f = img.astype(np.float32)
    p_low, p_high = np.percentile(img_f, [low, high])
    if (p_high - p_low) < 1e-6:
        p_high = p_low + 1.0
    return float(p_low), float(p_high)


def _normalize_image_with_range(img: np.ndarray, p2: float, p98: float) -> np.ndarray:
    """Normalize image to 0-1 using provided percentile range."""
    if img is None or img.size == 0:
        return img
    rng = p98 - p2
    if rng < 1e-6:
        rng = 1.0
    normalized = (img.astype(np.float32) - p2) / rng
    return np.clip(normalized, 0, 1)


_GLV_MASK_MIN_PIXELS = 50  # Minimum masked pixels required; fall back to full-image if below.


def _percentile_range_glv_masked(
    img: np.ndarray,
    glv_low: int,
    glv_high: int,
    low: float = 2.0,
    high: float = 98.0,
) -> Tuple[float, float]:
    """Return P2/P98 computed only from pixels whose value falls in [glv_low, glv_high].

    Statistics are derived exclusively from pixels inside the specified GLV range
    (e.g. MG: 110-145, EPI: 200-255) so that the normalization scale is anchored
    to the brightness distribution of that specific pattern, not the full image.

    If fewer than _GLV_MASK_MIN_PIXELS pixels satisfy the mask, the function
    falls back to full-image _percentile_range to avoid degenerate results.

    Args:
        img:      Input image (float32, values 0-255, already inverted if needed).
        glv_low:  Lower bound of GLV mask range (inclusive, 0-255).
        glv_high: Upper bound of GLV mask range (inclusive, 0-255).
        low:      Low percentile (default 2.0).
        high:     High percentile (default 98.0).

    Returns:
        (p_low, p_high) computed from the masked pixel subset.
    """
    if img is None or img.size == 0:
        return 0.0, 1.0

    img_f = img.astype(np.float32)
    mask = (img_f >= float(glv_low)) & (img_f <= float(glv_high))
    masked_pixels = img_f[mask]

    if masked_pixels.size < _GLV_MASK_MIN_PIXELS:
        # Not enough pixels in the specified range — fall back to full-image percentile.
        return _percentile_range(img_f, low, high)

    p_low, p_high = np.percentile(masked_pixels, [low, high])
    if (p_high - p_low) < 1e-6:
        p_high = p_low + 1.0
    return float(p_low), float(p_high)


def _apply_alignment(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Apply sub-pixel translation to image using affine warp (INTER_LINEAR)."""
    if image is None:
        return None

    h, w = image.shape[:2]
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return aligned


def _alignment_overlap_slices(shape: Tuple[int, int], dx: int, dy: int) -> Tuple[slice, slice, slice, slice]:
    """Return slices for overlapping regions given a shift."""
    h, w = shape
    if dx >= 0:
        base_x = slice(0, w - dx)
        target_x = slice(dx, w)
    else:
        base_x = slice(-dx, w)
        target_x = slice(0, w + dx)

    if dy >= 0:
        base_y = slice(0, h - dy)
        target_y = slice(dy, h)
    else:
        base_y = slice(-dy, h)
        target_y = slice(0, h + dy)

    return base_y, base_x, target_y, target_x


def _ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized cross-correlation score between two same-shape arrays."""
    a_mean = float(np.mean(a))
    b_mean = float(np.mean(b))
    a_z = a - a_mean
    b_z = b - b_mean
    denom = float(np.sqrt(np.sum(a_z ** 2) * np.sum(b_z ** 2))) + 1e-6
    return float(np.sum(a_z * b_z) / denom)


def _calculate_alignment_scores(
    base: np.ndarray,
    target: np.ndarray,
    dx: int,
    dy: int,
    phase_score: float = 0.0,
    method: str = "phase"
) -> Tuple[float, float, float]:
    """Calculate NCC, residual, and final score for a given shift."""
    base_norm = _normalize_image(base.astype(np.float32))
    target_norm = _normalize_image(target.astype(np.float32))
    aligned_target = _apply_alignment(target_norm, dx, dy)
    residual = float(np.mean(np.abs(base_norm - aligned_target)))
    score_residual = max(0.0, 1.0 - residual * 2.0)

    by, bx, ty, tx = _alignment_overlap_slices(base_norm.shape[:2], dx, dy)
    if by.stop <= by.start or bx.stop <= bx.start:
        score_ncc = 0.0
    else:
        score_ncc = (_ncc_score(base_norm[by, bx], target_norm[ty, tx]) + 1.0) / 2.0

    if method == "ncc":
        final_score = (0.6 * score_ncc + 0.4 * score_residual) * 100.0
    else:
        final_score = (0.4 * phase_score + 0.6 * score_residual) * 100.0

    return score_ncc, score_residual, final_score


def _calculate_alignment_ncc(
    base: np.ndarray,
    target: np.ndarray,
    search_radius: int = 40
) -> AlignResult:
    """NCC alignment with coarse-to-fine search for efficiency.
    
    Uses a two-stage approach:
    1. Coarse search with step=4 over full radius
    2. Fine search with step=1 in ±4 pixel neighborhood around best
    
    This reduces computation from ~6561 to ~400 NCC evaluations.
    """
    if base is None or target is None:
        return AlignResult(0, 0, 0.0, 0.0, 0.0, 0.0, 'fail', 'none')

    base_f = _normalize_image(base.astype(np.float32))
    target_f = _normalize_image(target.astype(np.float32))
    h, w = base_f.shape[:2]

    def search_best(cx, cy, radius, step):
        best_score = -1.0
        best_dx, best_dy = cx, cy
        for dy in range(cy - radius, cy + radius + 1, step):
            for dx in range(cx - radius, cx + radius + 1, step):
                by, bx, ty, tx = _alignment_overlap_slices((h, w), dx, dy)
                if by.stop <= by.start or bx.stop <= bx.start:
                    continue
                score = _ncc_score(base_f[by, bx], target_f[ty, tx])
                if score > best_score:
                    best_score = score
                    best_dx = dx
                    best_dy = dy
        return best_dx, best_dy, best_score
    
    # Stage 1: Coarse search (step=4)
    coarse_step = 4
    coarse_dx, coarse_dy, _ = search_best(0, 0, search_radius, coarse_step)

    # Stage 2: Fine search (step=1) around coarse result
    fine_radius = coarse_step
    best_dx, best_dy, best_score = search_best(coarse_dx, coarse_dy, fine_radius, 1)

    # Stage 3: Parabolic sub-pixel refinement on the NCC surface
    # Fit parabola along each axis at the integer peak.
    def _ncc_at(ddx, ddy):
        by, bx, ty, tx = _alignment_overlap_slices((h, w), ddx, ddy)
        if by.stop <= by.start or bx.stop <= bx.start:
            return -1.0
        return _ncc_score(base_f[by, bx], target_f[ty, tx])

    def _parabolic_offset(f_neg, f_0, f_pos):
        denom = 2.0 * (f_pos - 2.0 * f_0 + f_neg)
        if abs(denom) < 1e-8:
            return 0.0
        return -(f_pos - f_neg) / denom

    dx_sub = best_dx + _parabolic_offset(
        _ncc_at(best_dx - 1, best_dy),
        best_score,
        _ncc_at(best_dx + 1, best_dy),
    )
    dy_sub = best_dy + _parabolic_offset(
        _ncc_at(best_dx, best_dy - 1),
        best_score,
        _ncc_at(best_dx, best_dy + 1),
    )

    aligned_target = _apply_alignment(target_f, dx_sub, dy_sub)
    residual = float(np.mean(np.abs(base_f - aligned_target)))
    score_residual = max(0.0, 1.0 - residual * 2.0)
    score_ncc = (best_score + 1.0) / 2.0
    final_score = (0.6 * score_ncc + 0.4 * score_residual) * 100.0

    status = 'ok'
    if final_score < 75:
        status = 'warn'
    if final_score < 55:
        status = 'fail'

    return AlignResult(
        dx=int(best_dx),
        dy=int(best_dy),
        score_phase=0.0,
        score_ncc=score_ncc,
        score_residual=score_residual,
        final_score=final_score,
        status=status,
        method='ncc',
        dx_subpixel=float(dx_sub),
        dy_subpixel=float(dy_sub),
    )


def _calculate_alignment(
    base: np.ndarray,
    target: np.ndarray,
    method: str,
    search_radius: int
) -> AlignResult:
    """Select alignment method."""
    method_norm = (method or "phase").lower()
    if method_norm in ("phase", "hybrid"):
        return calculate_alignment_robust(base, target, search_radius)
    if method_norm == "ncc":
        return _calculate_alignment_ncc(base, target, search_radius)
    return calculate_alignment_robust(base, target, search_radius)


def compute_snr_map(
    diff_image: np.ndarray,
    window_size: int = 31,
    clip_sigma: float = 3.0,
    clip_percentile: float = 99.5,
    exclude_border: int = 16
) -> np.ndarray:
    """Compute local SNR map highlighting areas with strong signal.

    Parameters
    ----------
    exclude_border : int
        Pixels within this margin are zeroed before peak detection.
        cv2.filter2D uses BORDER_REFLECT_101 by default, which causes
        artificially low variance (and therefore inflated SNR) near
        image edges. Setting this to >= window_size avoids false peaks.
    """
    if diff_image is None or diff_image.size == 0:
        return np.zeros((100, 100), dtype=np.uint8), 0.0

    img_f = diff_image.astype(np.float32)
    if img_f.max() > 1.5:
        img_f = img_f / 255.0

    if window_size % 2 == 0:
        window_size += 1

    kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)

    local_mean = cv2.filter2D(img_f, -1, kernel)
    local_sq_mean = cv2.filter2D(img_f ** 2, -1, kernel)
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 1e-6)
    local_std = np.sqrt(local_var)

    global_mean = np.mean(img_f)
    snr = np.abs(local_mean - global_mean) / (local_std + 1e-6)

    # Zero out border region to suppress edge artifacts from reflected padding
    if exclude_border > 0:
        h, w = snr.shape
        snr[:exclude_border, :] = 0
        snr[h - exclude_border:, :] = 0
        snr[:, :exclude_border] = 0
        snr[:, w - exclude_border:] = 0

    snr_clipped = np.clip(snr, 0, clip_sigma)
    if clip_percentile is not None:
        scale = float(np.percentile(snr_clipped, clip_percentile))
    else:
        scale = float(clip_sigma)
    if scale < 1e-6:
        scale = float(clip_sigma)

    # Capture raw SNR max before normalization
    raw_snr_max = float(snr.max())

    snr_normalized = np.clip(snr_clipped / scale, 0, 1)
    snr_normalized = (snr_normalized * 255).astype(np.uint8)

    return snr_normalized, raw_snr_max


def compute_single_pair(
    base: np.ndarray,
    compare: np.ndarray,
    base_label: str = "Base",
    compare_label: str = "Compare",
    operation: str = "subtract",
    alpha: float = 0.5,
    beta: float = 0.5,
    invert_base: bool = False,
    invert_compare: bool = False,
    invert_result: bool = False,
    normalize: bool = True,
    normalize_method: str = 'percentile',
    glv_range: Optional[Tuple[int, int]] = None,
    clahe_clip_limit: float = 2.0,
    search_radius: int = 50,
    alignment_method: str = "phase",
    manual_offset: Optional[Tuple[int, int]] = None,
    preserve_positive_diff: bool = False,
    abs_no_gain: bool = False,
    snr_window_size: int = 31,
    roi_rect: Optional[Tuple[int, int, int, int]] = None,
    roi_match: bool = False,
) -> SinglePairResult:
    """Compute a single Base-Compare pair operation.

    Args:
        base: Base image (grayscale)
        compare: Compare image to align and process
        base_label: Label for base image
        compare_label: Label for compare image
        operation: 'subtract' or 'blend'
        alpha: Base weight for blending (0-1)
        beta: Compare weight for blending (0-1)
        invert_base: Apply 255-X to base before operation
        invert_compare: Apply 255-X to compare before operation
        invert_result: Apply 255-X to result after operation
        normalize: Whether to normalize images
        glv_range: Optional (low, high) tuple (0-255). When provided and normalize=True,
                   P2/P98 are computed only from pixels whose value falls in [low, high]
                   (GLV-Mask Normalization). Falls back to full-image percentile when
                   the masked region contains fewer than _GLV_MASK_MIN_PIXELS pixels.
        search_radius: Maximum alignment search radius
        preserve_positive_diff: Keep signal direction (B−C), clamp negatives to 0, no gain.
        abs_no_gain: Take |B−C| without the ×2 enhancement factor.
        snr_window_size: Box-filter window size for SNR map (must be odd, ≥3)
        roi_rect: (x, y, w, h) in pixel coords — reference ROI for ROI-Match (EPI Nulling).
        roi_match: When True, use ROI-Match mode: calibrate a scale factor (alpha) from the
                   ROI so that EPI-dominated regions cancel in subtraction, leaving residual
                   HK/Hf defect signals near inner spacer visible.

    Returns:
        SinglePairResult with operation result, SNR map, and statistics
    """
    # Validate inputs
    if base is None or compare is None:
        return _empty_single_result(base_label, compare_label, operation)
    
    # Ensure same shape
    if base.shape != compare.shape:
        return _empty_single_result(base_label, compare_label, operation, 
                                     error="Shape mismatch")
    
    # Step 1: Align compare to base
    alignment = _calculate_alignment(base, compare, alignment_method, search_radius)
    manual_dx, manual_dy = manual_offset if manual_offset is not None else (0, 0)
    # Sub-pixel warp uses float offsets; integer fields kept for display
    applied_dx_f = alignment.dx_subpixel + manual_dx
    applied_dy_f = alignment.dy_subpixel + manual_dy
    applied_dx = int(round(applied_dx_f))
    applied_dy = int(round(applied_dy_f))
    if manual_offset is not None and (manual_dx != 0 or manual_dy != 0):
        score_ncc, score_residual, final_score = _calculate_alignment_scores(
            base,
            compare,
            applied_dx,
            applied_dy,
            phase_score=alignment.score_phase,
            method=alignment.method
        )
        status = 'ok'
        if final_score < 75:
            status = 'warn'
        if final_score < 55:
            status = 'fail'
    else:
        score_ncc = alignment.score_ncc
        score_residual = alignment.score_residual
        final_score = alignment.final_score
        status = alignment.status
    alignment = AlignResult(
        dx=applied_dx,
        dy=applied_dy,
        score_phase=alignment.score_phase,
        score_ncc=score_ncc,
        score_residual=score_residual,
        final_score=final_score,
        status=status,
        method=alignment.method,
        dx_subpixel=applied_dx_f,
        dy_subpixel=applied_dy_f,
    )
    # Use sub-pixel float offset for actual warp (INTER_LINEAR handles fractional shifts)
    aligned_compare = _apply_alignment(compare, applied_dx_f, applied_dy_f)
    
    # Step 2: Apply inversion if requested
    base_proc = base.astype(np.float32)
    comp_proc = aligned_compare.astype(np.float32)
    
    if invert_base:
        base_proc = 255.0 - base_proc
    if invert_compare:
        comp_proc = 255.0 - comp_proc

    comp_aligned_uint8 = np.clip(comp_proc, 0, 255).astype(np.uint8)

    # Step 2b: ROI-Match (EPI Nulling) — calibrate scale from a user-chosen ROI.
    #
    # Purpose: In multi-landing-energy SEM, the EPI region is very bright in BSE at
    # high LE and can blind weaker HK/Hf defect signals near inner spacer.  By
    # computing a least-squares scale factor from an EPI-dominated ROI and applying
    # it to the full aligned compare image, the EPI response is nulled out after
    # subtraction, leaving residual anomalies (e.g. Hf extrusion paths) more visible.
    #
    # When roi_match=True and roi_rect is provided:
    #   1. Extract ROI from both base and aligned compare (float32).
    #   2. Compute alpha_roi = sum(B_roi * C_roi) / (sum(C_roi^2) + eps)  (LS scale).
    #   3. Clamp alpha_roi to [0.25, 4.0] to avoid blowups from noisy/flat ROIs.
    #   4. Apply: comp_proc = alpha_roi * comp_proc  (full image).
    #   5. Force keep-direction subtraction (clip >= 0) and skip independent
    #      per-image normalization so the energy-dependent signature is preserved.
    roi_match_alpha: Optional[float] = None
    if roi_match and roi_rect is not None:
        rx, ry, rw, rh = roi_rect
        h_img, w_img = base_proc.shape[:2]
        # Clamp ROI to image bounds
        rx = max(0, min(rx, w_img - 1))
        ry = max(0, min(ry, h_img - 1))
        rw = max(1, min(rw, w_img - rx))
        rh = max(1, min(rh, h_img - ry))

        b_roi = base_proc[ry:ry + rh, rx:rx + rw]
        c_roi = comp_proc[ry:ry + rh, rx:rx + rw]

        # Least-squares scale: minimise ||B_roi - alpha * C_roi||^2
        denom = float(np.sum(c_roi.astype(np.float64) * c_roi.astype(np.float64)))
        numer = float(np.sum(b_roi.astype(np.float64) * c_roi.astype(np.float64)))
        alpha_roi = numer / (denom + 1e-6)

        # Clamp to reasonable range to avoid blowups from noisy/flat ROIs
        _ALPHA_MIN, _ALPHA_MAX = 0.25, 4.0
        alpha_roi_clamped = float(np.clip(alpha_roi, _ALPHA_MIN, _ALPHA_MAX))
        if alpha_roi != alpha_roi_clamped:
            import logging
            logging.getLogger(__name__).warning(
                "ROI-match alpha %.4f clamped to [%.2f, %.2f] -> %.4f",
                alpha_roi, _ALPHA_MIN, _ALPHA_MAX, alpha_roi_clamped,
            )
        roi_match_alpha = alpha_roi_clamped

        # Scale full aligned compare so EPI region matches base intensity
        comp_proc = comp_proc * roi_match_alpha

        # Override subtraction mode: keep-direction (clip >= 0) is the physically
        # meaningful choice for ROI-Match because we want to see where Base > scaled
        # Compare, i.e. extra signal not explained by the reference LE.
        preserve_positive_diff = True
        abs_no_gain = False

        # Skip independent per-image normalization — we already matched intensities
        # via the ROI scale factor.  Independent normalization would erase the
        # energy-dependent signature we just calibrated.
        normalize = False

    # Step 3: Normalize if requested, capture coefficients for display.
    # Both images are independently mapped to [0, 1] using their own range,
    # so intensity/brightness differences between conditions are removed before the
    # subtract/blend operation.  comp_norm_uint8 is the normalized compare
    # exposed for visual verification via the "Normalize Preview" button in the UI.
    norm_a, norm_b = 1.0, 0.0  # Default coefficients (identity when normalize=False)
    _norm_nonlinear = False  # True for HEQ/CLAHE (no closed-form a, b)
    effective_method = normalize_method if normalize else 'skip'

    if effective_method == 'skip':
        base_norm = base_proc / 255.0
        comp_norm = comp_proc / 255.0

    elif effective_method == 'glv_mask' and glv_range is not None:
        # GLV-Mask Normalization: compute P2/P98 only from pixels within the
        # specified gray-level range (e.g. MG: 110-145, EPI: 200-255).
        glv_low, glv_high = int(glv_range[0]), int(glv_range[1])
        base_p2, base_p98 = _percentile_range_glv_masked(base_proc, glv_low, glv_high)
        comp_p2, comp_p98 = _percentile_range_glv_masked(comp_proc, glv_low, glv_high)
        base_norm = _normalize_image_with_range(base_proc, base_p2, base_p98)
        comp_norm = _normalize_image_with_range(comp_proc, comp_p2, comp_p98)
        comp_rng = max(comp_p98 - comp_p2, 1e-6)
        norm_a = (base_p98 - base_p2) / comp_rng
        norm_b = base_p2 - norm_a * comp_p2

    elif effective_method == 'heq':
        # Histogram Equalization – non-linear, for visual enhancement only
        _norm_nonlinear = True
        base_u8 = np.clip(base_proc, 0, 255).astype(np.uint8)
        comp_u8 = np.clip(comp_proc, 0, 255).astype(np.uint8)
        base_norm = cv2.equalizeHist(base_u8).astype(np.float32) / 255.0
        comp_norm = cv2.equalizeHist(comp_u8).astype(np.float32) / 255.0

    elif effective_method == 'clahe':
        # CLAHE – non-linear, for visual enhancement only
        _norm_nonlinear = True
        clahe_obj = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        base_u8 = np.clip(base_proc, 0, 255).astype(np.uint8)
        comp_u8 = np.clip(comp_proc, 0, 255).astype(np.uint8)
        base_norm = clahe_obj.apply(base_u8).astype(np.float32) / 255.0
        comp_norm = clahe_obj.apply(comp_u8).astype(np.float32) / 255.0

    else:
        # 'percentile' (default) – Standard full-image P2/P98 linear normalization
        base_p2, base_p98 = _percentile_range(base_proc)
        comp_p2, comp_p98 = _percentile_range(comp_proc)
        base_norm = _normalize_image_with_range(base_proc, base_p2, base_p98)
        comp_norm = _normalize_image_with_range(comp_proc, comp_p2, comp_p98)
        comp_rng = max(comp_p98 - comp_p2, 1e-6)
        norm_a = (base_p98 - base_p2) / comp_rng
        norm_b = base_p2 - norm_a * comp_p2

    # comp_norm_uint8: normalized compare image for display in "Normalize Preview"
    if _norm_nonlinear:
        comp_norm_uint8 = (comp_norm * 255).astype(np.uint8)
    else:
        comp_norm_uint8 = np.clip(norm_a * comp_proc + norm_b, 0, 255).astype(np.uint8)
    
    # Step 4: Apply operation
    #
    # Subtract modes (mutually exclusive, checked in priority order):
    #   preserve_positive_diff : clip(B−C, 0, 1)       – keep direction, no abs, no gain
    #   abs_no_gain            : |B−C|                  – absolute value, no ×2 gain
    #   default                : clip(|B−C| × 2, 0, 1) – absolute value + ×2 enhancement
    if operation == 'subtract':
        if preserve_positive_diff:
            result_float = np.clip(base_norm - comp_norm, 0, 1)
        elif abs_no_gain:
            result_float = np.clip(np.abs(base_norm - comp_norm), 0, 1)
        else:
            result_float = np.abs(base_norm - comp_norm)
            result_float = np.clip(result_float * 2.0, 0, 1)  # ×2 enhance
    elif operation == 'blend':
        # Weighted blend: α×Base + β×Compare
        result_float = alpha * base_norm + beta * comp_norm
        # Normalize to 0-1
        result_float = np.clip(result_float, 0, 1)
    else:
        # Default to subtract
        result_float = np.abs(base_norm - comp_norm)
    
    # Step 5: Apply result inversion if requested
    if invert_result:
        result_float = 1.0 - result_float
    
    # Convert to uint8
    result_uint8 = (result_float * 255).astype(np.uint8)
    
    # Step 6: Compute SNR map
    # NOTE:
    # Large alignment shifts can introduce strong border artifacts after warpAffine.
    # Increase excluded border margin based on absolute shift to reduce false border peaks.
    dynamic_exclude_border = max(16, int(max(abs(alignment.dx), abs(alignment.dy)) + 8))
    snr_map, raw_snr_max = compute_snr_map(
        result_float,
        window_size=snr_window_size,
        clip_sigma=3.0,
        exclude_border=dynamic_exclude_border,
    )
    
    # Step 7: Compute histogram
    hist_counts, hist_edges = np.histogram(result_uint8.flatten(), bins=256, range=(0, 256))
    
    # Step 8: Compute statistics
    diff_vals = result_float.flatten()
    threshold_95 = np.percentile(diff_vals, 95)
    hot_mask = result_float > threshold_95
    hot_pixel_count = int(np.sum(hot_mask))

    if np.max(snr_map) > 0:
        snr_max_idx = np.unravel_index(np.argmax(snr_map), snr_map.shape)
        snr_peak_y, snr_peak_x = snr_max_idx
        snr_peak_valid = True
    else:
        # All-zero maps can happen when diff is very flat or border suppression dominates.
        # In this case, avoid reporting a misleading corner location like (0, 0).
        snr_peak_y, snr_peak_x = result_uint8.shape[0] // 2, result_uint8.shape[1] // 2
        snr_peak_valid = False

    # Center ROI weighted metrics (Gaussian mask centred on image)
    center_mask = _center_gaussian_mask(result_float.shape)
    center_snr = snr_map.astype(np.float32) * center_mask
    center_snr_peak = float(center_snr.max())
    center_hot = float((result_float * center_mask > threshold_95 * center_mask.max()).sum())
    center_weight_sum = float(center_mask.sum())
    center_hot_ratio = center_hot / max(center_weight_sum, 1.0)

    stats = {
        'diff_mean': float(np.mean(diff_vals)),
        'diff_std': float(np.std(diff_vals)),
        'diff_max': float(np.max(diff_vals)),
        'diff_min': float(np.min(diff_vals)),
        'hot_pixels': hot_pixel_count,
        'snr_peak': raw_snr_max,  # Use raw SNR max before normalization
        'snr_peak_x': int(snr_peak_x),
        'snr_peak_y': int(snr_peak_y),
        'snr_peak_valid': bool(snr_peak_valid),
        'snr_exclude_border': int(dynamic_exclude_border),
        'alignment_score': float(alignment.final_score),
        # Sub-pixel alignment
        'dx_subpixel': float(alignment.dx_subpixel),
        'dy_subpixel': float(alignment.dy_subpixel),
        # Center ROI
        'center_snr_peak': center_snr_peak,
        'center_hot_ratio': center_hot_ratio,
        'center_focused': bool(center_snr_peak > 0.7 * float(snr_map.max()) if snr_map.max() > 0 else False),
    }

    # Step 9: Defect segmentation from SNR map
    defect_rois = segment_defects(snr_map, result_float)

    return SinglePairResult(
        base_label=base_label,
        compare_label=compare_label,
        operation=operation,
        result_image=result_uint8,
        snr_map=snr_map,
        histogram=(hist_counts, hist_edges),
        alignment=alignment,
        stats=stats,
        blend_alpha=alpha,
        blend_beta=beta,
        norm_a=norm_a,
        norm_b=norm_b,
        normalize_method=effective_method,
        normalized_compare=comp_norm_uint8,
        aligned_compare=comp_aligned_uint8,
        defect_rois=defect_rois,
        roi_match_alpha=roi_match_alpha,
    )


def _empty_single_result(base_label: str, compare_label: str, operation: str, 
                          error: str = "Invalid input") -> SinglePairResult:
    """Create empty result for error cases."""
    return SinglePairResult(
        base_label=base_label,
        compare_label=compare_label,
        operation=operation,
        result_image=np.zeros((100, 100), dtype=np.uint8),
        snr_map=np.zeros((100, 100), dtype=np.uint8),
        histogram=(np.zeros(256), np.arange(257)),
        alignment=AlignResult(
            dx=0, dy=0, 
            score_phase=0, score_ncc=0, score_residual=0,
            final_score=0, status='fail', method='none'
        ),
        stats={'error': error},
        blend_alpha=0.5,
        blend_beta=0.5,
        normalized_compare=np.zeros((100, 100), dtype=np.uint8),
        aligned_compare=np.zeros((100, 100), dtype=np.uint8)
    )


def compute_multi_pairs(
    base: np.ndarray,
    base_label: str,
    compare_images: Dict[str, np.ndarray],
    operation: str = "subtract",
    alpha: float = 0.5,
    beta: float = 0.5,
    invert_base: bool = False,
    invert_compare: bool = False,
    invert_result: bool = False,
    normalize: bool = True,
    normalize_method: str = 'percentile',
    glv_range: Optional[Tuple[int, int]] = None,
    clahe_clip_limit: float = 2.0,
    search_radius: int = 50,
    alignment_method: str = "phase",
    manual_offset: Optional[Tuple[int, int]] = None,
    preserve_positive_diff: bool = False,
    abs_no_gain: bool = False,
    snr_window_size: int = 31,
    roi_rect: Optional[Tuple[int, int, int, int]] = None,
    roi_match: bool = False,
) -> List[SinglePairResult]:
    """Compute operations for multiple Base-Compare pairs.

    Args:
        base: Base image
        base_label: Label for base image
        compare_images: Dict of {label: image} for compare images
        operation: 'subtract' or 'blend'
        alpha, beta: Blend coefficients
        invert_*: Inversion flags
        normalize: Whether to normalize
        glv_range: Optional (low, high) GLV mask range passed to each pair.
        search_radius: Alignment search radius
        preserve_positive_diff: Use base-compare (no abs/no gain) and clamp negatives to 0
        snr_window_size: Box-filter window size for SNR map (must be odd, ≥3)
        roi_rect: (x, y, w, h) in pixel coords for ROI-Match (EPI Nulling).
        roi_match: When True, use ROI-Match mode.

    Returns:
        List of SinglePairResult, one for each compare image
    """
    results = []

    for compare_label, compare_img in compare_images.items():
        result = compute_single_pair(
            base=base,
            compare=compare_img,
            base_label=base_label,
            compare_label=compare_label,
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
            search_radius=search_radius,
            alignment_method=alignment_method,
            manual_offset=manual_offset,
            preserve_positive_diff=preserve_positive_diff,
            abs_no_gain=abs_no_gain,
            snr_window_size=snr_window_size,
            roi_rect=roi_rect,
            roi_match=roi_match,
        )
        results.append(result)

    return results


# Keep legacy function for backward compatibility
def align_and_subtract(
    base: np.ndarray,
    compare_images: List[np.ndarray],
    method: str = 'hybrid',
    normalize: bool = True,
    search_radius: int = 50
) -> CombineResult:
    """Legacy function: Align compare images to base, blend them, and compute difference."""
    if base is None or not compare_images:
        return CombineResult(
            diff_image=np.zeros((100, 100), dtype=np.uint8),
            snr_map=np.zeros((100, 100), dtype=np.uint8),
            histogram=(np.zeros(256), np.arange(257)),
            alignments=[],
            stats={'error': 'Invalid input'},
            blended_image=np.zeros((100, 100), dtype=np.uint8)
        )
    
    h, w = base.shape[:2]
    alignments: List[AlignResult] = []
    aligned_images: List[np.ndarray] = []
    
    for i, comp in enumerate(compare_images):
        if comp is None or comp.shape != base.shape:
            alignments.append(AlignResult(
                dx=0, dy=0, 
                score_phase=0, score_ncc=0, score_residual=0,
                final_score=0, status='fail', method='none'
            ))
            continue
        
        align_result = _calculate_alignment(base, comp, method, search_radius)
        alignments.append(align_result)
        aligned = _apply_alignment(comp, align_result.dx, align_result.dy)
        aligned_images.append(aligned)
    
    if not aligned_images:
        blended = np.zeros_like(base, dtype=np.float32)
    else:
        stack = np.stack([img.astype(np.float32) for img in aligned_images], axis=0)
        blended = np.mean(stack, axis=0)
    
    if normalize:
        base_norm = _normalize_image(base)
        blend_norm = _normalize_image(blended)
    else:
        base_norm = base.astype(np.float32) / 255.0 if base.max() > 1 else base.astype(np.float32)
        blend_norm = blended / 255.0 if blended.max() > 1 else blended
    
    diff_float = np.abs(base_norm - blend_norm)
    diff_enhanced = np.clip(diff_float * 2.0, 0, 1)
    diff_uint8 = (diff_enhanced * 255).astype(np.uint8)
    
    snr_map, _raw_snr_max = compute_snr_map(diff_float, window_size=31, clip_sigma=3.0)
    hist_counts, hist_edges = np.histogram(diff_uint8.flatten(), bins=256, range=(0, 256))

    diff_vals = diff_float.flatten()
    threshold_95 = np.percentile(diff_vals, 95)
    hot_mask = diff_float > threshold_95
    hot_pixel_count = int(np.sum(hot_mask))
    
    snr_max_idx = np.unravel_index(np.argmax(snr_map), snr_map.shape)
    snr_peak_y, snr_peak_x = snr_max_idx
    
    stats = {
        'diff_mean': float(np.mean(diff_vals)),
        'diff_std': float(np.std(diff_vals)),
        'diff_max': float(np.max(diff_vals)),
        'diff_min': float(np.min(diff_vals)),
        'hot_pixels': hot_pixel_count,
        'hot_threshold': float(threshold_95),
        'snr_peak': float(snr_map.max()),
        'snr_peak_x': int(snr_peak_x),
        'snr_peak_y': int(snr_peak_y),
        'alignment_avg_score': float(np.mean([a.final_score for a in alignments]) if alignments else 0),
    }
    
    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8) if blended.max() > 1 else (blended * 255).astype(np.uint8)
    
    return CombineResult(
        diff_image=diff_uint8,
        snr_map=snr_map,
        histogram=(hist_counts, hist_edges),
        alignments=alignments,
        stats=stats,
        blended_image=blended_uint8
    )


def colorize_snr_map(snr_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Apply colormap to SNR map for visualization."""
    if snr_map is None or snr_map.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    return cv2.applyColorMap(snr_map, colormap)


# ---------------------------------------------------------------------------
# Feature: Center ROI Gaussian Mask
# ---------------------------------------------------------------------------

def _center_gaussian_mask(shape: Tuple[int, int], sigma_ratio: float = 0.35) -> np.ndarray:
    """Return a Gaussian weight mask centered on the image.

    Parameters
    ----------
    shape : (H, W)
    sigma_ratio : fraction of min(H, W) used as Gaussian sigma (default 0.35)

    Returns
    -------
    float32 array in [0, 1], peak = 1.0 at image center.
    """
    H, W = shape
    cy, cx = H / 2.0, W / 2.0
    sigma = min(H, W) * sigma_ratio
    Y, X = np.ogrid[:H, :W]
    mask = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2))
    return mask.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature: Defect Segmentation + Labeling
# ---------------------------------------------------------------------------

@dataclass
class DefectROI:
    """Bounding box and metrics for a single detected defect region."""
    x: int
    y: int
    w: int
    h: int
    cx: float                # centroid x
    cy: float                # centroid y
    area: int                # pixel count
    mean_signal: float       # mean diff_image value inside bbox
    snr_value: float         # max SNR inside bbox
    aspect_ratio: float      # w / h
    dist_to_center: float    # Euclidean distance from image center (pixels)


def segment_defects(
    snr_map: np.ndarray,
    diff_image: np.ndarray,
    min_area: int = 4,
    snr_threshold: Optional[float] = None,
) -> List[DefectROI]:
    """Segment defect regions from SNR map using adaptive thresholding.

    Parameters
    ----------
    snr_map : uint8 SNR map (0-255)
    diff_image : float32 or uint8 difference image (same shape as snr_map)
    min_area : minimum connected-component area (pixels) to keep
    snr_threshold : manual SNR threshold (0-255); if None, Otsu is used

    Returns
    -------
    List of DefectROI sorted by snr_value descending.
    """
    if snr_map is None or snr_map.size == 0:
        return []

    snr_u8 = snr_map if snr_map.dtype == np.uint8 else np.clip(snr_map, 0, 255).astype(np.uint8)

    # Threshold
    if snr_threshold is not None:
        thresh_val = int(np.clip(snr_threshold, 0, 255))
        _, binary = cv2.threshold(snr_u8, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        # Otsu on non-border region
        _, binary = cv2.threshold(snr_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological clean-up: remove isolated noise, fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Connected components
    n_labels, labels, stats_cc, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    H, W = snr_map.shape[:2]
    img_cx, img_cy = W / 2.0, H / 2.0

    diff_f = diff_image.astype(np.float32)
    if diff_f.max() > 1.5:
        diff_f = diff_f / 255.0

    rois: List[DefectROI] = []
    for label_idx in range(1, n_labels):  # skip background (label 0)
        area = int(stats_cc[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        bx = int(stats_cc[label_idx, cv2.CC_STAT_LEFT])
        by = int(stats_cc[label_idx, cv2.CC_STAT_TOP])
        bw = int(stats_cc[label_idx, cv2.CC_STAT_WIDTH])
        bh = int(stats_cc[label_idx, cv2.CC_STAT_HEIGHT])
        cx_roi = float(centroids[label_idx, 0])
        cy_roi = float(centroids[label_idx, 1])

        mask_roi = (labels[by:by + bh, bx:bx + bw] == label_idx)
        diff_crop = diff_f[by:by + bh, bx:bx + bw]
        snr_crop = snr_u8[by:by + bh, bx:bx + bw].astype(np.float32)

        mean_sig = float(diff_crop[mask_roi].mean()) if mask_roi.any() else 0.0
        snr_val = float(snr_crop[mask_roi].max()) if mask_roi.any() else 0.0
        aspect = bw / max(bh, 1)
        dist = float(np.hypot(cx_roi - img_cx, cy_roi - img_cy))

        rois.append(DefectROI(
            x=bx, y=by, w=bw, h=bh,
            cx=cx_roi, cy=cy_roi,
            area=area,
            mean_signal=mean_sig,
            snr_value=snr_val,
            aspect_ratio=aspect,
            dist_to_center=dist,
        ))

    rois.sort(key=lambda r: r.snr_value, reverse=True)
    return rois


# ---------------------------------------------------------------------------
# Feature: Multi-image PCA Fusion
# ---------------------------------------------------------------------------

@dataclass
class PCAFusionResult:
    """Result of multi-image PCA fusion for defect amplification."""
    residual_map: np.ndarray          # Defect signal: sum of |PC2..PCn| weighted by variance
    pc1_image: np.ndarray             # PC1 = common structure (background)
    snr_map: np.ndarray               # SNR map computed on residual_map
    defect_rois: List[DefectROI]      # Segmented defects from residual SNR
    explained_variance: List[float]   # Explained variance ratio per component
    n_images: int
    stats: Dict[str, float]


def compute_pca_fusion(
    images_dict: Dict[str, np.ndarray],
    base_label: Optional[str] = None,
    search_radius: int = 50,
    alignment_method: str = "phase",
    n_components: int = 3,
) -> Optional[PCAFusionResult]:
    """Multi-image PCA fusion to amplify defect signal.

    All images are first aligned to a reference (base_label or first image),
    then PCA decomposes the pixel matrix:
      PC1 ≈ common structure (background)
      PC2..PCn ≈ condition differences → defect signal

    Parameters
    ----------
    images_dict : {label: grayscale uint8 image}
    base_label : reference image for alignment; defaults to first key
    n_components : number of PCA components to compute (max = n_images)

    Returns
    -------
    PCAFusionResult or None if fewer than 2 images.
    """
    if len(images_dict) < 2:
        return None

    labels = list(images_dict.keys())
    ref_label = base_label if base_label in images_dict else labels[0]
    ref_img = images_dict[ref_label]
    H, W = ref_img.shape[:2]

    # Align all images to reference
    aligned_imgs: List[np.ndarray] = []
    for lbl in labels:
        img = images_dict[lbl]
        if lbl == ref_label:
            aligned_imgs.append(img.astype(np.float32) / 255.0)
            continue
        align = _calculate_alignment(ref_img, img, alignment_method, search_radius)
        warped = _apply_alignment(img, align.dx_subpixel, align.dy_subpixel)
        aligned_imgs.append(warped.astype(np.float32) / 255.0)

    N = len(aligned_imgs)
    n_comp = min(n_components, N)

    # Stack → (N, H*W), zero-mean
    X = np.stack([img.reshape(-1) for img in aligned_imgs], axis=0)  # (N, H*W)
    mean_vec = X.mean(axis=0)
    X_centered = X - mean_vec

    # SVD-based PCA (economy)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained = (S ** 2) / (np.sum(S ** 2) + 1e-12)

    # PC1 reconstruction = background
    pc1_flat = mean_vec + U[:, 0:1] @ (S[0:1, None] * Vt[0:1, :])
    pc1_mean = pc1_flat.mean(axis=0)
    pc1_img = np.clip(pc1_mean.reshape(H, W), 0, 1)
    pc1_u8 = (pc1_img * 255).astype(np.uint8)

    # Residual = weighted sum of |PC2..PCn| amplified by their singular values
    residual_flat = np.zeros(H * W, dtype=np.float32)
    for k in range(1, n_comp):
        weight = float(S[k]) / (float(S[1]) + 1e-12)  # normalise by second PC
        comp_contribution = np.abs(Vt[k, :]) * weight
        residual_flat += comp_contribution

    residual = residual_flat.reshape(H, W)
    # Percentile normalise
    p2, p98 = float(np.percentile(residual, 2)), float(np.percentile(residual, 98))
    rng = p98 - p2 if (p98 - p2) > 1e-8 else 1.0
    residual_norm = np.clip((residual - p2) / rng, 0, 1)
    residual_u8 = (residual_norm * 255).astype(np.uint8)

    # SNR map on residual
    snr_map, raw_snr_max = compute_snr_map(residual_norm, window_size=31, clip_sigma=3.0)

    # Defect segmentation
    rois = segment_defects(snr_map, residual_norm)

    stats = {
        'n_images': N,
        'explained_pc1': float(explained[0]),
        'explained_pc2': float(explained[1]) if N > 1 else 0.0,
        'explained_pc3': float(explained[2]) if N > 2 else 0.0,
        'residual_mean': float(residual_norm.mean()),
        'residual_std': float(residual_norm.std()),
        'snr_peak': raw_snr_max,
        'defect_count': len(rois),
    }

    return PCAFusionResult(
        residual_map=residual_u8,
        pc1_image=pc1_u8,
        snr_map=snr_map,
        defect_rois=rois,
        explained_variance=[float(e) for e in explained[:n_comp]],
        n_images=N,
        stats=stats,
    )


@dataclass
class QuadrantFusionResult:
    """Result of Quadrant Fusion (Illuminator + 4 Quadrant detectors)."""
    fused_image_uint8: np.ndarray          # Final display image (depends on output_type)
    topo_uint8: np.ndarray                 # Topography map (uint8)
    bse_clean_uint8: np.ndarray            # BSE enhanced map (uint8)
    composite_uint8: np.ndarray            # Composite map (uint8)
    alpha_used: float
    beta_used: float
    notes: str


def _percentile_to_uint8(img: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Normalize a float image to uint8 using percentile mapping."""
    p2, p98 = np.percentile(img, [p_low, p_high])
    rng = p98 - p2
    if rng < 1e-8:
        rng = 1.0
    normed = (img - p2) / rng
    return np.clip(normed * 255, 0, 255).astype(np.uint8)


def compute_quadrant_fusion(
    illum: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    output_type: str = "bse_clean",   # "bse_clean" | "topo" | "composite"
    alpha_mode: str = "auto",         # "auto" | "manual"
    alpha_manual: float = 1.0,
    beta: float = 0.3,
    gaussian_sigma: float = 0.0,      # 0 means no blur
    roi_rect: Optional[Tuple[int, int, int, int]] = None,  # (x, y, w, h) for alpha fit
) -> QuadrantFusionResult:
    """Compute Quadrant Fusion from Illuminator + 4 Quadrant detector images.

    Parameters
    ----------
    illum : Central / Illuminator (BSE-dominant) image.
    top, bottom, left, right : Quadrant detector images.
    output_type : Which map to use as the primary fused output.
    alpha_mode : 'auto' fits alpha from data; 'manual' uses alpha_manual.
    alpha_manual : Manual alpha value (used when alpha_mode='manual').
    beta : Weight for topography in composite output.
    gaussian_sigma : Gaussian smoothing sigma for topo map (0 = no blur).
    roi_rect : Optional (x, y, w, h) pixel ROI for alpha fitting.

    Returns
    -------
    QuadrantFusionResult with all output maps.
    """
    eps = 1e-7

    # 1) Convert to float32 and robust normalize (divide by median to reduce gain bias)
    def _gain_normalize(img: np.ndarray) -> np.ndarray:
        f = img.astype(np.float32)
        med = float(np.median(f))
        return f / (med + eps)

    illum_f = _gain_normalize(illum)
    top_f = _gain_normalize(top)
    bottom_f = _gain_normalize(bottom)
    left_f = _gain_normalize(left)
    right_f = _gain_normalize(right)

    # 2) Compute slope / topography
    tx = right_f - left_f
    ty = top_f - bottom_f
    topo = np.sqrt(tx * tx + ty * ty + eps)

    # Optional Gaussian smoothing on topo
    if gaussian_sigma > 0:
        ksize = int(np.ceil(gaussian_sigma * 3)) * 2 + 1  # ensure odd
        topo = cv2.GaussianBlur(topo, (ksize, ksize), gaussian_sigma)

    # 3) Alpha computation
    if alpha_mode == "manual":
        alpha = float(alpha_manual)
    else:
        # Auto fit
        if roi_rect is not None:
            x, y, w, h = roi_rect
            illum_roi = illum_f[y:y + h, x:x + w]
            topo_roi = topo[y:y + h, x:x + w]
        else:
            illum_roi = illum_f
            topo_roi = topo

        # Least-squares fit: alpha = sum(illum * topo) / sum(topo^2)
        topo_sq_sum = float(np.sum(topo_roi * topo_roi))
        if topo_sq_sum > eps:
            alpha = float(np.sum(illum_roi * topo_roi)) / topo_sq_sum
        else:
            # Fallback: median ratio
            topo_med = float(np.median(topo_roi))
            illum_med = float(np.median(illum_roi))
            alpha = illum_med / (topo_med + eps)

    # Clamp alpha to reasonable range
    alpha = float(np.clip(alpha, 0.0, 5.0))

    # 4) BSE enhanced (clean)
    bse_clean = illum_f - alpha * topo

    # 5) Normalize each map to uint8 for display
    topo_u8 = _percentile_to_uint8(topo)
    bse_clean_u8 = _percentile_to_uint8(bse_clean)

    # 6) Composite: normalize(bse_clean) + beta * normalize(topo), then clip
    composite_f = bse_clean_u8.astype(np.float32) + beta * topo_u8.astype(np.float32)
    composite_u8 = np.clip(composite_f, 0, 255).astype(np.uint8)

    # 7) Select fused image based on output_type
    _output_map = {
        "bse_clean": bse_clean_u8,
        "topo": topo_u8,
        "composite": composite_u8,
    }
    fused = _output_map.get(output_type, bse_clean_u8)

    notes = (
        f"output={output_type}, alpha_mode={alpha_mode}, "
        f"alpha={alpha:.4f}, beta={beta:.2f}, sigma={gaussian_sigma:.1f}"
    )

    return QuadrantFusionResult(
        fused_image_uint8=fused,
        topo_uint8=topo_u8,
        bse_clean_uint8=bse_clean_u8,
        composite_uint8=composite_u8,
        alpha_used=alpha,
        beta_used=beta,
        notes=notes,
    )


__all__ = [
    'CombineResult',
    'SinglePairResult',
    'OperationType',
    'DefectROI',
    'PCAFusionResult',
    'QuadrantFusionResult',
    'align_and_subtract',
    'compute_single_pair',
    'compute_multi_pairs',
    'compute_pca_fusion',
    'compute_quadrant_fusion',
    'compute_snr_map',
    'colorize_snr_map',
    'segment_defects',
]
