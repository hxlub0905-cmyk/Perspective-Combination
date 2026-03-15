"""E-beam SNR Analysis Module.

This module provides functionality for comparing defect SNR (Signal-to-Noise Ratio)
across different E-beam imaging conditions (kEV, current).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re
import os
import numpy as np
import cv2


@dataclass
class AlignResult:
    """Result of robust alignment process."""
    dx: int
    dy: int
    score_phase: float
    score_ncc: float
    score_residual: float
    final_score: float
    status: str  # 'ok', 'warn', 'fail'
    method: str  # 'phase', 'ncc', 'fallback'
    # Sub-pixel precision offsets (default to integer values for backward compat)
    dx_subpixel: float = 0.0
    dy_subpixel: float = 0.0


@dataclass
class EbeamCondition:
    """E-beam imaging condition parameters.
    
    Attributes:
        image_path: Full path to the image file.
        kev: E-beam voltage in kiloelectron volts.
        current_value: E-beam current value.
        current_unit: Current unit ('pA' or 'nA').
        label: Human-readable label for this condition.
        defect_id: Optional defect ID parsed from filename.
    """
    image_path: str
    kev: float
    current_value: float
    current_unit: str = "nA"  # Default to nA as per user's format
    label: str = ""
    defect_id: str = ""
    
    def __post_init__(self):
        if not self.label:
            self.label = f"{self.kev}keV/{self.current_value}{self.current_unit}"


@dataclass
class ROIRegion:
    """Region of Interest definition.
    
    Attributes:
        x: Top-left X coordinate.
        y: Top-left Y coordinate.
        width: Width of ROI.
        height: Height of ROI.
        name: Optional name/label for this ROI.
    """
    x: int
    y: int
    width: int
    height: int
    name: str = "ROI"
    
    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) tuple."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def slice(self) -> Tuple[slice, slice]:
        """Return numpy slicing tuple (row_slice, col_slice)."""
        return (slice(self.y, self.y + self.height), slice(self.x, self.x + self.width))


@dataclass
class SNRResult:
    """SNR calculation result for a single image/condition.
    
    Attributes:
        condition: The E-beam condition for this result.
        roi: The ROI used for calculation.
        snr: Calculated SNR value.
        defect_mean: Mean intensity in defect ROI.
        defect_std: Standard deviation in defect ROI.
        background_mean: Mean intensity in background region.
        background_std: Standard deviation in background region.
        contrast: Absolute contrast (defect_mean - background_mean).
        contrast_ratio: Ratio of defect to background intensity.
        edge_sharpness: Edge definition metric (Sobel gradient magnitude).
        dvi: Defect Visibility Index = SNR * sqrt(contrast_ratio).
    """
    condition: EbeamCondition
    roi: ROIRegion
    snr: float
    defect_mean: float
    defect_std: float
    background_mean: float
    background_std: float
    contrast: float = 0.0
    contrast_ratio: float = 1.0
    edge_sharpness: float = 0.0
    dvi: float = 0.0
    
    # Alignment diagnostics
    align_score: float = 100.0
    align_status: str = 'ok'  # 'ok', 'warn', 'fail'
    
    # Store all aligned ROIs for reporting/visualization
    all_rois: List[ROIRegion] = field(default_factory=list)
    shift_x: float = 0.0
    shift_y: float = 0.0
    
    def __post_init__(self):
        self.contrast = abs(self.defect_mean - self.background_mean)


@dataclass
class SNRAnalysisReport:
    """Aggregated SNR analysis results across multiple conditions.
    
    Attributes:
        results: List of individual SNR results.
        roi: The ROI used for analysis.
        summary_stats: Dictionary of aggregated statistics by kEV.
    """
    results: List[SNRResult] = field(default_factory=list)
    roi: Optional[ROIRegion] = None
    summary_stats: Dict[float, Dict[str, float]] = field(default_factory=dict)
    
    def compute_summary(self):
        """Compute summary statistics grouped by kEV."""
        kev_groups: Dict[float, List[float]] = {}
        
        for result in self.results:
            kev = result.condition.kev
            if kev not in kev_groups:
                kev_groups[kev] = []
            kev_groups[kev].append(result.snr)
        
        self.summary_stats = {}
        for kev, snr_values in kev_groups.items():
            arr = np.array(snr_values)
            self.summary_stats[kev] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr)),
                'count': len(snr_values),
            }
    
    def get_optimal_condition(self) -> Optional[SNRResult]:
        """Return the condition with highest SNR."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.snr)


def parse_filename_conditions(
    filepath: str,
    pattern: Optional[str] = None
) -> Optional[EbeamCondition]:
    """Extract E-beam conditions from image filename.
    
    Supports common patterns:
    - DID1_12keV_5nA.tif  (user's format)
    - DID3_19keV_7nA.png
    - sample_10kev_50pA.tif
    - image_15KEV_100PA.png
    
    Args:
        filepath: Path to the image file.
        pattern: Optional regex pattern with named groups 'kev', 'current', 'unit'.
    
    Returns:
        EbeamCondition if parsing successful, None otherwise.
    """
    basename = os.path.basename(filepath)
    
    # Default patterns to try (ordered by priority)
    # Pattern 1: DID_12keV_5nA format (user's primary format)
    # Pattern 2: General keV_currentnA/pA format
    # Pattern 3: kV with pA
    default_patterns = [
        # DID{n}_{kev}keV_{current}nA format (user's format with defect ID)
        r'(?P<did>[A-Za-z]*\d*)_(?P<kev>\d+(?:\.\d+)?)[kK][eE]?[vV]_(?P<current>\d+(?:\.\d+)?)(?P<unit>[npNP][aA])',
        # Generic: {kev}keV_{current}nA/pA
        r'(?P<kev>\d+(?:\.\d+)?)[kK][eE]?[vV][_\s]*(?P<current>\d+(?:\.\d+)?)(?P<unit>[npNP][aA])',
        # Generic: {kev}kV_{current}pA
        r'(?P<kev>\d+(?:\.\d+)?)[kK][vV][_\s]*(?P<current>\d+(?:\.\d+)?)(?P<unit>[pP][aA])',
        # Fallback: keV anywhere followed by current pA/nA
        r'(?P<kev>\d+(?:\.\d+)?)[kK][eE]?[vV].*?(?P<current>\d+(?:\.\d+)?)(?P<unit>[npNP][aA])',
    ]
    
    patterns_to_try = [pattern] if pattern else default_patterns
    
    for pat in patterns_to_try:
        if not pat:
            continue
        match = re.search(pat, basename)
        if match:
            try:
                kev = float(match.group('kev'))
                current = float(match.group('current'))
                
                # Get unit (default to nA)
                unit = 'nA'
                try:
                    unit_match = match.group('unit')
                    if unit_match:
                        unit = unit_match.lower()
                        unit = 'nA' if 'n' in unit else 'pA'
                except (IndexError, AttributeError):
                    pass
                
                # Get defect ID if available
                defect_id = ''
                try:
                    defect_id = match.group('did') or ''
                except (IndexError, AttributeError):
                    pass
                
                return EbeamCondition(
                    image_path=filepath,
                    kev=kev,
                    current_value=current,
                    current_unit=unit,
                    defect_id=defect_id
                )
            except (ValueError, IndexError):
                continue
    
    return None


def load_image_gray(filepath: str) -> Optional[np.ndarray]:
    """Load image as grayscale numpy array.
    
    Args:
        filepath: Path to the image file.
    
    Returns:
        Grayscale image as uint8 numpy array, or None if failed.
    """
    if not os.path.isfile(filepath):
        return None
    
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Try with different backends for special formats
            img = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
            if img is not None:
                # Normalize to uint8
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return img
    except Exception:
        return None


def calculate_roi_snr(
    image: np.ndarray,
    roi: ROIRegion,
    background_margin: int = 20
) -> Optional[Dict[str, float]]:
    """Calculate SNR for a defect ROI.
    
    SNR = (μ_defect - μ_background) / σ_background
    
    Args:
        image: Grayscale image as numpy array.
        roi: The defect region of interest.
        background_margin: Pixels to expand around ROI for background estimation.
    
    Returns:
        Dictionary with SNR and related statistics, or None if failed.
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # Validate ROI bounds
    x1, y1 = max(0, roi.x), max(0, roi.y)
    x2, y2 = min(w, roi.x + roi.width), min(h, roi.y + roi.height)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Extract defect region
    defect_region = image[y1:y2, x1:x2].astype(np.float32)
    
    if defect_region.size == 0:
        return None
    
    # Calculate defect statistics with numerical overflow protection
    defect_mean = float(np.clip(np.mean(defect_region), -1e6, 1e6))
    defect_std = float(np.clip(np.std(defect_region), 0, 1e6))
    
    # Define background region (expanded area around ROI, excluding ROI itself)
    bg_x1 = max(0, x1 - background_margin)
    bg_y1 = max(0, y1 - background_margin)
    bg_x2 = min(w, x2 + background_margin)
    bg_y2 = min(h, y2 + background_margin)
    
    # Create mask for background (expanded region minus defect region)
    bg_mask = np.zeros((bg_y2 - bg_y1, bg_x2 - bg_x1), dtype=bool)
    bg_mask[:, :] = True
    
    # Mask out the defect region
    inner_x1 = x1 - bg_x1
    inner_y1 = y1 - bg_y1
    inner_x2 = x2 - bg_x1
    inner_y2 = y2 - bg_y1
    bg_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
    
    background_region = image[bg_y1:bg_y2, bg_x1:bg_x2].astype(np.float32)
    background_values = background_region[bg_mask]
    
    if background_values.size < 10:
        # Not enough background pixels, use whole expanded region
        background_values = background_region.flatten()
    
    # Calculate background statistics with numerical overflow protection
    background_mean = float(np.clip(np.mean(background_values), -1e6, 1e6))
    background_std = float(np.clip(np.std(background_values), 0, 1e6))
    
    # Calculate SNR with enhanced boundary condition handling
    if background_std < 1e-8 or not np.isfinite(background_std):
        snr = 0.0
    else:
        contrast_value = abs(defect_mean - background_mean)
        snr = float(np.clip(contrast_value / background_std, 0, 1e6))
    
    # Calculate Contrast Ratio with numerical stability
    if abs(background_mean) > 1e-8 and np.isfinite(background_mean):
        contrast_ratio = float(np.clip(defect_mean / background_mean, -1e6, 1e6))
    else:
        contrast_ratio = 0.0
    
    # Calculate Edge Sharpness (using Sobel gradient magnitude at ROI boundary)
    try:
        # Get the defect patch for edge analysis
        defect_patch = image[y1:y2, x1:x2]
        if defect_patch.size > 0 and defect_patch.shape[0] >= 3 and defect_patch.shape[1] >= 3:
            sobel_x = cv2.Sobel(defect_patch, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(defect_patch, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_sharpness = float(np.clip(np.mean(gradient_mag), 0, 1e6))
        else:
            edge_sharpness = 0.0
    except Exception:
        edge_sharpness = 0.0
    
    # Calculate DVI (Defect Visibility Index) with numerical protection
    try:
        if contrast_ratio > 0 and np.isfinite(contrast_ratio):
            sqrt_contrast = np.sqrt(abs(contrast_ratio))
            dvi = float(np.clip(snr * sqrt_contrast, 0, 1e6))
        else:
            dvi = float(np.clip(snr, 0, 1e6))
    except (ValueError, OverflowError):
        dvi = float(np.clip(snr, 0, 1e6))
    
    # Final validation of all return values
    contrast = float(np.clip(abs(defect_mean - background_mean), 0, 1e6))
    
    return {
        'snr': snr,
        'defect_mean': defect_mean,
        'defect_std': defect_std,
        'background_mean': background_mean,
        'background_std': background_std,
        'contrast': contrast,
        'contrast_ratio': contrast_ratio,
        'edge_sharpness': edge_sharpness,
        'dvi': dvi,
    }


def batch_snr_analysis(
    conditions: List[EbeamCondition],
    roi: ROIRegion,
    background_margin: int = 20
) -> SNRAnalysisReport:
    """Perform SNR analysis across multiple E-beam conditions.
    
    Args:
        conditions: List of E-beam conditions with image paths.
        roi: The ROI to analyze (same position on all images).
        background_margin: Pixels for background estimation.
    
    Returns:
        SNRAnalysisReport with all results and summary statistics.
    """
    report = SNRAnalysisReport(roi=roi)
    
    for condition in conditions:
        image = load_image_gray(condition.image_path)
        if image is None:
            continue
        
        stats = calculate_roi_snr(image, roi, background_margin)
        if stats is None:
            continue
        
        result = SNRResult(
            condition=condition,
            roi=roi,
            snr=stats['snr'],
            defect_mean=stats['defect_mean'],
            defect_std=stats['defect_std'],
            background_mean=stats['background_mean'],
            background_std=stats['background_std'],
            contrast_ratio=stats.get('contrast_ratio', 1.0),
            edge_sharpness=stats.get('edge_sharpness', 0.0),
            dvi=stats.get('dvi', 0.0),
        )
        report.results.append(result)
    
    report.compute_summary()
    return report


def scan_folder_for_conditions(
    folder_path: str,
    extensions: Tuple[str, ...] = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
) -> List[EbeamCondition]:
    """Scan folder for images and try to parse E-beam conditions from filenames.
    
    Args:
        folder_path: Path to folder containing images.
        extensions: File extensions to include.
    
    Returns:
        List of successfully parsed EbeamCondition objects.
    """
    conditions = []
    
    if not os.path.isdir(folder_path):
        return conditions
    
    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(extensions):
            continue
        
        filepath = os.path.join(folder_path, filename)
        condition = parse_filename_conditions(filepath)
        
        if condition:
            conditions.append(condition)
    
    return conditions


def _preprocess_for_align(img: np.ndarray) -> np.ndarray:
    """Preprocess image for alignment (Robust Norm + Sobel)."""
    if img is None:
        return None
        
    img_f = img.astype(np.float32)
    # Robust normalization (5th-95th percentile)
    p5, p95 = np.percentile(img_f, [5, 95])
    rng = p95 - p5
    if rng < 1e-6:
        rng = 1.0
    img_n = np.clip((img_f - p5) / rng, 0, 1)
    
    # Sobel Edge Detection
    # Ensure explicit float32 for OpenCV compatibility
    gx = cv2.Sobel(img_n.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_n.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    
    # Normalize magnitude
    mmax = mag.max()
    if mmax > 1e-6:
        mag /= mmax
        
    return mag


def calculate_alignment_robust(
    base: np.ndarray,
    target: np.ndarray,
    search_radius: int = 40
) -> AlignResult:
    """Layered robust alignment strategy.
    
    Strategy:
    1. Preprocess (Edge map) to ignore brightness diffs.
    2. Phase Correlation (Layer 1) for fast translation estimate.
    3. Verification & Grading.
    """
    if base is None or target is None:
        return AlignResult(0, 0, 0.0, 0.0, 0.0, 0.0, 'fail', 'none')
        
    h, w = base.shape[:2]

    # Performance guard for very large images:
    # phaseCorrelate runs FFT internally, so runtime grows quickly with image size.
    # Downsample to a working resolution, then scale sub-pixel shift back.
    max_dim = max(h, w)
    align_max_dim = 2048
    scale_x = 1.0
    scale_y = 1.0
    if max_dim > align_max_dim:
        down = align_max_dim / float(max_dim)
        work_w = max(64, int(round(w * down)))
        work_h = max(64, int(round(h * down)))
        scale_x = w / float(work_w)
        scale_y = h / float(work_h)
    else:
        work_w = w
        work_h = h
    
    # 1. Preprocessing
    base_edge = _preprocess_for_align(base)
    target_edge = _preprocess_for_align(target)
    if (work_w, work_h) != (w, h):
        base_edge = cv2.resize(base_edge, (work_w, work_h), interpolation=cv2.INTER_AREA)
        target_edge = cv2.resize(target_edge, (work_w, work_h), interpolation=cv2.INTER_AREA)
    
    # 2. Phase Correlation (Layer 1)
    # Create Hanning window to reduce edge effects
    hann = cv2.createHanningWindow((work_w, work_h), cv2.CV_32F)
    (dx_p, dy_p), response_p = cv2.phaseCorrelate(base_edge, target_edge, window=hann)

    # Convert working-resolution shift back to original coordinates.
    dx_p *= scale_x
    dy_p *= scale_y
    
    # Convert phase shift (target->base)
    # cv2.phaseCorrelate returns shift of src2 relative to src1.
    # So dx, dy is the shift.
    
    # 3. Residual Verification
    # Shift target edge back by (-dx, -dy)
    M = np.float32([[1, 0, -(dx_p / scale_x)], [0, 1, -(dy_p / scale_y)]])
    aligned_edge = cv2.warpAffine(target_edge, M, (work_w, work_h), flags=cv2.INTER_LINEAR)
    
    # Calculate residual (Difference in overlap area)
    diff = np.abs(base_edge - aligned_edge)
    # Ignore border regions affected by shift
    border = 5 + int(max(abs(dx_p / scale_x), abs(dy_p / scale_y)))
    if border < work_h//2 and border < work_w//2:
        valid_diff = diff[border:-border, border:-border]
    else:
        valid_diff = diff
        
    residual_mean = float(np.mean(valid_diff))
    score_residual = max(0.0, 1.0 - residual_mean * 2.0) # Heuristic scaling
    
    # 4. Composite Score
    # Weights: Phase=0.4, NCC/Legacy=0.0 (using residual instead), Residual=0.6
    # Actually let's calculate NCC on the aligned patch as confirmation if needed
    # For now, use Phase + Residual
    final_score = (0.4 * response_p + 0.6 * score_residual) * 100.0
    
    # 5. Grading
    status = 'ok'
    if final_score < 75:
        status = 'warn'
    if final_score < 55:
        status = 'fail'
        
    # Limit shift to search radius
    if abs(dx_p) > search_radius or abs(dy_p) > search_radius:
        # If phase corr says huge shift, it might be wrong (or just huge).
        # We can trigger fallback here.
        status = 'warn' 
        # Clamp? No, report it but warn.
        
    return AlignResult(
        dx=int(round(dx_p)),
        dy=int(round(dy_p)),
        score_phase=float(response_p),
        score_ncc=0.0, # Not computed yet
        score_residual=score_residual,
        final_score=final_score,
        status=status,
        method='phase',
        dx_subpixel=float(dx_p),
        dy_subpixel=float(dy_p),
    )


def align_images_ncc(
    base_image: np.ndarray,
    target_image: np.ndarray,
    search_radius: int = 50
) -> Tuple[int, int, float]:
    """Legacy wrapper using robust alignment."""
    res = calculate_alignment_robust(base_image, target_image, search_radius)
    # Mapping score 0-100 back to 0-1 approx for legacy compatibility
    return (res.dx, res.dy, res.final_score / 100.0)


def apply_offset_to_roi(roi: ROIRegion, offset: Tuple[int, int]) -> ROIRegion:
    """Apply alignment offset to a ROI.
    
    Args:
        roi: Original ROI region.
        offset: (dx, dy) offset to apply.
    
    Returns:
        New ROIRegion with adjusted coordinates.
    """
    dx, dy = offset
    return ROIRegion(
        x=roi.x + dx,
        y=roi.y + dy,
        width=roi.width,
        height=roi.height,
        name=roi.name
    )


def group_conditions_by_did(
    conditions: List[EbeamCondition]
) -> Dict[str, List[EbeamCondition]]:
    """Group E-beam conditions by their Defect ID.
    
    Args:
        conditions: List of E-beam conditions.
    
    Returns:
        Dictionary mapping defect_id to list of conditions.
    """
    groups: Dict[str, List[EbeamCondition]] = {}
    
    for cond in conditions:
        did = cond.defect_id or 'Unknown'
        if did not in groups:
            groups[did] = []
        groups[did].append(cond)
    
    return groups
