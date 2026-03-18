# Plan: Compare dSNR (Normalized Compare SNR) Feature

## Background
When Compare (e.g. 15k) is normalized using Base's (e.g. 5k) GLV Mask p2/p98 range,
`clip()` introduces non-linearity — pixels outside [p2, p98] are clamped to 0 or 1.
This degrades the SNR of the Compare image itself. The user wants to quantify this
degradation per pair (6 conditions → 30 pairs) and display it in the existing UI.

## What "Compare dSNR" means
- Compute ROI-based SNR on the **normalized compare image** (before subtraction)
  using the same target/reference ROI structure:
  `comp_snr = (μ_target - μ_ref) / σ_ref` on `comp_norm`
- This differs from the existing diff SNR which measures signal in the **subtracted** image.

## Implementation Steps

### Step 1: Extend `ROIFullResult` data structure (`roi_set.py`)

- Add new field: `snr_per_compare: Dict[str, ROISNREntry]` (parallel to existing `snr_per_diff`)
- `ROISNREntry` already has all needed fields (`le_label`, `snr`, `mu_target`, `mu_ref`, `sigma_ref`)

### Step 2: Compute compare SNR in `compute_roi_full_stats()` (`perspective_combine.py`)

After computing `comp_norm` and building the COMPARE layer (line ~968), compute SNR:
- Extract target ROI stats from `comp_layer`
- Extract reference ROI stats from `comp_layer`
- Apply same SNR formula: `SNR = (μ_target - μ_ref) / (σ_ref + ε)`
- Store in `result.snr_per_compare[le_label]`

### Step 3: Add "Compare SNR Pair Matrix" tab in `ROIIntensityProfileDialog` (`dialog.py`)

- New method `_build_compare_snr_matrix_tab()` — same N×N heatmap style as `_build_matrix_tab()`
- Rows = Base (anchor), Columns = Compare (LE)
- Cell value = `snr_per_compare[compare_label].snr`
- Added in `_build_ui()` as a new tab next to "SNR Pair Matrix"

### Step 4: Add Compare SNR line to Intensity Profile (`_refresh_mean_plot`)

In `_refresh_mean_plot()`, add a 4th line series:
- "Comp SNR" line (or annotation) showing the compare SNR value at each LE x-position
- Use a secondary y-axis for SNR scale since units differ from mean intensity

### Step 5: Add compare SNR column to LE Summary table

In `_build_summary_tab()`, add a "Comp SNR" column showing `snr_per_compare` values
alongside the existing diff SNR column.

## Files to modify
1. `perscomb/core/roi_set.py` — add `snr_per_compare` field to `ROIFullResult`
2. `perscomb/core/perspective_combine.py` — compute compare SNR in `compute_roi_full_stats()`
3. `perscomb/ui/dialog.py` — new matrix tab + intensity profile line + summary column
