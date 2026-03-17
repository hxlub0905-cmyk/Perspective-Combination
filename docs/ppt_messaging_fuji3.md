# Fuji3 (Perspective Combination) — Diff-Centric PPT Messaging Pack

## Slide Title Options (English)
1. **Fuji3: Diff-First SEM Defect Detection**
2. **Fuji3: Align, Diff, Detect**
3. **Perspective Combination: SEM Defects Revealed by Image Difference**
4. **From Aligned SEM Images to Defect Diff Maps**
5. **Fuji3: Image Subtraction for Quantifiable Defect Analysis**

---

## Background (Why this tool was developed)
- In SEM inspection, subtle defects are often buried in raw grayscale texture and are hard to detect by visual review alone.
- Engineers typically compare images across conditions, but manual side-by-side checking is slow and subjective.
- Without accurate alignment, subtraction results can produce false differences, reducing trust in conclusions.
- Fuji3 was developed to make **aligned image subtraction (Diff)** the core workflow, then reinforce interpretation with SNR and ROI quantification.

---

## Goal
### One-line goal
**To build a reliable Diff-centric SEM analysis workflow that converts aligned image subtraction into clear, quantifiable defect evidence.**

### 3-bullet goal (for slide body)
- Maximize defect visibility using robust alignment followed by **Base − Compare** image subtraction.
- Convert Diff intensity into trustworthy metrics through SNR and ROI-based quantification.
- Provide an end-to-end desktop pipeline from image loading to Diff visualization and reporting.

---

## Image Process Flow (PPT-friendly, Diff-Centric)
### Full flow
**Input SEM Images (Base + Compare)**
→ **Preprocess / Normalize**
→ **Robust Alignment**
→ **Core Step: Diff (Base − Compare)**
→ **Optional Support Ops: Blend / Invert**
→ **Diff Map + SNR Enhancement**
→ **ROI-based Defect Quantification**
→ **Visualization & Report Output**

### Compact one-line flow
**Load → Align → Diff → Enhance (SNR) → Quantify (ROI) → Report**

---

## Ready-to-paste single-slide version
**Title:** *Fuji3: Diff-First SEM Defect Detection*

**Background:**
Manual SEM comparison across conditions is time-consuming and easily biased, while small misalignment can mask or fabricate defects. Fuji3 was created to center analysis on aligned image subtraction (Diff), so defect signals become clearer and more consistent.

**Goal:**
Establish a dependable pipeline that uses alignment + Diff maps + SNR/ROI quantification to identify and measure potential defects.

**Image Process Flow:**
Load SEM Images → Normalize → Align → **Diff (Base − Compare)** → SNR Enhancement → ROI Quantification → Visualization/Report

---

## 3-slide mini deck outline (optional)
### Slide 1 — Why Diff Matters
- Visual-only inspection misses subtle patterns.
- Diff suppresses common background and highlights local change.
- Accurate alignment is the prerequisite for trustworthy Diff.

### Slide 2 — Method (Fuji3 Diff Workflow)
- Inputs: Base + Compare SEM images.
- Pipeline: alignment → **Diff (core)** → SNR map → ROI quantification.
- Outputs: defect-focused maps and numerical evidence.

### Slide 3 — Value
- Faster and more objective defect screening.
- Stronger consistency across users and samples.
- Better decision confidence with Diff-centered quantitative evidence.
