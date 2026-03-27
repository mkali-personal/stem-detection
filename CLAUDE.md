# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CT scan stem cross-section area calculator. Input: `.tif` images of plant stem cross-sections on a roughly constant background. Output: stem cross-sectional area in pixels, saved to CSV.

## Data Flow

**Phase 1 — Manual annotation (`data/raw-images/` → `data/csv-outputs/annotations.csv`):**
1. Load `.tif` image as numpy array, apply Gaussian blur
2. Display with matplotlib; user clicks (1) stem center, (2) a point on the stem edge
3. Save coordinates + filename to `data/csv-outputs/annotations.csv` after each image (resumable)

**Phase 2 — Area calculation (`data/csv-outputs/annotations.csv` → `data/csv-outputs/areas.csv`):**
1. Load image, blur.
2. Convert to polar coordinates (origin = stem center)
3. Compute binary contrast image, giving more weight to contrast in the radial direction (e.g. by applying a filter that emphasizes radial gradients)
4. Rotate polar image so the clicked edge point is at θ=0
5. Run Viterbi/HMM to trace the stem edge in polar coordinates:
   - States = possible radii; observations = binary contrast pixels
   - Transition: Gaussian centered on same radius, σ=3px (tunable global)
   - Emission: P(contrast=1 | at edge)=0.9, P(contrast=1 | not edge)=0.5 (tunable globals)
   - Initial state fixed at clicked radius; final state forced back to clicked radius
6. Area = sum of traced radii × angle step
7. Overlay mask on original image for visual QC; save area to `data/csv-outputs/areas.csv`

## Key Parameters (global variables to expose)

| Parameter | Default | Role |
|---|---|---|
| Transition σ | 3 px | HMM state transition spread |
| Edge emission prob | 0.9 | P(contrast pixel at true edge) |
| Non-edge emission prob | 0.5 | P(contrast pixel away from edge) |

## Tech Stack

Python · numpy · matplotlib (interactive click capture) · scipy (Gaussian blur, polar transform) · HMM/Viterbi (custom implementation expected)

## Repository Layout

```
stem-detection/
  readme.md
  CLAUDE.md
  data/
    raw-images/          # input .tif files, named "{sample} - slice{slice}.tif"
    csv-outputs/
      annotations.csv    # Phase 1 output: filename, center coords, edge point coords
      areas.csv          # Phase 2 output: filename, stem area in pixels
```
