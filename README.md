## Project Overview

CT scan stem cross-section area calculator. Input: `.tif` images of plant stem cross-sections on a roughly constant background. Output: stem cross-sectional area in pixels, saved to CSV.

Example output (4-panel QC figure):
![Example image](data/figures/11%20-%20slice00659.png)

The four panels show (left to right): the original image with the annotated center (+) and edge point (×); the polar transform with the annotated radius (red dashed) and Viterbi path (green); the binary radial-contrast image used by the HMM; and the final mask overlay with the computed area.

---

## Setup

### 1. Install dependencies

```bash
pip install numpy tifffile matplotlib scipy python-dotenv
```

### 2. Configure the images path

Create a `.env` file in the project root (next to `annotate.py`) with the path to your folder of `.tif` images:

```
RAW_IMAGES_DIR=C:\Users\yourname\path\to\images
```

The folder should contain files named `{sample} - slice{slice}.tif` (e.g. `10 - slice00659.tif`).

> The `.env` file is gitignored — it is machine-specific and should not be committed.

---

## Usage

### Step 1 — Annotate images (`annotate.py`)

Run:

```bash
python annotate.py
```

A fullscreen window opens showing each unannotated image in turn. For each image:

1. **Click the stem center** (marked with a red `+`)
2. **Click a point on the stem edge** (marked with a blue `×`)

The annotation is saved immediately to `data/csv-outputs/annotations.csv`. The script then advances to the next image automatically.

**Keyboard shortcuts:**
| Key | Action |
|---|---|
| `Backspace` | Undo the last saved annotation and re-show that image |
| `Esc` | Pause and exit — re-run the script to resume where you left off |

Already-annotated images are skipped on re-run, so the session is fully resumable.

### Step 2 — Compute areas (`areas.py`)

Run:

```bash
python areas.py
```

For each annotated image not yet processed, the script:

1. Loads and blurs the image
2. Converts to polar coordinates centred on the annotated stem center
3. Computes a binary radial-contrast image
4. Runs the Viterbi algorithm to trace the stem edge
5. Computes the cross-sectional area (px²) from the traced contour
6. Saves a 4-panel QC figure to `data/figures/<filename>.png`
7. Writes the result to `data/csv-outputs/areas.csv` (updates existing row if present)

Progress is printed to the console. Already-processed images are skipped, so the script is also resumable.

Optional CLI arguments:

- `--filename "<name>.tif"` processes one annotated image by exact filename.
- `--force` recomputes even if that filename already exists in `areas.csv` (must be used with `--filename`).

Examples:

```bash
python areas.py
python areas.py --filename "10 - slice00659.tif"
python areas.py --filename "10 - slice00659.tif" --force
```

### Step 3 — Inspect results

**QC figures** — open any `.png` in `data/figures/` to visually verify the traced edge. The green line in the polar and contrast panels shows the Viterbi path; it should follow the bright boundary of the stem.

**Areas CSV** — `data/csv-outputs/areas.csv` contains one row per image:

| filename | area_px |
|---|---|
| 10 - slice00659.tif | 18423.50 |
| 11 - slice00659.tif | 17891.00 |
| ... | ... |

---

## Tunable parameters

These are global variables at the top of `areas.py`:

| Parameter | Default | Role |
|---|---|---|
| `TRANSITION_SIGMA` | 3 px | HMM state-transition spread — larger values allow the edge to move more between adjacent angles |
| `EDGE_EMISSION_PROB` | 0.9 | P(contrast pixel = 1 \| truly at edge) |
| `NON_EDGE_EMISSION_PROB` | 0.5 | P(contrast pixel = 1 \| not at edge) |
| `CONTRAST_RADIAL_WEIGHT` | 3.0 | Weight of radial vs angular gradient in the contrast image |
| `CONTRAST_THRESHOLD_PCTILE` | 95 | Percentile cutoff for binarising the contrast image |
| `GAUSSIAN_SIGMA` | 1 px | Pre-blur applied before all processing |

---

## Repository layout

```
stem-detection/
  annotate.py              # Phase 1: interactive annotation
  areas.py                 # Phase 2: area computation
  .env                     # (gitignored) RAW_IMAGES_DIR path
  data/
    csv-outputs/
      annotations.csv      # Phase 1 output: filename, center and edge coords
      areas.csv            # Phase 2 output: filename, area in px²
    figures/               # Phase 2 output: 4-panel QC images
```
