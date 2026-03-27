import csv
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

GAUSSIAN_SIGMA              = 1
TRANSITION_SIGMA            = 3    # HMM state transition spread (for later step)
EDGE_EMISSION_PROB          = 0.9  # P(contrast pixel at true edge)
NON_EDGE_EMISSION_PROB      = 0.5  # P(contrast pixel away from edge)
CONTRAST_RADIAL_WEIGHT      = 3.0  # radial gradient weight relative to angular
CONTRAST_THRESHOLD_PCTILE   = 95   # percentile cutoff for binarisation

RAW_IMAGES_DIR    = Path(__file__).parent / "data/raw-images"
ANNOTATIONS_CSV   = Path(__file__).parent / "data/csv-outputs/annotations.csv"
AREAS_CSV         = Path(__file__).parent / "data/csv-outputs/areas.csv"
AREAS_CSV_COLUMNS = ["filename", "area_px"]


def load_done_set() -> set:
    if not AREAS_CSV.exists():
        return set()
    with open(AREAS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            raise ValueError(
                f"Malformed CSV at {AREAS_CSV}: expected header with 'filename'"
            )
        return {row["filename"] for row in reader}


def load_annotations() -> list:
    if not ANNOTATIONS_CSV.exists():
        raise FileNotFoundError(f"Annotations file not found: {ANNOTATIONS_CSV}")
    with open(ANNOTATIONS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                "filename": row["filename"],
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "edge_x":   float(row["edge_x"]),
                "edge_y":   float(row["edge_y"]),
            }
            for row in reader
        ]


def get_pending_annotations(annotations: list, done_set: set) -> list:
    return [ann for ann in annotations if ann["filename"] not in done_set]


def load_and_blur(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        print(f"  Warning: {path.name} is a 3D stack — using first slice only.")
        arr = arr[0]
    return gaussian_filter(arr.astype(float), sigma=GAUSSIAN_SIGMA)


def compute_radius(ann: dict) -> float:
    dx = ann["edge_x"] - ann["center_x"]
    dy = ann["edge_y"] - ann["center_y"]
    return np.sqrt(dx**2 + dy**2)


def to_polar(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    r_annotated: float,
    n_angles: int = 360,
) -> tuple:
    """Convert image region to polar coordinates centred on (center_x, center_y).

    The radial range covers [0, min(2 * r_annotated, closest image boundary)].

    Returns (polar, angle_step_rad, radius_step_px) where polar has shape
    (n_angles, n_radii) with angle on axis 0.
    """
    h, w = image.shape
    max_radius = min(2.0 * r_annotated,
                     center_x, center_y, w - center_x, h - center_y)
    n_radii = int(np.ceil(max_radius))
    angle_step = 2 * np.pi / n_angles
    radius_step = max_radius / n_radii

    angles = np.arange(n_angles) * angle_step   # (n_angles,)
    radii  = np.arange(n_radii)  * radius_step  # (n_radii,)

    # indexing="ij" → output shape (n_angles, n_radii)
    angle_grid, radius_grid = np.meshgrid(angles, radii, indexing="ij")

    # Polar → Cartesian; sample_x maps to columns, sample_y maps to rows
    sample_x = center_x + radius_grid * np.cos(angle_grid)
    sample_y = center_y + radius_grid * np.sin(angle_grid)

    col_idx = np.clip(np.round(sample_x).astype(int), 0, w - 1)
    row_idx = np.clip(np.round(sample_y).astype(int), 0, h - 1)

    polar = image[row_idx, col_idx]
    return polar, angle_step, radius_step


def rotate_to_edge(polar: np.ndarray, ann: dict, angle_step: float) -> np.ndarray:
    """Cyclically rotate the polar array so the annotated edge point is at angle index 0.

    The edge angle is computed from the vector (center → edge) using atan2, then
    converted to the nearest angle bin and applied as a np.roll shift.
    """
    dx = ann["edge_x"] - ann["center_x"]
    dy = ann["edge_y"] - ann["center_y"]
    theta = np.arctan2(dy, dx) % (2 * np.pi)   # in [0, 2π)
    shift = int(np.round(theta / angle_step)) % polar.shape[0]
    return np.roll(polar, -shift, axis=0)


def compute_contrast(polar: np.ndarray) -> np.ndarray:
    """Compute a binary contrast image from a polar-coordinate intensity array.

    Radial gradients (axis 1) are weighted more heavily than angular gradients
    (axis 0) so that the stem boundary — which runs along the angular direction —
    produces a strong response.  Both positive and negative radial transitions
    (bright→dark and dark→bright) are captured via the absolute value.

    Returns a binary array of the same shape as `polar`.
    """
    grad_angle  = np.gradient(polar, axis=0)   # along angle axis
    grad_radius = np.gradient(polar, axis=1)   # along radius axis

    weighted = CONTRAST_RADIAL_WEIGHT * np.abs(grad_radius) + np.abs(grad_angle)
    weighted = np.maximum(weighted, 0.0)

    threshold = np.percentile(weighted, CONTRAST_THRESHOLD_PCTILE)
    return (weighted > threshold).astype(float)


def viterbi(contrast: np.ndarray, r0: int, radius_step: float) -> np.ndarray:
    """Trace the stem edge in polar coordinates using the Viterbi algorithm.

    States are radius indices. The path starts and ends at r0 (the annotated
    edge radius), enforcing a closed contour.

    Args:
        contrast:    Binary contrast array, shape (n_angles, n_radii).
        r0:          Initial (and forced final) radius state index.
        radius_step: Pixel width of each radius bin, used to convert
                     TRANSITION_SIGMA from pixels to bins.

    Returns:
        path: int array of shape (n_angles,) — the radius index at each angle.
    """
    n_angles, n_radii = contrast.shape
    sigma_bins = TRANSITION_SIGMA / radius_step

    # Log transition matrix: log_trans[r, r'] = log P(r'|r) up to a constant.
    # Normaliser is the same for every r, so it doesn't affect argmax.
    r_idx = np.arange(n_radii)
    diff = r_idx[:, None] - r_idx[None, :]          # (n_radii, n_radii)
    log_trans = -0.5 * (diff / sigma_bins) ** 2     # (n_radii, n_radii)

    # Log emission score for being at the edge vs not (relative, drops constants).
    # Shape (n_angles, n_radii); positive where contrast=1, negative where contrast=0.
    log_emit = np.where(
        contrast > 0.5,
        np.log(EDGE_EMISSION_PROB    / NON_EDGE_EMISSION_PROB),
        np.log((1 - EDGE_EMISSION_PROB) / (1 - NON_EDGE_EMISSION_PROB)),
    )

    # Initialise: only r0 is a valid starting state.
    log_v = np.full(n_radii, -np.inf)
    log_v[r0] = log_emit[0, r0]

    backtrack = np.zeros((n_angles, n_radii), dtype=np.int32)

    # Forward pass.
    for t in range(1, n_angles):
        # scores[r, r'] = best log-prob of reaching r' via r
        scores = log_v[:, None] + log_trans          # (n_radii, n_radii)
        backtrack[t] = np.argmax(scores, axis=0)     # (n_radii,)
        log_v = scores[backtrack[t], np.arange(n_radii)] + log_emit[t]

    # Backtrack from forced final state r0.
    path = np.empty(n_angles, dtype=np.int32)
    path[-1] = r0
    for t in range(n_angles - 2, -1, -1):
        path[t] = backtrack[t + 1, path[t + 1]]

    return path


def compute_area(path: np.ndarray, radius_step: float, angle_step: float) -> float:
    """Compute stem cross-sectional area in pixels² from the Viterbi path.

    Uses the standard polar area formula: A = 0.5 * Σ r_i² * dθ,
    where r_i = path[i] * radius_step is the radius in pixels at angle i.
    """
    radii_px = path * radius_step
    return 0.5 * np.sum(radii_px ** 2) * angle_step


def path_to_mask(
    image_shape: tuple,
    center_x: float,
    center_y: float,
    edge_x: float,
    edge_y: float,
    path: np.ndarray,
    radius_step: float,
    angle_step: float,
) -> np.ndarray:
    """Convert a Viterbi path in polar coordinates to a boolean mask in image space.

    For each image pixel, its polar coordinates are computed relative to the stem
    center, accounting for the cyclic rotation applied by rotate_to_edge. The pixel
    is inside the mask if its radius is less than the path radius at its angle.

    Returns a boolean array of shape image_shape.
    """
    h, w = image_shape
    ys, xs = np.mgrid[:h, :w]
    dx = xs.astype(float) - center_x
    dy = ys.astype(float) - center_y

    pixel_r     = np.sqrt(dx ** 2 + dy ** 2)
    pixel_theta = np.arctan2(dy, dx) % (2 * np.pi)

    # Subtract the edge-point angle to match the rotation in rotate_to_edge
    edge_theta     = np.arctan2(edge_y - center_y, edge_x - center_x) % (2 * np.pi)
    rotated_theta  = (pixel_theta - edge_theta) % (2 * np.pi)

    angle_idx  = np.round(rotated_theta / angle_step).astype(int) % len(path)
    path_r_px  = path[angle_idx] * radius_step          # path radius (px) at each pixel

    return pixel_r < path_r_px


def append_to_areas_csv(filename: str, area: float) -> None:
    AREAS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not AREAS_CSV.exists()
    with open(AREAS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AREAS_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({"filename": filename, "area_px": round(area, 2)})


def main():
    done_set    = load_done_set()
    annotations = load_annotations()
    pending     = get_pending_annotations(annotations, done_set)

    if not pending:
        print("All annotated images have areas computed.")
        return

    print(f"{len(pending)} to process, {len(done_set)} already done.")

    for ann in pending:
        path = RAW_IMAGES_DIR / ann["filename"]
        print(f"  Processing: {ann['filename']}")
        try:
            blurred = load_and_blur(path)
        except Exception as e:
            print(f"  Skipping {ann['filename']}: {e}")
            continue

        r_annotated = compute_radius(ann)

        polar, angle_step, radius_step = to_polar(
            blurred, ann["center_x"], ann["center_y"], r_annotated
        )
        polar = rotate_to_edge(polar, ann, angle_step)
        contrast = compute_contrast(polar)

        r0 = int(np.round(r_annotated / radius_step))
        r0 = np.clip(r0, 0, polar.shape[1] - 1)
        edge_path = viterbi(contrast, r0, radius_step)

        area = compute_area(edge_path, radius_step, angle_step)
        append_to_areas_csv(ann["filename"], area)
        print(f"  Area: {area:.1f} px^2  saved")


if __name__ == "__main__":
    main()
