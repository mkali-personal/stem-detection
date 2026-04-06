import argparse
import csv
import os
from pathlib import Path

import numpy as np
import tifffile
from dotenv import load_dotenv
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter

load_dotenv(Path(__file__).parent / ".env")

RAW_IMAGES_DIR    = Path(os.environ["RAW_IMAGES_DIR"])
ANNOTATIONS_CSV   = Path(__file__).parent / "data/csv-outputs/annotations.csv"
AREAS_CSV         = Path(__file__).parent / "data/csv-outputs/areas.csv"
FIGURES_DIR       = Path(__file__).parent / "data/figures"
AREAS_CSV_COLUMNS = ["filename", "area_px"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute stem cross-sectional areas from annotations."
    )
    parser.add_argument(
        "--filename",
        help="Process only this annotated image filename (exact match).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if filename already exists in areas.csv (use with --filename).",
    )
    parser.add_argument("--gaussian-sigma",            type=float, default=4,    metavar="N")
    parser.add_argument("--transition-sigma",          type=float, default=2,    metavar="N")
    parser.add_argument("--edge-emission-prob",        type=float, default=0.9,  metavar="P")
    parser.add_argument("--non-edge-emission-prob",    type=float, default=0.5,  metavar="P")
    parser.add_argument("--contrast-radial-weight",    type=float, default=1.0,  metavar="W")
    parser.add_argument("--contrast-threshold-pctile", type=float, default=97,   metavar="N")
    parser.add_argument("--radial-contrast-sign",      type=str,   default="||", choices=["+", "-", "||"])
    return parser.parse_args()


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


def load_image(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        print(f"  Warning: {path.name} is a 3D stack — using first slice only.")
        arr = arr[0]
    return arr.astype(float)


def blur_image(image: np.ndarray, gaussian_sigma: float) -> np.ndarray:
    if gaussian_sigma is not None:
        return gaussian_filter(image, sigma=gaussian_sigma)
    return image


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


def compute_contrast(
    polar: np.ndarray,
    contrast_radial_weight: float,
    contrast_threshold_pctile: float,
    radial_contrast_sign: str,
) -> np.ndarray:
    """Compute a binary contrast image from a polar-coordinate intensity array.

    Radial gradients (axis 1) are weighted more heavily than angular gradients
    (axis 0) so that the stem boundary — which runs along the angular direction —
    produces a strong response.  Both positive and negative radial transitions
    (bright→dark and dark→bright) are captured via the absolute value.

    radial_contrast_sign controls which radial transitions are emphasised:
      '+'  — bright→dark outward (positive gradient)
      '-'  — dark→bright outward (negative gradient, sign-flipped)
      '||' — both directions (absolute value)

    Returns a binary array of the same shape as `polar`.
    """
    grad_angle  = np.gradient(polar, axis=0) / np.maximum(np.arange(polar.shape[1], dtype=float)[None, :], 1.0)  # angular grad normalized by r (ds = r dθ)
    grad_radius = np.gradient(polar, axis=1)   # along radius axis

    if radial_contrast_sign == '+':
        radial_term = grad_radius
    elif radial_contrast_sign == '-':
        radial_term = -grad_radius
    elif radial_contrast_sign == '||':
        radial_term = np.abs(grad_radius)
    else:
        raise ValueError(f"radial_contrast_sign must be '+', '-' or '||', got {radial_contrast_sign!r}")

    weighted = contrast_radial_weight * radial_term + np.abs(grad_angle)

    threshold = np.percentile(weighted, contrast_threshold_pctile)
    return (weighted > threshold).astype(float)


def viterbi(
    contrast: np.ndarray,
    r0: int,
    radius_step: float,
    transition_sigma: float,
    edge_emission_prob: float,
    non_edge_emission_prob: float,
) -> np.ndarray:
    """Trace the stem edge in polar coordinates using the Viterbi algorithm.

    States are radius indices. The path starts and ends at r0 (the annotated
    edge radius), enforcing a closed contour.

    Args:
        contrast:              Binary contrast array, shape (n_angles, n_radii).
        r0:                    Initial (and forced final) radius state index.
        radius_step:           Pixel width of each radius bin, used to convert
                               transition_sigma from pixels to bins.
        transition_sigma:      HMM state transition spread in pixels.
        edge_emission_prob:    P(contrast pixel at true edge).
        non_edge_emission_prob: P(contrast pixel away from edge).

    Returns:
        path: int array of shape (n_angles,) — the radius index at each angle.
    """
    n_angles, n_radii = contrast.shape
    sigma_bins = transition_sigma / radius_step

    # Log transition matrix: log_trans[r, r'] = log P(r'|r) up to a constant.
    # Normaliser is the same for every r, so it doesn't affect argmax.
    r_idx = np.arange(n_radii)
    diff = r_idx[:, None] - r_idx[None, :]          # (n_radii, n_radii)
    log_trans = -0.5 * (diff / sigma_bins) ** 2     # (n_radii, n_radii)

    # Log emission score for being at the edge vs not (relative, drops constants).
    # Shape (n_angles, n_radii); positive where contrast=1, negative where contrast=0.
    log_emit = np.where(
        contrast > 0.5,
        np.log(edge_emission_prob    / non_edge_emission_prob),
        np.log((1 - edge_emission_prob) / (1 - non_edge_emission_prob)),
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


def save_qc_figure(
    filename: str,
    image: np.ndarray,
    ann: dict,
    contrast: np.ndarray,
    edge_path: np.ndarray,
    mask: np.ndarray,
    r_annotated: float,
    radius_step: float,
    angle_step: float,
    area: float,
) -> None:
    """Save a 4-panel QC figure to data/figures/<filename>.png.

    Display panels use the original (unblurred) image; the contrast panel
    reflects the blurred-derived computation used by Viterbi.
    """
    polar_orig, _, _ = to_polar(image, ann["center_x"], ann["center_y"], r_annotated)
    polar_orig = rotate_to_edge(polar_orig, ann, angle_step)

    img_norm = (image - image.min()) / (image.max() - image.min())
    img_rgb  = np.stack([img_norm] * 3, axis=-1)
    tint     = np.array([0.0, 0.6, 0.6])
    img_rgb[mask] = img_rgb[mask] * 0.65 + tint * 0.35

    tick_angles = [0, 90, 180, 270]
    tick_px     = [int(a / np.degrees(angle_step)) for a in tick_angles]
    tick_labels = [str(a) for a in tick_angles]
    angle_indices = np.arange(polar_orig.shape[0])
    r_line = r_annotated / radius_step

    fig = Figure(figsize=(24, 5))
    FigureCanvasAgg(fig)
    ax_img, ax_polar, ax_contrast, ax_mask = fig.subplots(1, 4)

    ax_img.imshow(image, cmap="gray")
    ax_img.plot(ann["center_x"], ann["center_y"], "r+", markersize=12, markeredgewidth=2)
    ax_img.plot(ann["edge_x"],   ann["edge_y"],   "bx", markersize=12, markeredgewidth=2)
    ax_img.set_title(filename)

    ax_polar.imshow(polar_orig.T, cmap="gray", aspect="auto", origin="upper")
    ax_polar.set_xlabel("angle (deg)")
    ax_polar.set_ylabel("radius (px)")
    ax_polar.set_xticks(tick_px)
    ax_polar.set_xticklabels(tick_labels)
    ax_polar.axhline(y=r_line, color="red", linestyle="--", linewidth=1)
    ax_polar.plot(angle_indices, edge_path, color="lime", linewidth=1)
    ax_polar.set_title("Polar transform")

    ax_contrast.imshow(contrast.T, cmap="gray", aspect="auto", origin="upper")
    ax_contrast.set_xlabel("angle (deg)")
    ax_contrast.set_ylabel("radius (px)")
    ax_contrast.set_xticks(tick_px)
    ax_contrast.set_xticklabels(tick_labels)
    ax_contrast.axhline(y=r_line, color="red", linestyle="--", linewidth=1)
    ax_contrast.plot(angle_indices, edge_path, color="lime", linewidth=1)
    ax_contrast.set_title("Radial contrast (binary)")

    ax_mask.imshow(img_rgb)
    ax_mask.plot(ann["center_x"], ann["center_y"], "r+", markersize=12, markeredgewidth=2)
    ax_mask.set_title(f"Mask overlay  -  area: {area:.0f} px^2")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / (Path(filename).stem + ".png")
    fig.savefig(out_path, dpi=100)


def upsert_area_in_csv(filename: str, area: float) -> None:
    AREAS_CSV.parent.mkdir(parents=True, exist_ok=True)
    area_value = str(round(area, 2))

    rows = []
    if AREAS_CSV.exists():
        with open(AREAS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "filename" not in reader.fieldnames:
                raise ValueError(
                    f"Malformed CSV at {AREAS_CSV}: expected header with 'filename'"
                )
            rows = list(reader)

    replaced = False
    for row in rows:
        if row.get("filename") == filename:
            row["area_px"] = area_value
            replaced = True

    if not replaced:
        rows.append({"filename": filename, "area_px": area_value})

    with open(AREAS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AREAS_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def process_annotation(
    ann: dict,
    gaussian_sigma: float,
    transition_sigma: float,
    edge_emission_prob: float,
    non_edge_emission_prob: float,
    contrast_radial_weight: float,
    contrast_threshold_pctile: float,
    radial_contrast_sign: str,
) -> dict:
    path = RAW_IMAGES_DIR / ann["filename"]
    print(f"  Processing: {ann['filename']}")
    try:
        image = load_image(path)
    except Exception as e:
        print(f"  Skipping {ann['filename']}: {e}")
        return {"filename": ann["filename"], "status": "skipped", "error": str(e)}

    blurred = blur_image(image, gaussian_sigma)
    r_annotated = compute_radius(ann)

    polar, angle_step, radius_step = to_polar(
        blurred, ann["center_x"], ann["center_y"], r_annotated
    )
    polar = rotate_to_edge(polar, ann, angle_step)
    contrast = compute_contrast(polar, contrast_radial_weight, contrast_threshold_pctile, radial_contrast_sign)

    r0 = int(np.round(r_annotated / radius_step))
    r0 = np.clip(r0, 0, polar.shape[1] - 1)
    edge_path = viterbi(contrast, r0, radius_step, transition_sigma, edge_emission_prob, non_edge_emission_prob)

    area = compute_area(edge_path, radius_step, angle_step)
    mask = path_to_mask(
        image.shape, ann["center_x"], ann["center_y"],
        ann["edge_x"], ann["edge_y"], edge_path, radius_step, angle_step,
    )
    save_qc_figure(
        ann["filename"], image, ann, contrast,
        edge_path, mask, r_annotated, radius_step, angle_step, area,
    )
    upsert_area_in_csv(ann["filename"], area)
    print(f"  Area: {area:.1f} px^2  saved")
    return {
        "filename": ann["filename"],
        "status": "saved",
        "area_px": float(round(area, 2)),
    }


def areas(
    filename: str = None,
    force: bool = False,
    gaussian_sigma: float = 4,
    transition_sigma: float = 2,
    edge_emission_prob: float = 0.9,
    non_edge_emission_prob: float = 0.5,
    contrast_radial_weight: float = 1.0,
    contrast_threshold_pctile: float = 97,
    radial_contrast_sign: str = '||',
) -> list:
    """Run area computation and return per-file statuses.

    Args:
        filename:                  Exact filename to process, or None to process all pending.
        force:                     Recompute even if already present in areas.csv (only with filename).
        gaussian_sigma:            Gaussian blur sigma applied before computation.
        transition_sigma:          HMM state transition spread in pixels.
        edge_emission_prob:        P(contrast pixel at true edge).
        non_edge_emission_prob:    P(contrast pixel away from edge).
        contrast_radial_weight:    Radial gradient weight relative to angular.
        contrast_threshold_pctile: Percentile cutoff for contrast binarisation.
        radial_contrast_sign:      Which radial transitions to emphasise: '+', '-' or '||'.
    """
    if force and not filename:
        print("force=True requires a filename to be specified.")
        return []

    done_set    = load_done_set()
    annotations = load_annotations()

    if filename:
        requested = [ann for ann in annotations if ann["filename"] == filename]
        if not requested:
            print(f"No annotation found for filename: {filename}")
            return []
        if force:
            pending = requested
        else:
            pending = [ann for ann in requested if ann["filename"] not in done_set]
        if not pending and not force:
            print(f"Area already computed for: {filename}")
            return []
    else:
        pending = get_pending_annotations(annotations, done_set)

    if not pending:
        print("All annotated images have areas computed.")
        return []

    print(f"{len(pending)} to process, {len(done_set)} already done.")

    results = []
    for ann in pending:
        results.append(process_annotation(
            ann,
            gaussian_sigma=gaussian_sigma,
            transition_sigma=transition_sigma,
            edge_emission_prob=edge_emission_prob,
            non_edge_emission_prob=non_edge_emission_prob,
            contrast_radial_weight=contrast_radial_weight,
            contrast_threshold_pctile=contrast_threshold_pctile,
            radial_contrast_sign=radial_contrast_sign,
        ))
    return results


def main():
    args = parse_args()
    areas(
        filename=args.filename,
        force=args.force,
        gaussian_sigma=args.gaussian_sigma,
        transition_sigma=args.transition_sigma,
        edge_emission_prob=args.edge_emission_prob,
        non_edge_emission_prob=args.non_edge_emission_prob,
        contrast_radial_weight=args.contrast_radial_weight,
        contrast_threshold_pctile=args.contrast_threshold_pctile,
        radial_contrast_sign=args.radial_contrast_sign,
    )


if __name__ == "__main__":
    main()
