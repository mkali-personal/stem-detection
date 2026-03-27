import csv
from pathlib import Path

import numpy as np
import tifffile
from matplotlib import use
use("TkAgg")  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

GAUSSIAN_SIGMA = 2
RAW_IMAGES_DIR = Path(__file__).parent / "data/raw-images"
ANNOTATIONS_CSV = Path(__file__).parent / "data/csv-outputs/annotations.csv"
CSV_COLUMNS = ["filename", "center_x", "center_y", "edge_x", "edge_y"]


def load_done_set() -> set:
    if not ANNOTATIONS_CSV.exists():
        return set()
    with open(ANNOTATIONS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            raise ValueError(
                f"Malformed CSV at {ANNOTATIONS_CSV}: expected header with 'filename'"
            )
        return {row["filename"] for row in reader}


def get_pending_images(done_set: set) -> list:
    return [p for p in sorted(RAW_IMAGES_DIR.glob("*.tif")) if p.name not in done_set]


def load_and_blur(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        print(f"  Warning: {path.name} is a 3D stack — using first slice only.")
        arr = arr[0]
    return gaussian_filter(arr.astype(float), sigma=GAUSSIAN_SIGMA)


def collect_two_clicks(filename: str, blurred: np.ndarray):
    plt.figure()
    plt.imshow(blurred, cmap="gray")
    plt.title(f"Image: {filename}\nClick 1: stem CENTER  |  Click 2: point on stem EDGE")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    clicks = plt.ginput(n=2, timeout=0)
    plt.close()
    if len(clicks) < 2:
        return None
    return clicks


def append_to_csv(filename: str, center: tuple, edge: tuple):
    ANNOTATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not ANNOTATIONS_CSV.exists()
    with open(ANNOTATIONS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "filename": filename,
            "center_x": round(center[0], 2),
            "center_y": round(center[1], 2),
            "edge_x": round(edge[0], 2),
            "edge_y": round(edge[1], 2),
        })


def main():
    done_set = load_done_set()
    pending = get_pending_images(done_set)
    if not pending:
        print("All images annotated.")
        return
    print(f"{len(pending)} to annotate, {len(done_set)} already done.")
    for path in pending:
        clicks = collect_two_clicks(path.name, load_and_blur(path))
        if clicks is None:
            print("Session paused. Re-run to continue.")
            break
        append_to_csv(path.name, clicks[0], clicks[1])
        print(f"  Saved: {path.name}")


if __name__ == "__main__":
    main()
