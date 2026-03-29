import csv
import os
from pathlib import Path

import numpy as np
import tifffile
from dotenv import load_dotenv
from matplotlib import use
use("TkAgg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

load_dotenv(Path(__file__).parent / ".env")

GAUSSIAN_SIGMA  = 2
RAW_IMAGES_DIR  = Path(os.environ["RAW_IMAGES_DIR"])
ANNOTATIONS_CSV = Path(__file__).parent / "data/csv-outputs/annotations.csv"
CSV_COLUMNS     = ["filename", "center_x", "center_y", "edge_x", "edge_y"]


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


def append_to_csv(filename: str, center: tuple, edge: tuple):
    ANNOTATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not ANNOTATIONS_CSV.exists()
    with open(ANNOTATIONS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "filename":  filename,
            "center_x":  round(center[0], 2),
            "center_y":  round(center[1], 2),
            "edge_x":    round(edge[0], 2),
            "edge_y":    round(edge[1], 2),
        })


def delete_last_annotation() -> str | None:
    """Remove the last row from the CSV and return its filename, or None if empty."""
    if not ANNOTATIONS_CSV.exists():
        return None
    with open(ANNOTATIONS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    deleted = rows[-1]["filename"]
    with open(ANNOTATIONS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[:-1])
    return deleted


def main():
    done_set = load_done_set()
    pending  = get_pending_images(done_set)
    if not pending:
        print("All images annotated.")
        return
    print(f"{len(pending)} to annotate, {len(done_set)} already done.")

    fig, ax = plt.subplots()
    plt.get_current_fig_manager().window.state("zoomed")

    # Shared state touched by event callbacks
    state = {"clicks": [], "action": None}

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        if len(state["clicks"]) >= 2:
            return
        state["clicks"].append((event.xdata, event.ydata))
        marker, color = ("r+", "red") if len(state["clicks"]) == 1 else ("bx", "blue")
        ax.plot(event.xdata, event.ydata, marker, markersize=14, markeredgewidth=2,
                color=color)
        fig.canvas.draw()
        if len(state["clicks"]) == 2:
            state["action"] = "annotated"

    def on_key(event):
        if event.key == "backspace":
            state["action"] = "backspace"
        elif event.key == "escape":
            state["action"] = "quit"

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    idx = 0
    while idx < len(pending):
        img_path = pending[idx]
        ax.clear()
        ax.imshow(load_and_blur(img_path), cmap="gray")
        ax.set_title(
            f"[{idx + 1}/{len(pending)}]  {img_path.name}\n"
            "Click 1: CENTER    Click 2: EDGE    Backspace: undo    Esc: quit"
        )
        fig.canvas.draw()
        fig.canvas.flush_events()

        state["clicks"] = []
        state["action"] = None

        while state["action"] is None:
            plt.pause(0.05)

        if state["action"] == "quit":
            print("Session paused. Re-run to continue.")
            break

        elif state["action"] == "backspace":
            deleted = delete_last_annotation()
            if deleted is None:
                print("  Nothing to undo.")
                state["action"] = None   # stay on current image
            elif idx == 0:
                # Undo from a previous session: prepend to pending list
                pending.insert(0, RAW_IMAGES_DIR / deleted)
                print(f"  Undone: {deleted}")
                # idx stays 0, loop restarts showing the re-inserted image
            else:
                # Undo the annotation we just saved; step back one image
                idx -= 1
                print(f"  Undone: {deleted}")

        elif state["action"] == "annotated":
            center, edge = state["clicks"]
            append_to_csv(img_path.name, center, edge)
            print(f"  Saved: {img_path.name}")
            idx += 1

    plt.close()


if __name__ == "__main__":
    main()
