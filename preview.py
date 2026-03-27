import sys
from pathlib import Path

import numpy as np
from matplotlib import use
use("TkAgg")
import matplotlib.pyplot as plt

from areas import (
    load_annotations,
    load_and_blur,
    compute_radius,
    to_polar,
    rotate_to_edge,
    compute_contrast,
    viterbi,
    RAW_IMAGES_DIR,
)


def find_annotation(filename: str, annotations: list) -> dict:
    for ann in annotations:
        if ann["filename"] == filename:
            return ann
    raise ValueError(f"No annotation found for '{filename}' in annotations.csv")


def main():
    annotations = load_annotations()

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        ann = find_annotation(path.name, annotations)
    else:
        ann = annotations[0]
        path = RAW_IMAGES_DIR / ann["filename"]

    blurred = load_and_blur(path)
    r_annotated = compute_radius(ann)
    polar, angle_step, radius_step = to_polar(
        blurred, ann["center_x"], ann["center_y"], r_annotated
    )
    polar = rotate_to_edge(polar, ann, angle_step)
    contrast = compute_contrast(polar)

    r0 = int(np.round(r_annotated / radius_step))
    r0 = np.clip(r0, 0, polar.shape[1] - 1)
    edge_path = viterbi(contrast, r0, radius_step)

    fig, (ax_img, ax_polar, ax_contrast) = plt.subplots(1, 3, figsize=(18, 5))

    ax_img.imshow(blurred, cmap="gray")
    ax_img.plot(ann["center_x"], ann["center_y"], "r+", markersize=12, markeredgewidth=2)
    ax_img.plot(ann["edge_x"],   ann["edge_y"],   "bx", markersize=12, markeredgewidth=2)
    ax_img.set_title(path.name)

    tick_angles = [0, 90, 180, 270]
    tick_px = [int(a / np.degrees(angle_step)) for a in tick_angles]

    angle_indices = np.arange(polar.shape[0])

    ax_polar.imshow(polar.T, cmap="gray", aspect="auto", origin="upper")
    ax_polar.set_xlabel("angle (deg)")
    ax_polar.set_ylabel("radius (px)")
    ax_polar.set_xticks(tick_px)
    ax_polar.set_xticklabels([str(a) for a in tick_angles])
    ax_polar.axhline(y=r_annotated / radius_step, color="red", linestyle="--", linewidth=1)
    ax_polar.plot(angle_indices, edge_path, color="lime", linewidth=1)
    ax_polar.set_title("Polar transform")

    ax_contrast.imshow(contrast.T, cmap="gray", aspect="auto", origin="upper")
    ax_contrast.set_xlabel("angle (deg)")
    ax_contrast.set_ylabel("radius (px)")
    ax_contrast.set_xticks(tick_px)
    ax_contrast.set_xticklabels([str(a) for a in tick_angles])
    ax_contrast.axhline(y=r_annotated / radius_step, color="red", linestyle="--", linewidth=1)
    ax_contrast.plot(angle_indices, edge_path, color="lime", linewidth=1)
    ax_contrast.set_title("Radial contrast (binary)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
