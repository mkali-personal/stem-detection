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
    compute_area,
    path_to_mask,
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

    area = compute_area(edge_path, radius_step, angle_step)
    mask = path_to_mask(
        blurred.shape, ann["center_x"], ann["center_y"],
        ann["edge_x"], ann["edge_y"], edge_path, radius_step, angle_step,
    )

    # Build RGB overlay: normalize image to [0,1], tint masked region cyan
    img_norm = (blurred - blurred.min()) / (blurred.max() - blurred.min())
    img_rgb  = np.stack([img_norm] * 3, axis=-1)
    tint     = np.array([0.0, 0.6, 0.6])   # cyan
    img_rgb[mask] = img_rgb[mask] * 0.65 + tint * 0.35

    fig, (ax_img, ax_polar, ax_contrast, ax_mask) = plt.subplots(1, 4, figsize=(24, 5))

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

    ax_mask.imshow(img_rgb)
    ax_mask.plot(ann["center_x"], ann["center_y"], "r+", markersize=12, markeredgewidth=2)
    ax_mask.set_title(f"Mask overlay  —  area: {area:.0f} px²")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
