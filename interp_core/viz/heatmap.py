from __future__ import annotations

import numpy as np
from PIL import Image


def patch_scores_to_grid(scores: np.ndarray, n_patches: int) -> np.ndarray:
    side = int(n_patches ** 0.5)
    if side * side != n_patches:
        raise ValueError(f"Cannot form a square grid from {n_patches} patches")
    return np.asarray(scores).reshape(side, side)


def heatmap_overlay(image: Image.Image, scores: np.ndarray, alpha: float = 0.5) -> Image.Image:
    import matplotlib.cm as cm
    grid = patch_scores_to_grid(np.asarray(scores, dtype=float), len(scores))
    grid = (grid - grid.min()) / (np.ptp(grid) + 1e-12)
    size = image.size
    heat = Image.fromarray((grid * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    colored = (np.array(cm.jet(np.array(heat) / 255.0))[:, :, :3] * 255).astype(np.uint8)
    return Image.blend(image.convert("RGB"), Image.fromarray(colored), alpha)
