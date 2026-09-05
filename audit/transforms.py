from __future__ import annotations

import io
import numpy as np
from PIL import Image, ImageEnhance

from audit.config import CROP_MIN_FRAC, IMAGE_SIZE, JPEG_Q_MAX, JPEG_Q_MIN, ROT_MAX_DEG, STRENGTHS

NAMES = ("rotation", "crop", "jitter", "jpeg")


def check_strength(s: float) -> float:
    s = float(s)
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {s}")
    return s


def ensure_size(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    if img.size != (size, size):
        return img.convert("RGB").resize((size, size), Image.BICUBIC)
    return img.convert("RGB")


def apply_transform(img: Image.Image, name: str, s: float, seed: int = 1337, size: int = IMAGE_SIZE) -> Image.Image:
    # Fixed deterministic schedule; seed is accepted for a stable
    # call signature and reserved for future stochastic schedules.
    # s = 0 is defined as identity (not a quality-100 JPEG round-trip),
    # which is what guarantees both curves equal 1.0 at s = 0.
    if name not in NAMES:
        raise ValueError(f"Unknown transform {name!r}; expected one of {NAMES}")
    s = check_strength(s)
    img = ensure_size(img, size)
    if s == 0.0:
        return img.copy()
    if name == "rotation":
        return _rotation(img, s, size)
    if name == "crop":
        return _crop(img, s, size)
    if name == "jitter":
        return _jitter(img, s)
    return _jpeg(img, s)


def _rotation(img: Image.Image, s: float, size: int = IMAGE_SIZE) -> Image.Image:
    angle = s * ROT_MAX_DEG
    arr = np.array(img)
    # Edge-replicated pad absorbs the rotated corners; the centre crop
    # then contains no fill pixels. Covering check at 30 deg: the crop's
    # corner sits ~0.707*size from centre while the padded half-width is
    # 0.85*size, so corners always land on real (edge) pixels.
    pad = int(size * 0.35)
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    big = Image.fromarray(padded)
    rot = big.rotate(angle, resample=Image.BICUBIC, expand=False)
    w, h = rot.size
    cx, cy = w // 2, h // 2
    half = size // 2
    return rot.crop((cx - half, cy - half, cx + half, cy + half))


def _crop(img: Image.Image, s: float, size: int = IMAGE_SIZE) -> Image.Image:
    frac = 1.0 - (1.0 - CROP_MIN_FRAC) * s
    side = int(round(size * frac))
    left = (size - side) // 2
    box = (left, left, left + side, left + side)
    return img.crop(box).resize((size, size), Image.BICUBIC)


def _jitter(img: Image.Image, s: float) -> Image.Image:
    out = ImageEnhance.Brightness(img).enhance(1.0 - 0.30 * s)
    out = ImageEnhance.Contrast(out).enhance(1.0 + 0.30 * s)
    return ImageEnhance.Color(out).enhance(1.0 + 0.40 * s)


def _jpeg(img: Image.Image, s: float) -> Image.Image:
    q = int(round(JPEG_Q_MAX - (JPEG_Q_MAX - JPEG_Q_MIN) * s))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=max(1, q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def sweep(img: Image.Image, name: str, seed: int = 1337, size: int = IMAGE_SIZE) -> list[Image.Image]:
    return [apply_transform(img, name, s, seed, size) for s in STRENGTHS]


def sweep_grid(img: Image.Image, seed: int = 1337, size: int = IMAGE_SIZE) -> Image.Image:
    rows = [sweep(img, name, seed, size) for name in NAMES]
    w, h = rows[0][0].size
    canvas = Image.new("RGB", (w * len(STRENGTHS), h * len(NAMES)), "white")
    for r, row in enumerate(rows):
        for c, frame in enumerate(row):
            canvas.paste(frame, (c * w, r * h))
    return canvas
