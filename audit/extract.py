from __future__ import annotations

import os
from PIL import Image

from audit.config import IMAGE_SIZE, K_MAX_STORE, SEED, STRENGTHS, TRANSFORMS
from audit.transforms import apply_transform
from interp_core.cache import write_parquet

MAX_IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 64
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_ROWS = 500_000
GATE_N_DEFAULT = 500


def set_seed(seed: int = SEED) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def folder_images(path: str, n: int | None = None, size: int | None = None) -> list[tuple[str, Image.Image]]:
    """Load images as (filename, PIL) pairs.

    `size` pre-resizes to size x size on load. apply_transform() does the same
    resize on every call anyway, so this is byte-identical downstream and keeps
    2000 decoded frames near 300 MB instead of well over a gigabyte.
    """
    from PIL import UnidentifiedImageError
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    try:
        names = sorted(os.listdir(path))
    except OSError as e:
        raise FileNotFoundError(f"Cannot list dataset directory {path!r}") from e
    out = []
    for f in names:
        if not f.lower().endswith(exts):
            continue
        full = os.path.realpath(os.path.join(path, f))
        if not os.path.isfile(full):
            continue
        try:
            if os.path.getsize(full) > MAX_FILE_BYTES:
                print(f"Warning: skipping oversized file {f}")
                continue
            with Image.open(full) as im:
                im.load()
                rgb = im.convert("RGB")
                if size is not None and rgb.size != (size, size):
                    rgb = rgb.resize((size, size), Image.BICUBIC)
                out.append((f, rgb))
        except (UnidentifiedImageError, OSError) as e:
            print(f"Warning: skipping unreadable file {f}: {e}")
        if n is not None and len(out) >= n:
            break
    if not out:
        raise FileNotFoundError(f"No readable images in {path}")
    return out


def synthetic_images(n: int = 8, seed: int = SEED) -> list[tuple[str, Image.Image]]:
    import numpy as np
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    out = []
    for i in range(n):
        freq, phase = 2 + (i % 6), r.random() * 6.28
        pattern = np.sin(2 * np.pi * (freq * xx / IMAGE_SIZE + freq * yy / IMAGE_SIZE / 2) + phase)
        base = 128 + 90 * pattern + 30 * np.sin(2 * np.pi * ((i + 1) * xx / IMAGE_SIZE) + phase / 2)
        noise = r.normal(0, 12, (IMAGE_SIZE, IMAGE_SIZE))
        arr = np.stack([base + noise, base * 0.7 + 40 + noise * 0.5, 255 - base * 0.5 + noise * 0.5], axis=-1)
        arr = arr + np.array([((i * 37) % 90) - 45, ((i * 53) % 90) - 45, ((i * 71) % 90) - 45])
        out.append((f"synth_{i:04d}", Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))))
    return out


def _embedding_and_activation(model_bundle, batch, hook_name: str | None, sae=None):
    import torch
    from interp_core.loaders import ActivationHook, resolve_hook_name
    if hook_name is None and sae is not None and getattr(sae, "hook_layer", -1) >= 0:
        hook_name = resolve_hook_name(model_bundle.spec, sae.hook_layer, sae.hook_component)
    if not hook_name:
        raise ValueError("No hook point resolved; refusing to encode the output embedding as features")
    model = model_bundle.model
    with torch.no_grad():
        with ActivationHook(model, hook_name) as hook:
            emb = _forward_embed(model_bundle, batch)
            act = hook.captured
        if act is None:
            raise RuntimeError(f"Hook {hook_name!r} captured nothing")
    return emb.detach().cpu(), act.detach().cpu()


def _forward_embed(bundle, batch):
    import torch
    model = bundle.model
    if bundle.spec.startswith("open_clip"):
        feats = model.encode_image(batch)
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    feats = model.forward_features(batch)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    if feats.dim() == 3:
        feats = feats[:, 0]
    elif feats.dim() == 4:
        feats = feats.mean(dim=(2, 3))
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _to_batch_first(act, batch_size: int):
    """Normalise a 3-D hook output to (batch, tokens, dim).

    open_clip's VisionTransformer permutes to LND before self.transformer, so a
    hook on visual.transformer.resblocks.N captures (tokens, batch, dim), while
    timm blocks and newer batch-first open_clip builds capture (batch, tokens,
    dim). Guessing wrong silently pools over the wrong axis -- with batch 1 the
    shapes still line up, so nothing raises and every number is quietly wrong.
    """
    if act.dim() != 3:
        return act
    if act.shape[0] == batch_size and act.shape[1] != batch_size:
        return act
    if act.shape[1] == batch_size and act.shape[0] != batch_size:
        return act.transpose(0, 1)
    if act.shape[0] == batch_size:  # batch == tokens; ambiguous, assume batch-first
        return act
    raise ValueError(
        f"Hook output {tuple(act.shape)} has no axis matching batch size {batch_size}; "
        "pass --pool explicitly or check the hook point")


def _pooled_activation(act, batch_size: int, pool: str = "auto") -> object:
    act = _to_batch_first(act, batch_size)
    if pool == "auto":
        pool = "cls" if act.dim() == 3 else "mean"
    if pool == "cls":
        if act.dim() == 3:
            return act[:, 0].float()
        raise ValueError("cls pooling needs a 3-D (batch, tokens, dim) hook output")
    if pool == "mean":
        if act.dim() == 4:
            return act.mean(dim=(2, 3)).float()
        if act.dim() == 3:
            return act.mean(dim=1).float()
        if act.dim() == 2:
            return act.float()
        raise ValueError(f"Cannot mean-pool {act.dim()}-D hook output")
    raise ValueError(f"Unknown pooling {pool!r}; expected 'cls', 'mean', or 'auto'")


def check_strengths(strengths) -> list[float]:
    strengths = [float(s) for s in strengths]
    if not strengths:
        raise ValueError("Need at least one strength")
    if any(not 0.0 <= s <= 1.0 for s in strengths):
        raise ValueError(f"Strengths must be in [0, 1]: {strengths}")
    if 0.0 not in strengths:
        raise ValueError("Strength grid must include 0.0 (the reference)")
    return sorted(set(strengths))


def gate_activations(images, model_bundle, sae, preprocess, hook_name: str | None,
                     n: int = GATE_N_DEFAULT, pool: str = "auto", device: str = "cpu"):
    import torch
    acts = []
    with torch.no_grad():
        for _, pil in images[:n]:
            batch = preprocess(pil).unsqueeze(0).to(device)
            _, act = _embedding_and_activation(model_bundle, batch, hook_name, sae)
            acts.append(_pooled_activation(act, batch.shape[0], pool))
    if not acts:
        raise ValueError("No images available for SAE gate")
    return torch.cat(acts, dim=0)


def extract(images, model_bundle, sae, preprocess, transforms=None, strengths=None,
            k_store: int = K_MAX_STORE, hook_name: str | None = None, pool: str = "auto",
            gate_n: int = 0, device: str = "cpu") -> object:
    import pandas as pd
    import torch
    images = list(images)
    if not images:
        raise ValueError("No images to extract")
    transforms = list(transforms) if transforms else list(TRANSFORMS)
    for t in transforms:
        if t not in TRANSFORMS:
            raise ValueError(f"Unknown transform {t!r}; expected one of {TRANSFORMS}")
    strengths = check_strengths(strengths or STRENGTHS)
    if len(images) * len(transforms) * len(strengths) > MAX_ROWS:
        raise ValueError("Extraction grid exceeds row cap; reduce images, transforms, or strengths")
    size = getattr(model_bundle, "image_size", IMAGE_SIZE)
    model_bundle.model.eval().to(device)
    records, gate_acts = [], []
    with torch.no_grad():
        for image_id, pil in images:
            for t in transforms:
                for s in strengths:
                    view = apply_transform(pil, t, s, size=size)
                    batch = preprocess(view).unsqueeze(0).to(device)
                    emb, act = _embedding_and_activation(model_bundle, batch, hook_name, sae)
                    pooled = _pooled_activation(act, batch.shape[0], pool)
                    if gate_n and s == 0.0 and len(gate_acts) < gate_n:
                        gate_acts.append(pooled)
                    idx, vals = sae.encode_topk(pooled[0], k_store)
                    records.append({"image_id": str(image_id), "transform": t, "strength": float(s),
                                    "top_k_indices": idx, "top_k_values": vals,
                                    "embedding": emb[0].cpu().tolist()})
    df = pd.DataFrame.from_records(records)
    if gate_n:
        return df, torch.cat(gate_acts, dim=0) if gate_acts else torch.empty(0)
    return df


def extract_to_parquet(images, model_bundle, sae, preprocess, out_dir: str, **kwargs):
    out = extract(images, model_bundle, sae, preprocess, **kwargs)
    df, gate_acts = out if isinstance(out, tuple) else (out, None)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for t, sub in df.groupby("transform"):
        p = os.path.join(out_dir, f"{t}.parquet")
        write_parquet(sub.reset_index(drop=True), p)
        paths.append(p)
    return (paths, gate_acts) if gate_acts is not None else paths


def load_results(out_dir: str, transforms=None):
    import pandas as pd
    from interp_core.cache import read_parquet
    frames = []
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".parquet"):
            if transforms is not None and os.path.splitext(f)[0] not in transforms:
                continue
            frames.append(read_parquet(os.path.join(out_dir, f)))
    if not frames:
        raise FileNotFoundError(f"No parquet files in {out_dir}")
    return pd.concat(frames, ignore_index=True)
