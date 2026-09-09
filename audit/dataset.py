from __future__ import annotations

import os

# Why this module exists: the old downloader called
# `load_dataset(..., streaming=True).shuffle(buffer_size=10000).take(n)`.
# A streaming shuffle fills its whole buffer before yielding the first row,
# so nothing reached disk until ~10k full-size JPEGs had been pulled. With
# the caller also swallowing stdout, that reads as a permanent hang.
#
# Rules kept here:
#   - bounded shuffle buffer, so the first image lands in seconds
#   - progress is reported through a callback, never only to stdout
#   - every write is atomic, so an interrupted run leaves no truncated JPEG
#   - resume is real: existing, readable frames are never re-fetched
#   - fixed seed -> the same n images in the same order on every run

SOURCES = {
    "imagenet-1k": {
        "repo": "ILSVRC/imagenet-1k",
        "splits": ("validation",),
        "prefix": "ILSVRC2012_val_",
        "streaming": True,
        "gated": True,
    },
    "tiny-imagenet": {
        "repo": "zh-plus/tiny-imagenet",
        # The valid split is sorted by label, so a prefix of it covers only a
        # handful of classes. The non-streaming path below takes a full random
        # permutation instead, which is why streaming is off for this source.
        "splits": ("valid", "validation", "test"),
        "prefix": "tiny_val_",
        "streaming": False,
        "gated": False,
    },
}

GATED_HELP = (
    "GATED DATASET: this token lacks access to ILSVRC/imagenet-1k.\n"
    "  1. Open https://huggingface.co/datasets/ILSVRC/imagenet-1k while logged in\n"
    "  2. Submit the access form + accept the ImageNet terms, wait for approval\n"
    "  3. Set HF_TOKEN and re-run, OR bypass now with --source tiny-imagenet\n"
    "     (public, no approval; 64x64 images upscaled to 224 -- state the swap\n"
    "      in the report, it is a different dataset, not the same run)."
)


class GatedDatasetError(RuntimeError):
    """Raised when the HF token cannot read a gated dataset repo."""


def _network_defaults() -> None:
    # Without these a stalled socket blocks forever instead of raising.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


def _is_gated_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(m in text for m in ("gated", "401", "403", "unauthorized", "forbidden",
                                   "private", "restricted", "authenticated"))


def resolve_source(source: str) -> dict:
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}; expected one of {sorted(SOURCES)}")
    return SOURCES[source]


def preflight(source: str, token: str | None = None) -> str:
    """Fail fast and loudly on an access problem, before any bulk download.

    `dataset_info` is not enough: imagenet-1k's metadata is public while its
    data files are gated, so it returns 200 for a token with no access.
    `auth_check` is the call that actually tests read access to the files.
    """
    from huggingface_hub import HfApi

    _network_defaults()
    spec = resolve_source(source)
    api = HfApi()
    token = token or os.environ.get("HF_TOKEN") or None
    try:
        if hasattr(api, "auth_check"):
            api.auth_check(spec["repo"], repo_type="dataset", token=token)
        else:  # huggingface_hub < 0.25
            api.dataset_info(spec["repo"], token=token, files_metadata=True)
    except Exception as e:
        if _is_gated_error(e):
            raise GatedDatasetError(GATED_HELP) from e
        raise RuntimeError(f"Cannot reach the Hugging Face Hub for {spec['repo']}: {e}") from e
    return spec["repo"]


def frame_name(source: str, i: int) -> str:
    return f"{resolve_source(source)['prefix']}{i:08d}.JPEG"


def _readable(path: str) -> bool:
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def assert_single_source(source: str, out: str) -> None:
    """Refuse to mix two datasets in one folder; the audit is scoped to one."""
    if not os.path.isdir(out):
        return
    foreign = sorted(SOURCES[s]["prefix"] for s in SOURCES if s != source)
    names = [f for f in os.listdir(out) if any(f.startswith(p) for p in foreign)]
    if names:
        raise ValueError(
            f"{out} already holds {len(names)} frames from a different dataset "
            f"(e.g. {names[0]}). Point --out at a fresh directory, or delete it, "
            "so one result set never mixes two datasets.")


def existing_frames(source: str, out: str, n: int) -> set[int]:
    """Positions already on disk as a readable JPEG. Truncated files are removed."""
    have = set()
    if not os.path.isdir(out):
        return have
    for i in range(n):
        p = os.path.join(out, frame_name(source, i))
        if not os.path.isfile(p):
            continue
        if _readable(p):
            have.add(i)
        else:
            os.remove(p)
    return have


def _save_atomic(pil, path: str) -> None:
    tmp = path + ".part"
    pil.convert("RGB").save(tmp, format="JPEG", quality=95)
    os.replace(tmp, path)


def _open_split(spec: dict, streaming: bool, token: str | None):
    from datasets import load_dataset

    last = None
    for split in spec["splits"]:
        try:
            ds = load_dataset(spec["repo"], split=split, streaming=streaming, token=token)
            return ds, split
        except Exception as e:
            if _is_gated_error(e):
                raise GatedDatasetError(GATED_HELP) from e
            last = e
    raise RuntimeError(f"Could not load {spec['repo']} (tried splits {spec['splits']}): {last}")


def _stream_rows(spec: dict, n: int, seed: int, shuffle_buffer: int, token: str | None):
    """Yield (position, row) from a streaming split with a bounded shuffle buffer."""
    ds, _ = _open_split(spec, True, token)
    if shuffle_buffer > 0:
        # Also shuffles shard order, so the sample is not one contiguous block.
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for i, row in enumerate(ds.take(n)):
        yield i, row


def _indexed_rows(spec: dict, n: int, seed: int, skip: set[int], token: str | None):
    """Yield (position, row) from a fully materialised split, touching only what is needed."""
    import numpy as np

    ds, _ = _open_split(spec, False, token)
    total = len(ds)
    if total < n:
        raise ValueError(f"{spec['repo']} split has {total} rows, need {n}")
    order = np.random.default_rng(seed).permutation(total)[:n]
    for i, row_idx in enumerate(order):
        if i in skip:
            continue
        yield i, ds[int(row_idx)]


def build_subset(source: str = "imagenet-1k", n: int = 2000, out: str = "data/imagenet-val-2k",
                 seed: int = 1337, shuffle_buffer: int | None = None, progress=None,
                 token: str | None = None, resume: bool = True,
                 streaming: bool | None = None) -> dict:
    """Materialise a deterministic n-image subset as JPEGs in `out`.

    Returns a summary dict: written, reused, total, labels, source, split.
    `progress(done, n, message)` is called as frames land.
    """
    _network_defaults()
    spec = resolve_source(source)
    if n < 1:
        raise ValueError("n must be >= 1")
    assert_single_source(source, out)
    os.makedirs(out, exist_ok=True)
    token = token or os.environ.get("HF_TOKEN") or None

    have = existing_frames(source, out, n) if resume else set()
    if len(have) >= n:
        _report(progress, n, n, f"{n} {source} images already on disk")
        return {"written": 0, "reused": n, "total": n, "labels": None,
                "source": source, "out": out, "split": None, "warning": None}

    if spec["gated"]:
        preflight(source, token)

    stream = spec["streaming"] if streaming is None else bool(streaming)
    if shuffle_buffer is None:
        # Bounded on purpose: the buffer must fill before the first yield.
        shuffle_buffer = min(n, 1000) if stream else 0

    labels: set = set()
    written = 0
    _report(progress, len(have), n, f"fetching {source} ({n - len(have)} to go)")

    rows = (_stream_rows(spec, n, seed, shuffle_buffer, token) if stream
            else _indexed_rows(spec, n, seed, have, token))
    for i, row in rows:
        if "label" in row:
            labels.add(int(row["label"]))
        if i in have:
            continue
        _save_atomic(row["image"], os.path.join(out, frame_name(source, i)))
        have.add(i)
        written += 1
        if written % 25 == 0 or len(have) == n:
            _report(progress, len(have), n, f"{len(have)}/{n} {source} images")

    got = existing_frames(source, out, n)
    if len(got) < n:
        raise RuntimeError(
            f"Only {len(got)}/{n} {source} images landed in {out}. "
            "Re-run to resume -- completed frames are kept and skipped.")
    # Labels are only counted over rows this pass actually visited, so on a
    # resumed run the tally would undercount. Report it only when complete.
    n_labels = len(labels) if labels and written == n else None
    # A streaming shuffle only mixes within its buffer. If the split happens to
    # be ordered by label, the sample can collapse onto a few classes -- that
    # would be a confound, so surface it rather than let it pass silently.
    warning = None
    if n_labels is not None and n_labels < min(50, n // 4):
        warning = (f"LOW CLASS DIVERSITY: only {n_labels} distinct labels across {n} images. "
                   "The split is likely label-ordered and the shuffle buffer is too small. "
                   "Re-run with a larger --shuffle-buffer (slower first image) before "
                   "reporting any number from this subset.")
    summary = {"written": written, "reused": n - written, "total": len(got),
               "labels": n_labels, "source": source, "out": out,
               "split": spec["splits"][0], "warning": warning}
    tail = f" ({n_labels} distinct labels)" if n_labels else " (resumed; label count not recomputed)"
    _report(progress, n, n, f"{n} {source} images ready{tail}")
    if warning:
        _report(progress, n, n, warning)
    return summary


def _report(progress, done: int, total: int, message: str) -> None:
    if progress is None:
        print(f"[{done}/{total}] {message}", flush=True)
    else:
        progress(done, total, message)
