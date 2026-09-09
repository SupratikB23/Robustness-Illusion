"""Covers the image-subset builder and the fixes around it. No network."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from audit import dataset as D
from audit import stability as S
from audit.extract import _pooled_activation, _to_batch_first


# --- hook axis order -------------------------------------------------------
# open_clip 2.x permutes NLD -> LND before self.transformer, so a hook on
# visual.transformer.resblocks.N returns (tokens, batch, dim); 3.x is
# batch-first. Pooling the wrong axis is silent at batch 1.

def test_batch_first_passthrough_and_transpose():
    nld = torch.randn(4, 50, 768)          # (batch, tokens, dim)
    assert _to_batch_first(nld, 4).shape == (4, 50, 768)
    lnd = nld.transpose(0, 1).contiguous()  # (tokens, batch, dim)
    assert _to_batch_first(lnd, 4).shape == (4, 50, 768)
    assert torch.equal(_to_batch_first(lnd, 4), nld)


def test_cls_pooling_picks_the_cls_token_in_both_layouts():
    act = torch.zeros(1, 50, 8)
    act[0, 0] = 7.0        # CLS token
    act[0, 1:] = -1.0      # patch tokens
    from_nld = _pooled_activation(act, 1, "cls")
    from_lnd = _pooled_activation(act.transpose(0, 1).contiguous(), 1, "cls")
    assert from_nld.shape == (1, 8) and from_lnd.shape == (1, 8)
    assert torch.allclose(from_nld, torch.full((1, 8), 7.0))
    assert torch.allclose(from_lnd, from_nld)


def test_pooling_rejects_an_unrelated_batch_axis():
    with pytest.raises(ValueError, match="no axis matching batch size"):
        _to_batch_first(torch.randn(50, 3, 8), 4)


# --- subset builder --------------------------------------------------------

def _write_jpeg(path, color=(10, 20, 30)):
    Image.new("RGB", (8, 8), color).save(path, format="JPEG")


def test_frame_names_are_stable_and_source_scoped():
    assert D.frame_name("imagenet-1k", 7) == "ILSVRC2012_val_00000007.JPEG"
    assert D.frame_name("tiny-imagenet", 7) == "tiny_val_00000007.JPEG"
    with pytest.raises(ValueError):
        D.frame_name("cifar", 0)


def test_existing_frames_counts_good_and_deletes_truncated(tmp_path):
    good = tmp_path / D.frame_name("tiny-imagenet", 0)
    bad = tmp_path / D.frame_name("tiny-imagenet", 1)
    _write_jpeg(good)
    bad.write_bytes(b"not a jpeg")
    have = D.existing_frames("tiny-imagenet", str(tmp_path), 3)
    assert have == {0}
    assert good.exists() and not bad.exists()


def test_refuses_to_mix_two_datasets_in_one_folder(tmp_path):
    _write_jpeg(tmp_path / D.frame_name("tiny-imagenet", 0))
    D.assert_single_source("tiny-imagenet", str(tmp_path))  # same source is fine
    with pytest.raises(ValueError, match="different dataset"):
        D.assert_single_source("imagenet-1k", str(tmp_path))


def test_complete_folder_short_circuits_without_touching_the_network(tmp_path):
    for i in range(3):
        _write_jpeg(tmp_path / D.frame_name("tiny-imagenet", i))
    seen = []
    out = D.build_subset("tiny-imagenet", 3, str(tmp_path),
                         progress=lambda d, n, m: seen.append(m))
    assert out["written"] == 0 and out["reused"] == 3 and out["total"] == 3
    assert any("already on disk" in m for m in seen)


def test_gated_error_detection():
    assert D._is_gated_error(Exception("401 Client Error"))
    assert D._is_gated_error(Exception("Access to this gated repo is restricted"))
    assert not D._is_gated_error(Exception("connection reset by peer"))


def test_network_timeouts_are_bounded(monkeypatch):
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    D._network_defaults()
    assert int(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]) > 0


# --- analysis fixes --------------------------------------------------------

def _frame(n_images=6, strengths=(0.0, 0.5, 1.0), seed=0):
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n_images):
        for s in strengths:
            rows.append({"image_id": f"img{i}", "transform": "rotation", "strength": float(s),
                         "top_k_indices": r.choice(200, 64, replace=False).tolist(),
                         "top_k_values": (r.random(64) + 0.1).tolist(),
                         "embedding": r.random(8).tolist()})
    return pd.DataFrame(rows)


def _rii_ci_reference(df, transform, k):
    """The pre-optimisation nested-loop version, kept as the oracle."""
    ref = df[(df["transform"] == transform) & (df["strength"] == 0.0)].set_index("image_id")
    others = sorted(s for s in df[df["transform"] == transform]["strength"].unique() if s > 0)
    gaps_per_image = []
    for i in sorted(set(ref.index)):
        gaps = []
        for s in others:
            row = df[(df["transform"] == transform) & (df["strength"] == s) & (df["image_id"] == i)]
            if row.empty:
                continue
            row = row.iloc[0]
            gaps.append(S.cosine(np.asarray(row["embedding"]), np.asarray(ref.loc[i, "embedding"]))
                        - S.jaccard(set(row["top_k_indices"][:k]), set(ref.loc[i, "top_k_indices"][:k])))
        if gaps:
            gaps_per_image.append(float(np.mean(gaps)))
    return float(np.mean(gaps_per_image))


def test_rii_ci_matches_the_unoptimised_version():
    df = _frame()
    assert S.rii_ci(df, "rotation", 32, n_boot=100)[0] == pytest.approx(
        _rii_ci_reference(df, "rotation", 32))


def test_rii_ci_ignores_images_with_no_reference_row():
    df = _frame()
    df = df[~((df["image_id"] == "img0") & (df["strength"] > 0))]
    mean, lo, hi = S.rii_ci(df, "rotation", 32, n_boot=100)
    assert mean == mean and lo <= mean <= hi


def test_find_cases_returns_distinct_images():
    rows = []
    for i in range(4):
        for s in (0.0, 0.5, 1.0):
            # identical embeddings -> output stability 1.0; disjoint features -> 0.0
            rows.append({"image_id": f"img{i}", "transform": "rotation", "strength": s,
                         "top_k_indices": list(range(32)) if s == 0.0 else list(range(100, 132)),
                         "top_k_values": [1.0] * 32, "embedding": [1.0, 0.0]})
    cases = S.find_cases(pd.DataFrame(rows), "rotation", 32, n=3)
    assert len(cases) == 3
    assert len({c["image_id"] for c in cases}) == 3


def test_folder_images_size_matches_transform_resize(tmp_path):
    from audit.extract import folder_images
    from audit.transforms import ensure_size
    src = Image.fromarray((np.random.default_rng(0).random((97, 133, 3)) * 255).astype("uint8"))
    src.save(tmp_path / "a.jpg", quality=95)
    loaded = folder_images(str(tmp_path))[0][1]
    presized = folder_images(str(tmp_path), size=224)[0][1]
    assert presized.size == (224, 224)
    assert np.array_equal(np.array(presized), np.array(ensure_size(loaded, 224)))
