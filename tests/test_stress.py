import io
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from audit import stability as S
from audit.extract import check_strengths, extract, set_seed
from audit.transforms import NAMES, apply_transform
from interp_core.nulls import stable_seed


def _img(seed=0, size=224, color=None):
    if color is not None:
        return Image.new("RGB", (size, size), color)
    r = np.random.default_rng(seed)
    return Image.fromarray((r.random((size, size, 3)) * 255).astype("uint8"))


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_byte_identical_across_calls():
    img = _img()
    for name in NAMES:
        a = _png_bytes(apply_transform(img, name, 0.375, seed=11))
        b = _png_bytes(apply_transform(img, name, 0.375, seed=11))
        assert a == b


def test_cross_process_determinism():
    code = ("import pandas as pd, numpy as np;"
            "from audit.stability import null_a_curve;"
            "r = np.random.default_rng(0); rows = [];"
            "[rows.append({'image_id': f'img{i}', 'transform': 'rotation', 'strength': s,"
            " 'top_k_indices': r.choice(200, 64, replace=False).tolist(),"
            " 'top_k_values': [1.0]*64, 'embedding': [1.0]})"
            " for i in range(6) for s in (0.0, 0.5, 1.0)];"
            "print(null_a_curve(pd.DataFrame(rows), 'rotation', 32))")
    outs = []
    for py_seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=py_seed)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=".")
        assert p.returncode == 0, p.stderr
        outs.append(p.stdout)
    assert outs[0] == outs[1]


def test_solid_rotation_has_no_black_corners():
    img = _img(color=(200, 30, 30))
    out = np.array(apply_transform(img, "rotation", 1.0))
    assert out.size == 224 * 224 * 3
    assert (out == np.array([200, 30, 30])).all()


def test_crop_keeps_centre():
    img = Image.new("RGB", (224, 224), (0, 0, 0))
    px = img.load()
    for x in range(90, 134):
        for y in range(90, 134):
            px[x, y] = (255, 255, 255)
    out = np.array(apply_transform(img, "crop", 1.0))
    assert out.shape == (224, 224, 3)
    assert (out[112, 112] == [255, 255, 255]).all()


def test_jpeg_bounds_and_differs():
    img = _img()
    lo = apply_transform(img, "jpeg", 1.0)
    assert lo.size == (224, 224)
    assert not np.array_equal(np.array(lo), np.array(img))


def test_weird_inputs_survive():
    gray = Image.fromarray((np.random.default_rng(0).random((100, 150)) * 255).astype("uint8"), mode="L")
    rgba = _img().convert("RGBA")
    for im in (gray, rgba):
        for name in NAMES:
            assert apply_transform(im, name, 0.5).size == (224, 224)


def test_custom_size_end_to_end():
    img = _img()
    for name in NAMES:
        assert apply_transform(img, name, 0.5, size=384).size == (384, 384)


def test_s0_scores_exactly_one():
    rows = []
    for i in range(3):
        for s in (0.0, 0.5):
            rows.append({"image_id": f"img{i}", "transform": "crop", "strength": s,
                         "top_k_indices": list(range(32)), "top_k_values": [1.0] * 32,
                         "embedding": [0.3, 0.9]})
    df = pd.DataFrame(rows)
    o, f = S.per_image_scores(df, "crop", 0.0, 32)
    assert (o == 1.0).all() and (f == 1.0).all()


def test_extract_rejects_empty_and_bad_grid():
    import torch
    from cli import _mock_bundle_and_sae
    set_seed(0)
    bundle, sae, prep = _mock_bundle_and_sae("cpu", 0)
    with pytest.raises(ValueError):
        extract([], bundle, sae, prep, hook_name="blocks.1")
    with pytest.raises(ValueError):
        extract([("a", _img())], bundle, sae, prep, transforms=["warp"], hook_name="blocks.1")
    with pytest.raises(ValueError):
        check_strengths([0.5, 1.0])
    with pytest.raises(ValueError):
        extract([("a", _img())], bundle, sae, prep, hook_name=None)


def test_extract_deterministic_parquet(tmp_path):
    import torch
    from cli import _mock_bundle_and_sae
    from audit.extract import extract_to_parquet, load_results
    set_seed(4)
    bundle, sae, prep = _mock_bundle_and_sae("cpu", 4)
    imgs = [(f"s{i}", _img(seed=i)) for i in range(2)]
    kw = dict(transforms=["jpeg"], strengths=[0.0, 1.0], hook_name="blocks.1")
    d1 = str(tmp_path / "a")
    d2 = str(tmp_path / "b")
    extract_to_parquet(imgs, bundle, sae, prep, d1, **kw)
    extract_to_parquet(imgs, bundle, sae, prep, d2, **kw)
    a = load_results(d1).sort_values(["image_id", "strength"]).reset_index(drop=True)
    b = load_results(d2).sort_values(["image_id", "strength"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_folder_images_skips_junk(tmp_path):
    from audit.extract import folder_images
    _img().save(tmp_path / "ok.png")
    (tmp_path / "junk.png").write_bytes(b"not an image")
    (tmp_path / "evil.jpg").mkdir()
    (tmp_path / "note.txt").write_text("hi")
    out = folder_images(str(tmp_path))
    assert [i for i, _ in out] == ["ok.png"]


def test_cli_rejects_missing_dataset():
    from cli import _images
    ns = type("NS", (), {"dataset": None, "mock": False, "n_images": 4, "seed": 0})()
    with pytest.raises(ValueError):
        _images(ns)


def test_stable_seed_process_independent():
    assert stable_seed("x", 1) == stable_seed("x", 1)
    assert len({stable_seed("img", i) for i in range(100)}) == 100


def test_rii_ci_all_missing_is_nan():
    rows = [{"image_id": "lonely", "transform": "jpeg", "strength": 0.0,
             "top_k_indices": [1], "top_k_values": [1.0], "embedding": [1.0]}]
    df = pd.DataFrame(rows)
    assert all(v != v for v in S.rii_ci(df, "jpeg", 32))


def test_load_sae_local_roundtrip(tmp_path):
    import torch
    from interp_core.sae import load_sae_local
    sd = {"encoder.weight": torch.randn(64, 8), "encoder.bias": torch.zeros(64),
          "decoder.weight": torch.randn(64, 8), "decoder.bias": torch.zeros(8)}
    p = str(tmp_path / "sae.pt")
    torch.save(sd, p)
    sae = load_sae_local(p, 3, "hook_resid_post")
    assert (sae.d_model, sae.d_sae) == (8, 64)
    assert sae.hook_layer == 3
