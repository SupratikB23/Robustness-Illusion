import numpy as np
from PIL import Image

from audit.transforms import NAMES, apply_transform, sweep


def _img(seed=0):
    r = np.random.default_rng(seed)
    return Image.fromarray((r.random((224, 224, 3)) * 255).astype("uint8"))


def test_identity_at_zero():
    img = _img()
    for name in NAMES:
        out = apply_transform(img, name, 0.0)
        assert out.size == (224, 224)
        assert np.array_equal(np.array(out), np.array(img))


def test_output_size_constant():
    img = _img()
    for name in NAMES:
        for s in (0.25, 0.5, 1.0):
            assert apply_transform(img, name, s).size == (224, 224)


def test_deterministic():
    img = _img()
    for name in NAMES:
        a = apply_transform(img, name, 0.625, seed=7)
        b = apply_transform(img, name, 0.625, seed=7)
        assert np.array_equal(np.array(a), np.array(b))


def test_sweep_length_and_rejects_bad_strength():
    img = _img()
    assert len(sweep(img, "jpeg")) == 8
    try:
        apply_transform(img, "jpeg", 1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_jpeg_in_memory_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = _img()
    apply_transform(img, "jpeg", 0.5)
    assert list(tmp_path.iterdir()) == []
