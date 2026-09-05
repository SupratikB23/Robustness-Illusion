import torch

from interp_core.loaders import ActivationHook, resolve_hook_name
from interp_core.sae import random_sae, validate_sae
from interp_core.nulls import jaccard, random_subset_from_pool, shuffled_partner, stable_seed, strict_subset


def test_sae_encode_decode_shapes():
    sae = random_sae(16, 64, seed=0)
    x = torch.randn(4, 16)
    z = sae.encode(x)
    assert z.shape == (4, 64)
    assert sae.decode(z).shape == (4, 16)
    assert (z >= 0).all()


def test_topk_returns_k():
    sae = random_sae(16, 64, seed=1)
    idx, vals = sae.encode_topk(torch.randn(16), 32)
    assert len(idx) == len(vals) <= 32
    assert all(v > 0 for v in vals)
    assert len(set(idx)) == len(idx)


def test_gate_runs():
    sae = random_sae(16, 64, seed=2)
    gate = validate_sae(torch.randn(100, 16), sae)
    assert set(gate) >= {"fve", "l0", "dead_frac", "verdict"}


def test_jaccard_edges():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1}, set()) == 0.0
    assert jaccard({1, 2}, {2, 3}) == 1 / 3


def test_null_helpers_seeded():
    a = random_subset_from_pool(list(range(100)), 8, seed=5)
    b = random_subset_from_pool(list(range(100)), 8, seed=5)
    assert a == b
    assert shuffled_partner(10, 3, 0) != 3
    assert strict_subset([1, 2], [0.5, 3.0]) == {2}
    assert stable_seed("a", 1) == stable_seed("a", 1)
    assert stable_seed("a", 1) != stable_seed("a", 2)


def test_encode_topk_caps_at_active():
    sae = random_sae(8, 64, seed=3)
    sae.activation = "topk"
    sae.k = 4
    idx, vals = sae.encode_topk(torch.randn(8), 128)
    assert len(idx) == 4 and len(vals) == 4
    assert all(v > 0 for v in vals)


def test_hook_numeric_index_and_resolve():
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    x = torch.randn(2, 4)
    with ActivationHook(m, "1") as h:
        m(x)
    assert h.captured is not None
    assert resolve_hook_name("open_clip:ViT-B-32", 9, "hook_resid_post") == "visual.transformer.resblocks.9"
    assert resolve_hook_name("open_clip:ViT-B-32", 9, "hook_mlp_out") == "visual.transformer.resblocks.9.mlp"
    assert resolve_hook_name("timm:vit_base_patch32_224", 3, "hook_resid_post") == "blocks.3"


def test_heatmap_runs():
    import numpy as np
    from PIL import Image
    from interp_core.viz.heatmap import heatmap_overlay
    img = Image.new("RGB", (224, 224), (10, 20, 30))
    out = heatmap_overlay(img, np.random.default_rng(0).random(49))
    assert out.size == (224, 224)
