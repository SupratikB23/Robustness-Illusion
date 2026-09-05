import numpy as np
import pandas as pd

from audit import stability as S


def _df(n=20, seed=0):
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base_emb = r.normal(size=8)
        base_emb /= np.linalg.norm(base_emb)
        base_idx = list(range(i * 100, i * 100 + 64))
        for t in ("rotation", "jpeg"):
            for s in (0.0, 0.5, 1.0):
                noise = r.normal(scale=0.05 * s, size=8)
                emb = base_emb + noise
                emb /= np.linalg.norm(emb)
                drop = int(40 * s)
                idx = (base_idx[drop:] + list(range(100000, 100000 + drop)))[:64]
                vals = list(np.linspace(3.0, 0.2, 64))
                rows.append({"image_id": f"img{i}", "transform": t, "strength": s,
                             "top_k_indices": idx[:64], "top_k_values": vals,
                             "embedding": emb.tolist()})
    return pd.DataFrame(rows)


def test_curves_bounded_and_one_at_zero():
    df = _df()
    c = S.curve(df, "rotation", 32, strengths=[0.0, 0.5, 1.0])
    assert c["strength"][0] == 0.0
    assert c["output_mean"][0] == 1.0
    assert c["feature_mean"][0] == 1.0
    for key in ("output_mean", "feature_mean"):
        assert all(-1e-9 <= v <= 1 + 1e-9 for v in c[key])


def test_rii_positive_when_features_turn_over():
    df = _df()
    assert S.rii(S.curve(df, "rotation", 32, strengths=[0.0, 0.5, 1.0])) > 0


def test_null_b_below_real():
    df = _df()
    real = S.curve(df, "rotation", 32, strengths=[0.0, 0.5, 1.0])["feature_mean"][1]
    floor = S.null_b_floor(df, "rotation", 32)[1]
    assert floor < real


def test_null_c_keys():
    df = _df()
    grid = S.null_c_grid(df, "rotation")
    assert sorted(grid) == [8, 16, 32, 64, 128]


def test_null_d_leq_or_valid():
    df = _df()
    d, fracs = S.null_d_curve(df, "rotation", 32)
    assert all((0.0 <= v <= 1.0 or v != v) for v in d)
    assert all(0.0 <= f <= 1.0 for f in fracs)


def test_null_a_deterministic_and_chance_level():
    df = _df()
    a1 = S.null_a_curve(df, "rotation", 32)
    a2 = S.null_a_curve(df, "rotation", 32)
    assert a1 == a2
    assert all(0.0 <= v <= 1.0 for v in a1)
    real0 = S.curve(df, "rotation", 32, strengths=[0.0, 0.5, 1.0])["feature_mean"][0]
    assert real0 == 1.0 and a1[0] < 0.6


def test_null_b_degrades_gracefully():
    df = _df()
    one = df[df["image_id"] == "img0"]
    assert all(v != v for v in S.null_b_floor(one, "rotation", 32))
    no_ref = df[df["strength"] > 0]
    assert all(v != v for v in S.null_b_floor(no_ref, "rotation", 32))


def test_null_d_all_filtered_is_nan_not_one():
    import pandas as pd
    rows = []
    for i in range(4):
        for s in (0.0, 1.0):
            rows.append({"image_id": f"img{i}", "transform": "rotation", "strength": s,
                         "top_k_indices": list(range(32)), "top_k_values": [0.1] * 32,
                         "embedding": [1.0, 0.0]})
    df = pd.DataFrame(rows)
    d, fracs = S.null_d_curve(df, "rotation", 32)
    assert all(v != v for v in d)
    assert all(f == 1.0 for f in fracs)
