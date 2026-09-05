from __future__ import annotations

import numpy as np
import pandas as pd

from audit.config import K_GRID, N_BOOTSTRAP, SEED, STRENGTHS
from interp_core.nulls import jaccard, random_subset_from_pool, shuffled_partner, strict_subset


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Clipped to [0, 1] so output stability shares the feature
    # statistic's bounds. Slightly inflates near-orthogonal pairs.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def _rows(df: pd.DataFrame, transform: str, strength: float) -> pd.DataFrame:
    return df[(df["transform"] == transform) & (df["strength"] == strength)]


def per_image_scores(df: pd.DataFrame, transform: str, strength: float, k: int) -> tuple[np.ndarray, np.ndarray]:
    ref = _rows(df, transform, 0.0).set_index("image_id")
    cur = _rows(df, transform, strength).set_index("image_id")
    ids = sorted(set(ref.index) & set(cur.index))
    if float(strength) == 0.0:
        ones = np.ones(len(ids))
        return ones, ones.copy()
    out, feat = [], []
    for i in ids:
        out.append(cosine(np.asarray(cur.loc[i, "embedding"]), np.asarray(ref.loc[i, "embedding"])))
        feat.append(jaccard(set(cur.loc[i, "top_k_indices"][:k]), set(ref.loc[i, "top_k_indices"][:k])))
    return np.array(out), np.array(feat)


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    r = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    means = np.array([vals[r.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def curve(df: pd.DataFrame, transform: str, k: int, strengths: list[float] | None = None,
          n_boot: int = N_BOOTSTRAP) -> dict:
    strengths = strengths or STRENGTHS
    present = set(df[df["transform"] == transform]["strength"])
    strengths = [s for s in strengths if s in present]
    out = {"strength": [], "output_mean": [], "output_lo": [], "output_hi": [],
           "feature_mean": [], "feature_lo": [], "feature_hi": [],
           "output_min": [], "feature_min": []}
    for s in strengths:
        o, f = per_image_scores(df, transform, s, k)
        o_lo, o_hi = bootstrap_ci(o, n_boot, SEED)
        f_lo, f_hi = bootstrap_ci(f, n_boot, SEED + 1)
        out["strength"].append(s)
        out["output_mean"].append(float(o.mean() if len(o) else float("nan")))
        out["output_lo"].append(o_lo)
        out["output_hi"].append(o_hi)
        out["feature_mean"].append(float(f.mean() if len(f) else float("nan")))
        out["feature_lo"].append(f_lo)
        out["feature_hi"].append(f_hi)
        out["output_min"].append(float(o.min() if len(o) else float("nan")))
        out["feature_min"].append(float(f.min() if len(f) else float("nan")))
    return out


def rii(curve_dict: dict) -> float:
    o = np.array(curve_dict["output_mean"], dtype=float)
    f = np.array(curve_dict["feature_mean"], dtype=float)
    s = np.array(curve_dict["strength"], dtype=float)
    mask = s > 0
    if not mask.any():
        return float("nan")
    return float(np.nanmean(o[mask] - f[mask]))


def rii_ci(df: pd.DataFrame, transform: str, k: int, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
    ref = _rows(df, transform, 0.0).set_index("image_id")
    others = sorted(s for s in df[df["transform"] == transform]["strength"].unique() if s > 0)
    ids = sorted(set(ref.index))
    per_img_gap = []
    skipped = 0
    for i in ids:
        gaps = []
        for s in others:
            row = df[(df["transform"] == transform) & (df["strength"] == s) & (df["image_id"] == i)]
            if row.empty:
                continue
            row = row.iloc[0]
            gaps.append(cosine(np.asarray(row["embedding"]), np.asarray(ref.loc[i, "embedding"]))
                        - jaccard(set(row["top_k_indices"][:k]), set(ref.loc[i, "top_k_indices"][:k])))
        if not gaps:
            skipped += 1
            continue
        per_img_gap.append(float(np.mean(gaps)))
    if skipped:
        print(f"Warning: {skipped} images with no s>0 rows skipped for {transform}")
    per_img_gap = np.array(per_img_gap)
    if not len(per_img_gap):
        return (float("nan"), float("nan"), float("nan"))
    lo, hi = bootstrap_ci(per_img_gap, n_boot, seed)
    return float(per_img_gap.mean()), lo, hi


def null_a_curve(df: pd.DataFrame, transform: str, k: int, seed: int = SEED) -> list[float]:
    # Chance overlap of two independent random draws from the per-image,
    # per-strength observed active set (the stored top-k). Independent
    # seeds at every strength, so s = 0 is chance like the rest.
    from interp_core.nulls import stable_seed
    sub = df[df["transform"] == transform]
    present = sorted(set(sub["strength"]))
    pool_of = {(i, s): list(inds) for (i, s), inds in
               sub.groupby(["image_id", "strength"])["top_k_indices"].first().items()}
    ref_pool = {i: pool_of.get((i, 0.0), []) for i in sub["image_id"].unique()}
    means = []
    for s in present:
        vals = []
        for img in sub["image_id"].unique():
            pool_s = pool_of.get((img, s), [])
            pool_0 = ref_pool.get(img, [])
            if len(pool_s) < k or len(pool_0) < k:
                continue
            vals.append(jaccard(random_subset_from_pool(pool_s, k, stable_seed(seed, img, s, "cur")),
                                random_subset_from_pool(pool_0, k, stable_seed(seed, img, s, "ref"))))
        means.append(float(np.mean(vals)) if vals else float("nan"))
    return means


def null_b_floor(df: pd.DataFrame, transform: str, k: int, seed: int = SEED) -> list[float]:
    from interp_core.nulls import stable_seed
    present = sorted(set(df[df["transform"] == transform]["strength"]))
    ref = _rows(df, transform, 0.0)
    if ref.empty:
        return [float("nan")] * len(present)
    ref_ids = list(ref["image_id"])
    ref_map = dict(zip(ref["image_id"], ref["top_k_indices"]))
    if len(ref_ids) < 2:
        return [float("nan")] * len(present)
    means = []
    for s in present:
        cur = _rows(df, transform, s)
        cur_map = dict(zip(cur["image_id"], cur["top_k_indices"]))
        ids = sorted(set(ref_map) & set(cur_map))
        n = len(ref_ids)
        vals = [jaccard(set(cur_map[i][:k]), set(ref_map[ref_ids[shuffled_partner(n, ref_ids.index(i), stable_seed(seed, i))]][:k])) for i in ids]
        means.append(float(np.mean(vals)) if vals else float("nan"))
    return means


def null_c_grid(df: pd.DataFrame, transform: str) -> dict[int, float]:
    return {k: rii(curve(df, transform, k)) for k in K_GRID}


def null_d_curve(df: pd.DataFrame, transform: str, k: int, factor: float = 2.0) -> tuple[list[float], list[float]]:
    # Pairs emptied by the strict filter contribute NaN (excluded), never
    # 1.0. Second return value is the fraction emptied per strength.
    means, fracs = [], []
    present = sorted(set(df[df["transform"] == transform]["strength"]))
    for s in present:
        ref = _rows(df, transform, 0.0).set_index("image_id")
        cur = _rows(df, transform, s).set_index("image_id")
        ids = sorted(set(ref.index) & set(cur.index))
        vals, emptied = [], 0
        for i in ids:
            r0 = strict_subset(list(ref.loc[i, "top_k_indices"][:k]), list(ref.loc[i, "top_k_values"][:k]), factor)
            r1 = strict_subset(list(cur.loc[i, "top_k_indices"][:k]), list(cur.loc[i, "top_k_values"][:k]), factor)
            if not r0 or not r1:
                emptied += 1
                continue
            vals.append(jaccard(r0, r1))
        means.append(float(np.mean(vals)) if vals else float("nan"))
        fracs.append(emptied / len(ids) if ids else float("nan"))
    return means, fracs


def find_cases(df: pd.DataFrame, transform: str, k: int, n: int = 3) -> list[dict]:
    strengths = [s for s in STRENGTHS if s > 0]
    scored = []
    for s in strengths:
        o, f = per_image_scores(df, transform, s, k)
        ref = _rows(df, transform, 0.0).set_index("image_id")
        cur = _rows(df, transform, s).set_index("image_id")
        ids = sorted(set(ref.index) & set(cur.index))
        for img, ov, fv in zip(ids, o, f):
            if ov > 0.95 and fv < 0.4:
                scored.append({"image_id": str(img), "strength": s,
                               "output_stability": float(ov), "feature_stability": float(fv),
                               "gap": float(ov - fv)})
    scored.sort(key=lambda d: -d["gap"])
    return scored[:n]
