from __future__ import annotations

import os

from audit import stability as S
from audit.config import K_DEFAULT, N_BOOTSTRAP, TRANSFORMS
from audit.transforms import apply_transform
from interp_core.viz.export import cases_payload, curves_payload, slider_payload, write_json


def export_site(df, images: dict, out_dir: str, k: int = K_DEFAULT, n_boot: int = N_BOOTSTRAP, dataset: str = "imagenet-val-2k") -> dict:
    data_dir = os.path.join(out_dir, "data")
    frames_dir = os.path.join(data_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    present_strengths = sorted(set(df["strength"]))
    config = {"k": k, "strengths": present_strengths, "transforms": TRANSFORMS, "dataset": dataset}
    curves, rii = {}, {}
    for t in TRANSFORMS:
        if t not in set(df["transform"]):
            continue
        c = S.curve(df, t, k, n_boot=n_boot)
        mean, lo, hi = S.rii_ci(df, t, k, n_boot=n_boot)
        null_d, null_d_emptied = S.null_d_curve(df, t, k)
        curves[t] = {**c, "null_a": S.null_a_curve(df, t, k), "null_b": S.null_b_floor(df, t, k)}
        rii[t] = {"mean": mean, "lo": lo, "hi": hi, "null_c": S.null_c_grid(df, t),
                  "null_d": null_d, "null_d_emptied": null_d_emptied}
    write_json(curves_payload(config, curves, rii), os.path.join(data_dir, "curves.json"))
    frames = []
    for t in TRANSFORMS:
        if t not in set(df["transform"]):
            continue
        src_id = S.find_cases(df, t, k, n=1)
        img_id = src_id[0]["image_id"] if src_id else sorted(df["image_id"].unique())[0]
        base = images.get(img_id)
        ref = df[(df["transform"] == t) & (df["strength"] == 0.0) & (df["image_id"] == img_id)]
        ref_set = set(int(i) for i in ref.iloc[0]["top_k_indices"][:k]) if not ref.empty else set()
        for s in present_strengths:
            row = df[(df["transform"] == t) & (df["strength"] == s) & (df["image_id"] == img_id)]
            if row.empty:
                continue
            row = row.iloc[0]
            cur_set = set(int(i) for i in row["top_k_indices"][:k])
            fname = f"frames/{t}_{s}.jpg"
            if base is not None:
                apply_transform(base, t, s).save(os.path.join(data_dir, fname), "JPEG", quality=88)
                image_ref = fname
            else:
                image_ref = None
            o, f = S.per_image_scores(df[df["image_id"] == img_id], t, s, k)
            frames.append({"transform": t, "strength": float(s), "image_id": img_id, "image": image_ref,
                           "output_stability": float(o[0]) if len(o) else 1.0,
                           "feature_stability": float(f[0]) if len(f) else 1.0,
                           "kept": sorted(cur_set & ref_set), "entered": sorted(cur_set - ref_set),
                           "left": sorted(ref_set - cur_set)})
    write_json(slider_payload(config, frames), os.path.join(data_dir, "slider.json"))
    cases = []
    for t in TRANSFORMS:
        for c in S.find_cases(df, t, k, n=3):
            cases.append({"transform": t, **c,
                          "description": f"DRAFT — replace with hand-written analysis. Output {c['output_stability']:.2f} while features {c['feature_stability']:.2f} at strength {c['strength']}.",
                          "description_draft": True})
    payload = cases_payload(config, cases)
    payload["pending_hand_analysis"] = not cases or all(c.get("description_draft") for c in cases)
    write_json(payload, os.path.join(data_dir, "cases.json"))
    return {"curves": curves, "rii": rii, "n_frames": len(frames), "n_cases": len(cases)}
