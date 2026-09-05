from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit import stability as S
from audit.config import K_DEFAULT, N_BOOTSTRAP, STRENGTHS, TRANSFORMS


def summarize(df, k: int = K_DEFAULT, n_boot: int = N_BOOTSTRAP) -> dict:
    summary = {}
    for t in TRANSFORMS:
        if t not in set(df["transform"]):
            continue
        c = S.curve(df, t, k, n_boot=n_boot)
        mean, lo, hi = S.rii_ci(df, t, k, n_boot=n_boot)
        null_d, null_d_emptied = S.null_d_curve(df, t, k)
        if any(v == 0.0 for v in c["output_min"]):
            print(f"Warning: {t} has zero output stability; cosine clip bound may bind")
        summary[t] = {"curve": c, "rii": mean, "rii_lo": lo, "rii_hi": hi,
                      "null_a": S.null_a_curve(df, t, k),
                      "null_b": S.null_b_floor(df, t, k),
                      "null_c": S.null_c_grid(df, t),
                      "null_d": null_d,
                      "null_d_emptied": null_d_emptied,
                      "cases": S.find_cases(df, t, k)}
    return summary


def write_tables(summary: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if not summary:
        raise ValueError("Empty summary; check that results contain known transforms")
    lines = ["| transform | RII | 95% CI |", "|---|---|---|"]
    for t, v in summary.items():
        lines.append(f"| {t} | {v['rii']:.3f} | [{v['rii_lo']:.3f}, {v['rii_hi']:.3f}] |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "rii.md"), "w") as f:
        f.write(md)
    with open(os.path.join(out_dir, "null_d.md"), "w") as f:
        f.write("| transform | null_d curve | frac emptied |\n|---|---|---|\n")
        for t, v in summary.items():
            curve_s = ", ".join(f"{x:.3f}" if x == x else "nan" for x in v["null_d"])
            emp_s = ", ".join(f"{x:.2f}" for x in v["null_d_emptied"])
            f.write(f"| {t} | {curve_s} | {emp_s} |\n")
    with open(os.path.join(out_dir, "null_c.md"), "w") as f:
        ks = sorted(next(iter(summary.values()))["null_c"])
        f.write("| transform | " + " | ".join(f"k={k}" for k in ks) + " |\n")
        f.write("|---|" + "---|" * len(ks) + "\n")
        for t, v in summary.items():
            f.write("| " + t + " | " + " | ".join(f"{v['null_c'][k]:.3f}" for k in ks) + " |\n")
    return md


def plot_curves(summary: dict, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for t, v in summary.items():
        c = v["curve"]
        s = np.array(c["strength"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(s, c["output_mean"], label="output stability", marker="o")
        ax.fill_between(s, c["output_lo"], c["output_hi"], alpha=0.2)
        ax.plot(s, c["feature_mean"], label="feature stability", marker="s")
        ax.fill_between(s, c["feature_lo"], c["feature_hi"], alpha=0.2)
        ax.plot(s, v["null_b"], label="null B floor", linestyle="--")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("strength")
        ax.set_ylabel("stability")
        ax.set_title(f"{t}: RII={v['rii']:.3f} [{v['rii_lo']:.3f}, {v['rii_hi']:.3f}]")
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, f"curve_{t}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def write_gate_card(gate: dict, out_dir: str, hook: str | None = None, pool: str = "auto") -> str:
    import re
    os.makedirs(out_dir, exist_ok=True)
    comp = re.sub(r"[^A-Za-z0-9_.+-]", "_", str(gate.get('hook_component', '')))
    p = os.path.join(out_dir, "sae_gate.md")
    with open(p, "w") as f:
        f.write(f"# SAE acceptance gate: {gate['verdict']}\n\n"
                f"- FVE: {gate['fve']:.3f} (accept > 0.7, reject < 0.5)\n"
                f"- L0: {gate['l0']:.1f} (accept 10-200)\n"
                f"- dead fraction: {gate['dead_frac']:.3f} (accept < 0.5)\n"
                f"- hook: layer `{gate['hook_layer']}` `{comp}` (module `{hook or 'n/a'}`, pool `{pool}`)\n")
    return p
