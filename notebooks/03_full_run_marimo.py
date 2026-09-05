import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import importlib
    import io
    import os
    import subprocess
    import sys
    import zipfile
    return importlib, io, mo, os, subprocess, sys, zipfile


@app.cell
def _(mo):
    mo.vstack([
        mo.md("# Invariance Audit — full run (Molab)\nSet the two fields, then press the stage buttons **1 → 5** in order. Each stage stops itself with a clear message if its gate fails."),
        repo_url := mo.ui.text(label="Repo URL (leave empty if audit/ sits next to this file)", full_width=True),
        hf_token := mo.ui.text(label="Hugging Face token, needs imagenet-1k access (password, never stored)", kind="password", full_width=True),
        n_full := mo.ui.number(label="Full images", start=8, stop=50000, step=100, value=2000),
        n_pilot := mo.ui.number(label="Pilot images", start=8, stop=1000, step=50, value=200),
        auto_continue := mo.ui.checkbox(label="Start full run automatically after pilot passes", value=True),
        setup_btn := mo.ui.run_button(label="1 · Setup environment"),
        data_btn := mo.ui.run_button(label="2 · Fetch images"),
        gate_btn := mo.ui.run_button(label="3 · SAE gate (500 images)"),
        pilot_btn := mo.ui.run_button(label="4 · Pilot + Null D"),
        full_btn := mo.ui.run_button(label="5 · Full run"),
    ])
    return (
        auto_continue,
        data_btn,
        full_btn,
        gate_btn,
        hf_token,
        n_full,
        n_pilot,
        pilot_btn,
        repo_url,
        setup_btn,
    )


@app.cell
def _(mo, setup_btn, hf_token, importlib, subprocess, sys):
    mo.stop(not setup_btn.value, mo.md("Press **1 · Setup environment** above."))
    missing = [p for p, m in [("open-clip-torch", "open_clip"), ("datasets", "datasets"), ("pyarrow", "pyarrow"), ("huggingface-hub", "huggingface_hub")] if importlib.util.find_spec(m) is None]
    if importlib.util.find_spec("torch") is None:
        missing += ["torch", "torchvision"]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    import torch
    assert torch.cuda.is_available(), "No GPU visible. Attach the RTX 6000 GPU to this session first."
    cap = torch.cuda.get_device_capability()
    mo.md(f"CUDA OK · {torch.cuda.get_device_name(0)} · capability {cap} · torch {torch.__version__}")
    assert hf_token.value, "Paste your Hugging Face token in the config cell first."
    from huggingface_hub import login
    login(hf_token.value)
    setup_done = True
    return (setup_done,)


@app.cell
def _(mo, setup_done, repo_url, os, subprocess, sys):
    cands = [os.getcwd(), "/content", "/content/repo", "/content/drive/MyDrive/invariance-audit"]
    try:
        cands.insert(0, os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    REPO = next((c for c in cands if os.path.isdir(os.path.join(c, "audit")) and os.path.isdir(os.path.join(c, "interp_core"))), None)
    if REPO is None:
        assert repo_url.value, "Set Repo URL or place audit/ next to this file."
        subprocess.run(["git", "clone", repo_url.value, "/tmp/invariance-audit"], check=True)
        REPO = "/tmp/invariance-audit"
    sys.path.insert(0, REPO)
    DATA = os.path.abspath("data/imagenet-val-2k")
    OUT = os.path.abspath("results")
    SITE = os.path.abspath("site")
    mo.md(f"Repo: `{REPO}`")
    return DATA, OUT, REPO, SITE


@app.cell
def _(REPO):
    from audit.extract import set_seed
    from audit.extract import extract_to_parquet, folder_images, gate_activations, load_results
    from audit.report import plot_curves, summarize, write_gate_card, write_tables
    from audit.sitegen import export_site
    from audit.transforms import sweep_grid
    from interp_core.loaders import load_model, resolve_hook_name
    from interp_core.sae import load_sae, validate_sae
    set_seed(1337)
    return (
        export_site,
        extract_to_parquet,
        folder_images,
        gate_activations,
        load_model,
        load_results,
        load_sae,
        plot_curves,
        resolve_hook_name,
        summarize,
        sweep_grid,
        validate_sae,
        write_gate_card,
        write_tables,
    )


@app.cell
def _(mo, data_btn, DATA, REPO, n_full, os, subprocess, sys):
    mo.stop(not data_btn.value, mo.md("Press **2 · Fetch images** above."))
    with mo.status.spinner(title="Downloading ImageNet subset (one time, ~500 MB)") as _s:
        subprocess.run([sys.executable, f"{REPO}/scripts/make_subset.py", "--n", str(int(n_full.value)), "--out", DATA], check=True)
    n_ready = len(os.listdir(DATA))
    mo.md(f"**{n_ready} images ready** in `{DATA}`")
    return (n_ready,)


@app.cell
def _(mo, DATA, n_ready, folder_images, sweep_grid):
    sweep_grid(folder_images(DATA, 1)[0][1]).save("sweep.png")
    mo.vstack([
        mo.md("**Look at this sweep.** Rotation must show no black corners and crop must stay centred. If it looks wrong, stop — no number below is trustworthy."),
        mo.image("sweep.png"),
    ])


@app.cell
def _(mo, gate_btn, DATA, folder_images, gate_activations, load_model, load_sae, resolve_hook_name, validate_sae, write_gate_card):
    mo.stop(not gate_btn.value, mo.md("Press **3 · SAE gate** above."))
    with mo.status.spinner(title="Loading CLIP ViT-B-32 + Prisma SAE, forwarding 500 clean images") as _s:
        bundle = load_model("open_clip:ViT-B-32", device="cuda")
        sae = load_sae("Prisma-Multimodal/sae-top_k-64-cls_only-layer_9-hook_resid_post", device="cuda")
        hook = resolve_hook_name(bundle.spec, sae.hook_layer, sae.hook_component)
        gate = validate_sae(gate_activations(folder_images(DATA, 500), bundle, sae, bundle.preprocess, hook, 500, "auto", "cuda"), sae)
    mo.ui.table([{"metric": k, "value": round(v, 4) if isinstance(v, float) else v} for k, v in gate.items()])
    mo.md(f"**Hook:** `{hook}` — gate verdict: **{gate['verdict']}**")
    mo.stop(gate["verdict"] == "reject", mo.md("**SAE GATE REJECTED — stopping.** Switch to the attention-head fallback (`interp_core.heads`). Do not train an SAE."))
    return bundle, gate, hook, sae


@app.cell
def _(mo, pilot_btn, DATA, bundle, sae, hook, n_pilot, extract_to_parquet, folder_images, load_results, summarize, write_tables):
    mo.stop(not pilot_btn.value, mo.md("Press **4 · Pilot + Null D** above."))
    with mo.status.spinner(title="Pilot extraction + analysis") as _s:
        extract_to_parquet(folder_images(DATA, int(n_pilot.value)), bundle, sae, bundle.preprocess, "results/pilot", hook_name=hook, pool="auto", device="cuda")
        ps = summarize(load_results("results/pilot"), 32, n_boot=200)
    mo.ui.table([{"transform": t, "RII": round(v["rii"], 3), "lo": round(v["rii_lo"], 3), "hi": round(v["rii_hi"], 3)} for t, v in ps.items()])
    max_emp = max(max(v["null_d_emptied"]) for v in ps.values())
    mo.md(f"Max Null-D emptied fraction: **{max_emp:.0%}**")
    mo.stop(max_emp > 0.5, mo.md("Null D empties **over half** the pairs — **stopping.** Reframe around threshold dependence."))
    pilot_ok = True
    return (pilot_ok,)


@app.cell
def _(mo, full_btn, auto_continue, pilot_ok, DATA, OUT, SITE, bundle, sae, hook, export_site, extract_to_parquet, folder_images, load_results, plot_curves, summarize, validate_sae, write_gate_card, write_tables):
    mo.stop(not (full_btn.value or (auto_continue.value and pilot_ok)), mo.md("Press **5 · Full run** above (or enable auto-continue and pass the pilot)."))
    with mo.status.spinner(title="Full extraction — 2000 images x 4 transforms x 8 strengths (1–3 h)") as _s:
        out = extract_to_parquet(folder_images(DATA, 2000), bundle, sae, bundle.preprocess, OUT, hook_name=hook, pool="auto", device="cuda", gate_n=500)
        paths, gacts = out if isinstance(out, tuple) else (out, None)
        gate2 = validate_sae(gacts, sae)
        write_gate_card(gate2, OUT, hook)
        df = load_results(OUT)
        summary = summarize(df, 32)
        write_tables(summary, OUT)
        plot_curves(summary, OUT)
        export_site(df, dict(folder_images(DATA)), SITE, 32)
    mo.ui.table([{"transform": t, "RII": round(v["rii"], 3), "lo": round(v["rii_lo"], 3), "hi": round(v["rii_hi"], 3)} for t, v in summary.items()])
    full_done = True
    return (full_done, summary)


@app.cell
def _(mo, full_done, summary, OUT, SITE, io, zipfile):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{SITE}/data/curves.json", "data/curves.json")
        z.write(f"{SITE}/data/slider.json", "data/slider.json")
        z.write(f"{SITE}/data/cases.json", "data/cases.json")
    mo.vstack([
        mo.md("**Done.** Pull `results/` (parquets) via the session file browser, unzip locally, and re-run `analyze` to confirm identical numbers. Then curate the Cases view (draft flags on), verify the second model, deploy `site/`."),
        mo.download(buf.getvalue(), filename="invariance-audit-site.zip", label="Download site JSON"),
    ])


if __name__ == "__main__":
    app.run()
