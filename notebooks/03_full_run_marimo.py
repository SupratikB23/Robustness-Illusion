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
def _(mo, os):
    mo.md("# The Robustness Illusion — full run\nRuns end to end on open: setup → images → sweep check → SAE gate → pilot → full extraction → site export. Stops itself if a gate fails.")
    REPO_URL = "https://github.com/SupratikB23/Robustness-Illusion"
    N_FULL = 2000
    N_PILOT = 200
    DATA = os.path.abspath("data/imagenet-val-2k")
    OUT = os.path.abspath("results")
    SITE = os.path.abspath("site")
    return DATA, N_FULL, N_PILOT, OUT, REPO_URL, SITE


@app.cell
def _(mo, importlib, os, subprocess, sys):
    missing = [p for p, m in [("open-clip-torch", "open_clip"), ("datasets", "datasets"), ("pyarrow", "pyarrow"), ("huggingface-hub", "huggingface_hub")] if importlib.util.find_spec(m) is None]
    if importlib.util.find_spec("torch") is None:
        missing += ["torch", "torchvision"]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    import torch
    assert torch.cuda.is_available(), "No GPU visible. Attach the RTX 6000 GPU to this session first."
    cap = torch.cuda.get_device_capability()
    mo.md(f"CUDA OK · {torch.cuda.get_device_name(0)} · capability {cap} · torch {torch.__version__}")
    token = os.environ.get("HF_TOKEN", "")
    if token:
        from huggingface_hub import login
        login(token)
    else:
        mo.md("No HF_TOKEN secret found. ImageNet will fail and the run will fall back to Tiny-ImageNet automatically.")
    setup_done = True
    return (setup_done,)


@app.cell
def _(mo, setup_done, REPO_URL, os, subprocess, sys):
    cands = [os.getcwd(), "/tmp/robustness-illusion", "/content/repo", "/content/drive/MyDrive/robustness-illusion"]
    try:
        cands.insert(0, os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    REPO = next((c for c in cands if os.path.isdir(os.path.join(c, "audit")) and os.path.isdir(os.path.join(c, "interp_core"))), None)
    if REPO is None:
        subprocess.run(["git", "clone", REPO_URL, "/tmp/robustness-illusion"], check=True)
        REPO = "/tmp/robustness-illusion"
    sys.path.insert(0, REPO)
    mo.md(f"Repo: `{REPO}`")
    return (REPO,)


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
def _(mo, DATA, REPO, N_FULL, os, subprocess, sys):
    script = f"{REPO}/scripts/make_subset.py"
    first = subprocess.run([sys.executable, script, "--n", str(N_FULL), "--out", DATA, "--source", "imagenet-1k"], capture_output=True, text=True)
    if first.returncode == 0:
        dataset_source = "imagenet-1k"
    elif "GATED DATASET" in (first.stdout + first.stderr):
        mo.md("ImageNet access not granted for this token — continuing automatically on Tiny-ImageNet (public). The swap is recorded in `results/dataset_source.txt` and the site JSON.")
        subprocess.run([sys.executable, script, "--n", str(N_FULL), "--out", DATA, "--source", "tiny-imagenet"], check=True)
        dataset_source = "tiny-imagenet"
    else:
        raise RuntimeError(f"Image fetch failed:\n{(first.stdout + first.stderr)[-3000:]}")
    n_ready = len(os.listdir(DATA))
    assert n_ready >= N_FULL, f"Only {n_ready}/{N_FULL} images downloaded; re-run this notebook to resume."
    mo.md(f"**{n_ready} {dataset_source} images ready** in `{DATA}`")
    return dataset_source, n_ready


@app.cell
def _(mo, DATA, n_ready, folder_images, sweep_grid):
    sweep_grid(folder_images(DATA, 1)[0][1]).save("sweep.png")
    mo.vstack([
        mo.md("**Check this sweep before trusting anything below.** Rotation must show no black corners and crop must stay centred."),
        mo.image("sweep.png"),
    ])


@app.cell
def _(mo, DATA, folder_images, gate_activations, load_model, load_sae, resolve_hook_name, validate_sae, write_gate_card):
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
def _(mo, DATA, bundle, sae, hook, N_PILOT, extract_to_parquet, folder_images, load_results, summarize, write_tables):
    with mo.status.spinner(title="Pilot extraction + analysis") as _s:
        extract_to_parquet(folder_images(DATA, N_PILOT), bundle, sae, bundle.preprocess, "results/pilot", hook_name=hook, pool="auto", device="cuda")
        ps = summarize(load_results("results/pilot"), 32, n_boot=200)
    mo.ui.table([{"transform": t, "RII": round(v["rii"], 3), "lo": round(v["rii_lo"], 3), "hi": round(v["rii_hi"], 3)} for t, v in ps.items()])
    max_emp = max(max(v["null_d_emptied"]) for v in ps.values())
    mo.md(f"Max Null-D emptied fraction: **{max_emp:.0%}**")
    mo.stop(max_emp > 0.5, mo.md("Null D empties **over half** the pairs — **stopping.** Reframe around threshold dependence."))
    return


@app.cell
def _(mo, DATA, OUT, SITE, bundle, sae, hook, dataset_source, export_site, extract_to_parquet, folder_images, load_results, os, plot_curves, summarize, validate_sae, write_gate_card, write_tables):
    with mo.status.spinner(title="Full extraction — 2000 images x 4 transforms x 8 strengths (1–3 h)") as _s:
        out = extract_to_parquet(folder_images(DATA, 2000), bundle, sae, bundle.preprocess, OUT, hook_name=hook, pool="auto", device="cuda", gate_n=500)
        paths, gacts = out if isinstance(out, tuple) else (out, None)
        gate2 = validate_sae(gacts, sae)
        write_gate_card(gate2, OUT, hook)
        with open(os.path.join(OUT, "dataset_source.txt"), "w") as f:
            f.write(dataset_source + "\n")
        df = load_results(OUT)
        summary = summarize(df, 32)
        write_tables(summary, OUT)
        plot_curves(summary, OUT)
        export_site(df, dict(folder_images(DATA)), SITE, 32, dataset=dataset_source)
    mo.ui.table([{"transform": t, "RII": round(v["rii"], 3), "lo": round(v["rii_lo"], 3), "hi": round(v["rii_hi"], 3)} for t, v in summary.items()])
    full_done = True
    return (full_done, summary)


@app.cell
def _(mo, full_done, summary, OUT, SITE, dataset_source, io, zipfile):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{SITE}/data/curves.json", "data/curves.json")
        z.write(f"{SITE}/data/slider.json", "data/slider.json")
        z.write(f"{SITE}/data/cases.json", "data/cases.json")
    mo.vstack([
        mo.md(f"**Done on `{dataset_source}`.** Pull `results/` (parquets) via the session file browser, unzip locally, re-run `analyze` to confirm identical numbers. Then curate the Cases view (draft flags on), verify the second model, deploy `site/`."),
        mo.download(buf.getvalue(), filename="robustness-illusion-site.zip", label="Download site JSON"),
    ])


if __name__ == "__main__":
    app.run()
