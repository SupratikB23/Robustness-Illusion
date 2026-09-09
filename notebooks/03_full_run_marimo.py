import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import io
    import os
    import subprocess
    import sys
    import zipfile

    return importlib, io, mo, os, subprocess, sys, zipfile


@app.cell
def _(mo):
    mo.md("""
    # The Robustness Illusion — full run

    Setup → images → sweep check → SAE gate → pilot → full extraction → site export.

    **Nothing heavy starts on its own.** Each stage waits for its button, so opening
    this notebook costs nothing. Press **1 → 5** in order. Stages 3 and 4 stop the
    notebook themselves if a gate fails.

    Set `HF_TOKEN` as a secret (or env var) before stage 2 if you have ImageNet
    access. Without it the run falls back to public Tiny-ImageNet and records the
    swap in `results/dataset_source.txt` and the site JSON.
    """)
    return


@app.cell
def _(os):
    N_FULL = 2000
    N_PILOT = 200
    K = 32
    GATE_N = 500
    SEED = 1337
    MODEL = "open_clip:ViT-B-32"
    SAE_REPO = "Prisma-Multimodal/sae-top_k-64-cls_only-layer_9-hook_resid_post"
    SAE_REVISION = None  # pin to a commit sha for a publishable run
    REPO_URL = "https://github.com/SupratikB23/Robustness-Illusion"
    OUT = os.path.abspath("results")
    PILOT_OUT = os.path.join(OUT, "pilot")
    SITE = os.path.abspath("site")
    WORK = os.path.abspath("work")
    return (
        GATE_N,
        K,
        MODEL,
        N_FULL,
        N_PILOT,
        OUT,
        PILOT_OUT,
        REPO_URL,
        SAE_REPO,
        SAE_REVISION,
        SEED,
        SITE,
        WORK,
    )


@app.cell
def _(mo):
    b_setup = mo.ui.run_button(label="1 · Set up environment")
    b_data = mo.ui.run_button(label="2 · Fetch images")
    b_gate = mo.ui.run_button(label="3 · SAE gate (500 images)")
    b_pilot = mo.ui.run_button(label="4 · Pilot + Null D (200 images)")
    b_full = mo.ui.run_button(label="5 · Full run (1–3 h)")
    mo.vstack([mo.md("### Stages"), mo.hstack([b_setup, b_data, b_gate, b_pilot, b_full], justify="start")])
    return b_data, b_full, b_gate, b_pilot, b_setup


@app.cell
def _(b_setup, importlib, mo, os, subprocess, sys):
    mo.stop(not b_setup.value, mo.md("*Stage 1 idle — press the button.*"))
    _need = [("open-clip-torch", "open_clip"), ("datasets", "datasets"),
             ("pyarrow", "pyarrow"), ("huggingface-hub", "huggingface_hub"),
             ("pandas", "pandas"), ("matplotlib", "matplotlib")]
    _missing = [pkg for pkg, mod in _need if importlib.util.find_spec(mod) is None]
    if importlib.util.find_spec("torch") is None:
        _missing += ["torch", "torchvision"]
    if _missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_missing], check=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _lines = [f"torch {torch.__version__} · device **{device}**"]
    if device == "cuda":
        _cap = torch.cuda.get_device_capability()
        _lines.append(f"{torch.cuda.get_device_name(0)} · capability {_cap[0]}.{_cap[1]}")
        if _cap[0] >= 12 and not any(f"sm_{_cap[0]}{_cap[1]}" == a for a in torch.cuda.get_arch_list()):
            _lines.append(
                f"**This torch build has no sm_{_cap[0]}{_cap[1]} kernels** "
                f"(has {', '.join(torch.cuda.get_arch_list())}). Install a matching "
                "CUDA build before stage 3, or the first forward pass will fail.")
    else:
        _lines.append("**No GPU visible.** Stages 3–5 will run on CPU and take many hours. "
                      "Attach a GPU and re-press stage 1 if that is not what you want.")

    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import login
        login(os.environ["HF_TOKEN"])
        _lines.append("HF token loaded.")
    else:
        _lines.append("No `HF_TOKEN` set — ImageNet is gated, so stage 2 will use Tiny-ImageNet.")
    setup_done = True
    mo.md("\n\n".join(_lines))
    return (device,)


@app.cell
def _(REPO_URL, mo, os, subprocess, sys):
    _cands = [os.getcwd(), "/tmp/robustness-illusion", "/content/repo"]
    try:
        _cands.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except NameError:
        pass
    REPO = next((c for c in _cands
                 if os.path.isdir(os.path.join(c, "audit"))
                 and os.path.isdir(os.path.join(c, "interp_core"))), None)
    if REPO is None:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/tmp/robustness-illusion"], check=True)
        REPO = "/tmp/robustness-illusion"
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    mo.md(f"Repo: `{REPO}`")
    return (REPO,)


@app.cell
def _(REPO, SEED):
    _ = REPO  # import only after sys.path points at the repo
    from audit.dataset import GatedDatasetError, build_subset
    from audit.extract import extract_to_parquet, folder_images, gate_activations, load_results, set_seed
    from audit.report import plot_curves, summarize, write_gate_card, write_tables
    from audit.sitegen import export_site
    from audit.transforms import sweep_grid
    from interp_core.loaders import load_model, resolve_hook_name
    from interp_core.sae import load_sae, validate_sae

    set_seed(SEED)
    return (
        GatedDatasetError,
        build_subset,
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
def _(GatedDatasetError, N_FULL, SEED, b_data, build_subset, mo, os):
    mo.stop(not b_data.value, mo.md("*Stage 2 idle — press the button.*"))
    with mo.status.progress_bar(total=N_FULL, title="Fetching images") as _bar:
        _seen = [0]

        def _tick(done, total, message):
            _bar.update(increment=max(0, done - _seen[0]), subtitle=message)
            _seen[0] = done

        try:
            fetch = build_subset("imagenet-1k", N_FULL, os.path.abspath("data/imagenet-val-2k"),
                                 SEED, progress=_tick)
        except GatedDatasetError:
            _seen[0] = 0
            _bar.update(subtitle="ImageNet is gated for this token — switching to Tiny-ImageNet")
            fetch = build_subset("tiny-imagenet", N_FULL, os.path.abspath("data/tiny-imagenet-2k"),
                                 SEED, progress=_tick)

    DATA = fetch["out"]
    dataset_source = fetch["source"]
    _notes = [f"**{fetch['total']} {dataset_source} images ready** in `{DATA}` "
              f"({fetch['written']} fetched, {fetch['reused']} reused from a previous run)."]
    if dataset_source == "tiny-imagenet":
        _notes.append("Tiny-ImageNet frames are 64×64 upscaled to 224. That is a real "
                      "limitation of the run, not a detail — say so in the write-up.")
    if fetch.get("warning"):
        _notes.append(f"**{fetch['warning']}**")
    mo.md("\n\n".join(_notes))
    return DATA, dataset_source


@app.cell
def _(DATA, WORK, folder_images, mo, os, sweep_grid):
    os.makedirs(WORK, exist_ok=True)
    _sweep_path = os.path.join(WORK, "sweep.png")
    sweep_grid(folder_images(DATA, 1)[0][1]).save(_sweep_path)
    mo.vstack([
        mo.md("**Look at this sweep before trusting anything below.** Rows are rotation, "
              "crop, jitter, jpeg; columns are strength 0 → 1. Rotation must show no black "
              "corners and crop must stay centred."),
        mo.image(_sweep_path, width=900),
    ])
    return


@app.cell
def _(
    DATA,
    GATE_N,
    MODEL,
    OUT,
    SAE_REPO,
    SAE_REVISION,
    b_gate,
    device,
    folder_images,
    gate_activations,
    load_model,
    load_sae,
    mo,
    resolve_hook_name,
    validate_sae,
    write_gate_card,
):
    mo.stop(not b_gate.value, mo.md("*Stage 3 idle — press the button.*"))
    with mo.status.spinner(title=f"Loading {MODEL} + SAE, forwarding {GATE_N} clean images"):
        bundle = load_model(MODEL, device=device)
        sae = load_sae(SAE_REPO, revision=SAE_REVISION, device=device)
        hook = resolve_hook_name(bundle.spec, sae.hook_layer, sae.hook_component)
        _imgs = folder_images(DATA, GATE_N, size=bundle.image_size)
        gate = validate_sae(
            gate_activations(_imgs, bundle, sae, bundle.preprocess, hook, GATE_N, "auto", device), sae)
        write_gate_card(gate, OUT, hook)
    _rows = [{"metric": m, "value": round(gate[m], 4)} for m in ("fve", "l0", "dead_frac")]
    _view = mo.vstack([
        mo.md(f"**Hook:** `{hook}` · verdict **{gate['verdict']}** "
              f"(accept: FVE > 0.7, L0 in 10–200, dead < 0.5)"),
        mo.ui.table(_rows, selection=None),
    ])
    mo.stop(gate["verdict"] == "reject", mo.vstack([_view, mo.md(
        "**SAE GATE REJECTED — stopping.** Switch to the attention-head fallback "
        "(`interp_core.heads`). Do not train an SAE.")]))
    _view
    return bundle, hook, sae


@app.cell
def _(
    DATA,
    K,
    N_PILOT,
    PILOT_OUT,
    b_pilot,
    bundle,
    device,
    extract_to_parquet,
    folder_images,
    hook,
    load_results,
    mo,
    sae,
    summarize,
    write_tables,
):
    mo.stop(not b_pilot.value, mo.md("*Stage 4 idle — press the button.*"))
    with mo.status.spinner(title=f"Pilot: {N_PILOT} images × 4 transforms × 8 strengths"):
        extract_to_parquet(folder_images(DATA, N_PILOT, size=bundle.image_size), bundle, sae,
                           bundle.preprocess, PILOT_OUT, hook_name=hook, pool="auto", device=device)
        pilot = summarize(load_results(PILOT_OUT), K, n_boot=200)
        write_tables(pilot, PILOT_OUT)
    max_emp = max(max(v["null_d_emptied"]) for v in pilot.values())
    _view = mo.vstack([
        mo.ui.table([{"transform": t, "RII": round(v["rii"], 3),
                      "lo": round(v["rii_lo"], 3), "hi": round(v["rii_hi"], 3)}
                     for t, v in pilot.items()], selection=None),
        mo.md(f"Max Null-D emptied fraction: **{max_emp:.0%}** (stop above 50%). "
              "Pilot CIs use 200 bootstrap draws; the full run uses 1000."),
    ])
    mo.stop(max_emp > 0.5, mo.vstack([_view, mo.md(
        "**Null D empties over half the pairs — stopping.** The turnover is borderline "
        "features flickering. Reframe around threshold dependence; do not run stage 5.")]))
    pilot_ok = True
    _view
    return (pilot_ok,)


@app.cell
def _(
    DATA,
    GATE_N,
    K,
    N_FULL,
    OUT,
    SITE,
    b_full,
    bundle,
    dataset_source,
    device,
    export_site,
    extract_to_parquet,
    folder_images,
    hook,
    load_results,
    mo,
    os,
    pilot_ok,
    plot_curves,
    sae,
    summarize,
    validate_sae,
    write_gate_card,
    write_tables,
):
    mo.stop(not (pilot_ok and b_full.value), mo.md("*Stage 5 idle — stage 4 must pass first, then press the button.*"))
    with mo.status.spinner(title=f"Full run — {N_FULL} images × 4 transforms × 8 strengths (1–3 h)"):
        images = folder_images(DATA, N_FULL, size=bundle.image_size)
        _out = extract_to_parquet(images, bundle, sae, bundle.preprocess, OUT,
                                  hook_name=hook, pool="auto", device=device, gate_n=GATE_N)
        _paths, _gacts = _out if isinstance(_out, tuple) else (_out, None)
        if _gacts is not None and len(_gacts):
            write_gate_card(validate_sae(_gacts, sae), OUT, hook)
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "dataset_source.txt"), "w") as _f:
            _f.write(f"{dataset_source}\n{N_FULL} images\nk={K}\n")
        df = load_results(OUT)
        summary = summarize(df, K)
        write_tables(summary, OUT)
        plot_curves(summary, OUT)
        export_site(df, dict(images), SITE, K, dataset=dataset_source)
    full_done = True
    mo.vstack([
        mo.ui.table([{"transform": t, "RII": round(v["rii"], 3),
                      "lo": round(v["rii_lo"], 3), "hi": round(v["rii_hi"], 3)}
                     for t, v in summary.items()], selection=None),
        mo.md(f"Wrote `{OUT}` (parquet + tables + curve PNGs) and `{SITE}/data`."),
    ])
    return (full_done,)


@app.cell
def _(SITE, dataset_source, full_done, io, mo, os, zipfile):
    mo.stop(not full_done, mo.md("*Download appears after stage 5.*"))
    _wanted = ["data/curves.json", "data/slider.json", "data/cases.json"]
    _buf = io.BytesIO()
    _missing = []
    with zipfile.ZipFile(_buf, "w", zipfile.ZIP_DEFLATED) as _z:
        for _rel in _wanted:
            _p = os.path.join(SITE, _rel)
            if os.path.isfile(_p):
                _z.write(_p, _rel)
            else:
                _missing.append(_rel)
        for _root, _, _files in os.walk(os.path.join(SITE, "data", "frames")):
            for _f in _files:
                _full = os.path.join(_root, _f)
                _z.write(_full, os.path.relpath(_full, SITE))
    _msg = [f"**Done on `{dataset_source}`.** Download the site JSON below; pull `results/` "
            "(parquet + tables + PNGs) from the session file browser, then re-run "
            "`python cli.py analyze --out results/` locally to confirm identical numbers.",
            "Still to do by hand: curate the Cases view (the drafts are flagged), verify the "
            "CLI on a second model, deploy `site/`, write up with the limitations stated."]
    if _missing:
        _msg.append(f"**Missing from the site export: {', '.join(_missing)}.**")
    mo.vstack([
        mo.md("\n\n".join(_msg)),
        mo.download(_buf.getvalue(), filename="robustness-illusion-site.zip",
                    label="Download site JSON + frames"),
    ])
    return


if __name__ == "__main__":
    app.run()
