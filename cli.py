from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit.config import K_DEFAULT, MODEL_DEFAULT, SAE_DEFAULT, SEED, STRENGTHS, TRANSFORMS


def _parse(args=None):
    p = argparse.ArgumentParser(prog="invariance-audit")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--model", default=MODEL_DEFAULT)
        q.add_argument("--sae", default=SAE_DEFAULT)
        q.add_argument("--sae-layer", type=int, default=None)
        q.add_argument("--sae-hook", default=None)
        q.add_argument("--sae-act", default=None)
        q.add_argument("--sae-k", type=int, default=None)
        q.add_argument("--sae-revision", default=None)
        q.add_argument("--hook", default=None)
        q.add_argument("--pool", default="auto", choices=["auto", "cls", "mean"])
        q.add_argument("--k", type=int, default=K_DEFAULT)
        q.add_argument("--out", default="results/")
        q.add_argument("--seed", type=int, default=SEED)
        q.add_argument("--device", default="cpu")
        q.add_argument("--mock", action="store_true")
        q.add_argument("--n-images", type=int, default=32)
        q.add_argument("--n-boot", type=int, default=1000)
        q.add_argument("--dataset", default=None)

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--transforms", default=",".join(TRANSFORMS))
    r.add_argument("--strengths", default=",".join(str(s) for s in STRENGTHS))
    r.add_argument("--site", default="site/")

    v = sub.add_parser("validate-sae")
    common(v)

    e = sub.add_parser("extract")
    common(e)
    e.add_argument("--transforms", default=",".join(TRANSFORMS))
    e.add_argument("--strengths", default=",".join(str(s) for s in STRENGTHS))

    a = sub.add_parser("analyze")
    a.add_argument("--out", default="results/")
    a.add_argument("--k", type=int, default=K_DEFAULT)
    a.add_argument("--n-boot", type=int, default=1000)

    s = sub.add_parser("export-site")
    s.add_argument("--out", default="results/")
    s.add_argument("--site", default="site/")
    s.add_argument("--k", type=int, default=K_DEFAULT)
    s.add_argument("--dataset", default=None)
    s.add_argument("--mock-images", action="store_true")

    w = sub.add_parser("render-sweep")
    w.add_argument("--image", default=None)
    w.add_argument("--out", default="results/sweep.png")
    w.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(args)


def _mock_bundle_and_sae(device, seed):
    import torch
    from torchvision import transforms as T
    from interp_core.sae import random_sae
    torch.manual_seed(seed)

    class MockBundle:
        spec = "mock:vit"
        image_size = 224
        embed_dim = 32

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(48, 48), torch.nn.Linear(48, 32)])

            def forward_features(self, x):
                h = torch.relu(self.blocks[0](self.pool(x).flatten(1)))
                return self.blocks[1](h)

        model = M()

    preprocess = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    return MockBundle(), random_sae(32, 256, seed, device), preprocess


MOCK_HOOK = "blocks.1"


def _real_bundle_and_sae(ns, device):
    from interp_core.loaders import load_model
    from interp_core.sae import load_sae
    bundle = load_model(ns.model, device=device)
    sae = load_sae(ns.sae, ns.sae_layer, ns.sae_hook, ns.sae_act, ns.sae_k, ns.sae_revision, device=device)
    return bundle, sae, bundle.preprocess


def resolve_dataset(name: str | None) -> str | None:
    if name is None:
        return None
    if os.path.isdir(name):
        return name
    kaggle_path = os.path.join("/kaggle", "input", name)
    if os.path.isdir(kaggle_path):
        return kaggle_path
    return None


def _hook_for(ns, bundle, sae):
    if ns.hook:
        return ns.hook
    if ns.mock:
        return MOCK_HOOK
    from interp_core.loaders import resolve_hook_name
    return resolve_hook_name(bundle.spec, sae.hook_layer, sae.hook_component)


def _dataset_label(ns) -> str:
    if getattr(ns, "mock", False) or getattr(ns, "mock_images", False):
        return "mock-synthetic"
    return ns.dataset or "unspecified"


def _images(ns):
    from audit.extract import folder_images, synthetic_images
    if ns.mock:
        if ns.dataset and not resolve_dataset(ns.dataset):
            print(f"Warning: --dataset {ns.dataset!r} not found; using synthetic images")
        return synthetic_images(ns.n_images, ns.seed)
    path = resolve_dataset(ns.dataset)
    if path is None:
        raise ValueError(f"--dataset {ns.dataset!r} not found; pass an image folder path, a /kaggle/input name, or use --mock")
    return folder_images(path, ns.n_images)


def _parse_strengths(text: str) -> list[float]:
    from audit.extract import check_strengths
    try:
        return check_strengths([float(x) for x in text.split(",")])
    except ValueError as e:
        raise ValueError(f"Bad --strengths {text!r}: {e}") from e


def cmd_run(ns):
    from audit.extract import GATE_N_DEFAULT, extract_to_parquet, load_results, set_seed
    from audit.report import plot_curves, summarize, write_gate_card, write_tables
    from audit.sitegen import export_site
    from interp_core.sae import validate_sae
    if ns.n_boot < 100 or ns.n_boot > 10000:
        raise ValueError("--n-boot must be in [100, 10000]")
    set_seed(ns.seed)
    names = ns.transforms.split(",")
    strengths = _parse_strengths(ns.strengths)
    bundle, sae, preprocess = _mock_bundle_and_sae(ns.device, ns.seed) if ns.mock else _real_bundle_and_sae(ns, ns.device)
    hook = _hook_for(ns, bundle, sae)
    images = _images(ns)
    out = extract_to_parquet(images, bundle, sae, preprocess, ns.out, transforms=names,
                             strengths=strengths, hook_name=hook, pool=ns.pool,
                             gate_n=GATE_N_DEFAULT, device=ns.device)
    paths, gate_acts = out if isinstance(out, tuple) else (out, None)
    print("\n".join(paths))
    if gate_acts is not None and len(gate_acts):
        gate = validate_sae(gate_acts, sae)
        print(gate)
        write_gate_card(gate, ns.out, hook, ns.pool)
        if gate["verdict"] == "reject":
            print("SAE gate REJECTED: numbers below use the attention-head/neuron fallback per Segment 8, or stop here")
    df = load_results(ns.out, names)
    summary = summarize(df, ns.k, n_boot=ns.n_boot)
    print(write_tables(summary, ns.out))
    plot_curves(summary, ns.out)
    # Label the site with the dataset actually used, so a mock or a
    # fallback run can never be read as the ImageNet run.
    export_site(df, dict(images), ns.site, ns.k, n_boot=ns.n_boot,
                dataset=_dataset_label(ns))
    for t, v in summary.items():
        print(f"{t}: RII={v['rii']:.3f} [{v['rii_lo']:.3f}, {v['rii_hi']:.3f}]")
    return 0


def cmd_validate(ns):
    from audit.extract import gate_activations, set_seed, synthetic_images
    from interp_core.sae import validate_sae
    set_seed(ns.seed)
    if ns.mock:
        bundle, sae, preprocess = _mock_bundle_and_sae(ns.device, ns.seed)
        images = synthetic_images(500, ns.seed)
        acts = gate_activations(images, bundle, sae, preprocess, _hook_for(ns, bundle, sae), 500, ns.pool, ns.device)
    else:
        from interp_core.loaders import load_model
        from interp_core.sae import load_sae
        bundle = load_model(ns.model, device=ns.device)
        sae = load_sae(ns.sae, ns.sae_layer, ns.sae_hook, ns.sae_act, ns.sae_k, ns.sae_revision, device=ns.device)
        images = _images(ns)
        acts = gate_activations(images, bundle, sae, bundle.preprocess, _hook_for(ns, bundle, sae), 500, ns.pool, ns.device)
    from audit.report import write_gate_card
    gate = validate_sae(acts, sae)
    print(gate)
    write_gate_card(gate, ns.out, _hook_for(ns, bundle, sae), ns.pool)
    return 0 if gate["verdict"] == "accept" else 1


def cmd_extract(ns):
    from audit.extract import extract_to_parquet, set_seed
    set_seed(ns.seed)
    bundle, sae, preprocess = _mock_bundle_and_sae(ns.device, ns.seed) if ns.mock else _real_bundle_and_sae(ns, ns.device)
    out = extract_to_parquet(_images(ns), bundle, sae, preprocess, ns.out,
                             transforms=ns.transforms.split(","), strengths=_parse_strengths(ns.strengths),
                             hook_name=_hook_for(ns, bundle, sae), pool=ns.pool, device=ns.device)
    paths = out[0] if isinstance(out, tuple) else out
    print("\n".join(paths))
    return 0


def cmd_analyze(ns):
    from audit.extract import load_results
    from audit.report import plot_curves, summarize, write_tables
    if ns.n_boot < 100 or ns.n_boot > 10000:
        raise ValueError("--n-boot must be in [100, 10000]")
    df = load_results(ns.out)
    summary = summarize(df, ns.k, n_boot=ns.n_boot)
    print(write_tables(summary, ns.out))
    plot_curves(summary, ns.out)
    return 0


def cmd_export_site(ns):
    from audit.extract import folder_images, load_results
    from audit.extract import synthetic_images
    from audit.sitegen import export_site
    df = load_results(ns.out)
    if ns.mock_images:
        images = dict(synthetic_images(8))
    elif resolve_dataset(ns.dataset):
        images = dict(folder_images(resolve_dataset(ns.dataset)))
    else:
        print("Warning: no source images given; slider will show stability numbers without frame images")
        images = {}
    print(export_site(df, images, ns.site, ns.k, dataset=_dataset_label(ns)))
    return 0


def cmd_sweep(ns):
    from PIL import Image
    from audit.transforms import sweep_grid
    if ns.out.lower().endswith((".png", ".jpg", ".jpeg")) is False:
        raise ValueError("--out must end with .png/.jpg for render-sweep")
    if ns.image and not os.path.isfile(os.path.realpath(ns.image)):
        raise ValueError(f"--image {ns.image!r} not found")
    img = Image.open(ns.image).convert("RGB") if ns.image else Image.new("RGB", (224, 224), (120, 140, 160))
    os.makedirs(os.path.dirname(os.path.abspath(ns.out)), exist_ok=True)
    sweep_grid(img, ns.seed).save(ns.out)
    print(ns.out)
    return 0


def main(args=None):
    ns = _parse(args)
    if ns.cmd == "run":
        return cmd_run(ns)
    if ns.cmd == "validate-sae":
        return cmd_validate(ns)
    if ns.cmd == "extract":
        return cmd_extract(ns)
    if ns.cmd == "analyze":
        return cmd_analyze(ns)
    if ns.cmd == "export-site":
        return cmd_export_site(ns)
    if ns.cmd == "render-sweep":
        return cmd_sweep(ns)
    raise ValueError(ns.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
