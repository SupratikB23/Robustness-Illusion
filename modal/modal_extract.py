import modal

app = modal.App("invariance-audit")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "torchvision", "open-clip-torch", "huggingface-hub",
    "numpy", "pandas", "pyarrow", "pillow", "tqdm", "matplotlib", "scikit-learn",
)

vol = modal.Volume.from_name("invariance-audit", create_if_missing=True)
REPO = "/root/repo"


@app.function(
    gpu="T4",
    image=image,
    volumes={"/mnt/cache": vol},
    mounts=[modal.Mount.from_local_dir(".", remote_path=REPO)],
    timeout=8 * 3600,
)
def extract_remote(n_images: int = 2000, k: int = 32, seed: int = 1337):
    import os
    import sys
    sys.path.insert(0, REPO)
    os.environ["HF_HOME"] = "/mnt/cache/hf"

    from audit.extract import extract_to_parquet, folder_images, set_seed
    from interp_core.loaders import load_model, resolve_hook_name
    from interp_core.sae import load_sae, validate_sae

    set_seed(seed)
    bundle = load_model("open_clip:ViT-B-32", device="cuda")
    sae = load_sae("Prisma-Multimodal/sae-top_k-64-cls_only-layer_9-hook_resid_post", device="cuda")
    hook = resolve_hook_name(bundle.spec, sae.hook_layer, sae.hook_component)
    images = folder_images("/mnt/cache/data/imagenet-val-2k", n_images)
    out = extract_to_parquet(images, bundle, sae, bundle.preprocess, "/mnt/cache/out",
                             hook_name=hook, device="cuda", gate_n=500)
    paths, gate_acts = out if isinstance(out, tuple) else (out, None)
    gate = validate_sae(gate_acts, sae) if gate_acts is not None else {}
    vol.commit()
    return {"paths": paths, "gate": gate, "hook": hook}


@app.local_entrypoint()
def main(n_images: int = 2000):
    print(extract_remote.remote(n_images))
