# The Robustness Illusion

Measures whether a vision transformer's internal features stay stable
under transformations that leave its output stable.

For transform `T` at strength `s`, against the same image at `s = 0`:

```
OutputStability(s)  = cosine similarity of image embeddings
FeatureStability(s) = Jaccard index of top-k active SAE feature sets
RII(T)              = mean over s > 0 of [OutputStability - FeatureStability]
```

Both curves equal `1.0` at `s = 0` by construction.

## Quickstart (CPU-only demo, no weights needed)

```bash
pip install -e ".[test,data]"
pytest
python cli.py run --mock --out results/ --n-images 32
python cli.py render-sweep --image examples/sample.jpg --out results/sweep.png
```

`--mock` is a CPU smoke test with seeded random weights and structured
synthetic images. It exercises the full pipeline but its numbers are not
results. Pinned working environment: `requirements-lock.txt`.

## Full run on Molab (or any marimo GPU session)

Open `notebooks/03_full_run_marimo.py` in Molab and press buttons **1 → 5**.
Nothing heavy runs on open — every stage waits for its button — so the
notebook costs nothing until you start it. Stages 3 and 4 halt the notebook
themselves if the SAE gate rejects or Null D empties over half the pairs.

Set `HF_TOKEN` first if you have ImageNet access. Without it, stage 2 detects
the gate in about two seconds and falls back to public Tiny-ImageNet,
recording the swap in `results/dataset_source.txt` and the site JSON.

## Building the image subset by hand

```bash
python scripts/make_subset.py --n 2000 --out data/imagenet-val-2k
python scripts/make_subset.py --n 2000 --out data/tiny-imagenet-2k --source tiny-imagenet
```

Deterministic (seed 1337), resumable, and it prints progress as frames land.
Writes are atomic, so an interrupted run leaves no truncated JPEG; re-running
keeps what is already on disk and fetches only the gap. `--fallback` switches
to Tiny-ImageNet automatically if the ImageNet gate blocks the token.

Then render one sweep and **look at it** before trusting any number:

```bash
python cli.py render-sweep --image data/imagenet-val-2k/ILSVRC2012_val_00000000.JPEG --out results/sweep.png
```

## Full run (CLI)

```bash
invariance-audit run \
  --model open_clip:ViT-B-32 \
  --sae Prisma-Multimodal/sae-top_k-64-cls_only-layer_9-hook_resid_post \
  --dataset data/imagenet-val-2k \
  --transforms rotation,crop,jitter,jpeg \
  --k 32 \
  --device cuda \
  --out results/
```

See `notebooks/` for thin GPU wrappers and `examples/second_model.md`
for a worked second-model example.

## Layout

- `interp_core/` — shared model-loading/hook/SAE/cache/null primitives (no transform, RII, or measurement code; loaders handle timm/open_clip weights by design)
- `audit/` — this project's measurement logic, plus `dataset.py` (image-subset fetching)
- `cli.py` — the tool entry point
- `site/` — static demo, reads precomputed JSON only
