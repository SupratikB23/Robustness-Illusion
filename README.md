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
pip install -e ".[test]"
pytest
python cli.py run --mock --out results/ --n-images 32
python cli.py render-sweep --image examples/sample.jpg --out results/sweep.png
```

`--mock` is a CPU smoke test with seeded random weights and structured
synthetic images. It exercises the full pipeline but its numbers are not
results. Pinned working environment: `requirements-lock.txt`.

## Full run (Kaggle GPU)

```bash
invariance-audit run \
  --model open_clip:ViT-B-32 \
  --sae Prisma-Multimodal/sae-top_k-64-cls_only-layer_9-hook_resid_post \
  --dataset imagenet-val-2k \
  --transforms rotation,crop,jitter,jpeg \
  --k 32 \
  --out results/
```

See `notebooks/` for thin Kaggle wrappers and `examples/second_model.md`
for a worked second-model example.

## Layout

- `interp_core/` — shared model-loading/hook/SAE/cache/null primitives (no transform, RII, or measurement code; loaders handle timm/open_clip weights by design)
- `audit/` — this project's measurement logic
- `cli.py` — the tool entry point
- `site/` — static demo, reads precomputed JSON only
