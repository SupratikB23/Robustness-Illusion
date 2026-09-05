# Second-model worked example

Proves the CLI accepts any timm / open_clip model plus a compatible SAE,
not only the `ViT-B-32` + Prisma Layer-9 pair.

## A: timm ViT with neuron-basis fallback (no SAE needed)

If the SAE gate rejects, keep the whole spec and swap only the
representation: top-k MLP neurons by activation instead of top-k SAE
features. The Jaccard / RII / null machinery is unchanged.

```bash
python cli.py run \
  --mock \
  --model timm:vit_base_patch32_224 \
  --k 32 \
  --n-images 64 \
  --out results/second-model/
```

`--mock` replaces the network + SAE with seeded random equivalents so the
pipeline runs CPU-only. Drop `--mock` on Kaggle with the real weights;
with a real timm model and no SAE, encode with
`interp_core.heads.topk_neurons_by_activation` at the same hook point.

## B: a second CLIP SAE (different layer)

Same model family, different instrument — still exercises the generic
`--sae` path including hook-point metadata validation:

```bash
python cli.py validate-sae \
  --sae Prisma-Multimodal/sae-top_k-64-cls_only-layer_9-hook_resid_post
```

Swap in any other `Prisma-Multimodal/sae-top_k-64-cls_only-*` repo; the
loader reads layer/component from `config.json` and refuses to run if the
hook point is undeclared, instead of guessing.
