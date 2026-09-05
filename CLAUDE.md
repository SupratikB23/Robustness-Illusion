# CLAUDE.md

Project instructions for coding agents working in this repository.
Read this file before writing code. Follow it over your own defaults.

---

## What this project is

**The Robustness Illusion** measures whether a vision transformer's internal
features stay stable under transformations that leave its output stable.

A model is expected to be invariant to rotation, cropping, colour
jitter, and JPEG compression. Benchmarks verify this at the output.
This project verifies it at the level of internal features.

The hypothesis: the output holds steady while the internal feature set
turns over. If that is true, output-level robustness numbers overstate
how stable the model is.

The deliverable is a measurement, a CLI, and a static web demo. It is
not a method paper and it does not propose a fix.

---

## The core measurement

For a transform `T` at strength `s`, against the same image at `s = 0`:

```
OutputStability(s)  = cosine similarity of image embeddings
FeatureStability(s) = Jaccard index of top-k active SAE feature sets
RII(T)              = mean over s of [OutputStability - FeatureStability]
```

`RII` is the **Representational Instability Index**. It is the single
headline number, reported once per transform with a bootstrap confidence
interval over images.

Both inputs to `RII` are bounded in `[0, 1]` and equal `1.0` at `s = 0`.
Keep that property. Any change that breaks it invalidates the statistic.

---

## Non-negotiable rules

1. **Never train an SAE in this repository.** The SAE is a published,
   frozen checkpoint. It is a fixed measuring instrument. If the
   checkpoint fails validation, switch the representation being measured
   (see Fallbacks), do not train a replacement.
2. **Never report a result without its null.** Four nulls are defined
   below. A number reported without them is not a result.
3. **Everything is precomputed.** The site is static JSON plus a viewer.
   There is no inference at serve time and no Python server. Do not add
   FastAPI, Flask, or Streamlit.
4. **Every seed is fixed.** Transform parameters, token sampling,
   bootstrap draws. The same image at the same strength must produce
   byte-identical output on every run.
5. **`interp_core/` is shared with a downstream project.** Keep it free
   of anything specific to this audit. No transform code, no RII, no
   CLIP assumptions inside `interp_core/`.

---

## Repository layout

```
interp_core/         # SHARED. Reused by the adapter-autopsy project.
  loaders.py         # load timm/open_clip models, attach hooks, return activations
  sae.py             # load frozen SAE checkpoint; encode, decode, top_k
  cache.py           # read/write activation caches as Kaggle Datasets
  nulls.py           # random-subset and shuffled-pairing generators
  viz/heatmap.py     # patch index -> image overlay
  viz/export.py      # write site JSON

audit/               # SPECIFIC to this project
  transforms.py      # the four transforms, strength-parameterised
  extract.py         # forward passes, SAE encoding, Parquet output
  stability.py       # the two curves, RII, bootstrap CIs
  report.py          # tables and charts

cli.py
notebooks/           # Kaggle-facing, thin wrappers over audit/ only
site/                # framework-free HTML and JS
  data/              # generated JSON, gitignored
tests/
```

Notebooks contain no logic. They import from `audit/` and call it. Any
logic written in a notebook is a bug.

---

## Environment

No local GPU. All GPU work runs on Kaggle free tier.

- Kaggle: T4 16GB, 12-hour session limit, 30 GPU-hours per week.
- `/kaggle/working` is 20 GB and is destroyed at session end.
- **Write all caches to a private Kaggle Dataset**, not to
  `/kaggle/working`. It mounts read-only at `/kaggle/input/<name>` in
  the next session.

Total compute for the whole project is under 5 GPU-hours. If a change
pushes it past 10, the change is wrong.

Analysis and site generation run on CPU. Keep them CPU-only so the
session limit never blocks the critical path.

---

## Reference configuration

| Item | Value |
|------|-------|
| Model | `open_clip:ViT-B-32` |
| SAE | Prisma checkpoint for CLIP ViT-B/32 |
| Hook point | whatever layer the SAE was trained on, no other |
| Dataset | ImageNet validation, 2000 images |
| Transforms | rotation, crop, jitter, jpeg |
| Strengths | 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0 |
| Default k | 32 |

An SAE is valid only at the hook point it was trained for. Applying it
elsewhere produces meaningless features. Read the checkpoint metadata,
do not guess the layer.

---

## Transform specifications

| Transform | s=0 | s=1 | Required detail |
|-----------|-----|-----|-----------------|
| rotation | 0 deg | 30 deg | Rotate about centre, fill with edge pixels, then centre crop. Black corners are a confound and will drive the result if left in. |
| crop | full frame | 60% centre crop | Rescale to 224x224 after cropping. Input size never changes. |
| jitter | unchanged | scheduled brightness, contrast, saturation | Use a fixed deterministic schedule. Do not sample per image. |
| jpeg | quality 100 | quality 20 | Encode and decode in memory. Do not touch disk. |

Render a strength sweep for one image and look at it before trusting any
number from a transform.

---

## The four nulls

All four are mandatory. Report each, including the ones that fail.

- **Null A, random subset.** Replace the top-k set with k features drawn
  at random from the active set. Confirms top-k selection does work.
- **Null B, shuffled pairing.** Jaccard between image `i` at strength
  `s` and a *different* image `j` at strength 0. This is the floor. Real
  overlap must sit clearly above it.
- **Null C, k-sensitivity.** Recompute at k = 8, 16, 32, 64, 128. If
  `RII` changes sign or ordering across k, the finding is a threshold
  artifact and must be reported as one.
- **Null D, threshold noise.** Recompute Jaccard using only features
  whose activation exceeds twice the threshold. This tests whether
  "turnover" is just borderline features flickering.

**Null D is the one most likely to kill the project.** Run it on 200
images early, before full extraction. If the effect vanishes under Null
D, stop and reframe around threshold dependence rather than continuing
to the full run.

---

## SAE acceptance gate

Run before anything else. On 500 clean images:

| Metric | Accept | Reject |
|--------|--------|--------|
| Fraction of variance explained | above 0.7 | below 0.5 |
| L0 (mean active features) | 10 to 200 | outside |
| Dead feature fraction | below 0.5 | above |

Publish these three numbers in the final report regardless of outcome.

---

## Fallbacks

If the SAE gate fails, keep the entire spec and swap only the
representation being measured:

- **Preferred: attention heads.** Replace "top-k SAE features" with
  "top-k attention heads by contribution to the output". Needs no SAE.
- **MLP neurons.** Weakest option. The neuron basis is polysemantic;
  state that limitation prominently if used.

Do not train an SAE as a fallback.

---

## CLI contract

```
invariance-audit run \
  --model open_clip:ViT-B-32 \
  --sae prisma/clip-vit-b-32-layer-9 \
  --dataset imagenet-val-2k \
  --transforms rotation,crop,jitter,jpeg \
  --k 32 \
  --out results/
```

The CLI must accept any timm or open_clip model paired with a compatible
SAE, not only the demo pair. Ship a worked example on a second model.
The CLI is what makes this a tool rather than a notebook. Treat it as a
deliverable, not as a wrapper.

---

## Site

Three views, all reading precomputed JSON:

1. **Curves.** Four charts, one per transform: both curves, confidence
   bands, null floor, `RII` printed above.
2. **Slider.** One slider per transform. Dragging updates three panels
   together: transformed image, prediction and confidence, and the
   top-32 feature list with entering and leaving features highlighted.
   Every frame is precomputed.
3. **Cases.** Three hand-picked images per transform where output
   stability stays above 0.95 while feature stability falls below 0.4,
   each with a written description.

The slider view is the demo. It has to be readable by a developer with
no interpretability background in about ten seconds. Optimise for that.

---

## Writing and reporting style

- Report effect sizes, not just significance.
- State negative results plainly. A small measured effect with clean
  nulls is worth more than a large unverified one.
- Never describe a feature as meaningful without showing its
  top-activating patches. If a contact sheet shows no pattern, say so.
- Scope every claim to one model, one SAE, and one dataset. Do not
  generalise to vision transformers as a class.

---

## Out of scope, refuse these

- Adversarial perturbations. Different literature, different controls.
- Any mitigation or fix for the instability.
- A second model family before the first result is complete.
- Training any SAE.
- A backend server of any kind.

---

## Definition of done

- [ ] SAE gate numbers published
- [ ] Four `RII` values with bootstrap confidence intervals
- [ ] Nulls A, B, C, D run and reported
- [ ] CLI verified on a second model
- [ ] Static site live on the custom domain
- [ ] Write-up published with limitations stated
- [ ] Seeds fixed, environment pinned, repository public
- [ ] `interp_core/` clean and importable by the downstream project
