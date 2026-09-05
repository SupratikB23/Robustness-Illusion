# P2: The Robustness Illusion

**Build this first.** It requires no training. It produces a deployed
public artifact in three weeks. Its code becomes the base for P1.

---

## Segment 1: The question

A vision transformer is expected to be stable under transformations that
do not change the content of an image. Rotate a photo of a dog by 8
degrees and the model should still say "dog". Compress it to JPEG
quality 40 and the model should still say "dog".

Model evaluations check this at the output. They report that accuracy
holds.

This project checks it at the level of internal features. It asks: when
the output stays the same, do the internal features stay the same?

The hypothesis is that they do not. The prediction is that the model
often produces a stable answer through an unstable internal route. If
that is true, then output-level robustness numbers overstate how stable
the model actually is.

### The measurable claim

For a transformation `T` with strength parameter `s`:

- `OutputStability(s)` = similarity between the model output on the
  original image and on `T(image, s)`.
- `FeatureStability(s)` = overlap between the top-k active SAE features
  on the original image and on `T(image, s)`.

The result of the project is the **gap** between these two curves,
reported per transformation as a single scalar.

### Why this is not already done

Robustness research measures outputs. SAE research measures features on
clean inputs. Papers exist on both sides. The intersection, measured as
paired curves with a defined gap statistic, is not a standard artifact
and has no tool.

Novelty level: moderate. The method is simple. The contribution is the
measurement, the statistic, and the tool. Do not overclaim it as a new
method. Claim it as a new measurement.

---

## Segment 2: Scope

### In scope

- One model family: CLIP ViT-B/32.
- One SAE source: published Prisma SAE checkpoints for CLIP ViT-B/32.
- Four transformations: rotation, centre crop and rescale, colour
  jitter, JPEG compression.
- One dataset: ImageNet validation subset, 2000 images.
- A CLI that audits any timm vision model paired with a compatible SAE.
- A static web demo with a per-transformation slider.

### Out of scope

- Training an SAE. If no usable published SAE exists, go to the fallback
  in Segment 8. Do not train one for this project.
- Adversarial perturbations. Those are a different literature with
  different controls.
- Fixing the instability. This project measures. It does not propose a
  mitigation.
- Multiple model families. Extend later if the result holds.

---

## Segment 3: Environment and compute

Inference only. No training.

| Step | Hardware | Estimated time |
|------|----------|----------------|
| SAE checkpoint validation | Kaggle T4 | 1 GPU-hour |
| Activation extraction, 2000 images x 4 transforms x 8 strengths | Kaggle T4 | 3 GPU-hours |
| Analysis and statistics | Local CPU | Minutes |
| Site build | Local CPU | Minutes |

Total under 5 GPU-hours. This fits inside one weekly Kaggle quota with
room to repeat.

Write the extracted activations and feature indices to a private Kaggle
Dataset named `invariance-audit-cache`. The analysis then runs locally
on a CPU, which removes the session limit from the critical path.

---

## Segment 4: Method, step by step

### Step 4.1: Validate the published SAE

Do this before writing anything else. It is the go/no-go gate.

1. Download the Prisma SAE checkpoint for CLIP ViT-B/32. Record the
   exact layer and hook point it was trained on. An SAE is valid only at
   the hook point it was trained for.
2. Run 500 clean ImageNet images through CLIP. Extract activations at
   that hook point.
3. Encode and decode with the SAE. Compute three numbers:
   - **Reconstruction fraction of variance explained.** Accept above
     0.7. Below 0.5, reject the checkpoint.
   - **L0**, the mean count of active features per input. Accept between
     10 and 200. Outside that range, the dictionary is either collapsed
     or not sparse.
   - **Dead feature fraction**, the share of features that never
     activate across 500 images. Accept below 0.5.
4. If any number fails, go to Segment 8.

Record all three numbers in the final report. Reviewers will ask.

### Step 4.2: Define the transformations

Each transformation takes an image and a strength `s` in `[0, 1]`, and
returns a transformed image. Use 8 strength levels: 0.0, 0.125, 0.25,
0.375, 0.5, 0.625, 0.75, 1.0. Strength 0.0 is the identity, and its
output is the reference for every comparison.

| Transform | s = 0.0 | s = 1.0 | Notes |
|-----------|---------|---------|-------|
| Rotation | 0 degrees | 30 degrees | Rotate about the centre, fill with edge pixels, then centre crop to remove black corners. Black corners are a confound. |
| Crop and rescale | full frame | 60% centre crop | Rescale back to 224x224 after cropping, so input size never changes. |
| Colour jitter | unchanged | brightness, contrast, and saturation each scaled by a factor drawn from a fixed schedule | Fix the schedule with a seed. Do not sample randomly per image. |
| JPEG compression | quality 100 | quality 20 | Encode and decode in memory. Do not write files. |

Fix the random seed. The same image at the same strength must produce
the same transformed image on every run.

### Step 4.3: Extract

For each image, each transformation, and each strength:

1. Run the forward pass. Capture the activation at the SAE hook point.
2. Encode with the SAE. Record the top-32 feature indices and their
   activation values.
3. Record the model output. For CLIP, the output is the image embedding.

Store per record: `image_id`, `transform`, `strength`, `top_k_indices`,
`top_k_values`, `image_embedding`.

Total records: 2000 x 4 x 8 = 64,000. Store as Parquet, one file per
transformation.

### Step 4.4: Compute the two stability curves

For each image `i`, transform `T`, strength `s`:

**Output stability.** Cosine similarity between the image embedding at
strength `s` and at strength 0.0.

```
OutputStability(i, T, s) = cos(emb(i, T, s), emb(i, T, 0))
```

**Feature stability.** Jaccard index between the top-k active feature
sets at strength `s` and at strength 0.0.

```
FeatureStability(i, T, s) = |F_s intersect F_0| / |F_s union F_0|
```

where `F_s` is the set of top-32 feature indices at strength `s`.

Report the mean across images, with a 95% bootstrap confidence interval
over images.

Both statistics are bounded in `[0, 1]` and equal 1.0 at `s = 0`, so the
curves are directly comparable.

### Step 4.5: Define the gap statistic

Call it the **Representational Instability Index**, or RII.

```
RII(T) = mean over s of [ OutputStability(T, s) - FeatureStability(T, s) ]
```

Interpretation:

- RII near 0: the internals move exactly as much as the output. Nothing
  surprising.
- RII large and positive: the output holds while the internals turn
  over. This is the finding the project is looking for.
- RII negative: the internals hold while the output moves. Report it if
  it happens. It would mean the instability is downstream of the hook
  point.

Report RII per transformation, with a confidence interval.

### Step 4.6: Run the null models

This step decides whether the result is real. Do not skip it and do not
run it last as an afterthought.

**Null A: random feature subset.** Replace the top-32 feature set with
32 features drawn at random from the active set. If Jaccard overlap
falls to chance under this null, the top-k choice is doing real work. If
the real curve looks like the null curve, the top-k statistic is
measuring nothing.

**Null B: shuffled pairing.** Compute Jaccard overlap between image `i`
at strength `s` and a different image `j` at strength 0.0. This gives
the floor. Any real overlap must sit clearly above this floor.

**Null C: k-sensitivity.** Recompute everything at k = 8, 16, 32, 64,
128. If RII changes sign or ordering across k, the result is an artifact
of the threshold and must be reported as such.

**Null D: dead-feature control.** Confirm that features counted as
"turning over" are not just features near the activation threshold
flickering. Recompute Jaccard using only features whose activation
exceeds twice the threshold. If the effect disappears, the finding is
threshold noise, not representational instability.

Null D is the one most likely to kill the result. Run it early, at Step
4.4, on 200 images, before extracting the full set.

### Step 4.7: Qualitative pass

Numbers alone will not carry the deployment. For each transformation,
find 3 images where output stability stays above 0.95 while feature
stability falls below 0.4. These are the demo cases. Inspect the
features that appear and disappear. Write a one-sentence description of
each.

---

## Segment 5: Deliverables

### 5.1 Repository

```
invariance-audit/
  README.md
  pyproject.toml
  interp_core/          # shared, later imported by P1
    loaders.py
    sae.py
    cache.py
    nulls.py
  audit/
    transforms.py
    extract.py
    stability.py
    report.py
  cli.py
  notebooks/
    01_validate_sae.ipynb
    02_extract.ipynb
  site/
    index.html
    app.js
    data/               # exported JSON
  tests/
```

### 5.2 CLI

The CLI is what separates this from a notebook.

```
invariance-audit run \
  --model open_clip:ViT-B-32 \
  --sae prisma/clip-vit-b-32-layer-9 \
  --dataset imagenet-val-2k \
  --transforms rotation,crop,jitter,jpeg \
  --k 32 \
  --out results/
```

It must accept any timm or open_clip model plus a compatible SAE, not
only the demo pair. Ship at least one worked example with a second model
to prove this.

### 5.3 Static site

Three views:

1. **Curve view.** Four charts, one per transformation. Each shows the
   two curves, the confidence bands, and the null floor. RII printed
   above each chart.
2. **Slider view.** The demo. One image, one slider per transformation.
   Dragging the slider updates three panels at once: the transformed
   image, the model prediction with confidence, and the top-32 feature
   list with entering and leaving features highlighted. All frames are
   precomputed, so the page needs no server.
3. **Case view.** The 3 hand-picked cases per transformation, with the
   written descriptions from Step 4.7.

Precompute every frame. The slider reads from a JSON file. There is no
inference at serve time.

### 5.4 Write-up

A page on the site, about 1200 words: question, method, the four RII
numbers, the null results, the limitations, and what it does not show.
State clearly that this is one model and one SAE, and that the finding
may not generalise.

---

## Segment 6: Timeline

| Block | Work |
|-------|------|
| Week 1, blocks 1-2 | SAE validation, Step 4.1. Go/no-go decision. |
| Week 1, blocks 3-4 | `transforms.py` plus visual check that each transform looks correct at every strength. |
| Week 1, block 5 | Null D on 200 images. Second go/no-go. |
| Week 2, blocks 1-2 | Full extraction on Kaggle, cached to a Dataset. |
| Week 2, blocks 3-4 | Stability computation, RII, nulls A to C. |
| Week 2, block 5 | Qualitative pass, pick demo cases. |
| Week 3, blocks 1-3 | Static site. |
| Week 3, blocks 4-5 | Write-up, CLI polish, deploy. |

Fifteen 3-hour blocks. About 45 hours of work.

---

## Segment 7: Risks

| Risk | Likelihood | Response |
|------|-----------|----------|
| Published SAE fails validation | Medium | Segment 8 fallback |
| Null D kills the effect | Medium | Report the negative result. It is still publishable and the tool still works. Reframe the site around "feature stability is threshold-dependent". |
| Effect is real but tiny | Medium | Report the effect size honestly. Do not inflate it. A small measured effect with clean nulls beats a large unverified one. |
| Black corners from rotation drive the result | High if unhandled | Already handled by the edge-fill plus centre crop in Step 4.2. Verify visually. |
| Feature indices are not comparable across runs | Low | The SAE is frozen, so indices are stable by construction. Confirm once. |

---

## Segment 8: Fallback if no usable SAE exists

If Step 4.1 fails, do not train an SAE. Switch the internal
representation being measured, and keep everything else:

**Fallback A: attention head stability.** Replace "top-k SAE features"
with "top-k attention heads by contribution to the output". Compute the
same Jaccard statistic over head indices. Needs no SAE at all. Novelty
drops slightly. Everything else in the spec holds unchanged.

**Fallback B: neuron basis.** Use the top-k MLP neurons by activation.
This is the weakest option because the neuron basis is polysemantic, but
it needs nothing beyond the model itself. Note the polysemanticity
limitation prominently.

**Fallback C: switch to a language SAE.** Gemma Scope SAEs are known
good. Replace image transformations with text paraphrases. The project
becomes "does paraphrasing change the internal route", which is a valid
version of the same question. This loses the vision angle, so prefer A
or B.

Choose Fallback A by default.

---

## Segment 9: Definition of done

- [ ] SAE validation numbers recorded and published
- [ ] Four RII values with confidence intervals
- [ ] Nulls A, B, C, D all run and reported, including any that failed
- [ ] CLI runs on a second model to prove generality
- [ ] Static site live on a custom domain
- [ ] Write-up published, limitations stated
- [ ] Repository public with a pinned environment and fixed seeds
- [ ] `interp_core/` extracted cleanly, ready for P1 to import
