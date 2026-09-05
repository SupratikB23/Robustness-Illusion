from __future__ import annotations

import argparse
import os


def build_stream(source: str, seed: int):
    from datasets import load_dataset
    if source == "imagenet-1k":
        try:
            ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
        except Exception as e:
            if any(m in str(e).lower() for m in ("gated", "401", "unauthorized", "private")):
                raise SystemExit(
                    "GATED DATASET: this token lacks access to ILSVRC/imagenet-1k.\n"
                    "  1. Open https://huggingface.co/datasets/ILSVRC/imagenet-1k while logged in\n"
                    "  2. Submit the access form + accept the ImageNet terms, wait for approval\n"
                    "  3. Re-run, OR bypass now with --source tiny-imagenet (public, no approval;\n"
                    "     state the dataset swap in the report — same pipeline, different images).")
            raise
        return ds, "ILSVRC2012_val_"
    for split in ("valid", "validation", "test"):
        try:
            return load_dataset("zh-plus/tiny-imagenet", split=split, streaming=True), "tiny_val_"
        except Exception:
            continue
    raise SystemExit("Could not load zh-plus/tiny-imagenet (tried splits valid/validation/test).")


def fetch(source: str, n: int, seed: int, out: str) -> None:
    import time
    for attempt in range(3):
        try:
            ds, prefix = build_stream(source, seed)
            count = 0
            for i, row in enumerate(ds.shuffle(seed=seed, buffer_size=10000).take(n)):
                row["image"].convert("RGB").save(os.path.join(out, f"{prefix}{i:08d}.JPEG"))
                count += 1
                if count % 250 == 0:
                    print(f"{count}/{n} ...", flush=True)
            print(f"wrote {count} {source} images to {out}", flush=True)
            return
        except SystemExit:
            raise
        except Exception as e:
            print(f"attempt {attempt + 1} failed ({e}); retrying in 10s", flush=True)
            time.sleep(10)
    raise SystemExit("Download failed 3 times; re-run this cell. Partial files reuse the same names, so resume is safe.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default="data/imagenet-val-2k")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--source", default="imagenet-1k", choices=["imagenet-1k", "tiny-imagenet"])
    ns = p.parse_args()

    from datasets import load_dataset  # noqa: F401 (ensures dep present before starting)
    os.makedirs(ns.out, exist_ok=True)
    fetch(ns.source, ns.n, ns.seed, ns.out)


if __name__ == "__main__":
    main()
