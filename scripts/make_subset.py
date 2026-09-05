from __future__ import annotations

import argparse
import os


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default="data/imagenet-val-2k")
    p.add_argument("--seed", type=int, default=1337)
    ns = p.parse_args()

    from datasets import load_dataset
    os.makedirs(ns.out, exist_ok=True)
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    ds = ds.shuffle(seed=ns.seed, buffer_size=10000)
    for i, row in enumerate(ds.take(ns.n)):
        row["image"].convert("RGB").save(os.path.join(ns.out, f"ILSVRC2012_val_{i:08d}.JPEG"))
    print(f"wrote {ns.n} images to {ns.out}")


if __name__ == "__main__":
    main()
