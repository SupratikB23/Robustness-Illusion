"""Thin CLI over audit.dataset.build_subset. No logic lives here."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit.dataset import GatedDatasetError, build_subset  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(prog="make_subset")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default="data/imagenet-val-2k")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--source", default="imagenet-1k", choices=["imagenet-1k", "tiny-imagenet"])
    p.add_argument("--shuffle-buffer", type=int, default=None,
                   help="streaming shuffle buffer; larger mixes more but delays the first image")
    p.add_argument("--fallback", action="store_true",
                   help="fall back to the public tiny-imagenet if imagenet-1k is gated")
    p.add_argument("--no-resume", action="store_true")
    ns = p.parse_args()

    try:
        summary = build_subset(ns.source, ns.n, ns.out, ns.seed,
                               shuffle_buffer=ns.shuffle_buffer, resume=not ns.no_resume)
    except GatedDatasetError as e:
        if not ns.fallback:
            print(e, file=sys.stderr)
            return 2
        alt = ns.out.rstrip("/\\") + "-tiny"
        print(f"{e}\n\nFalling back to tiny-imagenet (--fallback) in {alt}.", file=sys.stderr)
        summary = build_subset("tiny-imagenet", ns.n, alt, ns.seed,
                               shuffle_buffer=ns.shuffle_buffer, resume=not ns.no_resume)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
