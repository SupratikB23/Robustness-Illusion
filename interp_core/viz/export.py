from __future__ import annotations

import json
import os


SCHEMA_VERSION = 1


def write_json(obj: dict, path: str) -> str:
    import numpy as np
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(obj), f)
    return path


def _clean(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k) if isinstance(k, np.integer) else k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    return o


def curves_payload(config: dict, curves: dict, rii: dict) -> dict:
    return {"schema": SCHEMA_VERSION, "config": config, "curves": curves, "rii": rii}


def slider_payload(config: dict, frames: list[dict]) -> dict:
    return {"schema": SCHEMA_VERSION, "config": config, "frames": frames}


def cases_payload(config: dict, cases: list[dict]) -> dict:
    return {"schema": SCHEMA_VERSION, "config": config, "cases": cases}
