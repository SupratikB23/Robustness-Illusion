from __future__ import annotations

import os

REQUIRED_COLUMNS = ["image_id", "transform", "strength", "top_k_indices", "top_k_values", "embedding"]
MAX_ROWS = 500_000


def _is_int_list(v) -> bool:
    import numpy as np
    return isinstance(v, (list, np.ndarray)) and all(isinstance(i, (int, np.integer)) for i in v)


def _is_num_list(v) -> bool:
    import numpy as np
    return isinstance(v, (list, np.ndarray)) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in v)


def validate_schema(df) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Cache missing columns: {missing}")
    if len(df) > MAX_ROWS:
        raise ValueError(f"Cache has {len(df)} rows, cap is {MAX_ROWS}")
    if len(df):
        row = df.iloc[0]
        if not _is_int_list(row["top_k_indices"]):
            raise ValueError("top_k_indices must be a list of ints")
        if not _is_num_list(row["top_k_values"]) or not _is_num_list(row["embedding"]):
            raise ValueError("top_k_values and embedding must be numeric lists")


def write_parquet(df, path: str) -> str:
    import pandas as pd
    validate_schema(df)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: str):
    import pandas as pd
    df = pd.read_parquet(path)
    validate_schema(df)
    return df


def kaggle_input_path(dataset: str, filename: str) -> str:
    return os.path.join("/kaggle", "input", dataset, filename)


def kaggle_output_dir() -> str:
    return "/kaggle/working"
