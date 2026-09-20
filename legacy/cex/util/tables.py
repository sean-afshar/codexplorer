"""Small table IO helpers for normalized CEX files."""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

TABLE_EXTENSIONS = (".parquet", ".feather", ".csv", ".csv.gz", ".npz", ".pkl", ".pickle")


def find_table(fp_folder: str, name: str) -> str | None:
    """Return the first matching table path in `fp_folder`."""
    stem, ext = os.path.splitext(name)
    if ext:
        fp = os.path.join(fp_folder, name)
        return fp if os.path.exists(fp) else None

    for candidate_ext in TABLE_EXTENSIONS:
        fp = os.path.join(fp_folder, f"{name}{candidate_ext}")
        if os.path.exists(fp):
            return fp
    return None


def read_table(fp: str, columns: list[str] | None = None):
    """Read a supported table or pickle file."""
    ext = table_extension(fp)
    if ext == ".parquet":
        return pd.read_parquet(fp, columns=columns)
    if ext == ".feather":
        return pd.read_feather(fp, columns=columns)
    if ext in {".csv", ".csv.gz"}:
        # pd does not preserve column orders
        data = pd.read_csv(fp, usecols=columns)
        if columns is not None:
            data = data[columns]  # Ensure column order matches requested
        return data
    if ext == ".npz":
        return dict(np.load(fp))
    if ext in {".pkl", ".pickle"}:
        with open(fp, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported table extension: {ext}")


def write_table(data, fp: str) -> None:
    """Write a supported table or pickle file."""
    folder = os.path.dirname(fp)
    if folder:
        os.makedirs(folder, exist_ok=True)
    ext = table_extension(fp)
    if ext == ".parquet":
        data.to_parquet(fp, index=False)
    elif ext == ".feather":
        data.reset_index(drop=True).to_feather(fp)
    elif ext in {".csv", ".csv.gz"}:
        data.to_csv(fp, index=False)
    elif ext == ".npz":
        np.savez(fp, **data)
    elif ext in {".pkl", ".pickle"}:
        with open(fp, "wb") as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported table extension: {ext}")


def table_columns(fp: str) -> list[str]:
    """Return column names without loading full table contents."""
    if not os.path.exists(fp):
        return []
    ext = table_extension(fp)
    if ext == ".parquet":
        return pq.ParquetFile(fp).schema.names
    if ext == ".feather":
        with pa.memory_map(fp, "r") as source:
            return ipc.open_file(source).schema.names
    if ext in {".csv", ".csv.gz"}:
        return pd.read_csv(fp, nrows=0).columns.tolist()
    if ext == ".npz":
        return list(np.load(fp).files)
    table = read_table(fp)
    return list(table.columns) if hasattr(table, "columns") else []


def missing_columns(fp: str, required: list[str]) -> list[str]:
    """Return required columns missing from a table."""
    return sorted(set(required).difference(table_columns(fp)))


def preview_table(fp: str, n: int = 3) -> pd.DataFrame | None:
    """Return the first rows of a supported table."""
    if not os.path.exists(fp):
        fp = find_table(os.path.dirname(fp), os.path.splitext(os.path.basename(fp))[0])
    if fp is None:
        return None
    ext = table_extension(fp)
    if ext == ".parquet":
        return pq.ParquetFile(fp).read_row_group(0).to_pandas().head(n)
    if ext == ".feather":
        with pa.memory_map(fp, "r") as source:
            reader = ipc.open_file(source)
            if reader.num_record_batches == 0:
                return pd.DataFrame()
            return reader.get_batch(0).slice(0, n).to_pandas()
    if ext in {".csv", ".csv.gz"}:
        return pd.read_csv(fp, nrows=n)
    if ext == ".npz":
        data = np.load(fp)
        columns = [k for k in data.files if data[k].ndim == 1]
        return pd.DataFrame({k: data[k][:n] for k in columns})
    table = read_table(fp)
    return table.head(n) if hasattr(table, "head") else None


def table_extension(fp: str) -> str:
    """Return the supported table extension, including two-part suffixes."""
    fp = os.fspath(fp)
    if fp.endswith(".csv.gz"):
        return ".csv.gz"
    return os.path.splitext(fp)[1].lower()
