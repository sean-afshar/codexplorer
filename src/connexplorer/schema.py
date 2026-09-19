"""Normalized table schemas, invariants and the dataset manifest (schema v1).

A dataset on disk is ``<root>/tables/`` holding the Parquet files listed in
:data:`TABLES`, a ``type_matrix.npz`` and a ``manifest.json``. This module is
the single source of truth for column names and dtypes; ingestion writes
against it and the runtime reads against it. Everything here is polars,
numpy and scipy only.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import scipy.sparse as sp

SCHEMA_VERSION = 1

# Synapse tables are written in row groups of this size so that a lazy scan
# filtered on the sort column prunes to one group (0.5 ms per neuron).
ROW_GROUP_SIZE = 1_048_576

MANIFEST_FILE = "manifest.json"
TYPE_MATRIX_FILE = "type_matrix.npz"
SKELETON_INDEX_FILE = "skeletons/skeleton_index.parquet"

NT_VALUES = ("ACH", "GABA", "GLUT", "DA", "SER", "OCT", "HIS")
NT_SOURCES = ("prediction", "literature")
SIDES = ("L", "R", "M")

# Strings that vendor exports use in place of a real null. Ingestion turns
# them into nulls; validation rejects them.
NULL_SENTINELS = ("NAN", "NaN", "nan", "Unknown", "unknown", "uncertain", "")


@dataclass(frozen=True)
class TableSpec:
    """Contract for one Parquet table.

    ``required`` columns must exist with exactly the given dtype and, unless
    listed in ``nullable``, contain no nulls. ``optional`` columns may be
    absent; when present they must match the dtype and may contain nulls.
    Extra dataset-specific columns are allowed. ``sorted_by`` is checked on
    validation.
    """

    name: str
    file: str
    required: dict[str, pl.DataType]
    optional: dict[str, pl.DataType] = field(default_factory=dict)
    nullable: frozenset[str] = frozenset()
    sorted_by: tuple[str, ...] = ()
    required_table: bool = True

    @property
    def columns(self) -> dict[str, pl.DataType]:
        return {**self.required, **self.optional}


_STR = pl.String
_U32 = pl.UInt32

_SYNAPSE_COLS = {"pre": _U32, "post": _U32, "x_nm": _U32, "y_nm": _U32, "z_nm": _U32}

TABLES: dict[str, TableSpec] = {
    "cells": TableSpec(
        name="cells",
        file="cells.parquet",
        required={"id": _U32, "root_id": pl.Int64},
        optional={
            "type": _STR,
            "side": _STR,
            "nt": _STR,
            "nt_score": pl.Float32,
            "superclass": _STR,
            "class": _STR,
            "subclass": _STR,
            "hemilineage": _STR,
            "flow": _STR,
        },
        sorted_by=("id",),
    ),
    "types": TableSpec(
        name="types",
        file="types.parquet",
        required={"type": _STR, "type_idx": _U32, "n_cells": _U32},
        optional={"nt": _STR, "nt_source": _STR, "nt_predicted": _STR},
        sorted_by=("type_idx",),
    ),
    "edges": TableSpec(
        name="edges",
        file="edges.parquet",
        required={"pre": _U32, "post": _U32, "n_syn": _U32},
        sorted_by=("pre", "post"),
    ),
    "edges_by_neuropil": TableSpec(
        name="edges_by_neuropil",
        file="edges_by_neuropil.parquet",
        required={"pre": _U32, "post": _U32, "neuropil": _STR, "n_syn": _U32},
        nullable=frozenset({"neuropil"}),
        sorted_by=("pre", "post"),
        required_table=False,
    ),
    "synapses": TableSpec(
        name="synapses",
        file="synapses.parquet",
        required=dict(_SYNAPSE_COLS),
        optional={"neuropil": _STR},
        sorted_by=("pre",),
    ),
    "synapses_by_post": TableSpec(
        name="synapses_by_post",
        file="synapses_by_post.parquet",
        required=dict(_SYNAPSE_COLS),
        optional={"neuropil": _STR},
        sorted_by=("post",),
        required_table=False,
    ),
    "columns": TableSpec(
        name="columns",
        file="columns.parquet",
        required={"id": _U32, "p": pl.Int16, "q": pl.Int16, "side": _STR},
        sorted_by=("id",),
        required_table=False,
    ),
}


# --------------------------------------------------------------------------
# Table-level validation
# --------------------------------------------------------------------------


def validate_table(spec: TableSpec, df: pl.DataFrame) -> list[str]:
    """Return a list of problems; empty means the frame satisfies ``spec``."""
    problems: list[str] = []
    schema = df.schema
    for col, dtype in spec.required.items():
        if col not in schema:
            problems.append(f"{spec.name}: missing required column {col!r}")
        elif schema[col] != dtype:
            problems.append(f"{spec.name}.{col}: dtype {schema[col]} != {dtype}")
    for col, dtype in spec.optional.items():
        if col in schema and schema[col] != dtype:
            problems.append(f"{spec.name}.{col}: dtype {schema[col]} != {dtype}")
    if df.is_empty():
        return problems

    non_null = [c for c in spec.required if c in schema and c not in spec.nullable]
    if non_null:
        counts = df.select(pl.col(non_null).null_count()).row(0)
        for col, n in zip(non_null, counts):
            if n:
                problems.append(f"{spec.name}.{col}: {n} nulls in a non-null column")

    str_cols = [c for c, t in schema.items() if t == pl.String]
    if str_cols:
        hits = df.select(pl.col(str_cols).is_in(NULL_SENTINELS).sum()).row(0)
        for col, n in zip(str_cols, hits):
            if n:
                problems.append(f"{spec.name}.{col}: {n} sentinel strings {NULL_SENTINELS[:2]}... instead of null")

    keys = [c for c in spec.sorted_by if c in schema]
    if keys and len(keys) == len(spec.sorted_by):
        sorted_ok = df.select(pl.struct(keys).is_sorted() if len(keys) > 1 else pl.col(keys[0]).is_sorted()).item()
        if not sorted_ok:
            problems.append(f"{spec.name}: not sorted by {spec.sorted_by}")

    if "nt" in schema and spec.name in ("cells", "types"):
        bad = df.filter(pl.col("nt").is_not_null() & ~pl.col("nt").is_in(NT_VALUES)).height
        if bad:
            problems.append(f"{spec.name}.nt: {bad} values outside {NT_VALUES}")
    if "nt_source" in schema:
        bad = df.filter(pl.col("nt_source").is_not_null() & ~pl.col("nt_source").is_in(NT_SOURCES)).height
        if bad:
            problems.append(f"{spec.name}.nt_source: {bad} values outside {NT_SOURCES}")
    if "side" in schema:
        bad = df.filter(pl.col("side").is_not_null() & ~pl.col("side").is_in(SIDES)).height
        if bad:
            problems.append(f"{spec.name}.side: {bad} values outside {SIDES}")
    return problems


# --------------------------------------------------------------------------
# I/O helpers that keep the on-disk invariants
# --------------------------------------------------------------------------


def write_table(df: pl.DataFrame, path: Path, *, row_group_size: int = ROW_GROUP_SIZE) -> None:
    """Write a Parquet table with the row-group size the lazy scans rely on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, row_group_size=row_group_size, statistics=True)


def read_table(tables_dir: Path, name: str) -> pl.DataFrame | None:
    """Eagerly read one table; ``None`` if an optional table is absent."""
    spec = TABLES[name]
    path = Path(tables_dir) / spec.file
    if not path.exists():
        if spec.required_table:
            raise FileNotFoundError(f"required table {name!r} missing at {path}")
        return None
    return pl.read_parquet(path)


def save_type_matrix(path: Path, matrix: sp.csr_matrix, type_list: np.ndarray | list[str]) -> None:
    m = sp.csr_matrix(matrix)
    m.sort_indices()
    np.savez(
        path,
        data=m.data.astype(np.uint64),
        indices=m.indices.astype(np.int32),
        indptr=m.indptr.astype(np.int32),
        shape=np.asarray(m.shape, dtype=np.int64),
        type_list=np.asarray(type_list, dtype=str),
    )


def load_type_matrix(path: Path) -> tuple[sp.csr_matrix, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        m = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return m, z["type_list"]


def sum_u64(df: pl.DataFrame, col: str) -> int:
    """Sum a count column without the u32 overflow footgun."""
    return int(df.select(pl.col(col).cast(pl.UInt64).sum()).item() or 0)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Dataset-level invariants (section 2.3 of the plan)
# --------------------------------------------------------------------------


@dataclass
class InvariantReport:
    problems: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict:
        return asdict(self)


def _row_group_ranges(path: Path, col: str) -> list[tuple[int, int]]:
    meta = pq.ParquetFile(path).metadata
    idx = meta.schema.names.index(col)
    ranges = []
    for i in range(meta.num_row_groups):
        st = meta.row_group(i).column(idx).statistics
        if st is None or not st.has_min_max:
            raise ValueError(f"{path.name}: row group {i} has no statistics for {col!r}")
        ranges.append((st.min, st.max))
    return ranges


def check_invariants(tables_dir: Path) -> InvariantReport:
    """Validate every table and the cross-table invariants under ``tables_dir``.

    The synapse tables are scanned lazily; nothing else here needs more memory
    than the edge table.
    """
    tables_dir = Path(tables_dir)
    rep = InvariantReport()

    frames: dict[str, pl.DataFrame] = {}
    for name, spec in TABLES.items():
        if name.startswith("synapses"):
            continue
        df = read_table(tables_dir, name)
        if df is None:
            continue
        frames[name] = df
        rep.problems += validate_table(spec, df)
        rep.counts[f"{name}_rows"] = df.height

    cells, edges = frames["cells"], frames["edges"]
    n = cells.height
    rep.counts["n_cells"] = n
    if not cells.select((pl.col("id") == pl.int_range(0, n, dtype=pl.UInt32)).all()).item():
        rep.problems.append("cells.id is not arange(N)")
    if cells.select(pl.col("root_id").n_unique()).item() != n:
        rep.problems.append("cells.root_id is not unique")

    for name in ("edges", "edges_by_neuropil", "columns"):
        df = frames.get(name)
        if df is None:
            continue
        cols = ["pre", "post"] if "pre" in df.columns else ["id"]
        bad = df.filter(pl.any_horizontal(pl.col(cols) >= n)).height
        if bad:
            rep.problems.append(f"{name}: {bad} rows reference ids >= {n}")

    if edges.select(pl.struct("pre", "post").n_unique()).item() != edges.height:
        rep.problems.append("edges: duplicate (pre, post) pairs")
    edge_sum = sum_u64(edges, "n_syn")
    rep.counts["n_synapses_from_edges"] = edge_sum
    rep.counts["n_autapses"] = edges.filter(pl.col("pre") == pl.col("post")).height
    rep.counts["n_autapse_synapses"] = sum_u64(edges.filter(pl.col("pre") == pl.col("post")), "n_syn")

    if "edges_by_neuropil" in frames:
        by_np = frames["edges_by_neuropil"].group_by("pre", "post").agg(pl.col("n_syn").cast(pl.UInt64).sum())
        merged = edges.join(by_np, on=["pre", "post"], how="full", coalesce=True, suffix="_np")
        mism = merged.filter(pl.col("n_syn").cast(pl.UInt64) != pl.col("n_syn_np")).height
        if mism:
            rep.problems.append(f"edges_by_neuropil: {mism} pairs whose neuropil sum != edges.n_syn")

    # synapses (lazy)
    syn_path = tables_dir / TABLES["synapses"].file
    if not syn_path.exists():
        rep.problems.append(f"required table 'synapses' missing at {syn_path}")
        return rep
    lazy = pl.scan_parquet(syn_path)
    head = lazy.head(1).collect()
    rep.problems += validate_table(TABLES["synapses"], head)
    n_syn = lazy.select(pl.len()).collect().item()
    rep.counts["synapses_rows"] = n_syn
    if n_syn != edge_sum:
        rep.problems.append(f"edges.n_syn.sum() = {edge_sum} != len(synapses) = {n_syn}")
    stats = lazy.select(
        pl.col("pre").is_sorted().alias("sorted"),
        (pl.col("pre") >= n).sum().alias("bad_pre"),
        (pl.col("post") >= n).sum().alias("bad_post"),
    ).collect().row(0, named=True)
    if not stats["sorted"]:
        rep.problems.append("synapses: not sorted by pre")
    if stats["bad_pre"] or stats["bad_post"]:
        rep.problems.append(f"synapses: {stats['bad_pre']} pre / {stats['bad_post']} post ids >= {n}")
    derived = lazy.group_by("pre", "post").agg(pl.len().cast(pl.UInt32).alias("n_syn")).sort("pre", "post").collect()
    if not derived.equals(edges.select("pre", "post", "n_syn").sort("pre", "post")):
        rep.problems.append("edges does not equal synapses grouped by (pre, post)")
    ranges = _row_group_ranges(syn_path, "pre")
    rep.counts["synapse_row_groups"] = len(ranges)
    for (lo1, hi1), (lo2, hi2) in zip(ranges, ranges[1:]):
        if not (lo1 <= hi1 <= lo2 <= hi2):
            rep.problems.append("synapses: row-group pre ranges are not disjoint and monotone")
            break

    bp_path = tables_dir / TABLES["synapses_by_post"].file
    rep.counts["has_synapses_by_post"] = int(bp_path.exists())
    if bp_path.exists():
        lazy_bp = pl.scan_parquet(bp_path)
        rep.problems += validate_table(TABLES["synapses_by_post"], lazy_bp.head(1).collect())
        n_bp = lazy_bp.select(pl.len()).collect().item()
        if n_bp != n_syn:
            rep.problems.append(f"synapses_by_post has {n_bp} rows, synapses has {n_syn}")
        if not lazy_bp.select(pl.col("post").is_sorted()).collect().item():
            rep.problems.append("synapses_by_post: not sorted by post")
        bp_ranges = _row_group_ranges(bp_path, "post")
        for (lo1, hi1), (lo2, hi2) in zip(bp_ranges, bp_ranges[1:]):
            if not (lo1 <= hi1 <= lo2 <= hi2):
                rep.problems.append("synapses_by_post: row-group post ranges are not disjoint and monotone")
                break

    # types and the type matrix
    types = frames["types"]
    if not types.select((pl.col("type_idx") == pl.int_range(0, types.height, dtype=pl.UInt32)).all()).item():
        rep.problems.append("types.type_idx is not arange(n_types)")
    if "type" in cells.columns:
        counted = cells.drop_nulls("type").group_by("type").len().rename({"len": "n"})
        joined = types.join(counted, on="type", how="full", coalesce=True)
        bad = joined.filter(pl.col("n_cells").cast(pl.Int64).fill_null(-1) != pl.col("n").cast(pl.Int64).fill_null(-1)).height
        if bad:
            rep.problems.append(f"types.n_cells disagrees with cells for {bad} types")
        typed = cells.filter(pl.col("type").is_not_null()).select("id")
        typed_edges = edges.join(typed, left_on="pre", right_on="id", how="semi").join(typed, left_on="post", right_on="id", how="semi")
        typed_sum = sum_u64(typed_edges, "n_syn")
        rep.counts["n_synapses_between_typed_cells"] = typed_sum
        rep.counts["n_synapses_involving_untyped_cells"] = edge_sum - typed_sum
        tm_path = tables_dir / TYPE_MATRIX_FILE
        if tm_path.exists():
            m, type_list = load_type_matrix(tm_path)
            if list(type_list) != types["type"].to_list():
                rep.problems.append("type_matrix.type_list != types.type order")
            if int(m.sum()) != typed_sum:
                rep.problems.append(f"type_matrix.sum() = {int(m.sum())} != typed edge sum {typed_sum}")
        else:
            rep.problems.append(f"{TYPE_MATRIX_FILE} missing")
    if {"nt", "nt_source", "nt_predicted"} <= set(types.columns):
        bad = types.filter((pl.col("nt_source") == "prediction") & (pl.col("nt") != pl.col("nt_predicted"))).height
        if bad:
            rep.problems.append(f"types: {bad} rows labelled 'prediction' whose nt != nt_predicted")

    connected = pl.concat([edges.select(pl.col("pre").alias("id")), edges.select(pl.col("post").alias("id"))]).unique().height
    rep.counts["n_cells_without_edges"] = n - connected
    return rep


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


@dataclass
class Manifest:
    """What ``manifest.json`` records about a built dataset."""

    dataset: str
    version: str
    schema_version: int = SCHEMA_VERSION
    created: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    source_files: dict[str, dict] = field(default_factory=dict)  # name -> {path, sha256, bytes}
    tables: dict[str, dict] = field(default_factory=dict)  # name -> {file, rows}
    invariants: dict = field(default_factory=dict)
    nt_overrides_version: str | None = None
    extras: dict = field(default_factory=dict)

    @classmethod
    def read(cls, tables_dir: Path) -> "Manifest":
        with open(Path(tables_dir) / MANIFEST_FILE) as f:
            raw = json.load(f)
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{tables_dir}: schema version {raw.get('schema_version')} != {SCHEMA_VERSION}; rebuild with connexplorer ingest"
            )
        return cls(**raw)

    def write(self, tables_dir: Path) -> Path:
        path = Path(tables_dir) / MANIFEST_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
            f.write("\n")
        return path

    def record_tables(self, tables_dir: Path) -> None:
        tables_dir = Path(tables_dir)
        for name, spec in TABLES.items():
            path = tables_dir / spec.file
            if path.exists():
                self.tables[name] = {"file": spec.file, "rows": pq.ParquetFile(path).metadata.num_rows}
