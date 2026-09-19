"""Ingestion: turn vendor downloads into the normalized tables of schema v1.

A :class:`Source` knows how to read one vendor's raw files and hand back
three things in vendor-agnostic form: a cell table keyed by ``root_id``, a
stream of synapse batches keyed by pre/post ``root_id``, and (optionally) a
column-assignment table. :func:`build` does everything else the same way for
every dataset: id assignment, type summaries with the labelled
neurotransmitter override layer, synapse sorting and row-group layout, edge
and type-matrix derivation, invariant checks and the manifest.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import polars as pl
import scipy.sparse as sp

from connexplorer import schema

# --------------------------------------------------------------------------
# Raw files
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RawFile:
    key: str
    filename: str
    required: bool = True
    read: bool = True  # False: listed for completeness, never opened, never hashed
    description: str = ""
    alternatives: tuple[str, ...] = ()  # other accepted file names (e.g. an uncompressed variant)


class RawDir:
    """The raw download directory of one dataset, addressed by file key."""

    def __init__(self, path: Path, files: dict[str, RawFile]):
        self.root = Path(path)
        self.files = files

    def _find(self, key: str) -> Path | None:
        f = self.files[key]
        for name in (f.filename, *f.alternatives):
            if (self.root / name).exists():
                return self.root / name
        return None

    def has(self, key: str) -> bool:
        return self._find(key) is not None

    def path(self, key: str) -> Path:
        p = self._find(key)
        if p is None:
            raise FileNotFoundError(f"raw file {self.files[key].filename!r} ({key}) not found in {self.root}")
        return p

    def missing_required(self) -> list[str]:
        return [f.filename for f in self.files.values() if f.required and not self.has(f.key)]

    def present(self) -> dict[str, Path]:
        return {k: p for k in self.files if (p := self._find(k)) is not None}


# --------------------------------------------------------------------------
# Source protocol
# --------------------------------------------------------------------------


class Source(ABC):
    """One vendor dataset release. Subclasses live in ``flywire.py`` and ``mcns.py``.

    Contract for the frames a source returns:

    * ``cells``: one row per cell, ``root_id`` (i64, unique) plus any of
      ``type``, ``side``, ``nt``, ``nt_score`` and dataset extras. Raw vendor
      spellings are fine; :func:`build` normalizes side and transmitter labels
      and turns sentinel strings into nulls.
    * ``synapse_batches``: frames with ``pre_root_id``, ``post_root_id`` (i64),
      ``x_nm``, ``y_nm``, ``z_nm`` (integer nanometres) and optionally
      ``neuropil``. Any order; rows whose root ids are not cells are dropped
      and counted.
    * ``columns``: ``root_id``, ``p``, ``q``, ``side`` plus extras, or ``None``.
    * ``type_extras``: per-type columns to attach to ``types.parquet``, or ``None``.
    """

    name: str
    version: str
    raw_files: dict[str, RawFile]
    nt_overrides: dict[str, str] = {}
    nt_overrides_version: str | None = None
    type_summary_columns: tuple[str, ...] = ()

    @abstractmethod
    def cells(self, raw: RawDir) -> pl.DataFrame: ...

    @abstractmethod
    def synapse_batches(self, raw: RawDir) -> Iterator[pl.DataFrame]: ...

    def columns(self, raw: RawDir, cells: pl.DataFrame) -> pl.DataFrame | None:
        return None

    def type_extras(self, raw: RawDir, types: pl.DataFrame) -> pl.DataFrame | None:
        return None


# --------------------------------------------------------------------------
# Label normalization
# --------------------------------------------------------------------------

SIDE_MAP = {
    "l": "L", "left": "L", "-1": "L",
    "r": "R", "right": "R", "1": "R",
    "m": "M", "mid": "M", "middle": "M", "midline": "M", "center": "M", "centre": "M",
}
NT_SYNONYMS = {
    "ACH": "ACH", "ACETYLCHOLINE": "ACH",
    "DA": "DA", "DOPAMINE": "DA",
    "GABA": "GABA", "GAMMA-AMINOBUTYRIC ACID": "GABA",
    "GLUT": "GLUT", "GLUTAMATE": "GLUT",
    "OCT": "OCT", "OCTOPAMINE": "OCT",
    "SER": "SER", "SEROTONIN": "SER", "5HT": "SER",
    "HIS": "HIS", "HISTAMINE": "HIS",
}
_SENTINELS_LOWER = tuple(s.lower() for s in schema.NULL_SENTINELS)


def _map_or_null(expr: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    mapped = expr.replace(mapping)
    return pl.when(mapped.is_in(list(set(mapping.values())))).then(mapped).otherwise(None)


def normalize_side(expr: pl.Expr) -> pl.Expr:
    """``left``/``l``/``-1`` -> ``L``, ``right``/``r``/``1`` -> ``R``, midline -> ``M``, else null."""
    return _map_or_null(expr.cast(pl.String).str.strip_chars().str.to_lowercase(), SIDE_MAP)


def normalize_nt(expr: pl.Expr) -> pl.Expr:
    """Vendor transmitter spellings -> the codes in :data:`schema.NT_VALUES`, else null."""
    return _map_or_null(expr.cast(pl.String).str.strip_chars().str.to_uppercase(), NT_SYNONYMS)


def nullify_sentinels(df: pl.DataFrame) -> pl.DataFrame:
    """Strip string columns and turn ``"NAN"``, ``"Unknown"``, ``""`` ... into real nulls."""
    str_cols = [c for c, t in df.schema.items() if t == pl.String]
    if not str_cols:
        return df
    return df.with_columns(
        [
            pl.when(pl.col(c).str.strip_chars().str.to_lowercase().is_in(_SENTINELS_LOWER))
            .then(None)
            .otherwise(pl.col(c).str.strip_chars())
            .alias(c)
            for c in str_cols
        ]
    )


# --------------------------------------------------------------------------
# Building blocks (each testable on its own)
# --------------------------------------------------------------------------

_CELL_FRONT = ["id", "root_id", "type", "side", "nt", "nt_score"]


def assign_ids(cells: pl.DataFrame) -> pl.DataFrame:
    """Normalize labels, sort by (type, root_id) with untyped cells last, add ``id``.

    Sorting by type makes every type a contiguous id range, which the runtime
    uses to slice type blocks straight out of the CSR matrix.
    """
    if cells["root_id"].n_unique() != cells.height:
        raise ValueError("cells.root_id is not unique")
    cells = nullify_sentinels(cells).with_columns(pl.col("root_id").cast(pl.Int64))
    if "side" in cells.columns:
        cells = cells.with_columns(normalize_side(pl.col("side")).alias("side"))
    if "nt" in cells.columns:
        cells = cells.with_columns(normalize_nt(pl.col("nt")).alias("nt"))
    if "nt_score" in cells.columns:
        cells = cells.with_columns(pl.col("nt_score").cast(pl.Float32))
    if "type" not in cells.columns:
        cells = cells.with_columns(pl.lit(None, dtype=pl.String).alias("type"))
    cells = cells.sort(["type", "root_id"], nulls_last=True).with_row_index("id")
    front = [c for c in _CELL_FRONT if c in cells.columns]
    return cells.select(front + [c for c in cells.columns if c not in front])


def _majority(typed: pl.DataFrame, col: str) -> pl.DataFrame:
    """Most common non-null value of ``col`` per type; ties break alphabetically."""
    return (
        typed.drop_nulls(col)
        .group_by("type", col)
        .len()
        .sort(["type", "len", col], descending=[False, True, False])
        .group_by("type", maintain_order=True)
        .first()
        .select("type", col)
    )


def build_types(
    cells: pl.DataFrame,
    summary_columns: tuple[str, ...] = (),
    nt_overrides: dict[str, str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Per-type table with the labelled neurotransmitter layer.

    Returns the table and the override keys that matched no type.
    """
    nt_overrides = dict(nt_overrides or {})
    typed = cells.drop_nulls("type")
    types = typed.group_by("type").len().rename({"len": "n_cells"}).sort("type")
    if "nt" in typed.columns:
        types = types.join(_majority(typed, "nt").rename({"nt": "nt_predicted"}), on="type", how="left")
    else:
        types = types.with_columns(pl.lit(None, dtype=pl.String).alias("nt_predicted"))
    overrides = pl.DataFrame(
        {"type": list(nt_overrides), "nt_override": list(nt_overrides.values())},
        schema={"type": pl.String, "nt_override": pl.String},
    )
    types = (
        types.join(overrides, on="type", how="left")
        .with_columns(
            pl.coalesce("nt_override", "nt_predicted").alias("nt"),
            pl.when(pl.col("nt_override").is_not_null())
            .then(pl.lit("literature"))
            .when(pl.col("nt_predicted").is_not_null())
            .then(pl.lit("prediction"))
            .otherwise(None)
            .alias("nt_source"),
        )
        .drop("nt_override")
    )
    for col in summary_columns:
        if col in typed.columns and col not in types.columns:
            types = types.join(_majority(typed, col), on="type", how="left")
    types = types.with_row_index("type_idx").with_columns(pl.col("n_cells").cast(pl.UInt32))
    front = ["type", "type_idx", "n_cells", "nt", "nt_source", "nt_predicted"]
    types = types.select(front + [c for c in types.columns if c not in front])
    unmatched = sorted(set(nt_overrides) - set(types["type"].to_list()))
    return types, unmatched


def map_root_ids(batch: pl.DataFrame, lookup: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Replace ``pre_root_id``/``post_root_id`` with ``pre``/``post`` ids; drop unknown rows."""
    out = (
        batch.join(lookup, left_on="pre_root_id", right_on="root_id", how="left")
        .rename({"id": "pre"})
        .join(lookup, left_on="post_root_id", right_on="root_id", how="left")
        .rename({"id": "post"})
    )
    out = out.filter(pl.col("pre").is_not_null() & pl.col("post").is_not_null())
    keep = ["pre", "post", "x_nm", "y_nm", "z_nm"] + [
        c for c in out.columns if c not in ("pre", "post", "x_nm", "y_nm", "z_nm", "pre_root_id", "post_root_id")
    ]
    out = out.select(keep).with_columns(pl.col("x_nm", "y_nm", "z_nm").cast(pl.UInt32))
    return out, batch.height - out.height


def build_type_matrix(edges: pl.DataFrame, cells: pl.DataFrame, types: pl.DataFrame) -> sp.csr_matrix:
    """Sum ``edges.n_syn`` into a (n_types, n_types) CSR; untyped cells are skipped."""
    n_types = types.height
    type_idx = (
        cells.select("type")
        .join(types.select("type", "type_idx"), on="type", how="left")["type_idx"]
        .fill_null(-1)
        .cast(pl.Int64)
        .to_numpy()
    )
    pre_t = type_idx[edges["pre"].to_numpy()]
    post_t = type_idx[edges["post"].to_numpy()]
    ok = (pre_t >= 0) & (post_t >= 0)
    weights = edges["n_syn"].to_numpy().astype(np.uint64)[ok]
    m = sp.coo_array((weights, (pre_t[ok], post_t[ok])), shape=(n_types, n_types)).tocsr()
    m.sum_duplicates()
    return sp.csr_matrix(m)


# --------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------


class BuildError(RuntimeError):
    def __init__(self, report: schema.InvariantReport, tables_dir: Path):
        self.report = report
        self.tables_dir = tables_dir
        super().__init__(f"{len(report.problems)} invariant(s) failed in {tables_dir}:\n  " + "\n  ".join(report.problems))


@dataclass
class BuildResult:
    tables_dir: Path
    manifest: schema.Manifest
    report: schema.InvariantReport
    timings: dict[str, float] = field(default_factory=dict)


def build(
    source: Source,
    out_dir: Path,
    raw_dir: Path,
    *,
    by_post: bool = True,
    hash_sources: bool = True,
    log: Callable[[str], None] | None = print,
) -> BuildResult:
    """Build ``<out_dir>/tables`` from ``raw_dir`` with ``source``.

    Raises :class:`BuildError` after writing everything (manifest included)
    if any invariant fails, so the output can be inspected.
    """
    log = log or (lambda s: None)
    raw = RawDir(raw_dir, source.raw_files)
    if missing := raw.missing_required():
        raise FileNotFoundError(f"{source.name} {source.version}: required raw files missing from {raw.root}: {missing}")
    tables = Path(out_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    extras: dict = {"raw_files_present": sorted(raw.present())}

    def stage(name: str):
        t0 = time.perf_counter()

        def done(msg: str = ""):
            timings[name] = round(time.perf_counter() - t0, 2)
            log(f"[{source.name}] {name}: {timings[name]} s {msg}")

        return done

    done = stage("cells")
    cells = assign_ids(source.cells(raw))
    schema.write_table(cells, tables / schema.TABLES["cells"].file)
    done(f"({cells.height} cells, {cells['type'].null_count()} untyped)")

    done = stage("types")
    types, unmatched = build_types(cells, source.type_summary_columns, source.nt_overrides)
    if (extra := source.type_extras(raw, types)) is not None:
        types = types.join(nullify_sentinels(extra), on="type", how="left")
    schema.write_table(types, tables / schema.TABLES["types"].file)
    extras["nt_overrides_unmatched"] = unmatched
    extras["nt_overrides_applied"] = int(types.filter(pl.col("nt_source") == "literature").height)
    done(f"({types.height} types, {extras['nt_overrides_applied']} literature overrides, {len(unmatched)} unmatched)")

    done = stage("synapses")
    lookup = cells.select("root_id", "id")
    frames: list[pl.DataFrame] = []
    n_in = n_dropped = 0
    for batch in source.synapse_batches(raw):
        mapped, dropped = map_root_ids(batch, lookup)
        n_in += batch.height
        n_dropped += dropped
        frames.append(mapped)
    synapses = pl.concat(frames, rechunk=False).sort(["pre", "post"])
    del frames
    schema.write_table(synapses, tables / schema.TABLES["synapses"].file)
    extras["synapses_read"] = n_in
    extras["synapses_dropped_unknown_cell"] = n_dropped
    done(f"({synapses.height} kept of {n_in}; {n_dropped} dropped for unknown cells)")

    if by_post:
        done = stage("synapses_by_post")
        schema.write_table(synapses.sort(["post", "pre"]), tables / schema.TABLES["synapses_by_post"].file)
        done()

    done = stage("edges")
    edges = (
        synapses.group_by("pre", "post")
        .len()
        .rename({"len": "n_syn"})
        .with_columns(pl.col("n_syn").cast(pl.UInt32))
        .sort(["pre", "post"])
    )
    schema.write_table(edges, tables / schema.TABLES["edges"].file)
    if "neuropil" in synapses.columns:
        by_np = (
            synapses.group_by("pre", "post", "neuropil")
            .len()
            .rename({"len": "n_syn"})
            .with_columns(pl.col("n_syn").cast(pl.UInt32))
            .sort(["pre", "post", "neuropil"], nulls_last=True)
        )
        schema.write_table(by_np, tables / schema.TABLES["edges_by_neuropil"].file)
    del synapses
    done(f"({edges.height} pairs)")

    done = stage("type_matrix")
    matrix = build_type_matrix(edges, cells, types)
    schema.save_type_matrix(tables / schema.TYPE_MATRIX_FILE, matrix, types["type"].to_list())
    done(f"({matrix.nnz} nonzero type pairs)")

    done = stage("columns")
    cols = source.columns(raw, cells)
    if cols is not None:
        cols = nullify_sentinels(cols).join(lookup, on="root_id", how="left")
        extras["column_rows_dropped_unknown_cell"] = int(cols["id"].null_count())
        cols = (
            cols.drop_nulls("id")
            .drop("root_id")
            .with_columns(
                pl.col("p").cast(pl.Int16),
                pl.col("q").cast(pl.Int16),
                normalize_side(pl.col("side")).alias("side"),
            )
            .sort("id")
        )
        front = ["id", "p", "q", "side"]
        cols = cols.select(front + [c for c in cols.columns if c not in front])
        schema.write_table(cols, tables / schema.TABLES["columns"].file)
        done(f"({cols.height} column-assigned cells)")
    else:
        done("(none)")

    done = stage("manifest")
    manifest = schema.Manifest(dataset=source.name, version=source.version, nt_overrides_version=source.nt_overrides_version)
    for key, path in raw.present().items():
        if not source.raw_files[key].read:
            continue
        entry = {"path": str(path.resolve()), "bytes": path.stat().st_size}
        if hash_sources:
            entry["sha256"] = schema.file_sha256(path)
        manifest.source_files[key] = entry
    manifest.record_tables(tables)
    report = schema.check_invariants(tables)
    manifest.invariants = report.to_dict()
    manifest.extras = extras
    manifest.write(tables)
    done(f"({len(report.problems)} problems)")
    for k, v in report.counts.items():
        log(f"[{source.name}]   {k} = {v}")

    if not report.ok:
        raise BuildError(report, tables)
    return BuildResult(tables, manifest, report, timings)
