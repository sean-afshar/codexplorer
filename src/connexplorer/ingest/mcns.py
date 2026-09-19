"""Male CNS source: neuPrint flat-connectome feather exports -> schema v1 tables.

Reads ``body-annotations``, ``body-neurotransmitters`` and ``syn-partners``
for one release (files from
``gs://flyem-male-cns/<release>/connectome-data/flat-connectome/``). The
synapse partner file (300M+ rows, 6.8 GB) is streamed in record-batch
chunks; rows whose bodies are not annotated cells are dropped and counted.
Synapse positions are the midpoint of the pre and post sites, converted from
voxels to nanometres. The cell universe is every annotated body, glia and
orphans included; ``cells.status`` lets users filter.

Neurotransmitters: ``nt``/``nt_score`` are the per-body prediction
(``predicted_nt``, ``predicted_nt_confidence``); the vendor's per-type
``consensus_nt`` and the ``ground_truth`` column ride along as
``nt_consensus`` and ``nt_ground_truth``. Releases before v1.0 only have
``consensus_nt``, which then fills ``nt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from connexplorer.ingest.base import RawDir, RawFile, Source, normalize_nt

# Column name in the vendor annotation file -> our name. Missing ones are skipped.
ANNOTATION_COLUMNS = {
    "bodyId": "root_id",
    "type": "type",
    "somaSide": "side",
    "rootSide": "root_side",
    "superclass": "superclass",
    "class": "class",
    "subclass": "subclass",
    "flywireType": "flywire_type",
    "hemibrainType": "hemibrain_type",
    "mancType": "manc_type",
    "instance": "instance",
    "group": "group",
    "itoleeHl": "hemilineage",
    "entryNerve": "nerve",
    "dimorphism": "dimorphism",
    "status": "status",
    "assignedOlHex1": "p",
    "assignedOlHex2": "q",
}
NT_COLUMNS = {
    "body": "root_id",
    "predicted_nt": "nt",
    "predicted_nt_confidence": "nt_score",
    "consensus_nt": "nt_consensus",
    "ground_truth": "nt_ground_truth",
}
PARTNER_COLUMNS = ["body_pre", "body_post", "x_pre", "y_pre", "z_pre", "x_post", "y_post", "z_post"]
PARTNER_NEUROPIL_COLUMN = "primary_post"  # ROI of the postsynaptic site; absent before v1.0
_UNSPECIFIED = "<unspecified>"

# No literature overrides yet for the male CNS; the consensus column is used as is.
NT_OVERRIDES: dict[str, str] = {}
NT_OVERRIDES_VERSION = "mcns-none"


def raw_files_for(release: str) -> dict[str, RawFile]:
    return {
        "annotations": RawFile("annotations", f"body-annotations-male-cns-{release}-minconf-0.5.feather"),
        "neurotransmitters": RawFile("neurotransmitters", f"body-neurotransmitters-male-cns-{release}.feather"),
        "partners": RawFile("partners", f"syn-partners-male-cns-{release}-minconf-0.5.feather"),
        "weights": RawFile(
            "weights", f"connectome-weights-male-cns-{release}-minconf-0.5.feather", required=False, read=False,
            description="vendor pair weights; used only by verify_against_weights",
        ),
    }


def _read_feather(path: Path, rename: dict[str, str], required: tuple[str, ...]) -> pl.DataFrame:
    """Read the vendor columns in ``rename`` that exist; ``required`` names our must-haves."""
    names = ipc.open_file(pa.memory_map(str(path))).schema.names
    cols = [c for c in rename if c in names]
    missing = [r for r in required if r not in {rename[c] for c in cols}]
    if missing:
        raise ValueError(f"{path.name}: required columns missing: {missing}; has {names}")
    table = feather.read_table(str(path), columns=cols, memory_map=True)
    return pl.from_arrow(table).rename({c: rename[c] for c in cols})


class McnsSource(Source):
    name = "mcns"
    nt_overrides = NT_OVERRIDES
    nt_overrides_version = NT_OVERRIDES_VERSION
    type_summary_columns = ("superclass", "class", "subclass", "flywire_type", "hemibrain_type", "nt_consensus")

    def __init__(self, release: str = "v1.0", voxel_size_nm: int = 8, batches_per_chunk: int = 32):
        self.version = release
        self.raw_files = raw_files_for(release)
        self.voxel_size_nm = voxel_size_nm
        self.batches_per_chunk = batches_per_chunk

    def cells(self, raw: RawDir) -> pl.DataFrame:
        cells = _read_feather(raw.path("annotations"), ANNOTATION_COLUMNS, ("root_id", "type"))
        cells = cells.with_columns(pl.col("root_id").cast(pl.Int64))
        if "group" in cells.columns:
            cells = cells.with_columns(pl.col("group").cast(pl.Int64))
        nt = _read_feather(raw.path("neurotransmitters"), NT_COLUMNS, ("root_id",)).with_columns(pl.col("root_id").cast(pl.Int64))
        if "nt" not in nt.columns:  # pre-v1.0 files only carry the consensus
            nt = nt.with_columns(pl.col("nt_consensus").alias("nt"))
        for c in ("nt_consensus", "nt_ground_truth"):
            if c in nt.columns:
                nt = nt.with_columns(normalize_nt(pl.col(c)).alias(c))
        nt = nt.unique(subset="root_id", keep="first")
        cells = cells.join(nt, on="root_id", how="left")
        for c in ("p", "q"):
            if c in cells.columns:
                cells = cells.with_columns(pl.col(c).cast(pl.Float64).cast(pl.Int16))
        return cells

    def columns(self, raw: RawDir, cells: pl.DataFrame) -> pl.DataFrame | None:
        if not {"p", "q"} <= set(cells.columns):
            return None
        return cells.filter(pl.col("p").is_not_null() & pl.col("q").is_not_null()).select("root_id", "p", "q", "side")

    def synapse_batches(self, raw: RawDir) -> Iterator[pl.DataFrame]:
        half_voxel = self.voxel_size_nm / 2  # midpoint of pre and post, in nm: (a + b) / 2 * voxel
        reader = ipc.open_file(pa.memory_map(str(raw.path("partners"))))
        names = reader.schema.names
        missing = [c for c in PARTNER_COLUMNS if c not in names]
        if missing:
            raise ValueError(f"syn-partners file lacks columns {missing}; has {names}")
        has_neuropil = PARTNER_NEUROPIL_COLUMN in names
        cols = PARTNER_COLUMNS + ([PARTNER_NEUROPIL_COLUMN] if has_neuropil else [])
        idx = [names.index(c) for c in cols]
        n = reader.num_record_batches
        for start in range(0, n, self.batches_per_chunk):
            batches = [reader.get_batch(i).select(idx) for i in range(start, min(start + self.batches_per_chunk, n))]
            df = pl.from_arrow(pa.Table.from_batches(batches))
            exprs = [
                pl.col("body_pre").cast(pl.Int64).alias("pre_root_id"),
                pl.col("body_post").cast(pl.Int64).alias("post_root_id"),
                *[
                    ((pl.col(f"{ax}_pre").cast(pl.Float64) + pl.col(f"{ax}_post").cast(pl.Float64)) * half_voxel)
                    .round(0)
                    .cast(pl.UInt32)
                    .alias(f"{ax}_nm")
                    for ax in ("x", "y", "z")
                ],
            ]
            if has_neuropil:
                npil = pl.col(PARTNER_NEUROPIL_COLUMN).cast(pl.String)
                exprs.append(pl.when(npil == _UNSPECIFIED).then(None).otherwise(npil).alias("neuropil"))
            yield df.select(exprs)


def verify_against_weights(tables_dir: Path, raw_dir: Path, release: str = "v1.0") -> dict[str, int]:
    """Compare ``edges.parquet`` with the vendor ``connectome-weights`` file.

    Returns counts; ``mismatched_pairs`` and ``missing_pairs`` should be 0.
    """
    from connexplorer import schema

    raw = RawDir(raw_dir, raw_files_for(release))
    weights = pl.from_arrow(feather.read_table(str(raw.path("weights")), memory_map=True))
    cells = schema.read_table(tables_dir, "cells").select("root_id", "id")
    edges = schema.read_table(tables_dir, "edges")
    vendor = (
        weights.join(cells, left_on="body_pre", right_on="root_id", how="inner")
        .rename({"id": "pre"})
        .join(cells, left_on="body_post", right_on="root_id", how="inner")
        .rename({"id": "post"})
        .select("pre", "post", pl.col("weight").cast(pl.UInt32).alias("n_syn"))
    )
    merged = edges.join(vendor, on=["pre", "post"], how="full", coalesce=True, suffix="_vendor")
    return {
        "vendor_pairs_between_cells": vendor.height,
        "edges": edges.height,
        "mismatched_pairs": merged.filter(pl.col("n_syn") != pl.col("n_syn_vendor")).height,
        "missing_pairs": merged.filter(pl.col("n_syn").is_null() | pl.col("n_syn_vendor").is_null()).height,
    }
