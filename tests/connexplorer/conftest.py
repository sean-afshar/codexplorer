"""Synthetic four-cell dataset in the connexplorer on-disk layout.

Cells 0..3 (root ids 10..13), types A (0, 1) and B (2, 3). Edges, including
one autapse, total 15 synapses:

    pre -> post : n_syn   types
    0 -> 0 : 1            A -> A  (autapse)
    0 -> 2 : 3            A -> B
    1 -> 2 : 2            A -> B
    2 -> 0 : 4            B -> A
    3 -> 1 : 5            B -> A

Type matrix therefore is [[1, 5], [9, 0]]. Type B carries a literature
neurotransmitter override (predicted GABA, recorded GLUT) so the labelled
layer is exercised.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from connexplorer import schema

EDGES = [(0, 0, 1), (0, 2, 3), (1, 2, 2), (2, 0, 4), (3, 1, 5)]


def write_fixture(root: Path, *, by_post: bool = True, row_group_size: int = 4) -> Path:
    """Write the fixture under ``root/tables`` and return that directory.

    A tiny ``row_group_size`` gives the synapse files several row groups so
    the row-group invariants are actually exercised.
    """
    tables = Path(root) / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    cells = pl.DataFrame(
        {
            "id": pl.Series([0, 1, 2, 3], dtype=pl.UInt32),
            "root_id": pl.Series([10, 11, 12, 13], dtype=pl.Int64),
            "type": ["A", "A", "B", "B"],
            "side": ["L", "R", "L", "R"],
            "nt": ["ACH", "ACH", "GABA", "GABA"],
            "nt_score": pl.Series([0.9, 0.8, 0.7, 0.6], dtype=pl.Float32),
        }
    )
    types = pl.DataFrame(
        {
            "type": ["A", "B"],
            "type_idx": pl.Series([0, 1], dtype=pl.UInt32),
            "n_cells": pl.Series([2, 2], dtype=pl.UInt32),
            "nt": ["ACH", "GLUT"],
            "nt_source": ["prediction", "literature"],
            "nt_predicted": ["ACH", "GABA"],
        }
    )
    edges = pl.DataFrame(
        {
            "pre": pl.Series([e[0] for e in EDGES], dtype=pl.UInt32),
            "post": pl.Series([e[1] for e in EDGES], dtype=pl.UInt32),
            "n_syn": pl.Series([e[2] for e in EDGES], dtype=pl.UInt32),
        }
    )
    edges_by_neuropil = pl.DataFrame(
        {
            "pre": pl.Series([0, 0, 0, 1, 2, 3], dtype=pl.UInt32),
            "post": pl.Series([0, 2, 2, 2, 0, 1], dtype=pl.UInt32),
            "neuropil": ["ME", "ME", "LO", "ME", "LO", None],
            "n_syn": pl.Series([1, 2, 1, 2, 4, 5], dtype=pl.UInt32),
        }
    )
    pre = np.repeat([e[0] for e in EDGES], [e[2] for e in EDGES])
    post = np.repeat([e[1] for e in EDGES], [e[2] for e in EDGES])
    k = len(pre)
    synapses = pl.DataFrame(
        {
            "pre": pl.Series(pre, dtype=pl.UInt32),
            "post": pl.Series(post, dtype=pl.UInt32),
            "x_nm": pl.Series(np.arange(k) * 100, dtype=pl.UInt32),
            "y_nm": pl.Series(np.arange(k) * 200, dtype=pl.UInt32),
            "z_nm": pl.Series(np.arange(k) * 300, dtype=pl.UInt32),
            "neuropil": ["ME"] * 3 + ["LO"] * 1 + ["ME"] * 2 + ["LO"] * 4 + [None] * 5,
        }
    ).sort("pre", "post", maintain_order=True)
    columns = pl.DataFrame(
        {
            "id": pl.Series([0, 2], dtype=pl.UInt32),
            "p": pl.Series([7, 8], dtype=pl.Int16),
            "q": pl.Series([9, 10], dtype=pl.Int16),
            "side": ["L", "L"],
        }
    )

    schema.write_table(cells, tables / "cells.parquet")
    schema.write_table(types, tables / "types.parquet")
    schema.write_table(edges, tables / "edges.parquet")
    schema.write_table(edges_by_neuropil, tables / "edges_by_neuropil.parquet")
    schema.write_table(synapses, tables / "synapses.parquet", row_group_size=row_group_size)
    if by_post:
        schema.write_table(
            synapses.sort("post", "pre", maintain_order=True),
            tables / "synapses_by_post.parquet",
            row_group_size=row_group_size,
        )
    schema.write_table(columns, tables / "columns.parquet")

    type_matrix = sp.csr_matrix(np.array([[1, 5], [9, 0]], dtype=np.uint64))
    schema.save_type_matrix(tables / schema.TYPE_MATRIX_FILE, type_matrix, ["A", "B"])

    manifest = schema.Manifest(dataset="fixture", version="v0", nt_overrides_version="fixture-1")
    manifest.record_tables(tables)
    manifest.invariants = schema.check_invariants(tables).to_dict()
    manifest.write(tables)
    return tables


@pytest.fixture
def fixture_tables(tmp_path: Path) -> Path:
    """Path to a freshly written ``tables/`` directory of the fixture dataset."""
    return write_fixture(tmp_path)
