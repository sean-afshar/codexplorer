"""Graph statistics for a set of cells (ported from shayan's Circuit onto NeuronSet).

``within=False`` counts partners anywhere in the dataset; ``within=True``
restricts to the subgraph induced by the set. All respect ``connectivity.autapses``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from connexplorer.connectivity import ConnBlock
from connexplorer.dataset import Dataset

DEGREE_COLS = ("in_degree", "out_degree", "total_degree", "in_syn", "out_syn", "total_syn")


def degree(ds: Dataset, idx: np.ndarray, within: bool = False) -> pl.DataFrame:
    """Per-cell partner counts and synapse totals: root_id, type, in_degree, out_degree, total_degree, in_syn, out_syn, total_syn."""
    idx = np.asarray(idx, dtype=np.int64)
    conn = ds.connectivity
    if within:
        b = ConnBlock(conn, idx, idx).sparse
        out_deg, in_deg = np.diff(b.indptr), np.diff(b.tocsc().indptr)
        out_syn, in_syn = np.asarray(b.sum(axis=1)).ravel(), np.asarray(b.sum(axis=0)).ravel()
    else:
        t = conn.cell_totals
        out_deg, in_deg, out_syn, in_syn = t.out_deg[idx], t.in_deg[idx], t.out_syn[idx], t.in_syn[idx]
    return pl.DataFrame(
        {
            "root_id": ds.root_ids[idx],
            "type": ds._type_series.gather(idx),
            "in_degree": pl.Series(in_deg, dtype=pl.UInt32),
            "out_degree": pl.Series(out_deg, dtype=pl.UInt32),
            "total_degree": pl.Series(in_deg.astype(np.uint32) + out_deg.astype(np.uint32), dtype=pl.UInt32),
            "in_syn": pl.Series(in_syn, dtype=pl.UInt64),
            "out_syn": pl.Series(out_syn, dtype=pl.UInt64),
            "total_syn": pl.Series(in_syn.astype(np.uint64) + out_syn.astype(np.uint64), dtype=pl.UInt64),
        }
    )


def hubs(ds: Dataset, idx: np.ndarray, k: int = 10, by: str = "total_degree", within: bool = False) -> pl.DataFrame:
    if by not in DEGREE_COLS:
        raise ValueError(f"by must be one of {DEGREE_COLS}")
    return degree(ds, idx, within).sort(by, descending=True).head(k)


def degree_distribution(ds: Dataset, idx: np.ndarray, direction: str = "in", within: bool = False) -> pl.DataFrame:
    """Histogram of in- or out-degree: degree, n_cells."""
    col = {"in": "in_degree", "out": "out_degree", "total": "total_degree"}[direction]
    return degree(ds, idx, within).group_by(col).len().rename({col: "degree", "len": "n_cells"}).sort("degree")


def reciprocal(ds: Dataset, idx: np.ndarray, within: bool = True) -> pl.DataFrame:
    """Pairs connected in both directions: a, b (root ids, a < b), n_ab, n_ba, plus types.

    ``within=False`` pairs each cell of the set with any reciprocal partner in the dataset.
    """
    idx = np.asarray(idx, dtype=np.int64)
    conn = ds.connectivity
    m = conn.sparse
    if within:
        b = ConnBlock(conn, idx, idx).sparse.tocoo()
        r, c, w = idx[b.row], idx[b.col], b.data
    else:
        sub = (m[idx[0] : idx[-1] + 1] if len(idx) and idx[-1] - idx[0] + 1 == len(idx) else m[idx]).tocoo()
        r, c, w = idx[sub.row], sub.col.astype(np.int64), sub.data
    keep = r != c
    r, c, w = r[keep], c[keep], w[keep]
    if len(r) == 0:
        return pl.DataFrame(schema={"a": pl.Int64, "b": pl.Int64, "type_a": pl.String, "type_b": pl.String, "n_ab": pl.UInt32, "n_ba": pl.UInt32})
    back = np.asarray(m[c, r]).ravel()
    ok = back > 0
    if within:
        ok &= r < c  # each unordered pair once
    r, c, w, back = r[ok], c[ok], w[ok], back[ok]
    if not within:  # keep a < b but both orientations may appear from different set members; dedupe
        lo, hi = np.minimum(r, c), np.maximum(r, c)
        w_lo_hi = np.where(r < c, w, back)
        w_hi_lo = np.where(r < c, back, w)
        pairs = pl.DataFrame({"ia": lo, "ib": hi, "n_ab": w_lo_hi, "n_ba": w_hi_lo}).unique(subset=["ia", "ib"], maintain_order=True)
        r, c, w, back = pairs["ia"].to_numpy(), pairs["ib"].to_numpy(), pairs["n_ab"].to_numpy(), pairs["n_ba"].to_numpy()
    return pl.DataFrame(
        {
            "a": ds.root_ids[r],
            "b": ds.root_ids[c],
            "type_a": ds._type_series.gather(r),
            "type_b": ds._type_series.gather(c),
            "n_ab": pl.Series(w, dtype=pl.UInt32),
            "n_ba": pl.Series(back, dtype=pl.UInt32),
        }
    ).sort(["n_ab", "n_ba"], descending=True)


def summary(ds: Dataset, idx: np.ndarray) -> dict:
    idx = np.asarray(idx, dtype=np.int64)
    within = ConnBlock(ds.connectivity, idx, idx).sparse
    t = ds.connectivity.cell_totals
    types = ds._type_series.gather(idx)
    return {
        "n_cells": int(len(idx)),
        "n_types": int(types.drop_nulls().n_unique()),
        "n_edges_within": int(within.nnz),
        "n_syn_within": int(within.sum()),
        "n_syn_in": int(t.in_syn[idx].sum()),
        "n_syn_out": int(t.out_syn[idx].sum()),
        "n_partners_in": int(len(ds.connectivity._partner_weights(idx, "in")[0])) if len(idx) else 0,
        "n_partners_out": int(len(ds.connectivity._partner_weights(idx, "out")[0])) if len(idx) else 0,
    }
