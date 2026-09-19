"""Lazy synapse access: nothing is loaded until a query is collected.

``synapses.parquet`` is sorted by ``pre`` in 1M-row row groups, so a filter
on ``pre`` reads only the groups that can contain it. ``synapses_by_post``
does the same for ``post``; without it input-side queries fall back to a
full scan (about 50 ms on FlyWire).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from connexplorer import schema
from connexplorer.dataset import Dataset

XYZ = ["x_nm", "y_nm", "z_nm"]


def xyz(df: pl.DataFrame) -> np.ndarray:
    """(k, 3) uint32 array of synapse coordinates in nanometres."""
    return df.select(XYZ).to_numpy()


def _id_filter(col: str, idx: np.ndarray) -> pl.Expr:
    """Predicate for ``col in idx`` written so row-group statistics can prune."""
    if len(idx) == 1:
        return pl.col(col) == int(idx[0])
    lo, hi = int(idx.min()), int(idx.max())
    if hi - lo + 1 == len(idx):
        return pl.col(col).is_between(lo, hi)
    return pl.col(col).is_between(lo, hi) & pl.col(col).is_in(idx.tolist())


class Synapses:
    def __init__(self, ds: Dataset):
        self.ds = ds
        self._pre_path = ds.tables_dir / schema.TABLES["synapses"].file
        self._post_path = ds.tables_dir / schema.TABLES["synapses_by_post"].file

    @property
    def has_by_post(self) -> bool:
        return self._post_path.exists()

    def scan(self, by: str = "pre") -> pl.LazyFrame:
        """Raw LazyFrame over the pre-sorted (default) or post-sorted synapse table; ids are dense."""
        if by == "post" and self.has_by_post:
            return pl.scan_parquet(self._post_path)
        return pl.scan_parquet(self._pre_path)

    def load(self) -> pl.DataFrame:
        """The whole synapse table with dense ids (0.1 s, 1.6 GB on FlyWire)."""
        return pl.read_parquet(self._pre_path)

    # ---- queries -----------------------------------------------------------------

    def _finish(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """Collect, drop autapses unless enabled, and replace dense ids with root ids.

        The autapse mask is applied after collection: inside the lazy plan a
        ``pre != post`` predicate stops polars from pruning row groups.
        """
        df = lf.collect()
        if not self.ds.connectivity.autapses:
            df = df.filter(pl.col("pre") != pl.col("post"))
        rid = self.ds.root_ids
        return df.with_columns(
            pl.Series("pre", rid[df["pre"].to_numpy()]),
            pl.Series("post", rid[df["post"].to_numpy()]),
        )

    def of(self, idx: np.ndarray, direction: str = "out", partner=None) -> pl.DataFrame:
        """Synapses where cells ``idx`` are pre (``out``), post (``in``) or either (``both``)."""
        idx = np.asarray(idx, dtype=np.int64)
        p_idx = None if partner is None else self._resolve(partner)
        if direction == "out":
            lf = self.scan("pre").filter(_id_filter("pre", idx))
            if p_idx is not None:
                lf = lf.filter(pl.col("post").is_in(p_idx.tolist()))
        elif direction == "in":
            lf = self.scan("post").filter(_id_filter("post", idx))
            if p_idx is not None:
                lf = lf.filter(pl.col("pre").is_in(p_idx.tolist()))
        elif direction == "both":
            a = self.of(idx, "out", partner)
            b = self.of(idx, "in", partner)
            return pl.concat([a, b]).unique(maintain_order=True)
        else:
            raise ValueError("direction must be 'out', 'in' or 'both'")
        return self._finish(lf)

    def between(self, pre, post) -> pl.DataFrame:
        """Synapses from any cell in ``pre`` to any cell in ``post`` (types, NeuronSets or root ids)."""
        a, b = self._resolve(pre), self._resolve(post)
        lf = self.scan("pre").filter(_id_filter("pre", a)).filter(pl.col("post").is_in(b.tolist()))
        return self._finish(lf)

    def in_box(self, x=None, y=None, z=None) -> pl.DataFrame:
        """Synapses inside an axis-aligned box; each bound is ``(min, max)`` in nm or None."""
        lf = self.scan("pre")
        for col, bounds in zip(XYZ, (x, y, z)):
            if bounds is not None:
                lf = lf.filter(pl.col(col).is_between(int(bounds[0]), int(bounds[1])))
        df = lf.collect()
        rid = self.ds.root_ids
        return df.with_columns(pl.Series("pre", rid[df["pre"].to_numpy()]), pl.Series("post", rid[df["post"].to_numpy()]))

    def _resolve(self, key) -> np.ndarray:
        from connexplorer.connectivity import _as_idx

        return _as_idx(self.ds, key)
