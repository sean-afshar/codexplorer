"""Connectivity: CSR/CSC over the edge table, partner tables, cell blocks and the type matrix.

All hot paths run on scipy/numpy; polars frames are assembled only for the
rows returned. ``autapses`` (default False) masks self-connections in every
result, including the type matrix diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import polars as pl
import scipy.sparse as sp

from connexplorer import schema
from connexplorer.dataset import Dataset

_GROUP_COLS = ("type", "side", "neuropil")


@dataclass(frozen=True)
class Totals:
    """Per-cell (or per-type) whole-dataset totals used as normalization denominators."""

    out_syn: np.ndarray
    in_syn: np.ndarray
    out_deg: np.ndarray
    in_deg: np.ndarray


def _safe_frac(num, den) -> pl.Expr:
    return pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None)


def _as_idx(ds: Dataset, key) -> np.ndarray:
    """Resolve a type name, list of types, NeuronSet, root id or root-id array to dense ids."""
    from connexplorer.neurons import NeuronSet

    if isinstance(key, NeuronSet):
        return key.idx
    if isinstance(key, str):
        return ds.type_idx(key)
    if isinstance(key, (int, np.integer)):
        return ds.idx_of(int(key))
    if isinstance(key, pl.Series):
        key = key.to_list()
    key = list(key) if not isinstance(key, np.ndarray) else key
    if len(key) and all(isinstance(k, str) for k in key):
        return np.unique(np.concatenate([ds.type_idx(t) for t in key]))
    return np.unique(ds.idx_of(np.asarray(key, dtype=np.int64)))


def _is_range(idx: np.ndarray) -> bool:
    return len(idx) > 0 and idx[-1] - idx[0] + 1 == len(idx) and bool((np.diff(idx) == 1).all())


def _rows(m: sp.csr_matrix, idx: np.ndarray) -> sp.csr_matrix:
    return m[idx[0] : idx[-1] + 1] if _is_range(idx) else m[idx]


class Connectivity:
    def __init__(self, ds: Dataset):
        self.ds = ds
        self._autapses = False

    # ---- autapse flag ---------------------------------------------------------

    @property
    def autapses(self) -> bool:
        """Include self-connections in results. Default False."""
        return self._autapses

    @autapses.setter
    def autapses(self, value: bool) -> None:
        self._autapses = bool(value)

    # ---- matrices -------------------------------------------------------------

    @cached_property
    def _csr_full(self) -> sp.csr_matrix:
        e = self.ds.edges
        n = self.ds.n_cells
        pre = e["pre"].to_numpy()
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
        m = sp.csr_matrix((e["n_syn"].to_numpy().astype(np.uint32), e["post"].to_numpy().astype(np.int32), indptr), shape=(n, n))
        m.has_sorted_indices = True
        return m

    @cached_property
    def _csr_noauta(self) -> sp.csr_matrix:
        m = self._csr_full.copy()
        m.setdiag(0)
        m.eliminate_zeros()
        return m

    @property
    def sparse(self) -> sp.csr_matrix:
        """(N x N) CSR, rows = pre, cols = post, in ``ds.cells`` order; respects ``autapses``."""
        return self._csr_full if self._autapses else self._csr_noauta

    @cached_property
    def _csc_full(self) -> sp.csc_matrix:
        return self._csr_full.tocsc()

    @cached_property
    def _csc_noauta(self) -> sp.csc_matrix:
        return self._csr_noauta.tocsc()

    @property
    def sparse_t(self) -> sp.csc_matrix:
        """CSC view of :attr:`sparse` for input-side slicing."""
        return self._csc_full if self._autapses else self._csc_noauta

    # ---- normalization denominators ----------------------------------------------

    @cached_property
    def _totals_cache(self) -> dict:
        return {}

    @property
    def cell_totals(self) -> Totals:
        """Whole-dataset output/input synapses and partner counts per cell (respects ``autapses``)."""
        key = ("cell", self._autapses)
        if key not in self._totals_cache:
            m, t = self.sparse, self.sparse_t
            self._totals_cache[key] = Totals(
                out_syn=np.asarray(m.sum(axis=1)).ravel().astype(np.uint64),
                in_syn=np.asarray(t.sum(axis=0)).ravel().astype(np.uint64),
                out_deg=np.diff(m.indptr).astype(np.uint32),
                in_deg=np.diff(t.indptr).astype(np.uint32),
            )
        return self._totals_cache[key]

    @cached_property
    def _cell_type_idx(self) -> np.ndarray:
        """type_idx per cell, -1 for untyped."""
        return (
            self.ds.cells.select("type")
            .join(self.ds.types.select("type", "type_idx"), on="type", how="left")["type_idx"]
            .fill_null(-1)
            .cast(pl.Int64)
            .to_numpy()
        )

    @property
    def type_totals(self) -> Totals:
        """Per-type sums of :attr:`cell_totals` (partners to untyped cells included), in ``ds.types`` order."""
        key = ("type", self._autapses)
        if key not in self._totals_cache:
            c = self.cell_totals
            t_idx = self._cell_type_idx
            ok = t_idx >= 0
            n = self.ds.n_types

            def by_type(v):
                return np.bincount(t_idx[ok], weights=v[ok].astype(np.float64), minlength=n).astype(np.uint64)

            self._totals_cache[key] = Totals(by_type(c.out_syn), by_type(c.in_syn), by_type(c.out_deg), by_type(c.in_deg))
        return self._totals_cache[key]

    @cached_property
    def _autapse_syn(self) -> np.ndarray:
        """Autapse synapses per cell (length N)."""
        e = self.ds.edges.filter(pl.col("pre") == pl.col("post"))
        out = np.zeros(self.ds.n_cells, dtype=np.uint64)
        out[e["pre"].to_numpy()] = e["n_syn"].to_numpy()
        return out

    # ---- partner tables --------------------------------------------------------

    def _partner_weights(self, idx: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray]:
        """Partner dense ids and summed synapse counts for the set, sorted by count desc."""
        if direction == "out":
            m = self.sparse
            if len(idx) == 1:
                i = int(idx[0])
                cols, w = m.indices[m.indptr[i] : m.indptr[i + 1]], m.data[m.indptr[i] : m.indptr[i + 1]]
            else:
                sub = _rows(m, idx).tocoo()
                w_all = np.bincount(sub.col, weights=sub.data, minlength=self.ds.n_cells)
                cols = np.nonzero(w_all)[0]
                w = w_all[cols]
        elif direction == "in":
            t = self.sparse_t
            if len(idx) == 1:
                i = int(idx[0])
                cols, w = t.indices[t.indptr[i] : t.indptr[i + 1]], t.data[t.indptr[i] : t.indptr[i + 1]]
            else:
                sub = (t[:, idx[0] : idx[-1] + 1] if _is_range(idx) else t[:, idx]).tocoo()
                w_all = np.bincount(sub.row, weights=sub.data, minlength=self.ds.n_cells)
                cols = np.nonzero(w_all)[0]
                w = w_all[cols]
        else:
            raise ValueError("direction must be 'in' or 'out'")
        w = np.asarray(w, dtype=np.uint64)
        order = np.argsort(-w, kind="stable")
        return cols[order].astype(np.int64), w[order]

    def _partner_frame(self, partners: np.ndarray, w: np.ndarray, col: str) -> pl.DataFrame:
        cells = self.ds.cells
        data = {col: self.ds.root_ids[partners], "type": cells["type"].gather(partners)}
        if "side" in cells.columns:
            data["side"] = cells["side"].gather(partners)
        data["n_syn"] = pl.Series(w, dtype=pl.UInt64)
        data["_idx"] = partners
        return pl.DataFrame(data)

    def _set_total(self, idx: np.ndarray, direction: str) -> int:
        c = self.cell_totals
        return int((c.out_syn if direction == "out" else c.in_syn)[idx].sum())

    def _normalize(self, frame: pl.DataFrame, idx: np.ndarray, direction: str, level: str) -> pl.DataFrame:
        """Add the set-side fraction, the partner-side fraction and their geometric mean.

        ``frac_input``/``frac_output``: share of the set's total input/output.
        ``frac_partner_output``/``frac_partner_input``: share of the partner's
        (cell or type) whole-dataset output/input that lands on the set.
        ``weight_norm``: sqrt of the product (shayan's ``normalized_weight`` / 100).
        """
        set_total = self._set_total(idx, direction)
        own = "frac_input" if direction == "in" else "frac_output"
        other = "frac_partner_output" if direction == "in" else "frac_partner_input"
        if level == "cell":
            tot = self.cell_totals
            den = (tot.out_syn if direction == "in" else tot.in_syn)[frame["_idx"].to_numpy()]
        else:
            tot = self.type_totals
            t_idx = frame.select("type").join(self.ds.types.select("type", "type_idx"), on="type", how="left")["type_idx"].fill_null(-1).cast(pl.Int64).to_numpy()
            den = np.where(t_idx >= 0, (tot.out_syn if direction == "in" else tot.in_syn)[np.maximum(t_idx, 0)], 0)
        out = frame.with_columns(pl.Series("_den", den, dtype=pl.UInt64)).with_columns(
            (pl.col("n_syn") / max(set_total, 1)).alias(own),
            _safe_frac("n_syn", "_den").alias(other),
        )
        return out.with_columns((pl.col(own) * pl.col(other)).sqrt().alias("weight_norm")).drop("_den")

    @cached_property
    def _ebn_indptr(self) -> np.ndarray:
        pre = self.ds.edges_by_neuropil["pre"].to_numpy()
        indptr = np.zeros(self.ds.n_cells + 1, dtype=np.int64)
        np.cumsum(np.bincount(pre, minlength=self.ds.n_cells), out=indptr[1:])
        return indptr

    @cached_property
    def _ebn_by_post(self) -> tuple[pl.DataFrame, np.ndarray]:
        df = self.ds.edges_by_neuropil.sort("post", "pre")
        post = df["post"].to_numpy()
        indptr = np.zeros(self.ds.n_cells + 1, dtype=np.int64)
        np.cumsum(np.bincount(post, minlength=self.ds.n_cells), out=indptr[1:])
        return df, indptr

    def _neuropil_rows(self, idx: np.ndarray, direction: str) -> pl.DataFrame:
        if self.ds.edges_by_neuropil is None:
            raise ValueError(f"{self.ds.name} has no edges_by_neuropil table")
        if direction == "out":
            df, indptr = self.ds.edges_by_neuropil, self._ebn_indptr
        else:
            df, indptr = self._ebn_by_post
        rows = np.concatenate([np.arange(indptr[i], indptr[i + 1]) for i in idx]) if len(idx) > 1 else np.arange(indptr[idx[0]], indptr[idx[0] + 1])
        sub = df[rows]
        if not self._autapses:
            sub = sub.filter(pl.col("pre") != pl.col("post"))
        return sub

    def partners_of(self, idx: np.ndarray, direction: str, by=None, min_syn: int | None = None, normalize: bool = False) -> pl.DataFrame:
        """Partner table for a set of dense ids.

        ``by`` groups the partner-level table by any of ``type``, ``side``,
        ``neuropil`` (str or tuple). ``min_syn`` thresholds the per-partner
        total before grouping, never a neuropil fragment. ``normalize`` adds
        the set-side fraction, the partner-side fraction over the whole
        dataset, and their geometric mean (``weight_norm``); grouped tables
        always carry the set-side fraction.
        """
        idx = np.asarray(idx, dtype=np.int64)
        partner_col = "post" if direction == "out" else "pre"
        partners, w = self._partner_weights(idx, direction)
        if min_syn:
            keep = w >= min_syn
            partners, w = partners[keep], w[keep]
        frame = self._partner_frame(partners, w, partner_col)
        group = () if by is None else ((by,) if isinstance(by, str) else tuple(by))
        bad = [g for g in group if g not in _GROUP_COLS]
        if bad:
            raise ValueError(f"cannot group by {bad}; choose from {_GROUP_COLS}")
        if "neuropil" in group:
            rows = self._neuropil_rows(idx, direction)
            if min_syn:
                rows = rows.filter(pl.col(partner_col).is_in(partners))
            frame = self._neuropil_frame(rows, partner_col)
        if not group:
            if normalize:
                frame = self._normalize(frame, idx, direction, "cell")
            return frame.drop("_idx", strict=False)
        set_total = self._set_total(idx, direction)
        frac = "frac_output" if direction == "out" else "frac_input"
        out = (
            frame.group_by(list(group))
            .agg(pl.col("n_syn").sum(), pl.col(partner_col).n_unique().alias("n_partners"))
            .with_columns((pl.col("n_syn") / max(set_total, 1)).alias(frac))
            .sort("n_syn", descending=True)
        )
        if normalize:
            if group != ("type",):
                raise ValueError("normalize=True with grouping is only defined for by='type'")
            out = self._normalize(out.drop(frac), idx, direction, "type")
            out = out.select("type", "n_syn", "n_partners", frac, out.columns[-2], "weight_norm")
        return out

    def _neuropil_frame(self, rows: pl.DataFrame, partner_col: str) -> pl.DataFrame:
        """(partner, neuropil) rows -> partner root id, type, side, neuropil, n_syn."""
        agg = rows.group_by(partner_col, "neuropil").agg(pl.col("n_syn").cast(pl.UInt64).sum())
        p = agg[partner_col].cast(pl.Int64).to_numpy()
        cells = self.ds.cells
        data = {partner_col: self.ds.root_ids[p], "type": cells["type"].gather(p)}
        if "side" in cells.columns:
            data["side"] = cells["side"].gather(p)
        data["neuropil"] = agg["neuropil"]
        data["n_syn"] = agg["n_syn"]
        return pl.DataFrame(data).sort("n_syn", descending=True)

    def partners_both(self, idx: np.ndarray, min_syn: int | None = None) -> pl.DataFrame:
        po, wo = self._partner_weights(idx, "out")
        pi, wi = self._partner_weights(idx, "in")
        out = pl.DataFrame({"idx": po, "n_out": pl.Series(wo, dtype=pl.UInt64)})
        inp = pl.DataFrame({"idx": pi, "n_in": pl.Series(wi, dtype=pl.UInt64)})
        both = out.join(inp, on="idx", how="full", coalesce=True).fill_null(0).with_columns((pl.col("n_out") + pl.col("n_in")).alias("n_syn"))
        if min_syn:
            both = both.filter((pl.col("n_out") >= min_syn) | (pl.col("n_in") >= min_syn))
        p = both["idx"].to_numpy()
        cells = self.ds.cells
        data = {"partner": self.ds.root_ids[p], "type": cells["type"].gather(p)}
        if "side" in cells.columns:
            data["side"] = cells["side"].gather(p)
        return pl.DataFrame(data).hstack(both.select("n_out", "n_in", "n_syn")).sort("n_syn", descending=True)

    # ---- blocks -----------------------------------------------------------------

    def __getitem__(self, key) -> "ConnBlock":
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("index as connectivity[pre, post] with types, NeuronSets or root ids")
        return ConnBlock(self, _as_idx(self.ds, key[0]), _as_idx(self.ds, key[1]))

    @cached_property
    def types(self) -> "TypeMatrix":
        return TypeMatrix(self)


class ConnBlock:
    """Synapse counts from a set of pre cells (rows) to a set of post cells (columns)."""

    def __init__(self, conn: Connectivity, row_idx: np.ndarray, col_idx: np.ndarray):
        self.conn = conn
        self.ds = conn.ds
        self.row_idx = np.asarray(row_idx, dtype=np.int64)
        self.col_idx = np.asarray(col_idx, dtype=np.int64)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.row_idx), len(self.col_idx)

    @cached_property
    def sparse(self) -> sp.csr_matrix:
        m = _rows(self.conn.sparse, self.row_idx)
        c = self.col_idx
        return (m[:, c[0] : c[-1] + 1] if _is_range(c) else m[:, c]).tocsr()

    @property
    def values(self) -> np.ndarray:
        """Dense (n_pre, n_post) uint32 array, zero rows/cols included."""
        return self.sparse.toarray()

    @property
    def row_ids(self) -> np.ndarray:
        return self.ds.root_ids[self.row_idx]

    @property
    def col_ids(self) -> np.ndarray:
        return self.ds.root_ids[self.col_idx]

    @property
    def row_types(self) -> pl.Series:
        return self.ds._type_series.gather(self.row_idx)

    @property
    def col_types(self) -> pl.Series:
        return self.ds._type_series.gather(self.col_idx)

    def sum(self) -> int:
        return int(self.sparse.sum())

    @property
    def long(self) -> pl.DataFrame:
        """Nonzero entries: pre, post (root ids), n_syn."""
        coo = self.sparse.tocoo()
        return pl.DataFrame(
            {"pre": self.row_ids[coo.row], "post": self.col_ids[coo.col], "n_syn": pl.Series(coo.data, dtype=pl.UInt32)}
        ).sort("pre", "post")

    @property
    def normalized(self) -> pl.DataFrame:
        """:attr:`long` plus ``frac_output`` (of the pre cell's total output), ``frac_input`` (of the post cell's total input) and ``weight_norm``."""
        coo = self.sparse.tocoo()
        tot = self.conn.cell_totals
        r, c = self.row_idx[coo.row], self.col_idx[coo.col]
        w = coo.data.astype(np.float64)
        fo = np.divide(w, tot.out_syn[r], out=np.full(len(w), np.nan), where=tot.out_syn[r] > 0)
        fi = np.divide(w, tot.in_syn[c], out=np.full(len(w), np.nan), where=tot.in_syn[c] > 0)
        return pl.DataFrame(
            {"pre": self.ds.root_ids[r], "post": self.ds.root_ids[c], "n_syn": pl.Series(coo.data, dtype=pl.UInt32), "frac_output": fo, "frac_input": fi, "weight_norm": np.sqrt(fo * fi)}
        ).sort("pre", "post")

    @property
    def frame(self) -> pl.DataFrame:
        """Wide form: column ``pre`` (root ids) then one UInt32 column per post root id."""
        dense = self.values
        cols = {"pre": self.row_ids}
        for j, rid in enumerate(self.col_ids):
            cols[str(int(rid))] = dense[:, j]
        return pl.DataFrame(cols)

    def __repr__(self) -> str:
        return f"ConnBlock({self.shape[0]} pre x {self.shape[1]} post, {self.sum():,} synapses)"


class TypeMatrix:
    """Type-to-type synapse totals from the precomputed npz; axes are ``ds.types`` order."""

    def __init__(self, conn: Connectivity):
        self.conn = conn
        self.ds = conn.ds

    @cached_property
    def _loaded(self) -> tuple[sp.csr_matrix, list[str]]:
        m, names = schema.load_type_matrix(self.ds.tables_dir / schema.TYPE_MATRIX_FILE)
        names = [str(n) for n in names]
        if names != self.ds.type_names():
            raise RuntimeError("type_matrix.npz axes do not match types.parquet")
        return m, names

    @property
    def names(self) -> list[str]:
        return self._loaded[1]

    @cached_property
    def _index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.names)}

    @cached_property
    def _noauta(self) -> sp.csr_matrix:
        m = self._loaded[0].copy()
        cells = self.ds.cells.select("type")
        t_idx = cells.join(self.ds.types.select("type", "type_idx"), on="type", how="left")["type_idx"].fill_null(-1).cast(pl.Int64).to_numpy()
        per_cell = self.conn._autapse_syn
        ok = t_idx >= 0
        diag = np.bincount(t_idx[ok], weights=per_cell[ok].astype(np.float64), minlength=len(self.names)).astype(np.uint64)
        m = (m - sp.diags(diag, format="csr", dtype=np.uint64)).tocsr()
        m.eliminate_zeros()
        return m.astype(np.uint64)

    @property
    def sparse(self) -> sp.csr_matrix:
        return self._loaded[0] if self.conn.autapses else self._noauta

    def _ti(self, key) -> np.ndarray:
        keys = [key] if isinstance(key, str) else list(key)
        try:
            return np.asarray([self._index[k] for k in keys], dtype=np.int64)
        except KeyError as e:
            raise KeyError(f"unknown type {e.args[0]!r}") from None

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("index as types[pre_type, post_type]")
        a, b = key
        if isinstance(a, str) and isinstance(b, str):
            return int(self.sparse[self._ti(a)[0], self._ti(b)[0]])
        return TypeBlock(self, self._ti(a), self._ti(b))

    @property
    def long(self) -> pl.DataFrame:
        coo = self.sparse.tocoo()
        names = pl.Series(self.names)
        return pl.DataFrame({"pre_type": names.gather(coo.row), "post_type": names.gather(coo.col), "n_syn": pl.Series(coo.data, dtype=pl.UInt64)}).sort("pre_type", "post_type")

    @property
    def frame(self) -> pl.DataFrame:
        """Full wide matrix (n_types x n_types, UInt32) with a leading ``pre_type`` column."""
        return TypeBlock(self, np.arange(len(self.names)), np.arange(len(self.names))).frame

    @property
    def normalized(self) -> pl.DataFrame:
        """:attr:`long` plus ``frac_output`` (of the pre type's total output), ``frac_input`` (of the post type's total input) and ``weight_norm``."""
        return TypeBlock(self, np.arange(len(self.names)), np.arange(len(self.names))).normalized

    def __repr__(self) -> str:
        return f"TypeMatrix({len(self.names)} types, {int(self.sparse.sum()):,} synapses)"


class TypeBlock:
    def __init__(self, tm: TypeMatrix, rows: np.ndarray, cols: np.ndarray):
        self.tm = tm
        self.rows = rows
        self.cols = cols

    @property
    def row_types(self) -> list[str]:
        return [self.tm.names[i] for i in self.rows]

    @property
    def col_types(self) -> list[str]:
        return [self.tm.names[i] for i in self.cols]

    @cached_property
    def sparse(self) -> sp.csr_matrix:
        return self.tm.sparse[self.rows][:, self.cols].tocsr()

    @property
    def values(self) -> np.ndarray:
        return self.sparse.toarray()

    def sum(self) -> int:
        return int(self.sparse.sum())

    @property
    def long(self) -> pl.DataFrame:
        coo = self.sparse.tocoo()
        r, c = pl.Series(self.row_types), pl.Series(self.col_types)
        return pl.DataFrame({"pre_type": r.gather(coo.row), "post_type": c.gather(coo.col), "n_syn": pl.Series(coo.data, dtype=pl.UInt64)}).sort("pre_type", "post_type")

    @property
    def normalized(self) -> pl.DataFrame:
        coo = self.sparse.tocoo()
        tot = self.tm.conn.type_totals
        r, c = self.rows[coo.row], self.cols[coo.col]
        w = coo.data.astype(np.float64)
        fo = np.divide(w, tot.out_syn[r], out=np.full(len(w), np.nan), where=tot.out_syn[r] > 0)
        fi = np.divide(w, tot.in_syn[c], out=np.full(len(w), np.nan), where=tot.in_syn[c] > 0)
        names = pl.Series(self.tm.names)
        return pl.DataFrame(
            {"pre_type": names.gather(r), "post_type": names.gather(c), "n_syn": pl.Series(coo.data, dtype=pl.UInt64), "frac_output": fo, "frac_input": fi, "weight_norm": np.sqrt(fo * fi)}
        ).sort("pre_type", "post_type")

    @property
    def frame(self) -> pl.DataFrame:
        dense = self.values.astype(np.uint32)
        data = {"pre_type": self.row_types}
        for j, t in enumerate(self.col_types):
            data[t] = dense[:, j]
        return pl.DataFrame(data)

    def __repr__(self) -> str:
        return f"TypeBlock({len(self.rows)} pre types x {len(self.cols)} post types, {self.sum():,} synapses)"
