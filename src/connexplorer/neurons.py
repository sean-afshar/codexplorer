"""Neuron and NeuronSet: handles on cells that delegate to the dataset's sub-APIs."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import polars as pl

from connexplorer.dataset import Dataset


class NeuronSet:
    """An ordered set of cells (unique dense ids). Every query method also exists on ``Neuron``."""

    def __init__(self, ds: Dataset, idx: np.ndarray, label: str | None = None):
        self.ds = ds
        self.idx = np.asarray(idx, dtype=np.int64)
        self.label = label

    # ---- identity ----------------------------------------------------------

    @property
    def ids(self) -> np.ndarray:
        """Root ids (int64), in dense-id order."""
        return self.ds.root_ids[self.idx]

    @property
    def cells(self) -> pl.DataFrame:
        return self.ds.cells[self.idx]

    @property
    def types(self) -> list[str]:
        return self.ds._type_series.gather(self.idx).drop_nulls().unique().sort().to_list()

    def __len__(self) -> int:
        return len(self.idx)

    def __iter__(self) -> Iterator["Neuron"]:
        for i in self.idx:
            yield Neuron(self.ds, int(i))

    def __contains__(self, item) -> bool:
        if isinstance(item, Neuron):
            return bool(np.isin(item.idx, self.idx).all())
        return bool(np.isin(self.ds.idx_of(int(item)), self.idx).all())

    def __repr__(self) -> str:
        what = f" {self.label}" if self.label else ""
        return f"NeuronSet({self.ds.name}{what}: {len(self):,} cells)"

    def _same(self, other: "NeuronSet") -> None:
        if other.ds is not self.ds:
            raise ValueError("NeuronSets come from different datasets")

    def __and__(self, other: "NeuronSet") -> "NeuronSet":
        self._same(other)
        return NeuronSet(self.ds, np.intersect1d(self.idx, other.idx), f"({self.label} & {other.label})")

    def __or__(self, other: "NeuronSet") -> "NeuronSet":
        self._same(other)
        return NeuronSet(self.ds, np.union1d(self.idx, other.idx), f"({self.label} | {other.label})")

    def __sub__(self, other: "NeuronSet") -> "NeuronSet":
        self._same(other)
        return NeuronSet(self.ds, np.setdiff1d(self.idx, other.idx), f"({self.label} - {other.label})")

    def __eq__(self, other) -> bool:
        return isinstance(other, NeuronSet) and other.ds is self.ds and np.array_equal(self.idx, other.idx)

    def __hash__(self) -> int:
        return hash((id(self.ds), self.idx.tobytes()))

    # ---- connectivity ------------------------------------------------------

    def outputs(self, by=None, min_syn: int | None = None) -> pl.DataFrame:
        """Postsynaptic partners of the set (summed over members), strongest first."""
        return self.ds.connectivity.partners_of(self.idx, "out", by=by, min_syn=min_syn)

    def inputs(self, by=None, min_syn: int | None = None) -> pl.DataFrame:
        """Presynaptic partners of the set (summed over members), strongest first."""
        return self.ds.connectivity.partners_of(self.idx, "in", by=by, min_syn=min_syn)

    def partners(self, min_syn: int | None = None) -> pl.DataFrame:
        """Both directions in one table: partner, type, side, n_out, n_in, n_syn."""
        return self.ds.connectivity.partners_both(self.idx, min_syn=min_syn)

    def subgraph(self):
        """Edges within the set as a ConnBlock (rows and columns are the set)."""
        return self.ds.connectivity[self, self]

    # ---- synapses ------------------------------------------------------------

    def synapses(self, direction: str = "out", partner=None) -> pl.DataFrame:
        """Synapse locations where the set is pre (``"out"``), post (``"in"``) or either (``"both"``)."""
        return self.ds.synapses.of(self.idx, direction=direction, partner=partner)


class Neuron(NeuronSet):
    """A NeuronSet of size one with scalar conveniences."""

    def __init__(self, ds: Dataset, idx: int):
        super().__init__(ds, np.asarray([idx], dtype=np.int64))

    @property
    def id(self) -> int:
        """Dense id (row position in ``ds.cells``)."""
        return int(self.idx[0])

    @property
    def root_id(self) -> int:
        return int(self.ds.root_ids[self.id])

    def __int__(self) -> int:
        return self.root_id

    @property
    def cell(self) -> dict:
        return self.ds.cells.row(self.id, named=True)

    def _col(self, name: str):
        return self.ds.cells[name][self.id] if name in self.ds.cells.columns else None

    @property
    def type(self) -> str | None:
        return self._col("type")

    @property
    def side(self) -> str | None:
        return self._col("side")

    @property
    def nt(self) -> str | None:
        return self._col("nt")

    def __repr__(self) -> str:
        return f"Neuron({self.ds.name} {self.root_id}, type={self.type}, side={self.side}, nt={self.nt})"

    def __eq__(self, other) -> bool:
        return isinstance(other, NeuronSet) and other.ds is self.ds and np.array_equal(self.idx, other.idx)

    __hash__ = NeuronSet.__hash__
