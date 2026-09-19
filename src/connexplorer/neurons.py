"""Neuron and NeuronSet: handles on cells that delegate to the dataset's sub-APIs."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import polars as pl

from connexplorer import stats
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

    def __getitem__(self, i):
        """``grp[0]`` is a Neuron; a slice is a NeuronSet."""
        if isinstance(i, slice):
            return NeuronSet(self.ds, self.idx[i], f"{self.label}[{i.start or ''}:{i.stop or ''}]")
        return Neuron(self.ds, int(self.idx[i]))

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

    def outputs(self, by=None, min_syn: int | None = None, normalize: bool = False) -> pl.DataFrame:
        """Postsynaptic partners of the set (summed over members), strongest first.

        ``normalize=True`` adds ``frac_output`` (share of the set's output),
        ``frac_partner_input`` (share of the partner's total input) and ``weight_norm``.
        """
        return self.ds.connectivity.partners_of(self.idx, "out", by=by, min_syn=min_syn, normalize=normalize)

    def inputs(self, by=None, min_syn: int | None = None, normalize: bool = False) -> pl.DataFrame:
        """Presynaptic partners of the set (summed over members), strongest first.

        ``normalize=True`` adds ``frac_input`` (share of the set's input),
        ``frac_partner_output`` (share of the partner's total output) and ``weight_norm``.
        """
        return self.ds.connectivity.partners_of(self.idx, "in", by=by, min_syn=min_syn, normalize=normalize)

    def partners(self, min_syn: int | None = None) -> pl.DataFrame:
        """Both directions in one table: partner, type, side, n_out, n_in, n_syn."""
        return self.ds.connectivity.partners_both(self.idx, min_syn=min_syn)

    def subgraph(self):
        """Edges within the set as a ConnBlock (rows and columns are the set)."""
        return self.ds.connectivity[self, self]

    # ---- graph statistics -----------------------------------------------------

    def degree(self, within: bool = False) -> pl.DataFrame:
        """Per-cell partner counts and synapse totals; ``within=True`` counts only partners in the set."""
        return stats.degree(self.ds, self.idx, within)

    def hubs(self, k: int = 10, by: str = "total_degree", within: bool = False) -> pl.DataFrame:
        return stats.hubs(self.ds, self.idx, k, by, within)

    def degree_distribution(self, direction: str = "in", within: bool = False) -> pl.DataFrame:
        return stats.degree_distribution(self.ds, self.idx, direction, within)

    def reciprocal(self, within: bool = True) -> pl.DataFrame:
        """Pairs connected in both directions (inside the set, or with anyone when ``within=False``)."""
        return stats.reciprocal(self.ds, self.idx, within)

    def summary(self) -> dict:
        return stats.summary(self.ds, self.idx)

    # ---- synapses ------------------------------------------------------------

    def synapses(self, direction: str = "out", partner=None) -> pl.DataFrame:
        """Synapse locations where the set is pre (``"out"``), post (``"in"``) or either (``"both"``)."""
        return self.ds.synapses.of(self.idx, direction=direction, partner=partner)

    # ---- morphology and viewing ----------------------------------------------

    def skeletons(self, max_n: int | None = None, units: str = "um"):
        """navis NeuronList of the members' skeletons (micrometres by default)."""
        return self.ds.skeletons.load_many(self.ids, units=units, max_n=max_n)

    def view(self, viewer: str = "spelunker", partners: str | None = None, top: int = 5, synapses: bool = False, min_syn: int | None = None) -> str:
        """Neuroglancer (or Codex) URL for the set; never opens a browser. See ``viz.neuron_view``."""
        from connexplorer.viz.neuroglancer import neuron_view

        return neuron_view(self.ds, self.idx, viewer=viewer, partners=partners, top=top, synapses=synapses, min_syn=min_syn)


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

    def skeleton(self, units: str = "um"):
        """navis TreeNeuron in micrometres (or ``units="nm"``)."""
        return self.ds.skeletons.load(self.root_id, units=units)

    def __repr__(self) -> str:
        return f"Neuron({self.ds.name} {self.root_id}, type={self.type}, side={self.side}, nt={self.nt})"

    def __eq__(self, other) -> bool:
        return isinstance(other, NeuronSet) and other.ds is self.ds and np.array_equal(self.idx, other.idx)

    __hash__ = NeuronSet.__hash__
