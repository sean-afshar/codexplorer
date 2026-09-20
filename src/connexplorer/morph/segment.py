"""Compartmentalization of a skeleton for cable modelling.

``segment(sk, "natural")`` makes one compartment per navis segment (the
linear path between branch points). Unlike shayan's version, a compartment's
length includes the edge from its most proximal node to the parent
compartment, so lengths sum to the neuron's cable length and single-node
segments are not zero-length. Compartments shorter than ``min_length_um``
are merged into their parent (a zero-length root is absorbed by its largest
child), so the conductance matrix is never singular.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import scipy.sparse as sp

if TYPE_CHECKING:
    import navis

METHODS = ("natural",)


def _rows_of(sorted_ids: np.ndarray, order: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return order[np.searchsorted(sorted_ids, ids)]


@dataclass
class _Geometry:
    node_id: np.ndarray
    parent_row: np.ndarray  # -1 for root
    xyz: np.ndarray  # (N, 3) um
    radius: np.ndarray  # um
    edge_len: np.ndarray  # distance to parent, 0 for root


def _geometry(sk: "navis.TreeNeuron") -> _Geometry:
    nodes = sk.nodes
    node_id = nodes["node_id"].to_numpy().astype(np.int64)
    parent_id = nodes["parent_id"].to_numpy().astype(np.int64)
    xyz = nodes[["x", "y", "z"]].to_numpy().astype(np.float64)
    radius = nodes["radius"].to_numpy().astype(np.float64)
    order = np.argsort(node_id, kind="stable")
    sorted_ids = node_id[order]
    parent_row = np.full(len(node_id), -1, dtype=np.int64)
    has_parent = parent_id >= 0
    parent_row[has_parent] = _rows_of(sorted_ids, order, parent_id[has_parent])
    edge_len = np.zeros(len(node_id))
    edge_len[has_parent] = np.linalg.norm(xyz[has_parent] - xyz[parent_row[has_parent]], axis=1)
    return _Geometry(node_id, parent_row, xyz, radius, edge_len)


def _natural_assignment(sk: "navis.TreeNeuron", g: _Geometry) -> np.ndarray:
    """comp index per node row, following navis segment order; every node exactly once."""
    order = np.argsort(g.node_id, kind="stable")
    sorted_ids = g.node_id[order]
    comp = np.full(len(g.node_id), -1, dtype=np.int64)
    k = 0
    for seg in sk.segments:
        rows = _rows_of(sorted_ids, order, np.asarray(seg, dtype=np.int64))
        frontier = rows[comp[rows] == -1]
        if len(frontier) == 0:
            continue
        comp[frontier] = k
        k += 1
    for r in np.nonzero(comp == -1)[0]:  # isolated nodes (should not happen)
        comp[r] = k
        k += 1
    return comp


def _compact(comp: np.ndarray, g: _Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Relabel compartments as 0..K-1 with the root first and parents before children.

    Returns (comp per node, parent comp per compartment).
    """
    _, comp = np.unique(comp, return_inverse=True)
    k = comp.max() + 1
    # exit node of each compartment: the node whose parent lies outside (or the root)
    parent_comp = np.full(k, -1, dtype=np.int64)
    exits = np.nonzero((g.parent_row == -1) | (comp[np.maximum(g.parent_row, 0)] != comp))[0]
    for r in exits:
        parent_comp[comp[r]] = -1 if g.parent_row[r] == -1 else comp[g.parent_row[r]]
    # depth via parent chain
    depth = np.zeros(k, dtype=np.int64)
    changed = True
    while changed:
        changed = False
        nd = np.where(parent_comp >= 0, depth[np.maximum(parent_comp, 0)] + 1, 0)
        if not np.array_equal(nd, depth):
            depth, changed = nd, True
    new_order = np.lexsort((np.arange(k), depth))  # by depth, then original index
    relabel = np.empty(k, dtype=np.int64)
    relabel[new_order] = np.arange(k)
    comp = relabel[comp]
    parent_comp = np.where(parent_comp >= 0, relabel[np.maximum(parent_comp, 0)], -1)[new_order]
    return comp, parent_comp


def _properties(comp: np.ndarray, parent_comp: np.ndarray, g: _Geometry) -> pl.DataFrame:
    k = len(parent_comp)
    n = np.bincount(comp, minlength=k)
    length = np.bincount(comp, weights=g.edge_len, minlength=k)
    r_mean = np.bincount(comp, weights=g.radius, minlength=k) / n
    r_var = np.bincount(comp, weights=g.radius**2, minlength=k) / n - r_mean**2
    center = np.stack([np.bincount(comp, weights=g.xyz[:, i], minlength=k) / n for i in range(3)], axis=1)
    depth = np.zeros(k, dtype=np.int64)
    for i in range(k):  # parents precede children after _compact
        if parent_comp[i] >= 0:
            depth[i] = depth[parent_comp[i]] + 1
    return pl.DataFrame(
        {
            "compartment_id": pl.Series(np.arange(k), dtype=pl.UInt32),
            "parent_id": pl.Series(parent_comp, dtype=pl.Int64),
            "depth": pl.Series(depth, dtype=pl.UInt32),
            "n_nodes": pl.Series(n, dtype=pl.UInt32),
            "length": length,
            "radius_mean": r_mean,
            "radius_std": np.sqrt(np.maximum(r_var, 0)),
            "x": center[:, 0],
            "y": center[:, 1],
            "z": center[:, 2],
        }
    )


def _merge_short(comp: np.ndarray, parent_comp: np.ndarray, g: _Geometry, min_length: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Merge compartments shorter than ``min_length`` into their parent; returns (comp, parent_comp, n_merged)."""
    merged = 0
    for _ in range(1000):
        k = len(parent_comp)
        length = np.bincount(comp, weights=g.edge_len, minlength=k)
        n = np.bincount(comp, minlength=k)
        short = np.nonzero(length < min_length)[0]
        if len(short) == 0:
            break
        target = np.arange(k)
        for s in short:
            if parent_comp[s] >= 0:
                target[s] = parent_comp[s]
            else:  # zero/short root: absorb into the largest child, which becomes the root
                children = np.nonzero(parent_comp == s)[0]
                if len(children) == 0:
                    continue
                target[s] = children[np.argmax(n[children])]
        if np.array_equal(target, np.arange(k)):
            break
        for _ in range(k):  # resolve chains
            nt = target[target]
            if np.array_equal(nt, target):
                break
            target = nt
        merged += int((target != np.arange(k)).sum())
        comp, parent_comp = _compact(target[comp], g)
    return comp, parent_comp, merged


class Compartments:
    """A skeleton divided into compartments; ``table`` has one row per compartment.

    Columns: compartment_id, parent_id (-1 = root), depth, n_nodes, length (um),
    radius_mean, radius_std (um), x, y, z (centre, um). Compartment 0 is the
    root and parents always precede children.
    """

    def __init__(self, neuron: "navis.TreeNeuron", comp_of_node: np.ndarray, table: pl.DataFrame, method: str, min_length_um: float, n_merged: int):
        self.neuron = neuron
        self.comp_of_node = comp_of_node
        self.table = table
        self.method = method
        self.min_length_um = min_length_um
        self.n_merged = n_merged

    def __len__(self) -> int:
        return self.table.height

    @property
    def root(self) -> int:
        return 0

    @cached_property
    def parents(self) -> np.ndarray:
        return self.table["parent_id"].to_numpy()

    @cached_property
    def node_ids(self) -> np.ndarray:
        return self.neuron.nodes["node_id"].to_numpy().astype(np.int64)

    def nodes(self, compartment: int) -> np.ndarray:
        """Node ids belonging to one compartment."""
        return self.node_ids[self.comp_of_node == compartment]

    def children(self, compartment: int) -> np.ndarray:
        return np.nonzero(self.parents == compartment)[0]

    @property
    def centers(self) -> np.ndarray:
        return self.table.select("x", "y", "z").to_numpy()

    @property
    def lengths(self) -> np.ndarray:
        return self.table["length"].to_numpy()

    @property
    def diameters(self) -> np.ndarray:
        return 2.0 * self.table["radius_mean"].to_numpy()

    def nearest(self, xyz_um: np.ndarray) -> np.ndarray:
        """Compartment id of the skeleton node nearest to each point (k, 3) in micrometres.

        Use it to place synapses (``cnx.xyz(df) / 1e3``) onto compartments.
        """
        from scipy.spatial import cKDTree

        if not hasattr(self, "_tree"):
            self._tree = cKDTree(self.neuron.nodes[["x", "y", "z"]].to_numpy())
        _, rows = self._tree.query(np.asarray(xyz_um, dtype=np.float64).reshape(-1, 3))
        return self.comp_of_node[rows]

    def hines(self, Ra: float = 400.0) -> sp.csr_matrix:
        """Axial conductance structure (siemens): -g on parent/child pairs, row sums on the diagonal.

        g between compartment i and its parent p = 1e-4 * pi * ((d_i + d_p) / 2)^2 / 4 / (Ra * L_i),
        with d and L in micrometres and Ra in ohm cm.
        """
        k = len(self)
        i = np.nonzero(self.parents >= 0)[0]
        p = self.parents[i]
        d = self.diameters
        g = 1e-4 * np.pi * ((d[i] + d[p]) / 2.0) ** 2 / 4.0 / (Ra * self.lengths[i])
        rows = np.concatenate([i, p, i, p])
        cols = np.concatenate([p, i, i, p])
        vals = np.concatenate([-g, -g, g, g])
        return sp.coo_matrix((vals, (rows, cols)), shape=(k, k)).tocsr()

    def summary(self) -> str:
        t = self.table
        lines = [
            f"Compartments: {len(self)} ({self.method}, min length {self.min_length_um} um, {self.n_merged} merged)",
            f"  neuron {self.neuron.id}: {self.neuron.n_nodes} nodes, cable {float(t['length'].sum()):.1f} um, tree depth {int(t['depth'].max())}",
            f"  length um: min {t['length'].min():.2f}  median {t['length'].median():.2f}  mean {t['length'].mean():.2f}  max {t['length'].max():.2f}",
            f"  radius um: mean {t['radius_mean'].mean():.3f}  within-compartment std mean {t['radius_std'].mean():.3f}  max {t['radius_std'].max():.3f}",
        ]
        text = "\n".join(lines)
        print(text)
        return text

    def view(self, ds=None, root_id: int | None = None, viewer: str = "spelunker") -> str:
        from connexplorer.viz.neuroglancer import compartments_view

        return compartments_view(self, ds, root_id, viewer)

    def plot3d(self, color_by: str = "radius", **kw):
        from connexplorer.viz.plot import plot3d

        return plot3d(self, color_by=color_by, **kw)

    def plot2d(self, color_by: str = "radius", **kw):
        from connexplorer.viz.plot import plot2d

        return plot2d(self, color_by=color_by, **kw)

    def __repr__(self) -> str:
        return f"Compartments({len(self)} of neuron {self.neuron.id}, {self.method}, {float(self.table['length'].sum()):.1f} um)"


def segment(sk: "navis.TreeNeuron", method: str = "natural", min_length_um: float = 0.5) -> Compartments:
    """Divide a skeleton (micrometres) into compartments. See the module docstring."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    units = str(getattr(getattr(sk, "units", None), "units", "micrometer"))
    if units not in ("micrometer", "micron", "um", "dimensionless"):
        sk = sk.convert_units("micrometer")
    g = _geometry(sk)
    comp, parent_comp = _compact(_natural_assignment(sk, g), g)
    n_merged = 0
    if min_length_um and min_length_um > 0:
        comp, parent_comp, n_merged = _merge_short(comp, parent_comp, g, float(min_length_um))
    table = _properties(comp, parent_comp, g)
    return Compartments(sk, comp, table, method, float(min_length_um or 0), n_merged)
