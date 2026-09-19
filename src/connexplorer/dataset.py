"""Dataset: manifest, lazily cached tables, id lookups and neuron selection."""

from __future__ import annotations

import json
import os
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl

from connexplorer import schema

ALIASES = {"fafb": "flywire", "flywire_fafb": "flywire", "fw": "flywire", "male_cns": "mcns", "malecns": "mcns", "mcns": "mcns", "flywire": "flywire"}


class Config:
    """Process-wide settings. ``data_dir`` is searched first when opening by name."""

    data_dir: Path | None = None


config = Config()


def _version_key(v: str) -> tuple:
    parts = v.lstrip("vV").split(".")
    return tuple(int(p) if p.isdigit() else p for p in parts)


def _search_roots() -> list[Path]:
    roots = [config.data_dir, os.environ.get("CONNEXPLORER_DATA"), Path("data")]
    return [Path(r) for r in roots if r and Path(r).is_dir()]


def resolve(name: str, version: str | None = None) -> Path:
    """Find ``<root>/<dir>/tables/manifest.json`` whose dataset is ``name`` (latest version unless given)."""
    key = ALIASES.get(name.lower(), name.lower())
    found: list[tuple[tuple, Path]] = []
    roots = _search_roots()
    for root in roots:
        for d in sorted(root.iterdir()):
            mf = d / "tables" / schema.MANIFEST_FILE
            if not mf.is_file():
                continue
            try:
                raw = json.loads(mf.read_text())
            except (OSError, ValueError):
                continue
            if raw.get("dataset") == key and (version is None or str(raw.get("version")) == str(version)):
                found.append((_version_key(str(raw.get("version"))), d))
    if not found:
        raise FileNotFoundError(
            f"no built dataset {name!r}" + (f" version {version}" if version else "") + f" under {[str(r) for r in roots]}; "
            "build one with `python -m connexplorer.ingest build ...` or set CONNEXPLORER_DATA"
        )
    return max(found)[1]


def open(name_or_path: str | Path, version: str | None = None) -> "Dataset":
    """Open a built dataset by name (``"flywire"``, ``"mcns"``) or by directory."""
    p = Path(name_or_path)
    if (p / "tables" / schema.MANIFEST_FILE).is_file():
        return Dataset(p / "tables")
    if (p / schema.MANIFEST_FILE).is_file():
        return Dataset(p)
    return Dataset(resolve(str(name_or_path), version) / "tables")


class Dataset:
    """One built connectome. Opening reads the manifest only; tables load on first use."""

    def __init__(self, tables_dir: str | Path):
        self.tables_dir = Path(tables_dir)
        self.manifest = schema.Manifest.read(self.tables_dir)
        self.name: str = self.manifest.dataset
        self.version: str = str(self.manifest.version)

    def __repr__(self) -> str:
        return f"Dataset({self.name} {self.version}: {self.n_cells:,} cells, {self.n_synapses:,} synapses)"

    # ---- manifest-level facts -------------------------------------------

    @property
    def n_cells(self) -> int:
        return self.manifest.tables["cells"]["rows"]

    @property
    def n_synapses(self) -> int:
        return self.manifest.tables["synapses"]["rows"]

    @property
    def n_types(self) -> int:
        return self.manifest.tables["types"]["rows"]

    def has_table(self, name: str) -> bool:
        return name in self.manifest.tables

    def info(self) -> str:
        c = self.manifest.invariants.get("counts", {})
        lines = [
            f"{self.name} {self.version}  (schema v{self.manifest.schema_version}, built {self.manifest.created})",
            f"  cells {self.n_cells:,}   types {self.n_types:,}   edges {self.manifest.tables['edges']['rows']:,}   synapses {self.n_synapses:,}",
            f"  autapses {c.get('n_autapses', '?'):,} pairs / {c.get('n_autapse_synapses', '?'):,} synapses   cells without edges {c.get('n_cells_without_edges', '?'):,}",
            f"  tables: {', '.join(sorted(self.manifest.tables))}",
            f"  nt overrides: {self.manifest.nt_overrides_version}   invariant problems: {len(self.manifest.invariants.get('problems', []))}",
            f"  dir: {self.tables_dir}",
        ]
        text = "\n".join(lines)
        print(text)
        return text

    # ---- tables ------------------------------------------------------------

    def _read(self, name: str) -> pl.DataFrame | None:
        return schema.read_table(self.tables_dir, name)

    @cached_property
    def cells(self) -> pl.DataFrame:
        return self._read("cells")

    @cached_property
    def types(self) -> pl.DataFrame:
        return self._read("types")

    @cached_property
    def edges(self) -> pl.DataFrame:
        return self._read("edges")

    @cached_property
    def edges_by_neuropil(self) -> pl.DataFrame | None:
        return self._read("edges_by_neuropil")

    @cached_property
    def columns(self) -> pl.DataFrame | None:
        return self._read("columns")

    # ---- id lookups ---------------------------------------------------------

    @cached_property
    def root_ids(self) -> np.ndarray:
        """Root id of every dense id, in id order (int64)."""
        return self.cells["root_id"].to_numpy()

    @cached_property
    def _rid_index(self) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.root_ids, kind="stable")
        return self.root_ids[order], order

    def idx_of(self, root_ids) -> np.ndarray:
        """Dense ids for root ids (scalar or array). Raises KeyError for unknown ids."""
        rids = np.atleast_1d(np.asarray(root_ids, dtype=np.int64))
        sorted_rids, order = self._rid_index
        pos = np.searchsorted(sorted_rids, rids)
        pos[pos >= len(sorted_rids)] = 0
        bad = sorted_rids[pos] != rids
        if bad.any():
            raise KeyError(f"unknown root id(s): {rids[bad][:5].tolist()}{'...' if bad.sum() > 5 else ''}")
        return order[pos].astype(np.int64)

    @cached_property
    def _type_series(self) -> pl.Series:
        return self.cells["type"]

    @cached_property
    def _type_ranges(self) -> dict[str, tuple[int, int]]:
        """type -> (start, stop) dense-id range; cells are sorted by type so every type is contiguous."""
        g = self.cells.drop_nulls("type").group_by("type").agg(pl.col("id").min().alias("lo"), pl.col("id").max().alias("hi"), pl.len())
        if not g.select(((pl.col("hi") - pl.col("lo") + 1) == pl.col("len")).all()).item():
            raise RuntimeError("cells are not contiguous by type; rebuild the dataset")
        return {t: (int(lo), int(hi) + 1) for t, lo, hi in zip(g["type"], g["lo"], g["hi"])}

    def type_idx(self, type_name: str) -> np.ndarray:
        try:
            lo, hi = self._type_ranges[type_name]
        except KeyError:
            raise KeyError(f"unknown type {type_name!r} in {self.name}") from None
        return np.arange(lo, hi, dtype=np.int64)

    def type_names(self) -> list[str]:
        return self.types["type"].to_list()

    # ---- selection ----------------------------------------------------------

    def __getitem__(self, key):
        from connexplorer.neurons import Neuron, NeuronSet

        if isinstance(key, (int, np.integer)):
            return Neuron(self, int(self.idx_of(int(key))[0]))
        if isinstance(key, str):
            return NeuronSet(self, self.type_idx(key), label=key)
        if isinstance(key, NeuronSet):
            return key
        if isinstance(key, pl.Series):
            key = key.to_list()
        arr = list(key) if not isinstance(key, np.ndarray) else key
        if len(arr) == 0:
            return NeuronSet(self, np.empty(0, dtype=np.int64), label="empty")
        if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.integer) or all(isinstance(k, (int, np.integer)) for k in arr):
            return NeuronSet(self, np.unique(self.idx_of(np.asarray(arr, dtype=np.int64))), label=f"{len(arr)} ids")
        if all(isinstance(k, str) for k in arr):
            idx = np.concatenate([self.type_idx(t) for t in arr])
            return NeuronSet(self, np.unique(idx), label="|".join(arr))
        raise TypeError("index with a root id, a type name, or a list of either")

    def select(self, *exprs: pl.Expr, **equals):
        """NeuronSet of cells matching polars expressions and/or ``column=value`` (value may be a list)."""
        from connexplorer.neurons import NeuronSet

        conds = list(exprs)
        for col, val in equals.items():
            if col not in self.cells.columns:
                raise KeyError(f"cells has no column {col!r}")
            conds.append(pl.col(col).is_in(list(val)) if isinstance(val, (list, tuple, set, np.ndarray)) else pl.col(col) == val)
        sub = (self.cells.filter(conds) if conds else self.cells).select("id")
        label = ", ".join([f"{k}={v}" for k, v in equals.items()] + [str(e) for e in exprs]) or "all"
        return NeuronSet(self, sub["id"].cast(pl.Int64).to_numpy(), label=label)

    # ---- cross-dataset ---------------------------------------------------------

    @property
    def type_map(self) -> pl.DataFrame | None:
        """``type`` -> ``flywire_type`` when the vendor annotations carry one (Male CNS), else None."""
        from connexplorer.cross import type_map

        return type_map(self)

    def types_like(self, name: str) -> list[str]:
        """Types of this dataset matching a FlyWire type name (via the vendor map, or the name itself)."""
        from connexplorer.cross import types_like

        return types_like(self, name)

    # ---- sub-APIs ------------------------------------------------------------

    @cached_property
    def connectivity(self):
        from connexplorer.connectivity import Connectivity

        return Connectivity(self)

    @cached_property
    def synapses(self):
        from connexplorer.synapses import Synapses

        return Synapses(self)
