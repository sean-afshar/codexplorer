"""SWC skeleton store: one loader for every dataset, always returning micrometres by default.

Skeletons are looked up, in order, in ``config.skeletons[dataset]``, the
``CONNEXPLORER_SKELETONS_<DATASET>`` variable, ``<tables>/skeletons/`` (a
directory of ``<root_id>.swc`` files or a single ``.zip`` of them) and
``<dataset dir>/skeletons.zip``. Files are read straight out of the zip and
parsed with navis (about 10 ms per neuron); navis' own zip reader with its
progress bar is not used.

Datasets listed in :data:`REMOTE` (the Male CNS) can fetch one skeleton at a
time from the vendor's public Neuroglancer precomputed store; the result is
saved as ``<root_id>.swc`` under ``<tables>/skeletons/`` so it is only
downloaded once. Precomputed skeletons carry no radius, so ``radius_um``
(default 0.25) is assigned to every node and reported on the neuron.
"""

from __future__ import annotations

import io
import json
import os
import re
import struct
import warnings
import zipfile
from urllib.request import urlopen
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    import navis

    from connexplorer.dataset import Dataset

_META = re.compile(r"#\s*Meta:\s*(\{.*\})")

# dataset -> URL template of a per-body Neuroglancer precomputed skeleton (unsharded store)
REMOTE: dict[str, str] = {
    "mcns": "https://storage.googleapis.com/flyem-male-cns/{version}/segmentation/skeletons-malecns/skeletons-precomputed/{root_id}",
}


def parse_precomputed_skeleton(data: bytes, root_id: int, radius_um: float = 0.25, source_units_nm: float = 1.0):
    """Neuroglancer precomputed skeleton bytes -> navis TreeNeuron in micrometres.

    Layout: uint32 n_vertices, uint32 n_edges, float32 vertices (n, 3), uint32
    edges (m, 2). Edges are undirected; a breadth-first tree is taken per
    connected component (extra edges closing cycles are dropped).
    """
    import navis
    import pandas as pd
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    nv, ne = struct.unpack("<II", data[:8])
    verts = np.frombuffer(data, dtype="<f4", count=nv * 3, offset=8).reshape(nv, 3).astype(np.float64)
    edges = np.frombuffer(data, dtype="<u4", count=ne * 2, offset=8 + nv * 12).reshape(ne, 2).astype(np.int64)
    parent = np.full(nv, -1, dtype=np.int64)
    if nv and ne:
        adj = coo_matrix((np.ones(ne), (edges[:, 0], edges[:, 1])), shape=(nv, nv))
        adj = (adj + adj.T).tocsr()
        n_comp, labels = connected_components(adj, directed=False)
        for c in range(n_comp):
            root = int(np.nonzero(labels == c)[0][0])
            order, preds = breadth_first_order(adj, root, directed=False, return_predecessors=True)
            parent[order] = preds[order]
            parent[root] = -1
    nodes = pd.DataFrame(
        {
            "node_id": np.arange(1, nv + 1),
            "parent_id": np.where(parent >= 0, parent + 1, -1),
            "x": verts[:, 0] * source_units_nm / 1e3,
            "y": verts[:, 1] * source_units_nm / 1e3,
            "z": verts[:, 2] * source_units_nm / 1e3,
            "radius": float(radius_um),
        }
    )
    n = navis.TreeNeuron(nodes, id=int(root_id), name=str(root_id), units="micrometer")
    n.radius_assumed_um = float(radius_um)
    return n
_UNIT_ALIASES = {"um": "micrometer", "micron": "micrometer", "microns": "micrometer", "micrometer": "micrometer", "nm": "nanometer", "nanometer": "nanometer"}


class SkeletonStore:
    def __init__(self, ds: "Dataset", path: str | Path | None = None, source_units: str = "nm", fetcher=None):
        self.ds = ds
        self.path = Path(path) if path else self._resolve()
        self.source_units = source_units  # assumed when an SWC has no units header
        self.fetcher = fetcher or (lambda url: urlopen(url).read())
        self._fetched: dict[int, object] = {}

    @property
    def remote(self) -> str | None:
        """URL template for on-demand skeleton downloads, or None."""
        t = REMOTE.get(self.ds.name)
        return t.format(version=self.ds.version, root_id="{root_id}") if t else None

    def _resolve(self) -> Path | None:
        from connexplorer.dataset import config

        candidates = [
            config.skeletons.get(self.ds.name),
            os.environ.get(f"CONNEXPLORER_SKELETONS_{self.ds.name.upper()}"),
            self.ds.tables_dir / "skeletons",
            self.ds.tables_dir.parent / "skeletons.zip",
        ]
        for c in candidates:
            if not c:
                continue
            p = Path(c)
            if p.is_dir():
                zips = sorted(p.glob("*.zip"))
                if zips:
                    return zips[0]
                if any(p.glob("*.swc")):
                    return p
            elif p.is_file():
                return p
        return None

    def _require(self) -> Path:
        if self.path is None:
            hint = f"; {self.ds.name} skeletons can also be fetched one at a time with skeletons.fetch(root_id)" if self.remote else ""
            raise FileNotFoundError(
                f"no skeletons for {self.ds.name}: put <root_id>.swc files (or one zip of them) in "
                f"{self.ds.tables_dir / 'skeletons'}, or set config.skeletons['{self.ds.name}'] / "
                f"CONNEXPLORER_SKELETONS_{self.ds.name.upper()}{hint}"
            )
        return self.path

    @cached_property
    def _zip(self) -> zipfile.ZipFile | None:
        p = self._require()
        return zipfile.ZipFile(p) if p.is_file() else None

    @cached_property
    def _members(self) -> dict[int, str]:
        """root id -> member name / file path for every available skeleton."""
        if self.path is None:
            return {}
        p = self.path
        names = self._zip.namelist() if self._zip is not None else [f.name for f in p.glob("*.swc")]
        out = {}
        for n in names:
            stem = n.rsplit("/", 1)[-1]
            if stem.endswith(".swc") and stem[:-4].isdigit():
                out[int(stem[:-4])] = n
        return out

    def __len__(self) -> int:
        return len(self._members)

    def has(self, root_id: int) -> bool:
        return int(root_id) in self._members

    def ids(self) -> np.ndarray:
        return np.fromiter(self._members, dtype=np.int64, count=len(self._members))

    def read_text(self, root_id: int) -> str:
        if self.path is None:
            self._require()
        try:
            name = self._members[int(root_id)]
        except KeyError:
            raise KeyError(f"no skeleton for root id {root_id} in {self.path}") from None
        if self._zip is not None:
            return self._zip.read(name).decode()
        return (self.path / name).read_text()

    def fetch(self, root_id: int, radius_um: float = 0.25, save: bool = True) -> "navis.TreeNeuron":
        """Download one skeleton from the vendor store (see :data:`REMOTE`) and keep it as SWC."""
        import navis

        if self.remote is None:
            raise ValueError(f"no remote skeleton store for {self.ds.name}")
        url = self.remote.format(root_id=int(root_id))
        try:
            data = self.fetcher(url)
        except Exception as e:  # noqa: BLE001
            raise KeyError(f"could not fetch skeleton {root_id} from {url}: {e}") from None
        n = parse_precomputed_skeleton(data, int(root_id), radius_um=radius_um)
        if save and (self.path is None or self.path.is_dir()):
            folder = self.path or (self.ds.tables_dir / "skeletons")
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{int(root_id)}.swc"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                navis.write_swc(n, target)
            if self.path is None:
                self.path = folder
            self._members[int(root_id)] = target.name
        else:
            self._fetched[int(root_id)] = n
        return n

    def load(self, root_id: int, units: str = "um", fetch: bool = True, radius_um: float = 0.25) -> "navis.TreeNeuron":
        """One skeleton as a navis TreeNeuron in micrometres (or ``units="nm"``); ``.units`` is set.

        Missing skeletons are fetched from the vendor store when the dataset has one.
        """
        import navis

        rid = int(root_id)
        if rid in self._fetched:
            n = self._fetched[rid]
            return n if units == "um" else n.convert_units(_UNIT_ALIASES.get(units, units))
        if not self.has(rid) and fetch and self.remote is not None:
            n = self.fetch(rid, radius_um=radius_um)
            return n if units == "um" else n.convert_units(_UNIT_ALIASES.get(units, units))
        text = self.read_text(root_id)
        m = _META.search(text[:2000])
        src_units = None
        if m:
            try:
                src_units = json.loads(m.group(1)).get("units")
            except ValueError:
                src_units = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n = navis.read_swc(io.StringIO(text), id=int(root_id))
        if src_units is None:
            n.units = _UNIT_ALIASES.get(self.source_units, self.source_units)
        target = _UNIT_ALIASES.get(units, units)
        if str(n.units.units) != target:
            n = n.convert_units(target)
        n.id = int(root_id)  # navis takes a string id from the Meta header otherwise
        n.name = str(root_id)
        return n

    def load_many(self, root_ids: Iterable[int], units: str = "um", max_n: int | None = None) -> "navis.NeuronList":
        import navis

        rids = list(root_ids)[: max_n if max_n else None]
        return navis.NeuronList([self.load(r, units) for r in rids])

    def __repr__(self) -> str:
        remote = ", fetch on demand" if self.remote else ""
        return f"SkeletonStore({self.ds.name}: {self.path}, {len(self):,} skeletons{remote})"
