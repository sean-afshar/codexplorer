"""SWC skeleton store: one loader for every dataset, always returning micrometres by default.

Skeletons are looked up, in order, in ``config.skeletons[dataset]``, the
``CONNEXPLORER_SKELETONS_<DATASET>`` variable, ``<tables>/skeletons/`` (a
directory of ``<root_id>.swc`` files or a single ``.zip`` of them) and
``<dataset dir>/skeletons.zip``. Files are read straight out of the zip and
parsed with navis (about 10 ms per neuron); navis' own zip reader with its
progress bar is not used.
"""

from __future__ import annotations

import io
import json
import os
import re
import warnings
import zipfile
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    import navis

    from connexplorer.dataset import Dataset

_META = re.compile(r"#\s*Meta:\s*(\{.*\})")
_UNIT_ALIASES = {"um": "micrometer", "micron": "micrometer", "microns": "micrometer", "micrometer": "micrometer", "nm": "nanometer", "nanometer": "nanometer"}


class SkeletonStore:
    def __init__(self, ds: "Dataset", path: str | Path | None = None, source_units: str = "nm"):
        self.ds = ds
        self.path = Path(path) if path else self._resolve()
        self.source_units = source_units  # assumed when an SWC has no units header

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
            raise FileNotFoundError(
                f"no skeletons for {self.ds.name}: put <root_id>.swc files (or one zip of them) in "
                f"{self.ds.tables_dir / 'skeletons'}, or set config.skeletons['{self.ds.name}'] / "
                f"CONNEXPLORER_SKELETONS_{self.ds.name.upper()}"
            )
        return self.path

    @cached_property
    def _zip(self) -> zipfile.ZipFile | None:
        p = self._require()
        return zipfile.ZipFile(p) if p.is_file() else None

    @cached_property
    def _members(self) -> dict[int, str]:
        """root id -> member name / file path for every available skeleton."""
        p = self._require()
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
        try:
            name = self._members[int(root_id)]
        except KeyError:
            raise KeyError(f"no skeleton for root id {root_id} in {self.path}") from None
        if self._zip is not None:
            return self._zip.read(name).decode()
        return (self.path / name).read_text()

    def load(self, root_id: int, units: str = "um") -> "navis.TreeNeuron":
        """One skeleton as a navis TreeNeuron in micrometres (or ``units="nm"``); ``.units`` is set."""
        import navis

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
        return f"SkeletonStore({self.ds.name}: {self.path}, {len(self) if self.path else 0:,} skeletons)"
