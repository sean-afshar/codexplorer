"""Normalized cell table access."""

from __future__ import annotations

from functools import cached_property
import os

import numpy as np
import pandas as pd
from . import schema
from ..util import ScalarDict
from ..util.tables import read_table


class Cell:
    """Cell-level metadata and `id`/`rid` lookup tables."""

    __essential_columns = ("id", "rid", "type", "side")

    def __init__(
        self,
        fp_preprocessed: str,
        cell_table_name: str = schema.CELL_DATA_FILE,
        cell_type_info_name: str = schema.TYPE_DATA_FILE,
    ):
        self.fp_preprocessed = fp_preprocessed
        self.cell_table_name = cell_table_name
        self.cell_type_info_name = cell_type_info_name

    @cached_property
    def cell_table(self) -> pd.DataFrame:
        """Return the normalized cell table."""
        return read_table(os.path.join(self.fp_preprocessed, self.cell_table_name))

    @cached_property
    def _essential_table(self) -> pd.DataFrame:
        """Return a minimal table with only essential columns for lookups."""
        fp = os.path.join(self.fp_preprocessed, self.cell_table_name)
        table = read_table(fp, columns=list(self.__essential_columns))
        return table

    @cached_property
    def cell_type_info(self) -> pd.DataFrame:
        """Return the type-level summary table."""
        fp = os.path.join(self.fp_preprocessed, self.cell_type_info_name)
        return read_table(fp)

    @cached_property
    def ids(self) -> np.ndarray:
        """Assigned integer cell IDs."""
        ids = self._essential_table.id.values
        assert np.all(ids[1:] > ids[:-1]), "IDs are not strictly increasing, cannot build id_to_rid mapping"
        assert ids[0] == 0, "IDs do not start at 0, cannot build id_to_rid mapping"
        assert ids[-1] == ids.size - 1, "IDs do not end at max index, cannot build id_to_rid mapping"
        return ids

    @cached_property
    def rids(self) -> np.ndarray:
        """Dataset-provided root IDs."""
        return self._essential_table.rid.values

    @cached_property
    def cell_types(self) -> np.ndarray:
        """Cell type array indexed by assigned `id`."""
        return np.asarray(self._essential_table.type.fillna("Unknown"), dtype=str)

    @cached_property
    def id_to_rid(self) -> np.ndarray:
        """Map assigned `id` to dataset `rid`."""
        return self.rids

    @cached_property
    def rid_to_id(self) -> dict[int, int]:
        """Map dataset `rid` to assigned `id`."""
        return {int(k): int(v) for k, v in zip(self.rids, self.ids)}

    @cached_property
    def rid_to_id_lookup(self):
        """Vectorized `rid` to `id` lookup."""
        return ScalarDict(self.rids, self.ids.astype(np.int64, copy=False))

    @cached_property
    def id_to_rid_lookup(self):
        """Vectorized `id` to `rid` lookup."""
        return ScalarDict(self.ids, self.rids.astype(np.int64, copy=False))

    @cached_property
    def id_to_type(self) -> dict[int, str]:
        """Map assigned `id` to cell type."""
        return {int(k): str(v) for k, v in zip(self.ids, self.cell_types)}

    @cached_property
    def rid_to_type(self) -> dict[int, str]:
        """Map dataset `rid` to cell type."""
        return {int(k): str(v) for k, v in zip(self.rids, self.cell_types)}

    @cached_property
    def type_to_id(self) -> dict[str, np.ndarray]:
        """Map each cell type to assigned `id` values."""
        return self._values_by_type("id")

    @cached_property
    def type_to_rid(self) -> dict[str, np.ndarray]:
        """Map each cell type to dataset `rid` values."""
        return self._values_by_type("rid")

    @cached_property
    def cell_count(self) -> dict[str, int]:
        """Number of cells per type."""
        return {
            str(cell_type): int(num_cells)
            for cell_type, num_cells in zip(
                self.cell_type_info.type.values.astype(str),
                self.cell_type_info.num_cells.values.astype(int),
            )
        }

    @cached_property
    def type_to_neurotransmitter(self) -> dict[str, str]:
        """Map each cell type to its neurotransmitter."""
        if "nt" not in self.cell_type_info:
            return {
                str(cell_type): schema.DEFAULT_UNKNOWN
                for cell_type in self.cell_type_info.type.values.astype(str)
            }
        return {
            str(cell_type): str(nt)
            for cell_type, nt in zip(
                self.cell_type_info.type.values.astype(str),
                self.cell_type_info.nt.values.astype(str),
            )
        }

    @cached_property
    def id_to_side(self) -> np.ndarray:
        """Side code array indexed by assigned `id`."""
        return self._essential_table.side.values.astype(np.int8)

    @property
    def tot_num_cells(self) -> int:
        """Total number of cells."""
        return int(self.ids.size)

    def get_cell_type_id(self, cell_type: str, side: str | None = None) -> np.ndarray:
        """Return assigned IDs for a cell type, optionally filtered by side."""
        ids = self.type_to_id.get(cell_type, np.zeros(0, dtype=np.int64))
        if side is None or ids.size == 0:
            return ids
        side_values = self.id_to_side[ids]
        side_key = side.lower() if isinstance(side, str) else side
        side_idx = schema.SIDE_TO_IDX.get(side_key, schema.SIDE_UNKNOWN)
        if side_idx == schema.SIDE_UNKNOWN:
            raise ValueError(f"Unknown side: {side}")
        return ids[side_values == side_idx]

    def get_cell_type_rid(self, cell_type: str, side: str | None = None) -> np.ndarray:
        """Return dataset RIDs for a cell type, optionally filtered by side."""
        ids = self.get_cell_type_id(cell_type, side=side)
        if ids.size == 0:
            return np.zeros(0, dtype=self.rids.dtype)
        return self.id_to_rid_lookup.get_value(ids, not_found_val=-1)

    def _values_by_type(self, column: str, dtype=None) -> dict[str, np.ndarray]:
        table = self._essential_table.sort_values(["type", "id"])
        out = {}
        for cell_type, subtable in table.groupby("type", sort=True):
            values = subtable[column].to_numpy()
            out[str(cell_type)] = values.astype(dtype, copy=False) if dtype is not None else values
        return out
