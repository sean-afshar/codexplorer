"""Column assignment access for normalized cell IDs and RIDs."""

from collections import defaultdict
from functools import cached_property

import numpy as np
import pandas as pd

from . import schema
from ..util.tables import find_table, read_table


class Column:
    """Column assignments keyed by assigned `id` and dataset `rid`."""

    def __init__(
        self,
        fp_preprocessed: str,
        column_table_name: str = schema.COLUMN_DATA_FILE,
        cell=None,
    ):
        self.fp_preprocessed = fp_preprocessed
        self.column_table_name = column_table_name
        self.cell = cell

    @cached_property
    def column_table(self) -> pd.DataFrame:
        """Return the normalized column table, or an empty table if absent."""
        fp = find_table(self.fp_preprocessed, self.column_table_name)
        if fp is None:
            return pd.DataFrame(columns=["id", "rid", "type", "side", "p", "q"])
        if fp.endswith(".npz"):
            data = np.load(fp)
            row_count = data["id"].shape[0] if "id" in data.files else _most_common_1d_length(data)
            table = pd.DataFrame(
                {
                    k: data[k]
                    for k in data.files
                    if data[k].ndim == 1 and data[k].shape[0] == row_count
                }
            )
        else:
            table = read_table(fp)
        if "rid" not in table and "id" in table and self.cell is not None:
            table["rid"] = self.cell.id_to_rid_lookup.get_value(table["id"].to_numpy(), not_found_val=-1)
        required = {"id", "p", "q"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"Column table is missing required columns: {sorted(missing)}")
        return table.reset_index(drop=True)

    @cached_property
    def column_type_list(self) -> list[str]:
        """Cell types with column assignments."""
        if "type" not in self.column_table:
            return []
        return sorted(map(str, self.column_table["type"].dropna().unique()))

    @cached_property
    def col_pq_to_rid(self) -> dict:
        """Map side and pq coordinate to assigned RIDs."""
        out = {"left": defaultdict(list), "right": defaultdict(list)}
        if self.column_table.empty:
            return out
        side_col = _side_column(self.column_table)
        for row in self.column_table.itertuples(index=False):
            side = getattr(row, side_col, "")
            side = _normalize_side(side)
            if side not in out:
                continue
            out[side][(int(row.p), int(row.q))].append(int(row.rid))
        return out

    def get_columns_for_ids(self, ids, remove_invalid_Q: bool = False) -> dict:
        """Return column assignment for assigned IDs."""
        return self._get_columns(ids, key_col="id", remove_invalid_Q=remove_invalid_Q)

    def get_columns_for_rids(self, rids, remove_invalid_Q: bool = False) -> dict:
        """Return column assignment for dataset RIDs."""
        return self._get_columns(rids, key_col="rid", remove_invalid_Q=remove_invalid_Q)

    def col_pq_type_to_rid(self, side: str, pq, cell_type: str) -> list[int]:
        """Return RIDs for one side, pq coordinate, and cell type."""
        side = _normalize_side(side)
        pq = tuple(pq)
        rids = self.col_pq_to_rid.get(side, {}).get(pq, [])
        if "type" not in self.column_table or not rids:
            return list(rids)
        subtable = self.column_table.set_index("rid", drop=False)
        return [rid for rid in rids if str(subtable.loc[rid, "type"]) == cell_type]

    def is_column_type_Q(self, cell_type):
        """Return whether a cell type has column assignments."""
        if isinstance(cell_type, (list, np.ndarray)):
            return np.asarray([self.is_column_type_Q(ct) for ct in cell_type])
        return str(cell_type) in self.column_type_list

    def is_column_neuron_Q(self, rid_s):
        """Return whether each RID has a column assignment."""
        rids = np.atleast_1d(np.asarray(rid_s)).flatten()
        assigned = set(map(int, self.column_table["rid"].to_numpy())) if not self.column_table.empty else set()
        return np.asarray([int(rid) in assigned for rid in rids], dtype=bool)

    def _get_columns(self, keys, key_col: str, remove_invalid_Q: bool) -> dict:
        key_arr = np.atleast_1d(np.asarray(keys)).flatten()
        info = {"id": key_arr}
        info["xy"] = np.zeros((key_arr.size, 2), dtype=np.int32)
        info["pq"] = np.zeros((key_arr.size, 2), dtype=np.int32)
        info["left_Q"] = np.zeros(key_arr.size, dtype=bool)
        info["valid_Q"] = np.zeros(key_arr.size, dtype=bool)
        if self.column_table.empty or key_col not in self.column_table:
            return _filter_invalid(info, remove_invalid_Q)

        indexed = self.column_table.set_index(key_col, drop=False)
        side_col = _side_column(self.column_table)
        for i, key in enumerate(key_arr):
            if key not in indexed.index:
                continue
            row = indexed.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            info["valid_Q"][i] = True
            info["pq"][i] = [int(row["p"]), int(row["q"])]
            if {"x", "y"}.issubset(self.column_table.columns):
                info["xy"][i] = [int(row["x"]), int(row["y"])]
            info["left_Q"][i] = _normalize_side(row.get(side_col, "")) == "left"
        return _filter_invalid(info, remove_invalid_Q)


def _filter_invalid(info: dict, remove_invalid_Q: bool) -> dict:
    if not remove_invalid_Q:
        return info
    valid_Q = info["valid_Q"]
    return {k: v[valid_Q] for k, v in info.items()}


def _most_common_1d_length(data) -> int:
    lengths = [data[k].shape[0] for k in data.files if data[k].ndim == 1]
    if not lengths:
        return 0
    values, counts = np.unique(lengths, return_counts=True)
    return int(values[np.argmax(counts)])


def _side_column(table: pd.DataFrame) -> str:
    if "side" in table:
        return "side"
    if "hemisphere" in table:
        return "hemisphere"
    return "side"


def _normalize_side(value) -> str:
    if value == schema.SIDE_LEFT:
        return "left"
    if value == schema.SIDE_RIGHT:
        return "right"
    value = str(value).lower()
    if value in {"left", "l"}:
        return "left"
    if value in {"right", "r"}:
        return "right"
    return value
