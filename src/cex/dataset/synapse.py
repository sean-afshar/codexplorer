"""Normalized synapse table access."""

from functools import cached_property

import numpy as np
import pandas as pd
from . import schema
from ..util import bin_data_to_idx_list
from ..util.tables import find_table, read_table


SYNAPSE_ARRAY_COLUMNS = ["pre_id", "post_id", "x_nm", "y_nm", "z_nm"]


class Synapse:
    """Synapse table indexed by assigned pre/post cell IDs."""

    def __init__(
        self,
        fp_preprocessed: str,
        synapse_table_name: str = schema.SYNAPSE_DATA_FILE,
        cell=None,
    ):
        self.fp_preprocessed = fp_preprocessed
        self.synapse_table_name = synapse_table_name
        self.cell = cell

    @cached_property
    def synapse_table(self) -> pd.DataFrame:
        """Return the normalized synapse table."""
        fp = find_table(self.fp_preprocessed, self.synapse_table_name)
        if fp is None:
            raise FileNotFoundError(
                f"Missing normalized synapse table {self.synapse_table_name!r} in {self.fp_preprocessed}"
            )
        table = read_table(fp)
        required = set(SYNAPSE_ARRAY_COLUMNS)
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"Synapse table is missing required columns: {sorted(missing)}")
        return table.reset_index(drop=True)

    @cached_property
    def pre_to_idx(self) -> dict[int, np.ndarray]:
        """Map assigned presynaptic ID to synapse row indices."""
        return bin_data_to_idx_list(
            self.synapse_table["pre_id"].to_numpy(),
            return_type="dict",
        )

    @cached_property
    def post_to_idx(self) -> dict[int, np.ndarray]:
        """Map assigned postsynaptic ID to synapse row indices."""
        return bin_data_to_idx_list(
            self.synapse_table["post_id"].to_numpy(),
            return_type="dict",
        )

    @cached_property
    def pre_ids(self) -> np.ndarray:
        """Return the presynaptic assigned-ID column as an array."""
        return self.synapse_table["pre_id"].to_numpy(copy=False)

    @cached_property
    def post_ids(self) -> np.ndarray:
        """Return the postsynaptic assigned-ID column as an array."""
        return self.synapse_table["post_id"].to_numpy(copy=False)

    @cached_property
    def xyz_nm(self) -> np.ndarray:
        """Return synapse center coordinates in nm."""
        return self.synapse_table.loc[:, SYNAPSE_ARRAY_COLUMNS[2:]].to_numpy(copy=False)

    def get_for_ids(self, ids, side: str, return_type: str = "frame"):
        """Return synapses where assigned IDs appear on `side`."""
        if side in {"pre", "pre_id"}:
            mapping = self.pre_to_idx
        elif side in {"post", "post_id"}:
            mapping = self.post_to_idx
        else:
            raise ValueError(f"Unknown synapse side: {side}")
        return self._collect_rows(ids, mapping, return_type=return_type)

    def get_pair_indices(self, pre_id: int, post_id: int) -> np.ndarray:
        """Return row indices for synapses from `pre_id` to `post_id`."""
        pre_idx = self.pre_to_idx.get(int(pre_id), np.zeros(0, dtype=np.int64))
        post_idx = self.post_to_idx.get(int(post_id), np.zeros(0, dtype=np.int64))
        return np.intersect1d(pre_idx, post_idx, assume_unique=True)

    def get_pair_synapses(self, pre_id: int, post_id: int) -> pd.DataFrame:
        """Return synapse rows from `pre_id` to `post_id`."""
        return self.synapse_table.iloc[self.get_pair_indices(pre_id, post_id)]

    def get_pair_count(self, pre_id: int, post_id: int) -> int:
        """Return number of synapses from `pre_id` to `post_id`."""
        return int(self.get_pair_indices(pre_id, post_id).size)

    def get_pre_post_rows_for_ids(
        self,
        ids,
        remove_autapse_Q: bool = True,
    ) -> dict[str, dict[int, np.ndarray]]:
        """Return synapse row indices keyed by endpoint direction and assigned ID.

        ``"pre"`` contains rows where the queried ID is presynaptic, and
        ``"post"`` contains rows where it is postsynaptic.  Autapses are
        removed by default.
        """
        result = {"pre": {}, "post": {}}
        for cell_id in np.atleast_1d(np.asarray(ids)).ravel():
            cell_id = int(cell_id)
            pre_rows = self.pre_to_idx.get(cell_id, np.zeros(0, dtype=np.intp))
            post_rows = self.post_to_idx.get(cell_id, np.zeros(0, dtype=np.intp))
            if remove_autapse_Q:
                pre_rows = pre_rows[self.post_ids[pre_rows] != cell_id]
                post_rows = post_rows[self.pre_ids[post_rows] != cell_id]
            result["pre"][cell_id] = pre_rows
            result["post"][cell_id] = post_rows
        return result

    def get_synapse_xyz_for_ids(
        self,
        ids,
        remove_autapse_Q: bool = True,
    ) -> dict[int, np.ndarray]:
        """Return non-directional pre/post synapse xyz arrays keyed by assigned ID."""
        row_lookup = self.get_pre_post_rows_for_ids(ids, remove_autapse_Q=remove_autapse_Q)
        out = {}
        for cell_id in np.atleast_1d(np.asarray(ids)).ravel():
            cell_id = int(cell_id)
            pre_rows = row_lookup["pre"][cell_id]
            post_rows = row_lookup["post"][cell_id]
            rows = np.concatenate([pre_rows, post_rows])
            out[cell_id] = self.xyz_nm[rows]
        return out

    def _get_connection_rows_between_id_groups_without_grouping(
        self,
        self_ids,
        partner_ids,
        direction: str,
    ) -> np.ndarray:
        """Return rows matching one source-type/partner-type connection query.

        ``direction == "input"`` selects partner-to-source synapses.
        ``direction == "output"`` selects source-to-partner synapses.
        """
        self_ids = np.atleast_1d(np.asarray(self_ids)).ravel()
        partner_ids = np.atleast_1d(np.asarray(partner_ids)).ravel()
        if self_ids.size > 0 and partner_ids.size > 0:
            source_mapping, partner_mapping = self._connection_mappings(direction)
            source_rows = self._get_row_indices_for_ids(self_ids, source_mapping)
            partner_ids = self._get_row_indices_for_ids(partner_ids, partner_mapping)
            if source_rows.size > 0 and partner_ids.size > 0:
                return np.intersect1d(source_rows, partner_ids, assume_unique=True)
        return np.zeros(0, dtype=np.intp)

    def get_connection_rows_between_id_groups(
        self,
        self_ids,
        partner_ids,
        direction: str,
        group_by_source_Q: bool = False,
    ) -> np.ndarray | dict[int, np.ndarray]:
        """Return rows matching one source-type/partner-type connection query.

        ``direction == "input"`` selects partner-to-source synapses.
        ``direction == "output"`` selects source-to-partner synapses.
        When ``group_by_source_Q`` is true, return a dict keyed by source
        assigned ID instead of one merged row-index array.
        """
        self_ids = np.unique(np.atleast_1d(np.asarray(self_ids)).ravel())
        partner_ids = np.unique(np.atleast_1d(np.asarray(partner_ids)).ravel())
        if not group_by_source_Q:
            return self._get_connection_rows_between_id_groups_without_grouping(self_ids, partner_ids, direction)

        row_groups = {}
        if self_ids.size > 0 and partner_ids.size > 0:
            source_mapping, partner_mapping = self._connection_mappings(direction)
            # USER: not sure if intersection or boolean mask is more efficient here
            partner_row_mask = self._row_mask_for_ids(partner_ids, partner_mapping)
            if not np.any(partner_row_mask): return row_groups

            for source_id in self_ids:
                rows = source_mapping.get(int(source_id), np.zeros(0, dtype=np.intp))
                if rows.size == 0:
                    continue
                matched_rows = rows[partner_row_mask[rows]]
                if matched_rows.size > 0:
                    row_groups[int(source_id)] = matched_rows

        return row_groups

    @staticmethod
    def _get_row_indices_for_ids(ids, mapping) -> np.ndarray:
        row_parts = [mapping[int(query_id)] for query_id in ids if int(query_id) in mapping]
        if not row_parts:
            return np.zeros(0, dtype=np.intp)
        return np.concatenate(row_parts)

    def _collect_rows(self, ids, mapping, return_type: str):
        ids = np.atleast_1d(np.asarray(ids)).flatten()
        row_dict = {}
        for query_id in ids:
            rows = mapping.get(int(query_id), np.zeros(0, dtype=np.int64))
            if rows.size > 0:
                row_dict[int(query_id)] = self.synapse_table.iloc[rows]
        if return_type == "dict":
            return row_dict
        if return_type == "list":
            return list(row_dict.values())
        if return_type == "array":
            if not row_dict:
                return self.synapse_table.iloc[0:0].loc[:, SYNAPSE_ARRAY_COLUMNS].to_numpy()
            return np.vstack(
                [frame.loc[:, SYNAPSE_ARRAY_COLUMNS].to_numpy(copy=False) for frame in row_dict.values()]
            )
        if return_type == "frame":
            if not row_dict:
                return self.synapse_table.iloc[0:0]
            return pd.concat(row_dict.values(), ignore_index=True)
        raise ValueError(f"Unknown return_type: {return_type}")

    def _connection_mappings(self, direction: str) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        if direction == "input":
            return self.post_to_idx, self.pre_to_idx
        if direction == "output":
            return self.pre_to_idx, self.post_to_idx
        raise ValueError(f"Unknown connection direction: {direction}")

    def _row_mask_for_ids(self, ids, mapping) -> np.ndarray:
        row_mask = np.zeros(self.pre_ids.size, dtype=bool)
        for query_id in ids:
            rows = mapping.get(int(query_id), np.zeros(0, dtype=np.intp))
            if rows.size > 0:
                row_mask[rows] = True
        return row_mask
