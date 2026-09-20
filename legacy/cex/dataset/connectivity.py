"""Sparse connectivity access for assigned cell IDs."""

from functools import cached_property
import os

import numpy as np
import pandas as pd
import scipy.sparse as spsp

from . import schema
from ..util.tables import find_table, read_table


class Connectivity:
    """Synapse-count connectivity matrix built on assigned IDs."""

    def __init__(
        self,
        fp_preprocessed: str,
        cell,
        synapse=None,
        column=None,
        get_cell_type_id=None,
        connectivity_table_name: str = schema.CELL_TO_CELL_SYN_COUNT_FILE,
    ):
        self.fp_preprocessed = fp_preprocessed
        self.cell = cell
        self.synapse = synapse
        self.column = column
        self.get_cell_type_id = cell.get_cell_type_id if get_cell_type_id is None else get_cell_type_id
        self.connectivity_table_name = connectivity_table_name

    @cached_property
    def edge_table(self) -> pd.DataFrame:
        """Return pre/post ID edge counts."""
        fp = find_table(self.fp_preprocessed, self.connectivity_table_name)
        if fp is not None:
            table = read_table(fp)
        elif self.synapse is not None:
            table = _edges_from_synapses(self.synapse.synapse_table)
        else:
            raise FileNotFoundError(
                f"Missing connectivity table {self.connectivity_table_name!r} in {self.fp_preprocessed}"
            )
        required = {"pre_id", "post_id", "num_syn"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"Connectivity table is missing required columns: {sorted(missing)}")
        return table.reset_index(drop=True)

    @cached_property
    def count_matrix_csr(self):
        """CSR synapse-count matrix indexed by assigned IDs."""
        return self.construct_idx_pair_to_num_syn_matrix(self.edge_table, self.cell.tot_num_cells, "csr")

    @cached_property
    def count_matrix_csc(self):
        """CSC synapse-count matrix indexed by assigned IDs."""
        return self.construct_idx_pair_to_num_syn_matrix(self.edge_table, self.cell.tot_num_cells, "csc")

    @cached_property
    def type_count_matrix_csr(self):
        """Return total synapse counts indexed by the type-table row order."""
        fp = os.path.join(self.fp_preprocessed, schema.TYPE_TO_TYPE_SYN_COUNT_FILE)
        with np.load(fp, allow_pickle=False) as data:
            type_list = data["type_list"].astype(str)
            matrix = spsp.csr_array(
                (data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"]))
        assert np.array_equal(type_list, self.cell.cell_type_info["type"].to_numpy().astype(str))
        return matrix

    def counts_between_types(
        self,
        pre_types,
        post_types=None,
        pre_side: str | None = None,
        post_side: str | None = None,
        remove_autapse_Q: bool = True,
        include_column_Q: bool = False,
    ) -> dict:
        """Return cell-level synapse counts between ordered type selections.

        Args:
            pre_types: Ordered presynaptic cell types.
            post_types: Ordered postsynaptic cell types. Use ``None`` to reuse
                ``pre_types``.
            pre_side: Optional side used to select presynaptic cells.
            post_side: Optional side used to select postsynaptic cells.
            remove_autapse_Q: Remove entries whose pre- and postsynaptic IDs
                refer to the same cell.
            include_column_Q: Include downloaded column assignments when the
                dataset provides them.

        Returns:
            Dictionary containing the sparse cell-by-cell matrix and aligned
            pre/post IDs, RIDs, cell types, neurotransmitters, signs, and type
            index arrays. Optional ``pre_column`` and ``post_column`` entries
            preserve the downloaded column-assignment fields.
        """
        pre_types = np.atleast_1d(pre_types).astype(str)
        if post_types is None:
            post_types = pre_types.copy()
            post_side = pre_side if post_side is None else post_side
        else:
            post_types = np.atleast_1d(post_types).astype(str)
        result = {
            "pre_types": pre_types, "post_types": post_types, "pre_side": pre_side,
            "post_side": post_side, "remove_autapse_Q": bool(remove_autapse_Q),
        }
        nt_by_type = self.cell.type_to_neurotransmitter
        for prefix, types, side in [("pre", pre_types, pre_side), ("post", post_types, post_side)]:
            id_parts = [self.get_cell_type_id(cell_type, side=side) for cell_type in types]
            num_cells = np.asarray([part.size for part in id_parts])
            ids = np.concatenate(id_parts) if id_parts else np.zeros(0, dtype=np.int64)
            cell_types = np.repeat(types, num_cells)
            type_nt = np.asarray([nt_by_type.get(cell_type, schema.DEFAULT_UNKNOWN) for cell_type in types])
            type_sign = np.asarray([schema.NEUROTRANSMITTER_TO_SIGN.get(nt, 0) for nt in type_nt], dtype=np.int8)
            result.update({
                f"{prefix}_ids": ids, f"{prefix}_rids": self.cell.id_to_rid_lookup.get_value(ids, not_found_val=-1),
                f"{prefix}_cell_types": cell_types, f"{prefix}_type_to_idx": {
                    cell_type: np.flatnonzero(cell_types == cell_type) for cell_type in types},
                f"{prefix}_type_neurotransmitter": type_nt, f"{prefix}_type_sign": type_sign,
                f"{prefix}_neurotransmitter": np.repeat(type_nt, num_cells),
                f"{prefix}_sign": np.repeat(type_sign, num_cells),
            })

        matrix = self.num_syn_between_id_groups(result["pre_ids"], result["post_ids"]).tocoo()
        if remove_autapse_Q:
            keep_Q = result["pre_ids"][matrix.row] != result["post_ids"][matrix.col]
            matrix = spsp.coo_array(
                (matrix.data[keep_Q], (matrix.row[keep_Q], matrix.col[keep_Q])), shape=matrix.shape)
        result["matrix"] = matrix.tocsr()
        if include_column_Q:
            if self.column is None:
                raise ValueError("Column data are not available on this connectivity object")
            result["pre_column"] = self.column.get_columns_for_ids(result["pre_ids"])
            result["post_column"] = self.column.get_columns_for_ids(result["post_ids"])
        return result

    def connected_ids_and_counts(
        self,
        cell_id: int,
        direction: str,
        remove_autapse_Q: bool = False,
        min_syn_per_rid: int = 1,
    ):
        """Return connected assigned IDs and synapse counts for one direction."""
        assert direction in ["input", "output"], "direction must be 'input' or 'output'"
        cell_id = int(cell_id)
        if direction == "input":
            edge_data = self.count_matrix_csc[:, [cell_id]].tocoo()
            connected_id = edge_data.row
        else:
            edge_data = self.count_matrix_csr[[cell_id], :].tocoo()
            connected_id = edge_data.col
        conn_num_syn = edge_data.data
        valid_Q = conn_num_syn >= min_syn_per_rid
        if remove_autapse_Q:
            valid_Q &= connected_id != cell_id
        return connected_id[valid_Q].astype(np.int64, copy=False), conn_num_syn[valid_Q]

    @staticmethod
    def construct_idx_pair_to_num_syn_matrix(edge_table, num_cells: int, matrix_type: str = "csr"):
        """Build a sparse matrix from edge-count rows."""
        data = edge_table["num_syn"].to_numpy()
        row = edge_table["pre_id"].to_numpy(dtype=np.int64)
        col = edge_table["post_id"].to_numpy(dtype=np.int64)
        shape = (int(num_cells), int(num_cells))
        if matrix_type == "csr":
            return spsp.csr_array((data, (row, col)), shape=shape)
        if matrix_type == "csc":
            return spsp.csc_array((data, (row, col)), shape=shape)
        raise ValueError(f"Unknown matrix_type: {matrix_type}")

    def num_syn_between_id_pairs(self, pre_id, post_id) -> np.ndarray:
        """Return synapse counts for assigned ID pairs."""
        pre_arr = np.atleast_1d(np.asarray(pre_id, dtype=np.int64))
        post_arr = np.atleast_1d(np.asarray(post_id, dtype=np.int64))
        if pre_arr.size != post_arr.size:
            raise ValueError("pre_id and post_id must have the same size")
        valid_Q = (
            (pre_arr >= 0)
            & (post_arr >= 0)
            & (pre_arr < self.cell.tot_num_cells)
            & (post_arr < self.cell.tot_num_cells)
        )
        out = np.zeros(pre_arr.size, dtype=np.uint32)
        if np.any(valid_Q):
            values = self.count_matrix_csr[pre_arr[valid_Q], post_arr[valid_Q]]
            out[valid_Q] = np.asarray(values).ravel().astype(np.uint32)
        return out

    def num_syn_between_rid_pairs(self, pre_rid, post_rid) -> np.ndarray:
        """Return synapse counts for dataset RID pairs."""
        pre_id = self.cell.rid_to_id_lookup.get_value(pre_rid, not_found_val=-1)
        post_id = self.cell.rid_to_id_lookup.get_value(post_rid, not_found_val=-1)
        return self.num_syn_between_id_pairs(pre_id, post_id)

    def num_syn_between_id_groups(self, pre_ids, post_ids=None):
        """Return a sparse submatrix for assigned ID groups."""
        pre_arr = np.atleast_1d(np.asarray(pre_ids, dtype=np.int64)).flatten()
        post_arr = pre_arr if post_ids is None else np.atleast_1d(np.asarray(post_ids, dtype=np.int64)).flatten()
        _validate_ids(pre_arr, self.cell.tot_num_cells)
        _validate_ids(post_arr, self.cell.tot_num_cells)
        return self.count_matrix_csr[pre_arr][:, post_arr]

    def num_syn_between_rid_groups(self, pre_rids, post_rids=None):
        """Return a sparse submatrix for dataset RID groups."""
        pre_ids = self.cell.rid_to_id_lookup.get_value(pre_rids, not_found_val=-1)
        post_ids = pre_ids if post_rids is None else self.cell.rid_to_id_lookup.get_value(post_rids, not_found_val=-1)
        return self.num_syn_between_id_groups(pre_ids, post_ids)

    def num_syn_between_type_groups(self, pre_type_list, post_type_list=None):
        """Return total synapse counts between ordered cell-type groups.

        The sparse all-type matrix is built once during dataset preprocessing.
        Rows and columns follow ``cell.cell_type_info`` so the dataset's
        existing type indices can query it directly.

        Args:
            pre_type_list: Cell types for matrix rows.
            post_type_list: Cell types for matrix columns. Use ``None`` to use
                ``pre_type_list`` for both axes.

        Returns:
            Sparse total-synapse-count matrix in the requested type order.
        """
        pre_type_list = np.atleast_1d(pre_type_list).astype(str)
        post_type_list = pre_type_list if post_type_list is None else np.atleast_1d(post_type_list).astype(str)
        type_to_idx = {cell_type: i for i, cell_type in enumerate(self.cell.cell_type_info["type"].astype(str))}
        pre_idx = np.asarray([type_to_idx[cell_type] for cell_type in pre_type_list])
        post_idx = np.asarray([type_to_idx[cell_type] for cell_type in post_type_list])
        return self.type_count_matrix_csr[pre_idx][:, post_idx]


def _edges_from_synapses(synapses: pd.DataFrame) -> pd.DataFrame:
    valid_Q = (synapses["pre_id"] >= 0) & (synapses["post_id"] >= 0)
    return (
        synapses.loc[valid_Q]
        .groupby(["pre_id", "post_id"], sort=True)
        .size()
        .rename("num_syn")
        .reset_index()
    )


def _validate_ids(ids: np.ndarray, num_cells: int) -> None:
    if np.any((ids < 0) | (ids >= num_cells)):
        raise ValueError("Some assigned cell IDs are not valid")
