"""Neuron IO parsing backed by normalized `ConnectomeDataset` tables."""

from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
from ..dataset import flywire_links
from ..dataset.dataset import ConnectomeDataset
from ..util import select_val_by_num_repeat


def _conn_to_str(conn):
    """Convert `(direction, type)` to the legacy compact string."""
    assert isinstance(conn, tuple) and len(conn) == 2, "conn should be a tuple of (syn_dir, conn_type)."
    return f"{conn[1]}_{conn[0][0]}"


def _conn_str_to_conn_tuple(conn_str):
    """Convert a legacy compact connection string to `(direction, type)`."""
    conn_type, syn_dir = conn_str.split("_")
    syn_dir = "input" if syn_dir == "i" else "output" if syn_dir == "o" else None
    return syn_dir, conn_type


class NeuronBase:
    """Single-neuron IO view backed by a `ConnectomeDataset`.

    The constructor is assigned-`id` first; `root_id`/`rid` are
    derived from the dataset lookup table when needed.
    """

    def __init__(
        self,
        id,
        dataset: ConnectomeDataset,
        remove_autapse_Q: bool = False,
        min_syn_per_rid: int = 1,
        load_xyz_Q: bool = False,
        parse_io_Q: bool = True,
    ):
        self.id = id
        self.dataset = dataset
        self.load_xyz_Q = load_xyz_Q
        self._data = {}
        if parse_io_Q:
            self.__parse_io(
                remove_autapse_Q=remove_autapse_Q,
                min_syn_per_rid=min_syn_per_rid,
                load_xyz_Q=load_xyz_Q,
            )

    def __parse_io(
        self,
        remove_autapse_Q: bool = False,
        min_syn_per_rid: int = 1,
        load_xyz_Q: bool = False,
    ) -> None:
        self.input = NeuronBase._parse_synapse_data(
            self._id_array,
            self.dataset,
            "post",
            0,
            remove_autapse_Q,
            min_syn_per_rid,
            xyz_Q=load_xyz_Q,
        )
        self.output = NeuronBase._parse_synapse_data(
            self._id_array,
            self.dataset,
            "pre",
            1,
            remove_autapse_Q,
            min_syn_per_rid,
            xyz_Q=load_xyz_Q,
        )

    @cached_property
    def _id_array(self) -> np.ndarray:
        """Assigned IDs for this neuron handle."""
        id_arr = np.atleast_1d(np.asarray(self.id)).flatten().astype(np.int64, copy=False)
        if id_arr.size == 0:
            raise ValueError("Neuron id cannot be empty")
        invalid_Q = (id_arr < 0) | (id_arr >= self.dataset.cell.tot_num_cells)
        if np.any(invalid_Q):
            raise ValueError(f"Unknown assigned id: {self.id}")
        return id_arr

    @cached_property
    def root_id(self):
        """Dataset RID for this neuron handle."""
        rid = self.dataset.cell.id_to_rid_lookup.get_value(self._id_array, not_found_val=-1)
        if np.any(np.asarray(rid) < 0):
            raise ValueError(f"Unknown assigned id: {self.id}")
        if isinstance(self.id, (int, np.integer)):
            return int(np.asarray(rid).item())
        return np.asarray(rid)

    @property
    def rid(self):
        """Compatibility alias for `root_id`."""
        return self.root_id

    @cached_property
    def cell_type(self):
        """Return the cell type of this ID, or the common type for an ID list."""
        cell_type = [self.dataset.cell.id_to_type.get(int(id), "Unknown") for id in self._id_array]
        cell_type_u = np.unique(cell_type)
        if self._id_array.size == 1 or cell_type_u.size == 1:
            return cell_type_u[0]
        return cell_type

    @staticmethod
    def _parse_synapse_data(
        ids,
        dataset: ConnectomeDataset,
        side: str,
        rid_col: int,
        remove_autapse_Q: bool = False,
        min_syn_per_rid: int = 1,
        xyz_Q: bool = True,
    ):
        """Parse one input/output direction using the normalized dataset."""
        assert side in ["pre", "post"], "side must be 'pre' or 'post'"
        assert rid_col in [0, 1], "rid_col must be 0 (pre) or 1 (post)"
        if xyz_Q:
            return _parse_synapse_data_with_xyz(
                ids,
                dataset,
                side,
                rid_col,
                remove_autapse_Q=remove_autapse_Q,
                min_syn_per_rid=min_syn_per_rid,
            )

        ids = np.atleast_1d(np.asarray(ids)).flatten()
        if ids.size == 1:
            connected_id, conn_num_syn = dataset.connectivity.connected_ids_and_counts(
                int(ids[0]),
                "input" if side == "post" else "output",
                remove_autapse_Q=remove_autapse_Q,
                min_syn_per_rid=min_syn_per_rid,
            )
        else:
            if side == "post":
                count = np.asarray(dataset.connectivity.count_matrix_csc[:, ids].sum(axis=1)).ravel()
            elif side == "pre":
                count = np.asarray(dataset.connectivity.count_matrix_csr[ids, :].sum(axis=0)).ravel()
            else:
                raise ValueError(f"Invalid side: {side}")

            connected_id, conn_num_syn = _connected_group_counts(
                    count,
                    ids,
                    remove_autapse_Q=remove_autapse_Q,
                    min_syn_per_rid=min_syn_per_rid,
                )
        conn_rid = dataset.cell.id_to_rid_lookup.get_value(connected_id, not_found_val=-1)
        out = {"rid": conn_rid}
        out |= dataset._group_root_ids(out["rid"], rid_num_syn=conn_num_syn)
        return out

    def get_type_counts_combined_stat(self, direction, min_syn_count: int = 0, sort_key: str = "#s"):
        """Return per-type counts for one direction."""
        assert direction in ["input", "output"], "Direction must be 'input' or 'output'"
        if not hasattr(self, "_type_combined_stats"):
            self._type_combined_stats = {}

        if direction in self._type_combined_stats:
            table = self._type_combined_stats[direction]
        else:
            data = getattr(self, direction)
            table = pd.DataFrame(
                {
                    "type": data["type_u"],
                    "#s": data["t_num_s"],
                    "#c": data["t_num_c"],
                    "#ec": np.round(data["t_enum_c"], 2),
                    "#s_w1c": data["t_num_s_w1c"],
                }
            )
            if not table.empty:
                table["#s/c"] = np.round(table["#s"] / table["#c"], 2)
                table["f_s"] = np.round(table["#s"] / np.sum(table["#s"]), 4)
                table["f_c"] = np.round(table["#c"] / np.sum(table["#c"]), 4)
            self._type_combined_stats[direction] = table

        out = table.copy()
        if min_syn_count:
            out = out[out["#s"] >= min_syn_count]
        return out.sort_values(by=[sort_key], ascending=False).reset_index(drop=True)

    def get_connected_type_synapse_xyz(self, direction, cell_type):
        """Return xyz coordinates for synapses to/from one cell type."""
        assert direction in ["input", "output"], "Direction must be 'input' or 'output'"
        data = getattr(self, direction)
        type_idx = data["t2i"].get(cell_type, np.zeros(0, dtype=np.int64))
        return data["xyz"][type_idx]

    def get_top_n_input_type_stat(self, n: int = 10, sort_key: str = "#s"):
        """Return top input types by the selected count key."""
        return self.get_type_counts_combined_stat("input", sort_key=sort_key).head(n)

    def get_top_n_output_type_stat(self, n: int = 10, sort_key: str = "#s"):
        """Return top output types by the selected count key."""
        return self.get_type_counts_combined_stat("output", sort_key=sort_key).head(n)

    def get_type_neuron_w_most_synapse(self, conn_dir, cell_type, return_ns_Q: bool = False):
        """Return the connected RID with the largest synapse count."""
        assert conn_dir in ["input", "output"], f"Invalid connection direction: {conn_dir}"
        data = getattr(self, conn_dir)
        if cell_type not in data["t2rid"]:
            rid = None
            num_syn = 0
        else:
            rid_list = data["t2rid"][cell_type]
            num_syn_list = np.asarray([v.size for v in data["t2c_i"][cell_type]])
            if rid_list.size == 1:
                rid = int(rid_list[0])
                num_syn = int(num_syn_list[0])
            else:
                max_idx = num_syn_list.argmax()
                rid = int(rid_list[max_idx])
                num_syn = int(num_syn_list[max_idx])
        if return_ns_Q:
            return rid, num_syn
        return rid

    def _construct_synapse_annotation_layer(
        self,
        syn_dir: str,
        type_list: list,
        assign_random_color_Q: bool = False,
    ):
        """Return Neuroglancer annotation layers for connected synapse sets."""
        syn_data = getattr(self, syn_dir)
        if "xyz" not in syn_data:
            raise ValueError("Synapse annotation layers require Neuron(..., load_xyz_Q=True)")

        syn_ann_layer = {}
        for cell_type in type_list:
            type_idx = syn_data["t2i"].get(cell_type, np.zeros(0, dtype=np.int64))
            if type_idx.size == 0:
                continue
            syn_xyz = syn_data["xyz"][type_idx]
            if syn_dir == "input":
                layer_name = f"{cell_type}_to_{self.cell_type}_input"
            else:
                layer_name = f"{self.cell_type}_to_{cell_type}_output"
            syn_ann_layer[cell_type] = flywire_links.annotation_layer(
                syn_xyz,
                layer_name=layer_name,
                assign_random_color_Q=assign_random_color_Q,
            )
        return syn_ann_layer

    def _construct_connected_neuron_layer(self, syn_dir, type_list):
        """Return connected assigned IDs grouped by cell type for Neuroglancer."""
        syn_data = getattr(self, syn_dir)
        ct_layer = {}
        for cell_type in type_list:
            if cell_type not in syn_data["t2rid"]:
                print(f"Warning: Cell type {cell_type} not found in {syn_dir} data.")
                continue
            ids = self.dataset.cell.rid_to_id_lookup.get_value(
                syn_data["t2rid"][cell_type],
                not_found_val=-1,
            )
            ids = np.asarray(ids)
            ids = ids[ids >= 0]
            if ids.size > 0:
                ct_layer[cell_type] = ids
        return ct_layer

    def vis_self_w_top_n_conn_type_in_ng(
        self,
        syn_dir,
        n: int = 10,
        open_url_Q: bool = True,
        vis_syn_Q: bool = False,
        assign_random_color_Q: bool = False,
        return_payload_Q: bool = False,
    ):
        """Return a Neuroglancer URL for self plus top connected types."""
        if syn_dir in ["input", "output"]:
            type_stat_df = self.get_type_counts_combined_stat(syn_dir, sort_key="#s")
            top_n_types = type_stat_df["type"].head(n).tolist()
            d_dir2type = {syn_dir: top_n_types}
        else:
            assert isinstance(syn_dir, (list, tuple))
            d_dir2type = {}
            for tmp_dir in syn_dir:
                type_stat_df = self.get_type_counts_combined_stat(tmp_dir, sort_key="#s")
                d_dir2type[tmp_dir] = type_stat_df["type"].head(n).tolist()

        return self.vis_self_w_conn_type_and_synapse_in_ng(
            d_dir2type=d_dir2type,
            open_url_Q=open_url_Q,
            vis_syn_Q=vis_syn_Q,
            assign_random_color_Q=assign_random_color_Q,
            return_payload_Q=return_payload_Q,
        )

    def vis_self_w_conn_type_and_synapse_in_ng(
        self,
        syn_dir=None,
        type_list=None,
        d_dir2type=None,
        open_url_Q: bool = True,
        vis_syn_Q: bool = True,
        assign_random_color_Q: bool = False,
        return_payload_Q: bool = False,
    ):
        """Return a Neuroglancer URL for self, connected cells, and optional synapses."""
        if d_dir2type is not None:
            assert syn_dir is None and type_list is None
        elif syn_dir is not None:
            assert syn_dir in ["input", "output"], "syn_dir must be 'input' or 'output'"
            assert d_dir2type is None, "Conflicting input"
            d_dir2type = {syn_dir: type_list}
        else:
            raise ValueError("Unexpected inputs")

        syn_ann_layer = {} if vis_syn_Q else None
        conn_neuron_layer = {}
        add_prefix_Q = len(d_dir2type) > 1
        for tmp_dir, tmp_list in d_dir2type.items():
            if vis_syn_Q:
                tmp_syn_layers = self._construct_synapse_annotation_layer(
                    tmp_dir,
                    tmp_list,
                    assign_random_color_Q=assign_random_color_Q,
                )
                syn_ann_layer |= (
                    {f"{key}{tmp_dir[0]}": value for key, value in tmp_syn_layers.items()}
                    if add_prefix_Q
                    else tmp_syn_layers
                )
            tmp_conn_layers = self._construct_connected_neuron_layer(tmp_dir, tmp_list)
            conn_neuron_layer |= (
                {f"{key}{tmp_dir[0]}": value for key, value in tmp_conn_layers.items()}
                if add_prefix_Q
                else tmp_conn_layers
            )

        return self.vis_self_w_ids_in_ng(
            conn_neuron_layer,
            anno_layers=syn_ann_layer,
            open_url_Q=open_url_Q,
            return_payload_Q=return_payload_Q,
        )

    def vis_self_w_ids_in_ng(
        self,
        nb_id,
        anno_layers=None,
        same_color_Q: bool = True,
        open_url_Q: bool = True,
        match_id_anno_color_Q: bool = True,
        return_payload_Q: bool = False,
    ):
        """Return a Neuroglancer URL for self plus connected assigned IDs."""
        if not hasattr(self.dataset, "ng_url_for_ids"):
            raise NotImplementedError(
                "Neuroglancer URLs require a dataset that implements ng_url_for_ids"
            )

        vis_id = {f"Self {self.cell_type}": self._id_array}
        if isinstance(nb_id, dict):
            vis_id |= {
                str(key): np.atleast_1d(np.asarray(value)).flatten()
                for key, value in nb_id.items()
            }
        else:
            vis_id["connected"] = np.atleast_1d(np.asarray(nb_id)).flatten()

        return self.dataset.ng_url_for_ids(
            vis_id,
            anno_layers=anno_layers,
            same_color_Q=(same_color_Q and match_id_anno_color_Q),
            open_url_Q=open_url_Q,
            return_payload_Q=return_payload_Q,
        )

    def vis_self_w_rids_in_ng(
        self,
        nb_rid,
        anno_layers=None,
        same_color_Q: bool = True,
        open_url_Q: bool = True,
        match_id_anno_color_Q: bool = True,
        return_payload_Q: bool = False,
    ):
        """Compatibility wrapper that accepts dataset RIDs and converts to IDs."""
        def rid_to_id_array(rid):
            rid_arr = np.atleast_1d(np.asarray(rid)).flatten().astype(np.int64, copy=False)
            ids = self.dataset.cell.rid_to_id_lookup.get_value(rid_arr, not_found_val=-1)
            ids = np.asarray(ids)
            if np.any(ids < 0):
                raise ValueError(f"Some RIDs do not exist in this dataset: {rid}")
            return ids

        if isinstance(nb_rid, dict):
            nb_id = {
                str(key): rid_to_id_array(value)
                for key, value in nb_rid.items()
            }
        else:
            nb_id = rid_to_id_array(nb_rid)
        return self.vis_self_w_ids_in_ng(
            nb_id,
            anno_layers=anno_layers,
            same_color_Q=same_color_Q,
            open_url_Q=open_url_Q,
            match_id_anno_color_Q=match_id_anno_color_Q,
            return_payload_Q=return_payload_Q,
        )


class Neuron(NeuronBase):
    """IO-focused neuron view with one assigned integer ID."""

    def __init__(
        self,
        id: int,
        dataset: ConnectomeDataset,
        remove_autapse_Q: bool = False,
        min_syn_per_rid: int = 1,
        load_xyz_Q: bool = False,
        parse_io_Q: bool = True,
    ):
        if not isinstance(id, (int, np.integer)):
            raise TypeError("Neuron id must be an integer assigned id")
        super().__init__(
            int(id),
            dataset,
            remove_autapse_Q=remove_autapse_Q,
            min_syn_per_rid=min_syn_per_rid,
            load_xyz_Q=load_xyz_Q,
            parse_io_Q=parse_io_Q,
        )


def _parse_synapse_data_with_xyz(
    id,
    dataset: ConnectomeDataset,
    side: str,
    rid_col: int,
    remove_autapse_Q: bool = False,
    min_syn_per_rid: int = 1,
):
    id = np.atleast_1d(np.asarray(id)).flatten()
    synapses = dataset.synapse.get_for_ids(id, side, return_type="array")
    self_mask = np.isin(synapses[:, rid_col], id) if synapses.size else np.zeros(0, dtype=bool)

    if remove_autapse_Q:
        synapses = synapses[~self_mask]
    elif rid_col == 0 and self_mask.any():
        print(f"Neuron {id} has {self_mask.sum()} auto-synapses")

    if min_syn_per_rid > 1 and synapses.size:
        selected_Q = select_val_by_num_repeat(synapses[:, rid_col], num_repeat=min_syn_per_rid)
        synapses = synapses[selected_Q]

    connected_id = synapses[:, rid_col].astype(np.int64, copy=False)
    out = {
        "rid": dataset.cell.id_to_rid_lookup.get_value(connected_id, not_found_val=-1),
        "xyz": synapses[:, 2:5],
    }
    out |= dataset._group_root_ids(out["rid"])
    return out


def _connected_group_counts(
    count,
    ids,
    remove_autapse_Q: bool = False,
    min_syn_per_rid: int = 1,
):
    connected_id = np.nonzero(count >= min_syn_per_rid)[0].astype(np.int64, copy=False)
    conn_num_syn = count[connected_id]
    if remove_autapse_Q:
        keep_Q = ~np.isin(connected_id, ids)
        connected_id = connected_id[keep_Q]
        conn_num_syn = conn_num_syn[keep_Q]
    return connected_id, conn_num_syn
