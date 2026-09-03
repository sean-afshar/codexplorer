"""Dataset facade built on normalized preprocessing tables."""

from __future__ import annotations

from functools import cached_property
import os

import numpy as np

from .cell import Cell
from .column import Column
from .connectivity import Connectivity
from . import flywire_links, schema
from .paths import DatasetPaths, MCNS_RELEASE_BY_DATASET, dataset_family, normalize_dataset_name
from .stats import IOStatMixin
from .synapse import Synapse
from ..util import bin_data_to_idx_list
from ..util.tables import read_table


def get_dataset(name: str, fp_root: str | None = None):
    """Return a dataset instance by name."""
    dataset_name = normalize_dataset_name(name)
    if dataset_name == "flywire":
        return FlyWireDataset(fp_root=fp_root)
    if dataset_family(dataset_name) == "mcns":
        return MaleCNSDataset(fp_root=fp_root, name=dataset_name)
    raise ValueError(f"Unsupported dataset name: {name}")


class ConnectomeDataset(IOStatMixin):
    """Common normalized connectome dataset API."""

    def __init__(self, name: str, fp_root: str | None = None):
        self.paths = DatasetPaths.from_name(name, fp_root=fp_root)
        self.name = self.paths.name
        self.cell = Cell(self.paths.fp_preprocessed)
        self.synapse = Synapse(self.paths.fp_preprocessed, cell=self.cell)
        self.column = Column(self.paths.fp_preprocessed, cell=self.cell)
        self.connectivity = Connectivity(
            self.paths.fp_preprocessed, cell=self.cell, synapse=self.synapse, column=self.column,
            get_cell_type_id=self.get_cell_type_id)
        self._io_stat_cache = {}

    @cached_property
    def cell_type_to_type_idx(self):
        """Map cell type to type-table row index."""
        return {k: i for i, k in enumerate(self.cell.cell_type_info.type.values.astype(str))}

    def get_cell_type_id(self, cell_type: str, side: str | None = None):
        """Return assigned IDs for one cell type."""
        return self.cell.get_cell_type_id(cell_type, side=side)

    def _group_root_ids(self, rid_list, rid_num_syn=None):
        """Group RIDs by root ID and cell type, preserving legacy keys."""
        rid_list = np.atleast_1d(np.asarray(rid_list)).flatten()
        if rid_num_syn is not None:
            rid_num_syn = np.atleast_1d(np.asarray(rid_num_syn)).flatten()
            if rid_num_syn.size != rid_list.size:
                raise ValueError("weight size does not match rid_list size")

        info = {}
        info["type"] = np.asarray([self.cell.rid_to_type.get(int(k), "Unknown") for k in rid_list])
        info["type_u"] = np.unique(info["type"])
        info["rid2i"] = bin_data_to_idx_list(rid_list, return_type="dict")
        info["t2i"] = bin_data_to_idx_list(info["type"], return_type="dict")
        info["t2rid"] = {k: np.unique(rid_list[i]) for k, i in info["t2i"].items()}
        info["t2c_i"] = {k: [info["rid2i"][rid] for rid in rid_l] for k, rid_l in info["t2rid"].items()}
        if rid_num_syn is None:
            info["t2c_num_s"] = {k: np.asarray([len(c_i) for c_i in v]) for k, v in info["t2c_i"].items()}
            info["rid2ns"] = {rid: len(info["rid2i"][rid]) for rid in rid_list}
        else:
            info["t2c_num_s"] = {
                k: np.asarray([rid_num_syn[info["rid2i"][rid]].sum() for rid in rid_l])
                for k, rid_l in info["t2rid"].items()
            }
            info["rid2ns"] = {rid: rid_num_syn[info["rid2i"][rid]].sum() for rid in rid_list}
        info["t_num_s"] = np.asarray([info["t2c_num_s"][k].sum() for k in info["type_u"]], dtype=np.uint32)
        sort_idx = np.argsort(info["t_num_s"])[::-1]
        info["type_u"] = info["type_u"][sort_idx]
        info["t_num_s"] = info["t_num_s"][sort_idx]
        info["t_num_c"] = np.asarray([info["t2rid"][k].size for k in info["type_u"]], dtype=np.uint16)
        info["t_enum_c"] = np.asarray([_effective_cell_count(info["t2c_num_s"][k]) for k in info["type_u"]])
        info["t_num_s_w1c"] = np.asarray(
            [np.max(info["t2c_num_s"][k]) if info["t2c_num_s"][k].size else 0 for k in info["type_u"]],
            dtype=np.uint32,
        )
        return info


class FlyWireDataset(ConnectomeDataset):
    """Normalized FlyWire dataset."""

    version = "783"

    def __init__(self, fp_root: str | None = None):
        super().__init__("flywire", fp_root=fp_root)

    @cached_property
    def visual_types(self):
        """Return the normalized FlyWire visual-type table."""
        fp = os.path.join(self.paths.fp_preprocessed, schema.VISUAL_TYPE_DATA_FILE)
        return read_table(fp)

    def codex_url(self, rids, page_size: int = 10) -> str:
        """Return a Codex search URL for FlyWire RIDs."""
        return flywire_links.codex_url(rids, version=self.version, page_size=page_size)

    def codex_open(self, rids, page_size: int = 10) -> str:
        """Open and return a Codex search URL for FlyWire RIDs."""
        return flywire_links.codex_open(rids, version=self.version, page_size=page_size)

    def ng_url(self, rids, anno_layers=None, open_url_Q=False, same_color_Q=False, return_payload_Q=False):
        """Return a Neuroglancer URL for FlyWire RIDs."""
        return flywire_links.neuroglancer_url(
            rids,
            anno_layers=anno_layers,
            version=self.version,
            open_url_Q=open_url_Q,
            same_color_Q=same_color_Q,
            return_payload_Q=return_payload_Q,
        )

    def ng_url_for_ids(self, ids, **kwargs):
        """Return a Neuroglancer URL after converting assigned IDs to FlyWire RIDs."""
        if isinstance(ids, dict):
            rids = {
                str(key): self.cell.id_to_rid_lookup.get_value(value, not_found_val=-1)
                for key, value in ids.items()
            }
            invalid_Q = any(np.any(np.asarray(value) < 0) for value in rids.values())
        else:
            rids = self.cell.id_to_rid_lookup.get_value(ids, not_found_val=-1)
            invalid_Q = np.any(np.asarray(rids) < 0)
        if invalid_Q:
            raise ValueError("Some assigned IDs do not have corresponding FlyWire RIDs")
        return self.ng_url(rids, **kwargs)


class MaleCNSDataset(ConnectomeDataset):
    """Normalized Male CNS dataset."""

    family = "mcns"
    combined_photoreceptor_types = {
        "R7": ("R7d", "R7p", "R7y", "R7_unclear"),
        "R8": ("R8d", "R8p", "R8y", "R8_unclear"),
    }

    def __init__(self, fp_root: str | None = None, name: str = "mcns"):
        super().__init__(name, fp_root=fp_root)
        self.version = MCNS_RELEASE_BY_DATASET[self.name]

    def get_cell_type_id(self, cell_type: str, side: str | None = None):
        """Return assigned IDs, combining MCNS photoreceptor subtypes for R7/R8."""
        cell_type = str(cell_type)
        if cell_type in self.combined_photoreceptor_types:
            ids = [
                self.cell.get_cell_type_id(subtype, side=side)
                for subtype in self.combined_photoreceptor_types[cell_type]
            ]
            return np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
        return super().get_cell_type_id(cell_type, side=side)

    @cached_property
    def _type_conversion_dict(self):
        if hasattr(self.cell, "cell_type_info") and self.cell.cell_type_info is not None:
            mcns_ct_table = self.cell.cell_type_info
            mcns2fw = {}
            fw2mcns = {}
            for tmp_m, tmp_f in zip(mcns_ct_table['type'], mcns_ct_table['flywire_type']):
                tmp_f = tmp_m if tmp_m in ("Dm8a", "Dm8b") else tmp_f
                if str(tmp_f).upper() == "NAN": continue
                mcns2fw.setdefault(tmp_m, list()).append(tmp_f)
                fw2mcns.setdefault(tmp_f, list()).append(tmp_m)
            mcns2fw = {k: sorted(v) if len(v) > 1 else v[0] for k, v in mcns2fw.items()}
            assert 'R7' not in mcns2fw, "MCNS R7 photoreceptor type should not be present in the conversion table"
            mcns2fw['R7'] = ['R7']
            assert 'R8' not in mcns2fw, "MCNS R8 photoreceptor type should not be present in the conversion table"
            mcns2fw['R8'] = ['R8']
            fw2mcns = {k: sorted(v) if len(v) > 1 else v[0] for k, v in fw2mcns.items()}
        else:
            mcns2fw, fw2mcns = {}, {}
        return {"fw2mcns": fw2mcns, "mcns2fw": mcns2fw}

    def ct_fw2mcns(self, fw_type: str):
        """Convert a FlyWire type name to MCNS when known."""
        return _single_or_list(self._type_conversion_dict.get("fw2mcns", {}).get(fw_type, None))

    def ct_mcns2fw(self, mcns_type: str):
        """Convert an MCNS type name to FlyWire when known."""
        return _single_or_list(self._type_conversion_dict.get("mcns2fw", {}).get(mcns_type, None))

    def unique_flywire_type(self, cell_type: str):
        """Return the one FlyWire type matching an MCNS type, if unique."""
        fw_type = self.ct_mcns2fw(cell_type)
        if isinstance(fw_type, str):
            return fw_type
        if isinstance(fw_type, (list, tuple, np.ndarray)) and len(fw_type) == 1:
            return str(fw_type[0])
        return None

    def conn_fw2mcns(self, conn_dict):
        """Convert a connection dictionary with FlyWire types to MCNS types when known."""
        new_conn_dict = {}
        for k, v in conn_dict.items():
            new_pair = []
            if isinstance(v, str):
                v = [v]
            for t in v:
                new_t = self._single_conn_fw2mcns(t)
                if isinstance(new_t, list):
                    new_pair.extend(new_t)
                else:
                    new_pair.append(new_t)
            new_conn_dict[k] = new_pair
        return new_conn_dict

    def _single_conn_fw2mcns(self, conn_pair):
        CONN_SEP = ":"
        """Convert a single connection pair with FlyWire types to MCNS types when known."""
        t_list = []
        for i, tt in enumerate(conn_pair.split(CONN_SEP)):
            tt_mcns = self.ct_fw2mcns(tt)
            assert tt_mcns is not None, f"Cannot convert FlyWire type {tt} to MCNS type"

            if isinstance(tt_mcns, str):
                tt_mcns = [tt_mcns]
            else:
                assert isinstance(tt_mcns, list), f"Unexpected type {type(tt_mcns)} for {tt_mcns}"
            t_list.append(tt_mcns)

        new_pair = []
        for t1 in t_list[0]:
            for t2 in t_list[1]:
                new_pair.append(f"{t1}{CONN_SEP}{t2}")

        if len(new_pair) == 1:
            return new_pair[0]
        else:
            return new_pair


def _single_or_list(value):
    if value is not None and hasattr(value, "__len__") and not isinstance(value, str) and len(value) == 1:
        return value[0]
    return value


def _effective_cell_count(num_syn_per_cell) -> float:
    num_syn_per_cell = np.asarray(num_syn_per_cell)
    if num_syn_per_cell.size == 0:
        return 0.0
    if num_syn_per_cell.size == 1:
        return 1.0
    return float(np.sum(num_syn_per_cell) / np.max(num_syn_per_cell))
