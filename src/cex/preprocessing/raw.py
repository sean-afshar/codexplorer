"""One-time preprocessing helpers for normalized dataset tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as spsp

from ..dataset import paths, schema
from ..util import ScalarDict, minimize_int_dtype, reorder_pd_columns

def sort_table_and_add_id(table: pd.DataFrame, sort_cols, id_col_name="id") -> pd.DataFrame:
    """Sort a table by specified columns and add a stable integer ID column."""
    sorted_table = table.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    id = np.arange(sorted_table.shape[0], dtype=np.int64)
    id = minimize_int_dtype(id, allow_bool=False)
    sorted_table.insert(0, id_col_name, id)
    return sorted_table.reset_index(drop=True)

def build_flywire_cell_data(
    cell_classification_table: pd.DataFrame,
    cell_table: pd.DataFrame,
    neurotransmitter_table: pd.DataFrame,
) -> pd.DataFrame:
    """Merge FlyWire cell source tables and assign normalized IDs."""
    delete_columns = ["additional_type", "additional_types", "additional_type(s)"]
    source_table = (
        cell_classification_table
        .merge(cell_table, on="root_id", how="left")
        .merge(neurotransmitter_table, on="root_id", how="left")
    )
    source_table.drop(columns=[c for c in delete_columns if c in source_table], inplace=True)

    rename_cols = {"root_id": "rid", "primary_type": "type"}
    source_table.rename(columns=rename_cols, inplace=True)
    source_table = sort_table_and_add_id(source_table, sort_cols=["type", "rid"], id_col_name="id")
    if "side" in source_table:
        source_table["side"] = normalize_side_values(source_table["side"])
    source_table = reorder_pd_columns(source_table, ["id", "rid", "type", "side"])
    return source_table

def build_type_data(
    cell_data: pd.DataFrame,
    value_cols,
    col_name_map: dict[str, str] | None = None,
    nt_correction: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Summarize type-level attributes by majority value."""
    rows = []
    col_name_map = col_name_map or {}
    for cell_type, subtable in cell_data.groupby("type", sort=True):
        if str(cell_type).strip().lower() in schema.UNKNOWN_TYPE_NAMES:
            continue
        row = {"type": cell_type, "num_cells": int(len(subtable))}
        for source_col in value_cols:
            if source_col not in subtable.columns:
                continue
            out_col = col_name_map.get(source_col, source_col)
            value = _majority_value(subtable[source_col])
            if out_col == "nt" and nt_correction is not None:
                value = nt_correction.get(cell_type, value)
            row[out_col] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_type_connectivity_data(
    connectivity_table: pd.DataFrame,
    cell_data: pd.DataFrame,
    type_list,
) -> dict[str, np.ndarray]:
    """Aggregate cell-level edges into a sparse type connectivity matrix.

    Args:
        connectivity_table: Cell-level edge table with ``pre_id``, ``post_id``,
            and ``num_syn`` columns.
        cell_data: Normalized cell table whose ``id`` values index the edges.
        type_list: Ordered cell types used for both matrix axes.

    Returns:
        CSR arrays and ``type_list``. Matrix entry ``(i, j)`` is the total
        number of synapses from type ``type_list[i]`` to ``type_list[j]``.
    """
    type_list = np.asarray(type_list).astype(str)
    type_to_idx = {cell_type: i for i, cell_type in enumerate(type_list)}
    id_to_type_idx = np.full(cell_data["id"].max() + 1, -1, dtype=np.int64)
    id_to_type_idx[cell_data["id"].to_numpy()] = cell_data["type"].map(type_to_idx).fillna(-1).to_numpy(dtype=np.int64)
    pre_idx = id_to_type_idx[connectivity_table["pre_id"].to_numpy()]
    post_idx = id_to_type_idx[connectivity_table["post_id"].to_numpy()]
    valid_Q = (pre_idx >= 0) & (post_idx >= 0)
    matrix = spsp.coo_array(
        (connectivity_table.loc[valid_Q, "num_syn"].to_numpy(dtype=np.uint64), (pre_idx[valid_Q], post_idx[valid_Q])),
        shape=(type_list.size, type_list.size),
    ).tocsr()
    return {
        "type_list": type_list, "data": matrix.data, "indices": matrix.indices,
        "indptr": matrix.indptr, "shape": np.asarray(matrix.shape, dtype=np.int64),
    }


def build_column_data(
    column_table: pd.DataFrame,
    cell_data: pd.DataFrame,
    rid_col: str = "root_id",
    side_cols=("side", "hemisphere"),
    kept_cols=paths.FLYWIRE_COLUMN_COLUMNS,
) -> dict[str, np.ndarray]:
    """Build compact normalized column assignment arrays."""
    columns = _add_cell_id_column(column_table, cell_data, rid_col=rid_col, keep_unknown_Q=True)
    for side_col in side_cols:
        if side_col in columns:
            columns["side"] = normalize_side_values(columns[side_col])
            if side_col != "side":
                columns.drop(columns=[side_col], inplace=True)
            break
    data = {
        col: minimize_int_dtype(columns[col].to_numpy(), allow_bool=False)
        for col in kept_cols
        if col in columns
    }
    if {"p", "q"}.issubset(data):
        data["pq_min"] = np.asarray([data["p"].min(), data["q"].min()], dtype=np.int16)
        data["pq_max"] = np.asarray([data["p"].max(), data["q"].max()], dtype=np.int16)
    return data


def normalize_synapse_table(
    synapse_table: pd.DataFrame,
    cell_data: pd.DataFrame,
    pre_rid_col: str,
    post_rid_col: str,
    x_col: str,
    y_col: str,
    z_col: str,
    rid_add: int = 0,
    require_known_Q: bool = True,
    keep_unknown_synapses_Q: bool = False,
) -> pd.DataFrame:
    """Map source synapse RIDs to normalized pre/post IDs."""
    rid_to_id = ScalarDict(
        cell_data["rid"].to_numpy(),
        cell_data["id"].to_numpy(dtype=np.int64),
    )
    pre_rid = synapse_table[pre_rid_col].to_numpy(copy=False)
    post_rid = synapse_table[post_rid_col].to_numpy(copy=False)
    if rid_add:
        pre_rid = pre_rid + rid_add
        post_rid = post_rid + rid_add
    pre_id = rid_to_id.get_value(pre_rid, not_found_val=-1)
    post_id = rid_to_id.get_value(post_rid, not_found_val=-1)
    if require_known_Q:
        if np.any(pre_id < 0):
            raise ValueError("Some presynaptic RIDs do not have corresponding normalized cell IDs")
        if np.any(post_id < 0):
            raise ValueError("Some postsynaptic RIDs do not have corresponding normalized cell IDs")
    known_Q = (pre_id >= 0) & (post_id >= 0)
    if not keep_unknown_synapses_Q:
        synapse_table.drop(index=synapse_table.index[~known_Q], inplace=True)
        pre_id = pre_id[known_Q]
        post_id = post_id[known_Q]
    rename_cols = {x_col: "x_nm", y_col: "y_nm", z_col: "z_nm"}
    synapse_table.rename(columns={k: v for k, v in rename_cols.items() if k != v}, inplace=True)
    synapse_table.insert(0, "post_id", minimize_int_dtype(post_id, allow_bool=False))
    synapse_table.insert(0, "pre_id", minimize_int_dtype(pre_id, allow_bool=False))
    drop_cols = [c for c in [pre_rid_col, post_rid_col] if c in synapse_table]
    synapse_table.drop(columns=drop_cols, inplace=True)
    xyz_col = ["x_nm", "y_nm", "z_nm"]
    synapse_table[xyz_col] = minimize_int_dtype(
        synapse_table[xyz_col].to_numpy(copy=False), allow_bool=False)
    keep_cols = ["pre_id", "post_id", "x_nm", "y_nm", "z_nm"]
    extra_cols = [c for c in synapse_table.columns if c not in keep_cols]
    if extra_cols:
        synapse_table.drop(columns=extra_cols, inplace=True)
    synapse_table.sort_values(["pre_id", "post_id"], inplace=True, ignore_index=True)
    return synapse_table


def build_mcns_synapse_source_data(synapse_partner_table: pd.DataFrame, voxel_size_nm=8) -> pd.DataFrame:
    """Build MCNS synapse source rows from partner coordinates and body IDs."""
    if synapse_partner_table["body_pre"].min() <= 0 or synapse_partner_table["body_post"].min() <= 0:
        raise ValueError("MCNS synapse partner body IDs must be positive")
    if (
        synapse_partner_table["body_pre"].max() >= 2**31
        or synapse_partner_table["body_post"].max() >= 2**31
    ):
        raise ValueError("MCNS synapse partner body IDs exceed int32 range")
    ctr_x = (synapse_partner_table["x_pre"].to_numpy() + synapse_partner_table["x_post"].to_numpy()) / 2 * voxel_size_nm
    ctr_y = (synapse_partner_table["y_pre"].to_numpy() + synapse_partner_table["y_post"].to_numpy()) / 2 * voxel_size_nm
    ctr_z = (synapse_partner_table["z_pre"].to_numpy() + synapse_partner_table["z_post"].to_numpy()) / 2 * voxel_size_nm

    return pd.DataFrame(
        {
            "pre_root_id": synapse_partner_table["body_pre"].to_numpy(),
            "post_root_id": synapse_partner_table["body_post"].to_numpy(),
            "ctr_x": ctr_x.astype(np.int32),
            "ctr_y": ctr_y.astype(np.int32),
            "ctr_z": ctr_z.astype(np.int32),
        }
    )

def build_connectivity_edges(normalized_synapses: pd.DataFrame) -> pd.DataFrame:
    """Count synapses for each assigned pre/post ID pair."""
    valid_Q = (normalized_synapses["pre_id"] >= 0) & (normalized_synapses["post_id"] >= 0)
    table = (
        normalized_synapses.loc[valid_Q]
        .groupby(["pre_id", "post_id"], sort=True)
        .size()
        .rename("num_syn")
        .reset_index()
    )
    for col in table:
        table[col] = minimize_int_dtype(table[col].to_numpy(), allow_bool=False)
    return table


def normalize_io_stat_table(table: pd.DataFrame, drop_unknown_Q: bool = True) -> pd.DataFrame:
    """Normalize IO-stat fields that should not be patched at load time."""
    out = table.copy()
    if "key" in out:
        out.rename(columns={"key": "type"}, inplace=True)
    if "N_s" in out and "F_s" not in out:
        out["F_s"] = out["N_s"] / out["N_s"].sum()
    if drop_unknown_Q and "type" in out:
        type_key = out["type"].astype(str).str.strip().str.lower()
        out = out[~type_key.isin(schema.UNKNOWN_TYPE_NAMES)]
    return out.reset_index(drop=True)


def normalize_side_values(side_values: pd.Series) -> np.ndarray:
    """Normalize side labels to int8 codes: left=-1, right=1, unknown=0."""
    side = normalized_categorical_values(side_values, schema.SIDE_TO_IDX, default_val=0)
    side = side.astype(np.int8)
    return side

def normalized_categorical_values(values, dict_map, default_val=np.nan) -> np.ndarray:
    """Normalize categorical values using a mapping dictionary."""
    values = pd.Series(values).astype(str).str.strip().str.lower()
    normalized = np.full(values.shape[0], fill_value=default_val, dtype=object)
    for key, val in dict_map.items():
        normalized[values == key] = val
    return normalized

def _add_cell_id_column(
    table: pd.DataFrame,
    cell_data: pd.DataFrame,
    rid_col: str = "root_id",
    keep_unknown_Q: bool = True,
) -> pd.DataFrame:
    out = table.copy()
    if rid_col != "rid":
        out.rename(columns={rid_col: "rid"}, inplace=True)
    rid_to_id = ScalarDict(
        cell_data["rid"].to_numpy(),
        cell_data["id"].to_numpy(dtype=np.int64),
    )
    out.insert(0, "id", rid_to_id.get_value(out["rid"].to_numpy(), not_found_val=-1))
    if not keep_unknown_Q:
        out = out.loc[out["id"] >= 0].copy()
    first_cols = ["id", "rid"]
    rest_cols = [c for c in out.columns if c not in first_cols]
    return out.loc[:, first_cols + rest_cols]


def _majority_value(values) -> object:
    clean_values = pd.Series(values).dropna().to_numpy()
    if clean_values.size == 0:
        return schema.DEFAULT_UNKNOWN
    values_u, counts = np.unique(clean_values, return_counts=True)
    return values_u[np.argmax(counts)]
