"""Generate per-cell input/output statistics from normalized datasets."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from ..dataset import paths, schema
from ..dataset import ConnectomeDataset, get_dataset
from ..util.tables import write_table

COUNT_KEYS = ("#c", "#ec", "#s", "#s_w1c")
STAT_COLUMNS = (
    "type",
    "f_c",
    "n_c",
    "cv_c",
    "en_c",
    "cv_en_c",
    "N_s",
    "F_s",
    "n_s",
    "n_s_mw1c",
    "cv_s",
    "n_spc",
    "cv_spc",
    "n_c_spc",
    "cv_c_spc",
    "NT",
)
_WORKER_DATASET = None


def compute_per_cell_io_counts(
    cell_type,
    dataset: ConnectomeDataset,
    remove_autapse_Q: bool = True,
    min_syn_per_rid: int = 1,
    max_cells: int | None = None,
    side: str | None = None,
):
    """Return per-cell input/output count arrays for one cell type."""
    cell_ids = dataset.get_cell_type_id(cell_type, side=side)
    if max_cells is not None:
        cell_ids = cell_ids[:max_cells]
    if cell_ids.size == 0:
        raise ValueError(f"No cells found for cell type {cell_type!r}")

    init_fun = lambda: np.zeros(cell_ids.size)
    stat = {key: {"input": defaultdict(init_fun), "output": defaultdict(init_fun)} for key in COUNT_KEYS}

    for row_idx, cell_id in enumerate(cell_ids):
        for direction in ("input", "output"):
            data = _type_counts_for_cell(
                dataset,
                cell_id,
                direction,
                dataset.cell.cell_types,
                remove_autapse_Q=remove_autapse_Q,
                min_syn_per_rid=min_syn_per_rid,
            )
            for type_idx, conn_type in enumerate(data["type_u"]):
                conn_type = str(conn_type)
                if conn_type.strip().lower() in schema.UNKNOWN_TYPE_NAMES:
                    continue
                stat["#c"][direction][conn_type][row_idx] = data["t_num_c"][type_idx]
                stat["#ec"][direction][conn_type][row_idx] = data["t_enum_c"][type_idx]
                stat["#s"][direction][conn_type][row_idx] = data["t_num_s"][type_idx]
                stat["#s_w1c"][direction][conn_type][row_idx] = data["t_num_s_w1c"][type_idx]

    return stat


def compute_per_cell_io_stat(
    cell_count: dict,
    eff_cell_count: dict,
    syn_count: dict,
    syn_w1c_count: dict,
) -> dict[str, pd.DataFrame]:
    """Compute per-direction summary statistics from per-cell count arrays."""
    per_cell_stat = {}
    for direction in cell_count:
        rows = []
        for conn_type in cell_count[direction]:
            num_cell = cell_count[direction][conn_type]
            eff_num_cell = eff_cell_count[direction][conn_type]
            num_syn = syn_count[direction][conn_type]
            num_syn_w1c = syn_w1c_count[direction][conn_type]
            syn_per_cell = np.divide(
                num_syn,
                num_cell,
                out=np.full_like(num_syn, 0, dtype=np.float32),
                where=num_cell != 0,
            )
            row = {
                "type": conn_type,
                "f_c": (num_cell > 0).mean(),
                "n_c": num_cell.mean(),
                "cv_c": _cv(num_cell),
                "en_c": eff_num_cell.mean(),
                "cv_en_c": _cv(eff_num_cell),
                "N_s": num_syn.sum(),
                "n_s": num_syn.mean(),
                "n_s_mw1c": num_syn_w1c.mean(),
                "cv_s": _cv(num_syn),
                "n_spc": syn_per_cell.mean(),
                "cv_spc": _cv(syn_per_cell),
            }
            connected_syn_per_cell = syn_per_cell[syn_per_cell > 0]
            row["n_c_spc"] = connected_syn_per_cell.mean() if connected_syn_per_cell.size else np.nan
            row["cv_c_spc"] = _cv(connected_syn_per_cell)
            rows.append(row)

        table = pd.DataFrame(rows)
        if not table.empty:
            table.sort_values(by="N_s", ascending=False, inplace=True, ignore_index=True)
        per_cell_stat[direction] = table
    return per_cell_stat


def finalize_io_stat_tables(
    dataset: ConnectomeDataset,
    per_cell_stat: dict[str, pd.DataFrame],
    min_frac: float = 1e-2,
    filter_directions=("input",),
) -> dict[str, pd.DataFrame]:
    """Add derived columns and optional notebook row filtering."""
    nt_by_type = dataset.cell.type_to_neurotransmitter
    tables = {}
    for direction, table in per_cell_stat.items():
        out = table.copy()
        if out.empty:
            tables[direction] = _empty_stat_table()
            continue
        if direction in filter_directions:
            out = out[out["f_c"] > min_frac].copy()
        if out.empty:
            tables[direction] = _empty_stat_table()
            continue
        total_syn = out["N_s"].sum()
        out["F_s"] = out["N_s"] / total_syn if total_syn > 0 else np.nan
        out["NT"] = [nt_by_type.get(str(cell_type), schema.DEFAULT_UNKNOWN) for cell_type in out["type"]]
        out = out.loc[:, [col for col in STAT_COLUMNS if col in out.columns]]
        tables[direction] = out.reset_index(drop=True)
    return tables


def write_cell_type_io_stat(
    dataset: ConnectomeDataset,
    cell_type,
    tables,
    stat_type: str = paths.DEFAULT_IO_STAT_TYPE,
) -> dict[str, str]:
    """Write per-direction IO-stat CSVs using canonical dataset paths."""
    written = {}
    for direction, table in tables.items():
        fp = dataset.paths.fp_cell_type_stat_file(cell_type, stat_type=stat_type, dir=direction)
        write_table(table, fp)
        written[direction] = fp
    dataset.clear_io_stat_cache()
    return written


def compute_cell_type_io_stat(
    cell_type,
    dataset: ConnectomeDataset,
    stat_type: str = paths.DEFAULT_IO_STAT_TYPE,
    remove_autapse_Q: bool = True,
    min_syn_per_rid: int = 1,
    min_frac: float = 1e-2,
    max_cells: int | None = None,
    side: str | None = None,
    write_Q: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compute and optionally write IO-stat tables for one cell type."""
    counts = compute_per_cell_io_counts(
        cell_type,
        dataset,
        remove_autapse_Q=remove_autapse_Q,
        min_syn_per_rid=min_syn_per_rid,
        max_cells=max_cells,
        side=side,
    )
    per_cell_stat = compute_per_cell_io_stat(counts["#c"], counts["#ec"], counts["#s"], counts["#s_w1c"])
    tables = finalize_io_stat_tables(dataset, per_cell_stat, min_frac=min_frac)
    if write_Q:
        write_cell_type_io_stat(dataset, cell_type, tables, stat_type=stat_type)
    return tables


def compute_all_io_stats(
    dataset: ConnectomeDataset,
    cell_types=None,
    stat_type: str = paths.DEFAULT_IO_STAT_TYPE,
    remove_autapse_Q: bool = True,
    min_syn_per_rid: int = 1,
    min_frac: float = 1e-2,
    num_workers: int = 1,
    write_Q: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Compute per-cell IO stats for many cell types.

    `num_workers > 1` uses processes. Each process loads its own dataset
    instance, so use multiprocessing only when the machine has enough memory for
    replicated connectivity matrices.
    """
    if cell_types is None:
        cell_types = dataset.cell.cell_type_info.type.values.astype(str)
    cell_types = [
        str(cell_type) for cell_type in cell_types
        if str(cell_type).strip().lower() not in schema.UNKNOWN_TYPE_NAMES
    ]

    dataset.connectivity.count_matrix_csr
    dataset.connectivity.count_matrix_csc
    dataset.cell.cell_types

    kwargs = dict(
        stat_type=stat_type,
        remove_autapse_Q=remove_autapse_Q,
        min_syn_per_rid=min_syn_per_rid,
        min_frac=min_frac,
        write_Q=write_Q,
    )
    if num_workers <= 1:
        return {cell_type: compute_cell_type_io_stat(cell_type, dataset, **kwargs) for cell_type in cell_types}

    results = {}
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker_dataset,
        initargs=(dataset.name, dataset.paths.fp_root),
    ) as pool:
        futures = {
            pool.submit(_compute_cell_type_io_stat_worker, cell_type, kwargs): cell_type
            for cell_type in cell_types
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def _type_counts_for_cell(
    dataset: ConnectomeDataset,
    cell_id: int,
    direction: str,
    id_to_type_arr: np.ndarray,
    remove_autapse_Q: bool,
    min_syn_per_rid: int,
):
    connected_id, conn_num_syn = dataset.connectivity.connected_ids_and_counts(
        cell_id,
        direction,
        remove_autapse_Q=remove_autapse_Q,
        min_syn_per_rid=min_syn_per_rid,
    )
    return _compute_type_counts_fast(connected_id, conn_num_syn, id_to_type_arr)


def _compute_type_counts_fast(connected_id, conn_num_syn, id_to_type_arr: np.ndarray) -> dict:
    """Compute per-type aggregates without per-neuron index dictionaries."""
    connected_id = np.asarray(connected_id, dtype=np.int64)
    conn_num_syn = np.asarray(conn_num_syn)
    if connected_id.size == 0:
        return _empty_group_info()

    conn_types = id_to_type_arr[connected_id]
    type_u, inverse = np.unique(conn_types, return_inverse=True)
    t_num_c = np.bincount(inverse).astype(np.uint32, copy=False)
    t_num_s = np.bincount(inverse, weights=conn_num_syn).astype(np.uint32, copy=False)
    t_num_s_w1c = np.zeros(type_u.size, dtype=np.uint32)
    np.maximum.at(t_num_s_w1c, inverse, conn_num_syn.astype(np.uint32, copy=False))
    t_enum_c = np.divide(
        t_num_s,
        t_num_s_w1c,
        out=np.zeros(type_u.size, dtype=np.float64),
        where=t_num_s_w1c != 0,
    )

    sort_idx = np.argsort(t_num_s)[::-1]
    return {
        "type_u": type_u[sort_idx],
        "t_num_c": t_num_c[sort_idx],
        "t_enum_c": t_enum_c[sort_idx],
        "t_num_s": t_num_s[sort_idx],
        "t_num_s_w1c": t_num_s_w1c[sort_idx],
    }


def _init_worker_dataset(dataset_name: str, fp_root: str) -> None:
    global _WORKER_DATASET
    _WORKER_DATASET = get_dataset(dataset_name, fp_root=fp_root)


def _compute_cell_type_io_stat_worker(cell_type: str, kwargs: dict) -> dict[str, pd.DataFrame]:
    if _WORKER_DATASET is None:
        raise RuntimeError("Worker dataset was not initialized")
    return compute_cell_type_io_stat(cell_type, _WORKER_DATASET, **kwargs)


def _cv(values) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return np.nan
    mean_val = values.mean()
    if mean_val <= 0:
        return np.nan
    return float(values.std() / mean_val)


def _empty_group_info() -> dict:
    return {
        "type_u": np.zeros(0, dtype=str),
        "t_num_c": np.zeros(0, dtype=np.uint32),
        "t_enum_c": np.zeros(0, dtype=np.float64),
        "t_num_s": np.zeros(0, dtype=np.uint32),
        "t_num_s_w1c": np.zeros(0, dtype=np.uint32),
    }


def _empty_stat_table() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype="object") for col in STAT_COLUMNS})
