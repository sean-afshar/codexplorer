# CEX Normalized Data Layout

`cex` reads a dataset root containing `download_data/` and `preprocessed/`.
When no root is supplied, `get_dataset("flywire")` uses
`./data/flywire/`; `mcns` and `mcns_v1` use the corresponding lower-case
dataset folders. The preprocessing notebooks instead resolve their default
root from the installed CEX package, as `<codexplorer>/data/<dataset_name>/`,
so they do not depend on the notebook working directory. The `00` and `01`
notebooks can optionally download and extract a user-provided Dropbox archive
into `download_data/`, then delete the archive after successful extraction.
They do not bundle connectome data.

| File | Required columns or arrays | Purpose |
| --- | --- | --- |
| `cell_data.parquet` | `id`, `rid`, `type`, `side` | Dense assigned IDs, dataset root IDs, cell types, and side codes (`left=-1`, `right=1`). |
| `type_data.parquet` | `type`, `num_cells`, optional `nt` | Type-table order and type metadata. |
| `synapses.parquet` | `pre_id`, `post_id`, `x_nm`, `y_nm`, `z_nm` | Synapse locations in nanometers. |
| `cell_to_cell_syn_count.parquet` | `pre_id`, `post_id`, `num_syn` | Directed total counts between assigned IDs. |
| `type_to_type_syn_count.npz` | `type_list`, CSR `data`, `indices`, `indptr`, `shape` | Total counts between types in the exact `type_data.type` order. |
| `columns_data.npz` | optional `id`, `p`, `q`, `side` | Downloaded FlyWire column assignments only. |

The preprocessing notebooks create the first five files. `id` must be dense,
start at zero, and match the row order of `cell_data.parquet`; sparse matrices
use it directly as their index. `rid` remains the dataset-provided external
identifier.

Use the notebooks in `notebooks/cex/preprocessing/` in order: prepare FlyWire
or Male CNS tables, including per-cell I/O statistics. The dataset-specific `00` and `01` notebooks own the entire
preprocessing workflow.
