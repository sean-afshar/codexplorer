# connexplorer on-disk schema (v1)

The contract lives in `src/connexplorer/schema.py`. This page is the
human-readable version. A dataset is a directory:

```
data/<dataset>/
  raw/          vendor downloads; never read at runtime
  tables/       the only runtime input
    manifest.json
    cells.parquet  types.parquet  edges.parquet  synapses.parquet  type_matrix.npz
    edges_by_neuropil.parquet  synapses_by_post.parquet  columns.parquet   (optional)
    skeletons/    SWC files + skeleton_index.parquet                       (optional)
```

## Tables

Required columns must exist with exactly the listed polars dtype and contain
no nulls (unless marked `?`). Optional columns may be absent; when present the
dtype must match and nulls are allowed. Extra dataset-specific columns are
fine. No string column may hold a sentinel such as `"NAN"` or `"Unknown"`;
ingestion turns those into real nulls.

| table | required | optional | sorted by | present |
|---|---|---|---|---|
| `cells` | `id u32` (row position), `root_id i64` | `type`, `side` (`L`/`R`/`M`), `nt`, `nt_score f32`, `superclass`, `class`, `subclass`, `hemilineage`, `flow` | `id` | always |
| `types` | `type str`, `type_idx u32` (row position), `n_cells u32` | `nt`, `nt_source` (`prediction`/`literature`), `nt_predicted` | `type_idx` | always |
| `edges` | `pre u32`, `post u32`, `n_syn u32` | | `pre, post` | always |
| `edges_by_neuropil` | `pre`, `post`, `neuropil str?`, `n_syn` | | `pre, post` | when the dataset has neuropil labels |
| `synapses` | `pre u32`, `post u32`, `x_nm u32`, `y_nm u32`, `z_nm u32` | `neuropil` | `pre` | always |
| `synapses_by_post` | same as `synapses` | same | `post` | by default; skippable |
| `columns` | `id u32`, `p i16`, `q i16`, `side str` | | `id` | optical-lobe datasets |

Neurotransmitter values are one of `ACH GABA GLUT DA SER OCT HIS`.

`type_matrix.npz` holds a CSR matrix (`data u64`, `indices i32`, `indptr i32`,
`shape`, `type_list`) whose axes are `types.parquet` row order.

## Invariants

`schema.check_invariants(tables_dir)` checks and the manifest records:

- `cells.id == arange(N)`, `root_id` unique.
- Every `pre`/`post`/`id` in every table is `< N`.
- `edges` has no duplicate pairs and equals `synapses` grouped by `(pre, post)`;
  hence `edges.n_syn.sum() == len(synapses)`.
- `edges_by_neuropil` sums to `edges` per pair.
- `synapses` is sorted by `pre` and written in 1,048,576-row row groups with
  disjoint, monotone `pre` ranges (so a lazy scan prunes to one group).
  `synapses_by_post`, when present, has the same rows sorted by `post`.
- `types.type_idx == arange`, `n_cells` matches `cells`, and any type labelled
  `prediction` has `nt == nt_predicted`.
- `type_matrix.sum()` equals the edge sum between typed cells; the untyped
  remainder is reported as a count, not an error.
- Autapses are kept in the data and counted; cells with no edges are counted.

## Manifest

```json
{
  "dataset": "flywire", "version": "783", "schema_version": 1,
  "created": "...", "source_files": {"neurons.csv.gz": {"path": "...", "sha256": "...", "bytes": 0}},
  "tables": {"cells": {"file": "cells.parquet", "rows": 139255}},
  "invariants": {"problems": [], "counts": {"n_autapses": 138971}},
  "nt_overrides_version": "flywire-2026-09", "extras": {}
}
```

Opening a dataset reads only the manifest; a `schema_version` mismatch raises
with the rebuild command.

## Neurotransmitter override layer

Each ingestion `Source` owns `NT_OVERRIDES: dict[str, str]` with a citation
per entry. Ingestion writes `nt_predicted` (majority vote over cells with a
non-null prediction), then `nt` = override if present else `nt_predicted`, and
`nt_source` accordingly. FlyWire starts from cex's 13 entries (R1-6, R7, R8
to HIS; Dm1/6/8/9/12/16/17/19/20 and T1 to GLUT); MCNS starts empty.
