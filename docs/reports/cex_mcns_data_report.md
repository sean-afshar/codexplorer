# cex MCNS (v0.9) normalized-data build report

Date: 2026-09-10. Machine: macOS, 48 GB RAM. Env: `/Users/shayanafshar/Code/codexplorer/.venv` (pandas 3.0.5, pyarrow 25.0.1, numpy 2.5.2, scipy 1.18.1).

## Result

All seven notebook outputs were built and validated through the public `cex` API. Nothing was
skipped. Total wall time ~1.5 min, peak RSS 12.5 GB (vs. an estimated >35 GB for the notebook as
written; see "Gaps").

```python
from cex import get_dataset
ds = get_dataset("mcns", "/Users/shayanafshar/Code/codexplorer/data/mcns")
```

Files:

- Script: `/private/tmp/claude-501/-Users-shayanafshar-Code-codexplorer/4072b4ec-7f81-48a6-8191-113d24e8b3ea/scratchpad/preprocess_mcns.py`
- Validation: `.../scratchpad/validate_mcns.py` -> `.../scratchpad/validate_mcns_result.json`
- Stage timings: `.../scratchpad/preprocess_mcns_timing.json`; run log `.../scratchpad/preprocess_mcns_full.log`
- Outputs: `/Users/shayanafshar/Code/codexplorer/data/mcns/preprocessed/` (1.1 GB total)

## Exact steps

1. Dumped all cells of `notebooks/cex/preprocessing/01_preprocess_mcns_tables.ipynb`; read
   `src/cex/preprocessing/raw.py`, `dataset/paths.py`, `dataset/schema.py`, `preprocessing/io_stat.py`,
   `dataset/{dataset,cell,synapse,column,connectivity,stats}.py`, `util/{core,tables}.py`, `docs/cex_data_layout.md`.
2. Confirmed the notebook's write paths: every `rt.write_table` targets `fp_preprocess`; the only path that
   touches `download_data` is `dprep.download_and_extract`, gated by `DOWNLOAD_URL = None`. Reads go through
   `mcns_files` (download_data) only.
3. Determined input needs: the notebook reads only `body_annotations`, `body_neurotransmitters`, and
   `syn_partners`. `syn_points`, `tbar_neurotransmitters`, `connectome_weights`, `body_stats` are listed in
   `MCNS_DOWNLOAD_FILES` but used by nothing in `src/` or the notebook. So the three missing files block nothing.
4. Created `/Users/shayanafshar/Code/codexplorer/data/mcns/download_data/` with `ln -s` symlinks to the four
   read-only files in `~/Code/connect_tome/data/male_cns/` (no copies). Source mtimes verified unchanged afterwards.
5. Wrote `preprocess_mcns.py`, reproducing the notebook cell-for-cell, except that the synapse stage streams the
   6.8 GB feather in pyarrow record-batch chunks (32 batches = 2,097,152 rows per chunk, 149 chunks), calling the
   same library functions per chunk (`build_mcns_synapse_source_data` -> `normalize_synapse_table`), then
   concatenates and does one global stable `np.lexsort((post_id, pre_id))`. Edge counts are a run-length pass over
   the sorted arrays; equivalence with `dprep.build_connectivity_edges` was asserted on the first 3M rows (True).
6. Dry run (`--no-write --max-chunks 3`), then full run with `--check-weights` in the background.
7. IO stats via `io_stat.compute_all_io_stats(ds, cell_types=["Mi1","T4a","Dm9"], num_workers=1, write_Q=True)`
   plus the notebook's combined photoreceptor step (`compute_cell_type_io_stat` for `R7`, `R8`).
8. Validation script exercising `get_dataset`, matrices, synapse table, type conversion, `counts_between_types`,
   columns, and `load_cell_type_io_stat`.

## Per-stage wall time and memory

`peak RSS` is `ru_maxrss` (bytes on macOS), cumulative for the process.

| Stage | Wall | Cumulative peak RSS | Notes |
| --- | ---: | ---: | --- |
| cell_data | 0.1 s | 0.50 GB | 211,743 rows; 47,252 null `type` |
| type_data | 2.4 s | 0.51 GB | 11,732 types |
| columns_data | 0.0 s | 0.51 GB | 23,720 column cells |
| synapses: stream+normalize | 56.5 s | 10.28 GB | 311,833,243 rows in; 125,351,887 kept; 186,481,356 dropped (pre or post body not in annotations); 0 non-positive body IDs; 0 ValueErrors |
| synapses: assemble+write | 6.0 s | 10.28 GB | parquet write |
| cell_to_cell_syn_count | 1.1 s | 11.33 GB | 26,036,056 edges |
| type_to_type_syn_count | 0.8 s | 11.33 GB | |
| check vs connectome_weights (extra) | 22.4 s | 12.49 GB | exact match, see below |
| io_stat subset (5 types) | 1.5 s | 12.49 GB | |
| **Total** | **~91 s** | **12.49 GB** | |

Validation script: loading `count_matrix_csr` 0.18 s; `synapse_table` (125M rows) 0.35 s; peak RSS 7.4 GB.

## Output files and schemas

| File | Size | Rows | Schema |
| --- | ---: | ---: | --- |
| `cell_data.parquet` | 2.68 MB | 211,743 | `id:uint32, rid:uint64, type:string (null for 47,252), side:int8 (-1: 75,370; 1: 75,233; 0: 61,140)` |
| `type_data.parquet` | 0.14 MB | 11,732 | `type:string, num_cells:int64, nt:string, superclass:string, flywire_type:string`; nt counts ACH 5953, GABA 2778, GLUT 2381, NAN 518, DA 58, HIS 16, OCT 15, SER 13 |
| `columns_data.npz` | 0.17 MB | 23,720 | `id:uint32, side:int8, p:uint8, q:uint8, pq_min:int16[2]=[1,1], pq_max:int16[2]=[36,39]` |
| `synapses.parquet` | 1007.65 MB | 125,351,887 | `pre_id:uint32, post_id:uint32, x_nm:uint32, y_nm:uint32, z_nm:uint32`; sorted by (pre_id, post_id); x 20,152..744,400 nm, y 38,304..569,300, z 53,232..1,075,620 (voxel size 8 nm, center = mean of pre/post voxel coords) |
| `cell_to_cell_syn_count.parquet` | 81.88 MB | 26,036,056 | `pre_id:uint32, post_id:uint32, num_syn:uint16`; sum 125,351,887 |
| `type_to_type_syn_count.npz` | 62.8 MB | 11,732 x 11,732 | `type_list:<U32[11732], data:uint64[3,824,967], indices:int64, indptr:int64[11733], shape:int64[2]`; total 122,233,047 (excludes the 3,118,840 synapses touching untyped cells) |
| `stat/basic_pc/{input,output}/{Mi1,T4a,Dm9,R7,R8}.csv` | 8-75 KB each | | standard `io_stat.STAT_COLUMNS` |

## Validation (public API)

```
ds.name, ds.version                         -> ("mcns", "v0.9")
ds.cell.tot_num_cells                       -> 211743
ds.cell.cell_type_info.shape                -> (11732, 5)   columns: type, num_cells, nt, superclass, flywire_type
ds.connectivity.count_matrix_csr            -> shape (211743, 211743), nnz 26036056, sum 125351887, dtype uint16
ds.connectivity.type_count_matrix_csr       -> shape (11732, 11732), nnz 3824967, sum 122233047
ds.synapse.synapse_table.shape              -> (125351887, 5)
ds.get_cell_type_id("Mi1").size             -> 1773     (T4a 1684, Dm9 273, R7 1300 [combined subtypes], R8 1329, L1 1776)
ds.ct_mcns2fw("Mi1")                        -> "Mi1"    (T4a->"T4a", Dm9->"Dm9", Dm8a->"Dm8a", R7->"R7", TmY5a->"TmY5a")
ds.ct_fw2mcns("Mi1")                        -> "Mi1"
ds.connectivity.counts_between_types(["Mi1"], ["T4a"])["matrix"]  -> shape (1773, 1684), sum 115153, nnz 11336; pre/post NT ACH/ACH
ds.connectivity.counts_between_types(["Mi1","Dm9"], ["T4a","Mi1"])["matrix"] -> shape (2046, 3457), sum 123820
ds.connectivity.num_syn_between_type_groups(["Mi1"], ["T4a"]).sum() -> 115153   (consistent with cell-level)
ds.column.column_table.shape                -> (23720, 5)  columns id, side, p, q, rid (rid derived from id)
ds.load_cell_type_io_stat("T4a", dir="input").head(3) -> Mi1 (N_s 115153, ACH), Tm3 (50450, ACH), Mi9 (37468, GLUT)
ds.load_cell_type_io_stat("Mi1", dir="output").head(3) -> Pm2a (260568, GABA), Pm2b (229906, GABA), Pm1 (222616, GABA)
```

Extra cross-check (not in the notebook): mapping the released `connectome-weights-male-cns-v0.9-minconf-0.5.feather`
(151,871,794 body pairs) to normalized ids and keeping pairs where both bodies are annotated gives 26,036,056 pairs
with weight sum 125,351,887, and the table equals `cell_to_cell_syn_count.parquet` exactly (row-for-row, count-for-count).
So the synapse-to-edge pipeline is consistent with the upstream release.

## What was skipped and why

- Nothing required was skipped. The three absent raw files (`syn-points`, `tbar-neurotransmitters`, `body-stats`)
  are not read by the notebook or by any code under `src/`; they only appear in `paths.MCNS_DOWNLOAD_FILES`.
- IO stats were computed for 5 types only (Mi1, T4a, Dm9 + notebook's R7/R8 combined), not all 11,732, per the task
  brief. Extrapolating from 1.5 s for ~6,400 cells, the full pass (~164k typed cells) would take roughly a minute
  single-process, so `IO_STAT_NUM_WORKERS = 8` in the notebook is not needed on this machine.

## Bugs / gaps found in cex's MCNS path

1. **Notebook-only logic (no library counterpart).** The MCNS cell/type/column build exists only in notebook cells
   8, 10, 12. `raw.py` has `build_flywire_cell_data` but no `build_mcns_cell_data`; the MCNS column arrays are built
   inline rather than via `build_column_data` (whose default `kept_cols=FLYWIRE_COLUMN_COLUMNS` and `rid_col="root_id"`
   do not fit MCNS). Any script (like this one) has to copy the cells.
2. **Duplicated / unused constants.** The notebook hard-codes values that also live in `paths.py`, and the library
   versions are used by nothing: `MCNS_BODY_ANNOTATION_NAME_MAP`, `MCNS_BODY_TRANSMITTER_NAME_MAP`,
   `MCNS_TYPE_VALUE_COLUMNS` (notebook: `mcns_valud_col`, typo), `MCNS_CELL_SAVE_COLUMNS`, `MCNS_COLUMN_SOURCE_COLUMNS`,
   `MCNS_VOXEL_SIZE_NM` (notebook: `MCNS_SYN_VOXEL_SIZE_NM = 8`). Also unused anywhere: `MCNS_BODY_NEUROTRANSMITTER_COLUMNS`,
   `MCNS_VISUAL_SUPERCLASSES`, `MCNS_NT_TO_IDX`, `MCNS_VISUAL_TYPE_COLUMNS`, `MCNS_COLUMN_COLUMNS`, `MCNS_SYNAPSE_COLUMNS`.
   The script asserts the inline values equal the library constants (they do).
3. **Typo in `MCNS_VISUAL_SUPERCLASSES`:** `"visual_centrifiugal"`; the data value is `visual_centrifugal` (563 cells).
   Harmless today only because the constant is unused.
4. **Memory profile of the notebook synapse stage.** Cell 14 loads 8 columns x 311.8M rows into pandas (~12.5 GB),
   cell 15 builds float64 intermediates (~7.5 GB transient) plus a 28 B/row source frame (~8.7 GB), and cell 16 then
   inserts/drops columns and sorts in place. Estimated peak well above 35 GB on a 48 GB machine; streaming brings it to
   10.3 GB with identical output. Consider moving the chunked loop into `raw.py`.
5. **`normalize_synapse_table` mutates its input** (`drop`/`rename`/`insert` on the caller's frame). Fine in the
   notebook, surprising as a library API.
6. **`build_mcns_synapse_source_data` raises on any body id <= 0.** No such rows exist in v0.9, but if a future
   release contains body 0 (unassigned) the whole notebook stage fails instead of dropping those rows.
7. **Side codes.** `somaSide` has 392 `'M'` (midline) cells; `SIDE_TO_IDX` has no `m` key so they become unknown (0),
   indistinguishable from the 60,748 NaN. `rootSide` in the annotations is ignored.
8. **`docs/cex_data_layout.md` mismatch:** says `columns_data.npz` holds "Downloaded FlyWire column assignments only",
   but the MCNS notebook writes it from `assignedOlHex1/2` and `Column` loads it (23,720 rows) for MCNS.
9. **`type_count_matrix_csr` vs `count_matrix_csr` totals differ** (122,233,047 vs 125,351,887): synapses on the
   47,252 untyped cells are in the cell matrix but not the type matrix. This is by design (`build_type_data` skips
   unknown types) but undocumented.
10. **pandas 3 note:** string columns come back as the new `str` dtype (`large_string` in parquet). Everything in
    the loader path (`fillna("Unknown")`, `_majority_value`, `groupby`) behaved correctly, but `type` nulls are stored as
    parquet nulls, not the string `"NAN"`.
11. **`get_dataset` has side effects:** `DatasetPaths.from_name` calls `ensure_generated_folders`, creating
    `download_data/`, `preprocessed/`, `analysis/`, `preprocessed/stat/` on every construction, even read-only use.
12. Minor: `sort_table_and_add_id` sorts with NaN `type` last, so untyped cells get the highest ids (164,491..211,742).
    `Cell.cell_types` maps them to `"Unknown"`, which `io_stat` filters by `UNKNOWN_TYPE_NAMES`.

## Reproduce

```bash
cd /Users/shayanafshar/Code/codexplorer
.venv/bin/python -u /private/tmp/claude-501/-Users-shayanafshar-Code-codexplorer/4072b4ec-7f81-48a6-8191-113d24e8b3ea/scratchpad/preprocess_mcns.py --check-weights
.venv/bin/python -u /private/tmp/claude-501/-Users-shayanafshar-Code-codexplorer/4072b4ec-7f81-48a6-8191-113d24e8b3ea/scratchpad/validate_mcns.py
```

Options: `--skip-synapses` (reuse existing `synapses.parquet`), `--io-types ''` (skip IO stats), `--batches-per-chunk N`,
`--no-write`, `--max-chunks N` (debug).
