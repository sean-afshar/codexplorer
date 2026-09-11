# cex FlyWire preprocessed data: download + validation report

Date: 2026-09-10. Repo: /Users/shayanafshar/Code/codexplorer (.venv/bin/python, Python 3.13).
Nothing under src/, tests/, notebooks/ or ~/Code/connect_tome was touched; nothing was git-added. All writes went to data/ (gitignored) and the scratchpad.

## 1. What was downloaded, and how

Source (from `notebooks/cex/00_basic_package_usage.ipynb`, "Optional Preprocessed Data Download" cell):

    https://www.dropbox.com/scl/fi/jo5yfjy6z9v7854f0blc7/flywire_preprocessed.zip?rlkey=<redacted>&dl=1

Pre-flight `curl -I` on the link: HTTP 302 -> 302 -> 200, `content-length: 737496477` (703 MiB). Link is live, no auth needed.

Exact call executed (from repo root, via `.venv/bin/python`; identical to the notebook cell):

```python
import cex, cex.preprocessing as dprep
from pathlib import Path
REPO_ROOT = Path(cex.__file__).resolve().parents[2]      # -> /Users/shayanafshar/Code/codexplorer
PREPROCESSED_ROOT = REPO_ROOT / "data" / "flywire" / "preprocessed"
dprep.download_and_extract(
    "https://www.dropbox.com/scl/fi/jo5yfjy6z9v7854f0blc7/flywire_preprocessed.zip?rlkey=<redacted>&dl=1",
    PREPROCESSED_ROOT,
    "flywire_preprocessed.zip",
    False,                      # overwrite_Q
    strip_single_root_Q=True,   # removes the one enclosing `preprocessed/` folder inside the zip
)
```

- Download + extraction wall time: **30.5 s** (urlretrieve of 737 MB, then unpack into a temp dir under data/flywire/ and copy into place).
- The archive was deleted after extraction (as `download_and_extract` does); no `__MACOSX`, `.DS_Store` or stray `.zip` remain under data/.
- Destination is exactly `/Users/shayanafshar/Code/codexplorer/data/flywire/preprocessed/` (files at top level, no nested `preprocessed/preprocessed`).
- `get_dataset` also auto-creates the empty sibling folders `data/flywire/download_data/` and `data/flywire/analysis/` (DatasetPaths.ensure_generated_folders); that is expected.

## 2. Where and sizes

`du -sh`: data/ = **1.5G**; data/flywire/preprocessed = 1.5G; data/flywire/preprocessed/stat = 784M.

Full listing of `data/flywire/preprocessed/`:

| path | bytes | notes |
|---|---:|---|
| cell_data.parquet | 2,874,067 | 139,255 rows x 19 cols |
| type_data.parquet | 93,235 | 8,547 rows x 10 cols |
| visual_type_data.parquet | 10,415 | 742 rows x 5 cols  (**present**) |
| columns_data.npz | 502,988 | 45,528 column-assigned cells |
| cell_to_cell_syn_count.parquet | 62,030,071 | 19,912,747 edges |
| type_to_type_syn_count.npz | 45,899,088 | 8547 x 8547 CSR |
| synapses.parquet | 637,632,155 | 80,215,790 synapses |
| stat/basic_pc/input/  | 391M | **8,547 CSVs** (one per type) |
| stat/basic_pc/output/ | 393M | **8,547 CSVs** (one per type) |

So the IO-stat CSVs **are included** (17,094 files, only the `basic_pc` stat type, both directions). No other stat types are shipped.

## 3. Load validation with `cex`

Working call:

```python
from cex import get_dataset
ds = get_dataset("flywire", "/Users/shayanafshar/Code/codexplorer/data/flywire")
# -> cex.dataset.dataset.FlyWireDataset; ds.paths.fp_preprocessed == .../data/flywire/preprocessed
```

Timings (warm OS cache, first access of each cached_property; script: scratchpad/verify.py, log: verify.log):

| access | result | time |
|---|---|---:|
| `get_dataset(...)` | FlyWireDataset (lazy, nothing read) | 0.000 s |
| `ds.cell.tot_num_cells` | **139255** | 0.073 s (reads cell_data) |
| `ds.cell.cell_type_info.shape` | **(8547, 10)** | 0.001 s |
| `ds.cell.cell_table.shape` | (139255, 19) | 0.006 s |
| `ds.connectivity.edge_table.shape` | (19912747, 3) | 0.040 s |
| `ds.connectivity.count_matrix_csr.shape` / nnz | **(139255, 139255)**, nnz 19,912,747 | 0.153 s |
| `ds.connectivity.type_count_matrix_csr.shape` | **(8547, 8547)** | 0.011 s |
| `ds.synapse.synapse_table.shape` | **(80215790, 5)**, 1604 MB in RAM | **0.509 s** (standalone cold `pd.read_parquet`: 0.30 s) |
| `ds.column.column_table.shape` | (45528, 8), cols id,column_id,x,y,p,q,side,rid | 0.008 s |
| `ds.column.column_type_list` | `[]` (see quirks) | 0.000 s |
| `ds.visual_types.shape` | (742, 5) | 0.001 s |
| `ds.show_cell_type_io_stat("Mi1","input",num_rows=5)` | header "Mi1 (ACH, # 1581) input (basic_pc)", top input L1 f_c 0.997, N_s 192286 | 0.141 s |
| `ds.connectivity.counts_between_types(["L1","Mi1"],["L5"],"right","right",include_column_Q=True)` | 24 keys incl. `matrix` (1588, 785), `pre_column`, `post_column` | 0.038 s |
| `ds.get_cell_type_id("Mi1", side="right")[0]` -> `ds.ng_url_for_ids([id])` | id 59582, URL 1355 chars | ok |

Every notebook-00 step that does not need a browser ran without error.

## 4. Schemas

### cell_data.parquet (139,255 rows, 1 row group) — arrow type / pandas dtype
- id: uint32 / uint32 (contiguous 0..139254)
- rid: int64 (FlyWire root id, unique)
- type: large_string / str (1,578 NaN = untyped cells)
- side: int8 (-1 left: 69,959; 1 right: 69,093; 0 unknown: 203)
- flow, super_class, class, sub_class, hemilineage, nerve, group: large_string / str (NaN counts: class 32,286; sub_class 44,309; hemilineage 102,358; nerve 129,616)
- nt_type: large_string / str (values ACH, DA, GABA, GLUT, OCT, SER; 19,658 NaN)
- nt_type_score, da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg: double / float64

### type_data.parquet (8,547 rows) — one row per type; row order == type index used by type matrices
- type: large_string
- num_cells: int64 (sum 137,677 = 139,255 - 1,578 untyped; matches cell_data value_counts exactly)
- nt: large_string (ACH, DA, GABA, GLUT, HIS, NAN, OCT, SER; 191 "NAN"; HIS comes from the schema.TYPE_NEUROTRANSMITTER_GT override for R1-6/R7/R8)
- flow, super_class, class, sub_class, hemilineage, nerve, group: large_string (missing encoded as the string "NAN", not NaN)

### visual_type_data.parquet (742 rows)
- type: large_string (all 742 are present in type_data)
- num_cells: int64 (count from the visual-types source; <= type_data.num_cells, differs for 68/742 types, e.g. R1-6 7240 vs 8525, R7 1259 vs 1333, R8 1278 vs 1339)
- family: large_string (LT 136, LC 56, MT 55, Serpentine Medulla 43, ...)
- subsystem: large_string ("NAN" for boundary types)
- category: large_string (boundary 510, intrinsic 232)

### cell_to_cell_syn_count.parquet (19,912,747 rows, 19 row groups)
- pre_id: uint32, post_id: uint32, num_syn: uint16 (min 1; 138,971 autapse rows; sum 80,215,790 == number of synapses)

### synapses.parquet (80,215,790 rows, 77 row groups)
- pre_id: uint32, post_id: uint32, x_nm: uint32, y_nm: uint32, z_nm: uint32
- ranges: x 82,041–912,416; y 33,808–458,448; z 760–281,720 (nm); all ids < 139,255

### columns_data.npz
- id: (45528,) uint32 (unique, all valid)
- column_id: (45528,) uint16
- x, y, p, q, side: (45528,) int8 each
- pq_min: (2,) int16 = [-19, -17]; pq_max: (2,) int16 = [18, 17]
- (no `type` or `rid` array; Column derives rid via cell.id_to_rid_lookup)

### type_to_type_syn_count.npz (scipy CSR components)
- type_list: (8547,) <U31 — identical, same order, to type_data.type
- data: (2798101,) uint64 (sum 78,899,678)
- indices: (2798101,) int64
- indptr: (8548,) int64
- shape: (2,) int64 = [8547, 8547]

### stat/basic_pc/{input,output}/<type>.csv (e.g. input/4A0.csv: 444 rows x 16 cols)
- type: str; NT: str
- f_c, n_c, cv_c, en_c, cv_en_c, N_s, F_s, n_s, n_s_mw1c, cv_s, n_spc, cv_spc, n_c_spc, cv_c_spc: float64

## 5. Cross-table consistency checks (all pass; log: scratchpad/integrity.log)
- cell ids contiguous 0..n-1; rid unique.
- Non-null cell_data.type set == type_data.type set (8,547); per-type counts match exactly.
- type_data.type order == npz type_list order.
- All edge and synapse pre/post ids < 139,255; edge num_syn sum == synapse row count (80,215,790).
- Type matrix total (78,899,678) == edge sum over edges where both endpoints are typed — i.e. the type matrix excludes the 1,578 untyped cells by design.
- Column ids valid/unique; assignments cover 36 optic-lobe types (Mi1 1581, L1 1570, L2 1555, L5 1554, T4c 1553, Tm1 1549, Tm2 1542, Mi9 1541, ...).
- visual types are a subset of type_data.

## 6. Missing / broken / quirks
- Nothing missing or broken. All 7 table files named in `cex.dataset.schema` are present, plus stat/basic_pc in both directions.
- Quirk: `ds.column.column_type_list` returns `[]` because columns_data.npz has no `type` array (Column.column_type_list guards on `"type" in column_table`). Column lookups by id/rid and `include_column_Q=True` work fine regardless.
- Quirk: missing values are NaN in cell_data.parquet but the literal string "NAN" in type_data.parquet / visual_type_data.parquet (this is the cex schema.DEFAULT_UNKNOWN convention).
- Quirk: `visual_type_data.num_cells` is not the same quantity as `type_data.num_cells` (see section 4).
- Only stat type `basic_pc` is shipped; `fp_cell_type_stat_dir` for any other stat_type will be empty.
- Memory: the synapse table is 1.6 GB in RAM once touched; everything else is small.

## 7. Files produced (scratchpad)
- dl.py / dl.log — download script and log
- verify.py / verify.log — cex load + schema dump
- integrity.log — cross-table checks
- cex_flywire_data_report.md — this report
