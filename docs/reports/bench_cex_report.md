# `cex` hands-on benchmark (FlyWire v783 + MCNS v0.9)

Date 2026-09-10. Repo `/Users/shayanafshar/Code/codexplorer`, `.venv/bin/python` 3.13.9, pandas 3.0.5, pyarrow 25.0.1 (mimalloc pool), macOS, 48 GB RAM, other agents running concurrently.
Script: `scratchpad/bench_cex.py` (every task's user code is a string that is `exec`'d, so the code quoted in section 3 is exactly what was timed). Raw results: `scratchpad/bench_cex.json` (FlyWire) and `scratchpad/bench_cex_mcns.json` (MCNS). Logs: `bench_cex_flywire.log`, `bench_cex_mcns.log`.
Nothing under `src/`, `tests/`, `notebooks/`, or `~/Code/connect_tome` was modified; nothing committed.

Timing = `time.perf_counter()`; "repeat" = min of 2-5 warm re-executions of the same code; RSS = `ps -o rss` delta around the task; peak = `resource.getrusage(RUSAGE_SELF).ru_maxrss` (bytes on macOS) at the end of the task. LOC = non-comment lines of user code.

Reference neuron: FlyWire root_id 720575940599755718 -> assigned id **97249**, type T4a (right, column p,q = (-6,-10)).

## 1. Timing table, FlyWire

| task | first call (s) | repeat (s) | RSS delta (GB) | peak RSS (GB) | rows / shape / nnz | return type | LOC |
|---|---:|---:|---:|---:|---|---|---:|
| T1 `import cex` (fresh interpreter) | 0.202 | - | - | 0.13 | - | module | 1 |
| T1 `get_dataset("flywire", root)` | 0.00005 | - | - | 0.13 | - | `cex.dataset.dataset.FlyWireDataset` | 1 |
| T1 first access `ds.cell.tot_num_cells` | 0.0158 | - | - | 0.13 | 139,255 | int | 1 |
| T2 `ds.cell.cell_table` | 0.0206 | - | +0.074 | 0.19 | 139,255 x 19 | pd.DataFrame | 1 |
| T2 `ds.cell.cell_type_info` | 0.0020 | - | +0.001 | 0.19 | 8,547 x 10 | pd.DataFrame | 1 |
| T2 `ds.cell.rid_to_id_lookup` (build) | 0.0083 | - | +0.004 | 0.19 | - | `cex.util.core.ScalarDict` | 1 |
| T2 `ds.cell.rid_to_id` (dict build) | 0.0110 | - | +0.014 | 0.21 | 139,255 keys | dict[int,int] | 1 |
| T2 `rid_to_id_lookup.get_value(rid)` scalar | 0.0001 | 0.00001 | 0 | 0.21 | - | **np.ndarray, shape ()** | 1 |
| T3 `ds.connectivity.edge_table` | 0.0334 | - | +0.503 | 0.71 | 19,912,747 x 3 (uint32,uint32,uint16; 200 MB) | pd.DataFrame | 1 |
| T3 `count_matrix_csr` | 0.1127 | - | +0.518 | 1.23 | 139,255^2, nnz 19,912,747, uint16, 200 MB | scipy `csr_array` | 1 |
| T3 `count_matrix_csc` | 0.0922 | - | +0.199 | 1.43 | same | scipy `csc_array` | 1 |
| T4 rid -> id | 0.0001 | 0.00001 | 0 | 1.43 | - | int | 1 |
| T4 via `Neuron(cid, ds, min_syn_per_rid=5).output` | 0.0281 | 0.00034 | +0.017 | 1.45 | 19 partners | **dict** (13 keys of ndarray/dict) | 8 |
| T4 via `connected_ids_and_counts` | 0.0003 | 0.00006 | 0 | 1.45 | 19 partners | tuple(ndarray int64, ndarray uint16) | 7 |
| T4 raw scipy+pandas (comparison) | 0.0009 | 0.00022 | 0 | 1.45 | 19 x 3 | pd.DataFrame | 3 |
| T5 `counts_between_types("T4a","LPi14")["matrix"]` | 0.1318 | 0.00033 | +0.011 | 1.46 | (1457, 4), nnz 2,525, uint16 | scipy `csr_array` (inside a 22-key dict) | 4 |
| T5 `num_syn_between_type_groups("T4a","LPi14")` | 0.0107 | 0.00247 | +0.023 | 1.48 | (1, 1), nnz 1 | scipy `csr_array` | 3 |
| T5 raw `csr[pre][:,post].sum()` | 0.0005 | 0.00022 | 0 | 1.48 | - | int | 2 |
| T6 `show_cell_type_io_stat("Dm9","input",10)` | 0.0054 | 0.00025 | 0 | 1.48 | 10 x 10 | pd.DataFrame (+ a print) | 2 |
| T6 `load_cell_type_io_stat("Dm9", dir="input")` | 0.0001 (cached) | 0.00001 | 0 | 1.48 | 149 x 16 | pd.DataFrame | 1 |
| T6 live, type matrix column | 0.0061 | 0.00249 | +0.001 | 1.48 | 10 x 3 | pd.DataFrame | 6 |
| T6 live, cell matrix block + groupby | 0.0042 | 0.00270 | +0.001 | 1.48 | 10 x 4 | pd.DataFrame | 7 |
| T6 live, cell matrix, autapses removed | 0.0033 | 0.00219 | 0 | 1.49 | 10 | pd.Series | 7 |
| T7 `type_count_matrix_csr` load | 0.0076 | 0 (cached) | 0 | 1.49 | 8547^2, nnz 2,798,101, uint64, density 3.8 % | scipy `csr_array` | 1 |
| T7 `.toarray()` | 0.0320 | 0.0339 | +0.580 | 2.06 | 8547 x 8547 uint64, 584 MB | np.ndarray | 1 |
| T10 `ds.ng_url_for_ids([cid])` | 0.0003 | 0.00003 | 0 | 3.23* | 1,355 chars | str | 1 |
| T10 `ds.codex_url(rid)` | 0.0001 | 0.00001 | 0 | 3.23* | 107 chars | str | 1 |
| T10 `Neuron.vis_self_w_top_n_conn_type_in_ng("output", 5, open_url_Q=False)` | 0.0252 | 0.0017 | +0.016 | 3.24 | 3,937 chars | str | 2 |
| T10 same with `vis_syn_Q=True` (needs `load_syn_Q=True`) | 0.0022 | 0.00077 | 0 | 9.32 | 19,234 chars | str | 1 |
| T11 `get_columns_for_ids([cid])` | 0.0027 | 0.00021 | 0 | 3.24 | pq = (-6,-10), xy = (-2,-16), right | dict of 5 arrays | 1 |
| T11 `get_columns_for_ids(all 1581 Mi1)` | 0.0262 | 0.0251 | 0 | 3.24 | 1581 valid, 1581 distinct (p,q,side) | dict of 5 arrays | 2 |
| T11 raw pandas `column_table.set_index("id").loc[mi1]` | 0.0012 | 0.00046 | 0 | 3.24 | 1581 x 3 | pd.DataFrame | 2 |
| T8 `ds.synapse.synapse_table` load | **0.222** | - | **+3.77** | 7.01 | 80,215,790 x 5 uint32 (1,604 MB frame) | pd.DataFrame | 1 |
| T8 `ds.synapse.pre_to_idx` build | 0.407 | - | +1.32 | 8.33 | 139,164 keys | dict[int, ndarray] | 1 |
| T8 `ds.synapse.post_to_idx` build | **5.65** | - | +0.99 | 9.32 | 139,171 keys | dict[int, ndarray] | 1 |
| T8 xyz where cid is pre, `get_for_ids(cid,"pre")` | 0.0009 | 0.00019 | 0 | 9.32 | 289 x 3 | np.ndarray (from DataFrame) | 3 |
| T8 `get_synapse_xyz_for_ids(cid)[cid]` | 0.0007 | 0.00002 | 0 (+0.96 GB `xyz_nm` cached) | 9.32 | **546** x 3 (pre+post, autapses removed) | np.ndarray inside dict | 1 |
| T8 `get_pre_post_rows_for_ids(cid)["pre"][cid]` -> `xyz_nm[rows]` | 0.0001 | 0.00002 | 0 | 9.32 | 282 x 3 (289 minus 7 autapses) | np.ndarray | 2 |
| T8 baseline boolean mask, no index | 0.0191 | 0.0155 | 0 | 9.32 | 289 x 3 | np.ndarray | 2 |
| T8 `Neuron(cid, ds, load_syn_Q=True, min_syn_per_rid=5, remove_autapse_Q=True).output` | 0.0015 | 0.00068 | 0 | 9.32 | 158 synapse rows, 18 partners | dict (14 keys incl. `xyz`) | 2 |
| T8 release (pop 6 cached_property entries + gc + arrow `release_unused`) | 0.037 | - | RSS 9.29 after (no drop) | 9.32 | - | - | 3 |
| T9 morphology | n/a | | | | | not supported | 0 |

\* The 2.06 -> 3.23 GB peak jump before T10 is a harness artefact (three 584 MB dense arrays alive during the T7 repeats), not cex.

Final peak RSS of the FlyWire process: **9.32 GB** (limit was ~10 GB; almost all of it is T8).

Results of record:
- **T4**: 19 output partners with >= 5 synapses, 165 synapses total, strongest partner root_id **720575940632504874 (LPi14), 30 synapses**. Partner types: T4a 3, LLPC1 2, LPi14 2, Y1 2, Y11 2, CT1 1, LPi15 1, Mi9 1, ... (19). The partner list includes the neuron itself (7 autaptic synapses) because `remove_autapse_Q` defaults to False on both APIs; both ways agree exactly. With `remove_autapse_Q=True` at the synapse level: 18 partners, 158 synapses.
- **T5**: T4a -> LPi14 = **49,069 synapses** from the cell-level matrix (1457 T4a x 4 LPi14, 2,525 non-zero pairs), and **49,069** from the precomputed type matrix: they agree, and so does the raw `csr[pre][:,post].sum()`.
- **T6**: Dm9 top-10 inputs from the precomputed CSV (N_s, F_s, f_c, n_c): R8 49,524 (0.335, 1.00, 9.18); R7 28,699 (0.194, 0.995, 8.76); L3 27,045 (0.183, 0.99, 7.62); Dm8b 10,454 (0.071, 0.942); Dm8a 6,105 (0.041, 0.942); C2 3,720 (0.025, 0.99); Dm9 1,651 (0.011, 0.956); T1 1,533; L1 1,278; aMe4 1,087. Live from the matrices the same ranking appears **except Dm9 -> Dm9 = 10,779** (4th) because the matrices include **9,128 autaptic Dm9 synapses**; with autapses removed the live N_s matches the CSV exactly for all 149 input types (F_s then differs only by the 101 synapses from untyped cells that the CSV drops).
- **T7**: 8547 x 8547 uint64 CSR, nnz 2,798,101, total 78,899,678 synapses (typed cells only, autapses included), densifies in 32 ms to 584 MB.
- **T10**: host `spelunker.cave-explorer.org` (Neuroglancer state in the URL fragment); codex host `codex.flywire.ai` (`https://codex.flywire.ai/app/search?filter_string=720575940599755718&sort_by=&page_size=10&data_version=783`).
- **T11**: T4a 97249 -> (p,q) = (-6,-10), right side; all 1581 Mi1 have a column and the 1581 (p,q,side) triples are distinct.

Import: `import cex` 0.20 s cold (`-X importtime`: cex.dataset 214 ms, of which `flywire_links` 169 ms because it pulls `cex.util.tables` -> pandas 129 ms + pyarrow; scipy.sparse 44 ms). No warnings, no prints at import or dataset construction. `from cex.neurons import Neuron` is a separate import (0.3 ms once cex is loaded).

## 2. Timing table, MCNS (`get_dataset("mcns", ".../data/mcns")`, v0.9)

`data/mcns/preprocessed/cell_data.parquet` existed, and the five core tables had been stable since 22:31, so T1-T3 and T5 were run (T6/T7 ran as well; the `stat/` CSVs were still being written by the other agent at run time, so the MCNS T6 rows are provisional). Type names: `ds.ct_fw2mcns("T4a") = "T4a"`, `ds.ct_fw2mcns("LPi14") = "LPi12"`, Mi1 -> Mi1, Dm9 -> Dm9.

| task | first call (s) | repeat (s) | RSS delta (GB) | peak RSS (GB) | rows / shape / nnz | return type | LOC |
|---|---:|---:|---:|---:|---|---|---:|
| T1 `import cex` | 0.205 | - | - | 0.14 | - | module | 1 |
| T1 `get_dataset("mcns", root)` | 0.00005 | - | - | 0.14 | - | `MaleCNSDataset` | 1 |
| T1 first access `tot_num_cells` | 0.0172 | - | - | 0.14 | 211,743 | int | 1 |
| T2 `cell_table` | 0.0030 | - | +0.017 | 0.14 | 211,743 x **4** (id, rid, type, side only) | pd.DataFrame | 1 |
| T2 `cell_type_info` | 0.0001 | - | 0 | 0.14 | 11,732 x 5 (type, num_cells, nt, superclass, flywire_type) | pd.DataFrame | 1 |
| T2 `rid_to_id_lookup` / `rid_to_id` | 0.011 / 0.016 | - | +0.014 / +0.029 | 0.19 | - | ScalarDict / dict | 1 |
| T3 `edge_table` | 0.0489 | - | +0.656 | 0.84 | 26,036,056 x 3 (260 MB) | pd.DataFrame | 1 |
| T3 `count_matrix_csr` | 0.1478 | - | +0.677 | 1.52 | 211,743^2, nnz 26,036,056, uint16 | csr_array | 1 |
| T3 `count_matrix_csc` | 0.1182 | - | +0.260 | 1.78 | same | csc_array | 1 |
| T5 `counts_between_types("T4a","LPi12")` | 0.1841 | 0.00038 | +0.019 | 1.80 | (1684, 4), nnz 2,520; total **52,814** | csr_array in dict | 4 |
| T5 `num_syn_between_type_groups("T4a","LPi12")` | 0.0191 | 0.00317 | +0.001 | 1.80 | (1,1); **52,814**, agrees | csr_array | 3 |
| T6 `show_cell_type_io_stat("Dm9","input")` | 0.0087 | 0.00026 | 0 | 1.80 | 10 x 10 | pd.DataFrame | 2 |
| T6 live cell matrix | 0.0195 | 0.0025 | +0.03 | 1.84 | 10 x 4; N_s agrees with CSV for all 141 types (MCNS Dm9 has 0 autapses) | pd.DataFrame | 7 |
| T7 `type_count_matrix_csr` | 0.0102 | 0 | 0 | 1.84 | 11732^2, nnz 3,824,967, total 122,233,047 | csr_array | 1 |
| T7 `.toarray()` | 0.0505 | 0.049 | +1.00 | 2.84 | 11732 x 11732 uint64, 1.1 GB | np.ndarray | 1 |

Same code path as FlyWire, no dataset-specific branches needed by the user except the type-name translation. MCNS peak RSS 4.85 GB (T7 repeats). Synapse table (1.0 GB on disk) was not loaded.

## 3. Ergonomics (T12)

### Exact user code for T4-T6

T4, id conversion (1 step in; 1 step back out unless you use `Neuron`):
```python
rid = 720575940599755718
cid = ds.cell.rid_to_id[rid]          # step 1: rid -> assigned id (python dict)
```
T4 way A, `Neuron`:
```python
neuron = Neuron(cid, ds, min_syn_per_rid=5)        # parses input+output at construction
out = neuron.output                                # dict of arrays
partner_rids = out["rid"]                          # np.ndarray of FlyWire root ids (already converted back)
n_partners = partner_rids.size
total_syn = int(sum(out["rid2ns"].values()))
best_rid, best_w = max(out["rid2ns"].items(), key=lambda kv: kv[1])
partner_types = out["type"]
```
T4 way B, `connected_ids_and_counts`:
```python
ids, w = ds.connectivity.connected_ids_and_counts(cid, "output", min_syn_per_rid=5)   # (assigned ids, counts)
partner_rids = ds.cell.id_to_rid[ids]              # step 2: ids -> rids (numpy fancy index)
partner_types = ds.cell.cell_types[ids]
n_partners = ids.size
total_syn = int(w.sum())
best_rid, best_w = int(partner_rids[w.argmax()]), int(w.max())
```
T5:
```python
res = ds.connectivity.counts_between_types("T4a", "LPi14")   # dict
M = res["matrix"]                       # scipy csr_array, shape (n_T4a, n_LPi14)
total = int(M.sum())
Mt = ds.connectivity.num_syn_between_type_groups("T4a", "LPi14")   # sparse 1x1
total_type = int(Mt.toarray()[0, 0])
```
T6, precomputed:
```python
tbl = ds.show_cell_type_io_stat("Dm9", "input", num_rows=10)   # prints a header line, returns DataFrame
full = ds.load_cell_type_io_stat("Dm9", dir="input")          # 149 x 16, cached
```
T6, live from the type matrix:
```python
types = ds.cell.cell_type_info["type"].astype(str).to_numpy()
j = ds.cell_type_to_type_idx["Dm9"]
col = ds.connectivity.type_count_matrix_csr[:, [j]].toarray().ravel()
live_type = pd.DataFrame({"type": types, "N_s": col}).query("N_s > 0").sort_values("N_s", ascending=False)
live_type["F_s"] = live_type["N_s"] / live_type["N_s"].sum()
```
T6, live from the cell matrix (gives f_c / n_c too):
```python
post = ds.get_cell_type_id("Dm9")
sub = ds.connectivity.count_matrix_csc[:, post].tocoo()          # pre-cell x Dm9-cell block
df = pd.DataFrame({"pre_type": ds.cell.cell_types[sub.row], "pre_id": sub.row, "post": sub.col, "n": sub.data})
g = df.groupby("pre_type")
live_cell = pd.DataFrame({"N_s": g["n"].sum(), "f_c": g["post"].nunique() / post.size, "n_c": g.size() / post.size})
live_cell["F_s"] = live_cell["N_s"] / live_cell["N_s"].sum()
```

### What felt natural for a pandas/numpy user
- The whole runtime is lazy `cached_property`s over a handful of parquet/npz files; `ds.cell.cell_table`, `ds.connectivity.edge_table`, `ds.synapse.synapse_table` are plain DataFrames, `count_matrix_csr/csc` and `type_count_matrix_csr` are plain scipy arrays indexed by the dense `id`. Once you know the dense id, everything is direct numpy fancy indexing: `csr[pre][:, post]`, `id_to_rid[ids]`, `cell_types[ids]`. Every "raw" alternative in the table was 1-3 lines and as fast or faster than the wrapper.
- Sizes are small and loads are quick: the full cell x cell matrix costs ~0.15 s and ~0.7 GB; the type x type matrix densifies in 30 ms. Everything except the synapse table fits in 1.5 GB.
- `get_dataset(name, root)` + identical code for FlyWire and MCNS worked without change; `ds.ct_fw2mcns` handled the LPi14 -> LPi12 rename.
- `Neuron.get_top_n_output_type_stat()` gives a tidy DataFrame (`type, #s, #c, #ec, #s_w1c, #s/c, f_s, f_c`).
- The precomputed IO stats (`show_cell_type_io_stat`) are the fastest way to answer "who talks to type X" and come with the normalizations already computed.

### What felt foreign
- **dict-of-arrays returns.** `Neuron.output` is a 13-key dict (`rid, rid2i, rid2ns, t2c_i, t2c_num_s, t2i, t2rid, t_enum_c, t_num_c, t_num_s, t_num_s_w1c, type, type_u`) with abbreviated keys; getting "strongest partner" needs `max(out["rid2ns"].items(), ...)`. `counts_between_types` returns a 22-key dict around the matrix (`pre_ids, pre_rids, pre_cell_types, pre_type_to_idx, pre_type_neurotransmitter, pre_type_sign, ...`), `get_columns_for_ids` returns `{"id","xy","pq","left_Q","valid_Q"}`, `get_synapse_xyz_for_ids` returns `{id: xyz}`. A pandas user expects DataFrames (or at least a small dataclass) for all of these.
- **`_Q` suffix on every boolean** (`remove_autapse_Q`, `open_url_Q`, `load_syn_Q`, `parse_io_Q`, `include_column_Q`, `same_color_Q`, `return_payload_Q`, `vis_syn_Q`, `valid_Q`, `left_Q`, `is_column_type_Q`). Readable once learned but non-idiomatic.
- **id vs rid.** The public entry points (`Neuron`, `connected_ids_and_counts`, `get_for_ids`, `ng_url_for_ids`, `get_columns_for_ids`, `num_syn_between_id_groups`) take the dense `id`; FlyWire users think in root_ids. The rid path exists for a subset only (`num_syn_between_rid_pairs/groups`, `get_columns_for_rids`, `vis_self_w_rids_in_ng`, `ng_url`). For T4 that is 1 conversion in (`ds.cell.rid_to_id[rid]`) and, for the tuple API, 1 conversion out (`ds.cell.id_to_rid[ids]`); `Neuron` converts back for you. The vectorized `rid_to_id_lookup.get_value(rid)` returns a 0-d `np.ndarray`, not an int, and `Neuron(np.int64 0-d array)` fails the `isinstance(id, (int, np.integer))` check (`neurons/neuron.py:434`) unless you wrap it in `int(...)`. There are also two parallel mapping objects (`rid_to_id` dict vs `rid_to_id_lookup` ScalarDict) with no guidance on which to use.
- **Side effects on construction.** `get_dataset` creates `download_data/`, `analysis/` and `preprocessed/stat/` under the root (`dataset/paths.py:111,140`), even for a read-only session, and even when the root path is wrong (you then get a fresh empty tree and a `FileNotFoundError` only on first table access).
- **Browser side effects by default.** `Neuron.vis_self_w_top_n_conn_type_in_ng`, `vis_self_w_conn_type_and_synapse_in_ng`, `vis_self_w_ids_in_ng`, `vis_self_w_rids_in_ng` default to `open_url_Q=True` (`neurons/neuron.py:280,310,359,391`), whereas `ds.ng_url` / `ng_url_for_ids` default to False. `codex_open` also opens a browser. Easy to trigger from a script.
- **Prints instead of logging / return values.** `show_cell_type_io_stat` prints its header (`dataset/stats.py:74`); `_parse_synapse_data_with_xyz` prints "Neuron ... has N auto-synapses" (`neurons/neuron.py:460`); `_construct_connected_neuron_layer` prints warnings.
- **Semantics hidden in defaults.** `remove_autapse_Q` defaults to False in `connected_ids_and_counts`/`Neuron` but True in `counts_between_types`, `get_pre_post_rows_for_ids`, `get_synapse_xyz_for_ids`. `get_synapse_xyz_for_ids` is non-directional (pre + post concatenated, 546 rows for our T4a vs 289 presynaptic) even though the name suggests "synapses of these ids"; to get "where the neuron is presynaptic" you use `get_for_ids(cid, "pre")` (a DataFrame) or `get_pre_post_rows_for_ids(cid)["pre"][cid]` (row indices).
- `num_syn_between_type_groups("T4a","LPi14")` returns a 1x1 sparse matrix rather than a scalar; `Neuron` parses both input and output even if you only want one.
- `Neuron(cid, ds)` with `load_syn_Q=True` needs the synapse table plus both per-id indexes (~6 s and +6 GB RSS the first time) with no warning that this is what `load_syn_Q` triggers.

### Missing
- No skeleton / mesh / morphology support at all (T9): `grep -ri "skeleton|swc|navis|morpholog" src/cex` returns nothing. A user would fetch skeletons themselves (navis/fafbseg `flywire.get_skeletons(root_id)` or the FlyWire 783 SWC dump), keyed by `ds.cell.id_to_rid[cid]`, and would get no help placing `xyz_nm` synapses onto them.
- No DataFrame-returning single-neuron partner table (the natural `neuron.outputs(min_syn=5) -> DataFrame[post_rid, type, num_syn]`); you assemble it from the dict or from the raw CSR.
- No public way to release the synapse table (had to `ds.synapse.__dict__.pop("synapse_table")` etc.), no `columns=` / row-group / memory-mapped access to `synapses.parquet`, and no pre-built per-id synapse index on disk (both indexes are rebuilt from an 80 M-row argsort per process).
- No `rid`-first convenience on `Neuron` (e.g. `Neuron.from_rid`), no vectorized `rid -> type` (`rid_to_type` is a 139k-entry python dict).
- `column_type_list` is `[]` for FlyWire because `columns_data.npz` has no `type` array (`dataset/column.py:53`), so `is_column_type_Q` is always False.
- No logging configuration; no `__repr__` on `FlyWireDataset`/`Neuron` (printing `ds` gives the default object repr).

### Exceptions / wrong results
- No exceptions in the benchmarked paths; every task ran on the first try. No warnings emitted anywhere.
- **Autapse inconsistency (data-semantics, not a crash):** the cell x cell matrix, the type x type matrix, and the default `connected_ids_and_counts` include autapses, while the shipped `stat/basic_pc` CSVs exclude them (and drop untyped partners). For Dm9 the effect is large: Dm9 -> Dm9 is 10,779 in the matrices but 1,651 in the CSV (9,128 autaptic synapses; `F_s` denominators 156,956 vs 147,727). A user mixing `show_cell_type_io_stat` with `type_count_matrix_csr` will get different numbers and different rankings without being told why. (MCNS has no autapses in its tables, so there the two agree.)

### Places where I had to read source
- To learn that `Neuron` wants the dense id and rejects a 0-d array (`neurons/neuron.py:434`), and what the 13 output-dict keys mean (`dataset/dataset.py:54 _group_root_ids`).
- To learn that `get_synapse_xyz_for_ids` is non-directional (`dataset/synapse.py:121`) and that `get_for_ids(ids, side)` is the directional call.
- To find the default of `open_url_Q` on `Neuron.vis_*` (True) before calling it from a script.
- To find `ds.cell_type_to_type_idx` for the live type-matrix query (`num_syn_between_type_groups` rebuilds its own `type_to_idx` dict on every call, `dataset/connectivity.py:213`).
- To understand why the CSV and matrix disagreed for Dm9 (`dataset/stats.py:45` drops NaN types; autapse handling lives in preprocessing).
- To release memory (which `cached_property` names to pop from `ds.synapse.__dict__`).

## 4. Bugs, surprises, performance notes (file:line)

1. **Autapses in matrices vs. excluded in IO stats** (see above). `dataset/connectivity.py:52,62` (matrices straight from `cell_to_cell_syn_count` / `type_to_type_syn_count`, which include the 138,971 autapse edges) vs `preprocessed/stat/basic_pc/*.csv` produced by the preprocessing notebooks. Either the matrices should expose an `autapse-free` view or the docs should state the difference.
2. **Synapse-table memory:** `util/tables.py:36 pd.read_parquet` costs +3.8-4.0 GB RSS for a 1.6 GB frame (arrow pool retains the 1.6 GB of column buffers, pandas consolidates into one block = a second copy, and mimalloc never returns pages on macOS: RSS stayed at 9.3 GB after popping every cached value and `gc.collect()`). A standalone probe with `pq.read_table(fp).to_pandas(self_destruct=True, split_blocks=True)` loaded the same table with +1.0 GB RSS in 0.34 s. Then `pre_to_idx` (+1.3 GB) and `post_to_idx` (+1.0 GB) via `util/core.py:12 np.argsort(kind="stable")` over 80 M rows, and `xyz_nm` (`dataset/synapse.py:70`, +0.96 GB copy of three columns). Total T8 footprint ~6 GB on top of 3.2 GB.
3. **`post_to_idx` is 14x slower than `pre_to_idx`** (5.65 s vs 0.41 s): the synapse parquet is sorted by `pre_id`, so the stable argsort is O(n) for `pre_id` and a full 80 M-element sort for `post_id`. Sorting once in preprocessing (or storing both index arrays) would remove the 5.6 s.
4. **`get_columns_for_ids` is a python loop** with `DataFrame.set_index` per call and `.loc` per id (`dataset/column.py:114-116`): 26 ms for 1581 Mi1 vs 1.2 ms for the equivalent `column_table.set_index("id").loc[ids]`. Also `is_column_neuron_Q` rebuilds a `set` of all rids on every call.
5. **`rid_to_id_lookup.get_value(scalar)` returns a 0-d `np.ndarray`** (`util/core.py:109`), and `Neuron.__init__` raises `TypeError` on it (`neurons/neuron.py:434`); `int(...)` is required. Contrast `ds.cell.rid_to_id[rid]` which returns int but is a 139k-entry python dict built on first use.
6. **Constructor side effects:** `DatasetPaths.from_name` -> `ensure_generated_folders` (`dataset/paths.py:111,140`) creates `download_data/`, `analysis/`, `preprocessed/stat/` on every `get_dataset`, including for a mistyped root.
7. **Browser-opening defaults** on `Neuron.vis_*` (`neurons/neuron.py:280,310,359,391`) vs URL-returning defaults on the dataset (`dataset/dataset.py:114` `open_url_Q=False`). Inconsistent and dangerous from batch code.
8. **`print` in library code**: `dataset/stats.py:74`, `neurons/neuron.py:460`, `neurons/neuron.py` `_construct_connected_neuron_layer` warning print.
9. **`Neuron` always parses both directions** (`neurons/neuron.py:54 __parse_io`), 28 ms on first call (measured: building the `rid_to_type` 35 ms and `id_to_type` 22 ms python dicts in `dataset/cell.py:73-100`), 0.3-0.6 ms afterwards; `connected_ids_and_counts` alone is 0.06 ms.
10. **`num_syn_between_type_groups` rebuilds `type_to_idx`** from `cell_type_info["type"].astype(str)` on every call (`dataset/connectivity.py:213-219`): 2.5 ms per call for a 1x1 answer, while `ds.cell_type_to_type_idx` (`dataset/dataset.py:41`) already caches the same dict.
11. **`counts_between_types` first call 0.13 s** is dominated by building `type_to_id` (`dataset/cell.py:_values_by_type`, a pandas sort+groupby over all cells: 0.122 s measured); afterwards 0.3-0.5 ms.
12. **MCNS `cell_data.parquet` has only 4 columns** (`id, rid, type, side`; `MCNS_CELL_SAVE_COLUMNS` in `dataset/paths.py`), while FlyWire has 19 (flow, super_class, nt scores, ...). The type table carries `superclass`/`flywire_type`, but per-cell superclass / NT are not available for MCNS through `cell_table`.
13. **`column_type_list == []` for FlyWire** (`dataset/column.py:53`) because the npz has no `type` array; `is_column_type_Q("Mi1")` returns False even though all 1581 Mi1 have columns.
14. Minor: `read_table` has no column projection for `synapse_table` (always all 5 columns); `edge_table` is kept alive (200 MB) after the CSR/CSC are built (`dataset/connectivity.py:52-60`), and the CSR + CSC are two full copies (400 MB) instead of `csr.tocsc()` on demand or a shared COO.

## 5. Bottom line for the merge

- Strengths to keep: tiny normalized on-disk layout, dense-id design that makes every query a scipy/numpy one-liner, sub-second load of everything except synapses, precomputed type matrix and per-type IO CSVs, FlyWire/MCNS parity with a type-name translator, Neuroglancer/Codex URL builders that work headless.
- Weaknesses to fix in a merged API: DataFrame (not dict) returns for partner tables, columns and synapse queries; rid-first entry points (`Neuron.from_rid`, `outputs(min_syn=)`); consistent autapse handling (or an explicit flag) across matrices and IO stats; no filesystem or browser side effects by default; a memory-conscious synapse loader (`to_pandas(self_destruct=True)`, column projection, on-disk per-id index or a pre-sorted table so `post_to_idx` is not a 5.6 s / 1 GB rebuild) and a `release()`; morphology has to come from the sibling package (`shayan` has navis skeleton loading).
