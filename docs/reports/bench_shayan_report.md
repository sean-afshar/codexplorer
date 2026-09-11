# `shayan` hands-on benchmark (FlyWire FAFB v783)

Script: `scratchpad/bench_shayan.py`, raw results: `scratchpad/bench_shayan.json`.
Environment: Python 3.13.9, polars 1.44.1, numpy 2.5.2, scipy 1.18.1, navis 1.12.0, pandas 3.0.5, pyarrow 25.0.1; macOS arm64, 48 GB RAM, other agents running concurrently.
Total wall time for the whole suite: **49 s**. Main-process peak RSS 10.85 GB (that peak is the 152M-row Male-CNS feather in T11, not shayan; the shayan-only peak was 6.7 GB at T7). Cold tasks (T1, T3-eager, T8) ran in fresh interpreters whose RSS was polled from the parent. `ru_maxrss` is bytes on macOS; "rss" is `/bin/ps -o rss`.

Data facts established along the way: connections table = 22,285,323 rows = (pre, post, **neuropil**) triples covering 19,773,733 unique pre->post pairs, 76.9M synapses; neuron table = 139,255 rows, 8,548 types (1,578 untyped); synapse table = 80,215,790 rows (7.37 GB uncompressed CSV, 2.7 GB gz); skeleton zip = 139,273 SWC members.

## Timing table

| Task | What | time_s (first) | repeat_s | peak RSS (GB) | rows | return type | user LOC |
|---|---|---|---|---|---|---|---|
| T1 | `import shayan` + `Connectome(FlyWireDataset(dir), mode="lazy")` (fresh interp.) | **0.067** (import 0.067, construct 0.000) | - | 0.058 (child) | - | `shayan.core.connectome.Connectome` | 3 |
| T1b | `import shayan.morphology` (navis/plotly/scipy) after `shayan` | **1.12** | - | 0.277 (child) | - | module | 1 |
| T2 | `ct.neurons` | 0.022 | 1e-6 (cached property) | 0.138 | 139,255 x 6 | `polars.DataFrame` | 1 |
| T3 | `ct.connections` | **1.39** | cached | 4.38 peak, +1.33 GB RSS delta (DF est. 0.62 GB) | 22,285,323 x 5 | `polars.DataFrame` | 1 |
| T3b | `Connectome(mode="eager")` fresh interpreter | 1.43 (+0.064 import) | - | 2.54 (child) | 22.3M | `Connectome` | 1 |
| T4 | outputs of 720575940599755718, weight>=5 | 0.009 | 0.006 (no caching; polars is just fast) | 4.38 | 18 rows / 155 syn | `Circuit` (polars DFs inside) | 7 |
| T5 | T4a -> LPi14 total + 1457x4 matrix | 0.029 | 0.011 | 4.38 | 2,525 edge rows; total 49,069 | `int` + hand-built `numpy.ndarray` | 9 |
| T6 | Dm9 input profile, top-10 with % input | 0.51 | 0.52 (no caching) | 4.38 | 203 pre types | `polars.DataFrame` | 3 |
| T7 | full type x type | 0.70 (as_circuit 0.12 + aggregate_by_type 0.58); normalized variant 1.20 | - | 6.72 | 2,811,570 long-form rows; 8,547 types | `polars.DataFrame` **long-form only** (user scipy CSR build +0.85 s, nnz 2.8M) | 2 |
| T8 | `ct.dataset.load_synapses()` (fresh interp.) | **18.0** load; 0.007 per xyz query; 0.001 to_numpy | 0.007 | **11.9 (child peak)**, 6.6 after load, 5.4 after `del` | 80,215,790 x 6; 289 synapses for the T4a | `polars.DataFrame` | 3 |
| T8-alt | user pyarrow streaming fallback (selected cols, Int32 xyz, Categorical neuropil) | 16.0 load; 0.005 per query | - | 3.6 (child peak) | same 80.2M / 289 | `polars.DataFrame` | 16 |
| T9 | Dm9 720575940606097068: load zip (0.61) + segment_natural (0.61) + CompartmentalModel (0.019) + solve (0.0003) | 1.24 (+1.01 morphology import) | - | 6.72 | 5,937 nodes, 549 segments = 549 compartments, 8 zero-length | `navis.NeuronList` -> `Compartmentalization` -> `CompartmentalModel` -> `np.ndarray` (mV) | 5 |
| T10 | `neuron_to_neuroglancer_url(nid)` / `to_neuroglancer_url(comp)` | 0.00009 / 0.14 | - | 6.72 | URL 1,476 chars / 171,603 chars (549 point annotations); all-nodes variant 1.8 MB | `str` | 2 |
| T11 | MCNS `pl.read_ipc` weights / annotations | 1.11 / 0.023 (via pyarrow; polars errors) | scan_ipc len 0.009, filtered scan 0.12 | 10.85 (+7.4 GB RSS delta for a 3.6 GB DF) | 151,871,794 x 3 / 211,743 x 35 | `polars.DataFrame` | 2 |

Import-time note: `import shayan` = 0.067 s and pulls in only polars/numpy; the heavy names (`load_skeleton`, `CompartmentalModel`, ...) are resolved lazily via `shayan.__getattr__`, and `import shayan.morphology` costs 1.0-1.1 s (navis + plotly + scipy + pandas). No warnings are printed on import. The only warnings seen in the whole run were from T9 (zero-length compartments -> `MatrixRankWarning: Matrix is exactly singular`). navis prints a tqdm "Importing: 100%" progress bar to stderr on every `load_skeleton` call.

## Task details / results

- **T1**: `Connectome(..., mode="lazy")` opens no files (only `data_dir.exists()`), so it is instantaneous; `repr` reports `neurons=not loaded, connections=not loaded`.
- **T2**: columns `neuron_id Int64, type String, region String, nt_type String, nt_type_score Float64, side String`. `side` is a constant null column; `nt_type_score` is not in `NEURON_SCHEMA`; `validate_schema/ensure_schema` are never called by the loader.
- **T3**: `pre_id Int64, post_id Int64, weight Int32, neuropil String, nt_type String`. Rows are neuropil-split; 22.3M rows over 19.8M unique pairs (11% of pairs are split across neuropils). gz-CSV parse 1.4 s; process RSS +1.3 GB (peak 4.4 GB during decompression+parse of a 1.1 GB CSV).
- **T4** (package semantics, `min_weight(5)` applied per neuropil row): 18 rows, 155 synapses, strongest partner 720575940632504874 (30 syn, LOP_R). Partner types: LPi14 35, LLPC1 16, Y1 16, VCH 15, TmY16 12 ... (13 types). Per-partner semantics (sum over neuropils then threshold): 18 partners, **158** synapses, top types LPi14 35, LLPC1 16, Y1 16, VCH 15, **T4a 14** — i.e. the package answer silently drops the sub-threshold neuropil slices of otherwise >=5 partners and reorders the type ranking. Neuron 720575940599755718 is confirmed T4a.
- **T5**: total T4a->LPi14 = 49,069 synapses; 1,453 of 1,457 T4a cells connect to all 4 LPi14 cells; 2,525 (pre,post) pairs after summing over neuropils. Matrix built by hand (numpy index maps) — package has no adjacency/pivot method. `aggregate_by_type()` returns the single row `T4a, LPi14, 49069`.
- **T6**: Dm9 (206 cells, 147,791 input synapses, 21,789 input edges) top-10 inputs by `total_weight` with `percent_input`: R8 49,524 (33.5%), R7 28,662 (19.4%), L3 27,045 (18.3%), Dm8b 10,454 (7.1%), Dm8a 6,105 (4.1%), C2 3,720 (2.5%), Dm9 1,651 (1.1%), T1 1,533 (1.0%), L1 1,278 (0.9%), aMe4 1,087 (0.7%). `percent_input` sums to exactly 100.0 and matches `100*total_weight/sum` to 0.0 deviation. One `pre_type = null` row (26 synapses from untyped neurons) with `percent_output = null`. The 0.5 s per call is `aggregate_by_type_normalized` re-joining the entire 22M-row table with neuron types twice to recompute global per-type totals — every call, no caching.
- **T7**: `ct.query().as_circuit()` (0.12 s) then `aggregate_by_type()` (0.58 s) -> long-form 2.81M rows over 8,547 types; 745k synapses have null pre_type and 570k null post_type. Package produces no matrix; the user must pivot (scipy CSR 8547x8547, nnz 2.8M, 0.85 s; dense int64 would be 0.58 GB).
- **T8**: `load_synapses()` succeeded in 18.0 s but peaked at **11.9 GB** in the child (7.4 GB decompressed CSV held in RAM + 13 parsed columns, then a 6-column projection); resident 6.6 GB after load, 5.4 GB after `del` (allocator does not return it). Result 80,215,790 x (pre_id, post_id, x, y, z Int64; neuropil String), est. 3.56 GB. xyz query for the T4a: 289 presynaptic sites, 7 ms. Ids are rebuilt by `concat_str("720575940", suffix)` — correct here because every suffix is 9 digits (verified: min suffix digits = 9, no zero/null roots), but it would silently corrupt ids with leading-zero suffixes. Not exposed through `Connectome` at all (`_synapses` slot exists but no accessor). The pyarrow-streaming user fallback loads the same rows in 16 s at 3.6 GB peak.
- **T9**: `load_skeleton` default `convert_units=False` returns **nanometres** (`1 nanometer`), while `CompartmentalModel` documents/assumes micrometres — must pass `convert_units=True` (then `1.0 micrometer`; x range 695-763 um, cable 1,439 um). segment_natural: 549 compartments (one per navis segment), **8 zero-length** (single-node segments); `CompartmentalModel` warns, `solve_steady_state` then emits `MatrixRankWarning` and returns **all-NaN** (nan fraction 1.0). Compartment 0 does contain the root node. Workaround (set zero lengths to 1e-3 um, rebuild) gives V0 = 6.22 mV for 10 pA (R_in 0.62 GOhm), Vmin 2.26 mV, no NaN. Compartment dtypes: ids Int64, length/radius Float32 (navis precision=32).
- **T10**: default viewer host `ngl.cave-explorer.org` with `precomputed://gs://flywire_v141_m783` (+ brain mesh + BossDB EM); `viewer='flywire'` -> `ngl.flywire.ai` with graphene prodv1 (auth-gated). Pure string building, no network, no state-server shortening. Compartment overlay URL is 171 kB (549 point annotations); `use_centers_only=False` yields 1.8 MB — both far beyond what browsers/servers accept in a URL fragment.
- **T11**: weights feather: `body_pre Int64, body_post Int64, weight Int64`, 151.9M rows, read_ipc 1.1 s (3.6 GB DF, +7.4 GB RSS), `scan_ipc` schema 0.4 ms, `select(pl.len())` 9 ms, filtered scan for one body 0.12 s. Annotations feather: 211,743 x 35; **polars 1.44.1 `read_ipc`/`scan_ipc` both fail** (`ComputeError: The dictionary key must fit in a usize, but -1 does not`) because the `statusLabel` column is dictionary-encoded with -1 null indices; `pyarrow.feather.read_table` + `pl.from_arrow` works (23 ms). Key columns: `bodyId UInt64`, `type String` (22% null, 11,733 unique), `flywireType`, `hemibrainType`, `instance`, `somaSide String` (R/L/...), `superclass` (27), `class`, `subclass`, `somaNeuromere`, `status`, `somaLocation List(Int64)`.

### T11: MCNS `AbstractDataset` subclass estimate (not written)

`cli/cli.py:18-24` already tries `from shayan.datasets.male_cns import MaleCNSDataset`, so only the class is missing. Mirroring `datasets/flywire.py` (91 lines) it needs roughly **60-90 lines**:

- `load_neurons`: read `body-annotations-*.feather` via pyarrow (polars fails, see above) -> `bodyId -> neuron_id (cast Int64)`, `type -> type`, `somaSide -> side`, `superclass` (or `somaNeuromere`) `-> region`; join `body-neurotransmitters-male-cns-v0.9.feather` on bodyId for `nt_type` (columns not inspected here). ~20 lines.
- `load_connections`: `pl.scan_ipc(weights).select(body_pre->pre_id, body_post->post_id, weight.cast(Int32)->weight, lit(None)->neuropil, lit(None)->nt_type)` — 3 columns only, 152M rows; `has_neuropil` must return False so `ConnectionQuery.neuropil()` warns. ~10 lines. Note weight is Int64 in the file; `CONNECTION_SCHEMA` says Int32 (max weight 2,591 in head, fits).
- `load_synapses`: `syn-partners-*.feather` (6.8 GB) is the candidate but its columns were not inspected; return None or lazy scan. ~10-15 lines.
- properties (`organism`, `sex='male'`, `version='v0.9'`, `has_synapses`) ~15 lines.
- Also: `Connectome._get_connection_scanner` (`core/connectome.py:95-102`) is a stub, so `mode="scan"` would still fully load the 152M rows; a real scan mode would be the natural fit for MCNS.

## T12 Ergonomics

### Exact user code written

T4 (7 lines; the package-semantic version):
```python
import polars as pl
circ = ctome.query().pre_id(720575940599755718).min_weight(5).as_circuit()
n_out, total = circ.n_connections, circ.total_weight
top = circ.connections.sort("weight", descending=True).row(0, named=True)
partner_types = (circ.connections
                 .join(circ.neurons.select("neuron_id", "type"), left_on="post_id", right_on="neuron_id")
                 .group_by("type").agg(pl.col("weight").sum()).sort("weight", descending=True))
```
T4 per-partner variant I had to write to get the *intended* answer (5 lines):
```python
out = (ctome.query().pre_id(720575940599755718).collect()
       .group_by("post_id").agg(pl.col("weight").sum())
       .filter(pl.col("weight") >= 5).sort("weight", descending=True))
types = out.join(ctome.neurons.select("neuron_id", "type"), left_on="post_id", right_on="neuron_id")
```
T5 (9 lines):
```python
import numpy as np, polars as pl
circ = ctome.query().pre_type("T4a").post_type("LPi14").as_circuit()
total = circ.total_weight
pre_ids, post_ids = ctome.get_cell_type_ids("T4a"), ctome.get_cell_type_ids("LPi14")
pairs = circ.connections.group_by("pre_id", "post_id").agg(pl.col("weight").sum())
pi = {v: i for i, v in enumerate(pre_ids.tolist())}
qi = {v: i for i, v in enumerate(post_ids.tolist())}
M = np.zeros((len(pre_ids), len(post_ids)), dtype=np.int32)
M[[pi[v] for v in pairs["pre_id"]], [qi[v] for v in pairs["post_id"]]] = pairs["weight"].to_numpy()
```
T6 (3 lines):
```python
circ = ctome.query().post_type("Dm9").as_circuit()
prof = circ.aggregate_by_type_normalized().sort("total_weight", descending=True)
top10 = prof.select("pre_type", "post_type", "n_connections", "total_weight", "percent_input", "percent_output").head(10)
```

### What felt natural (for a pandas/numpy user)
- Every table is a plain `polars.DataFrame` with stable, documented column names (`pre_id/post_id/weight`, `neuron_id/type`), and `Circuit.to_pandas()` exists. Nothing is hidden in custom containers.
- `ct.get_cell_type_ids("T4a")` -> `np.ndarray` of int64; `ct.neuron_index[id] -> type` dict. Direct and cheap (0.05 s to build).
- The `query().pre_type().post_type().min_weight().as_circuit()` chain reads well for the common filters and is fast (all sub-second; the 22M-row table is only 0.6 GB in polars).
- `aggregate_by_type_normalized()` gives exactly the % input / % output normalisation a connectomics user wants, and the numbers check out (sum 100.0, zero deviation from manual).
- Lazy import keeps `import shayan` at 67 ms.

### What felt foreign
- **Vocabulary**: "Circuit" for "the filtered edge list plus its neuron rows", `as_circuit()` vs `collect()`, `total_weight` for "synapse count". The `weight` column is a synapse count, but `Circuit.empty` types it Int64 while real data is Int32 (`core/circuit.py:72` vs `datasets/flywire.py:70`).
- **Builder pattern mutates in place**: `ConnectionQuery` methods return `self` (`core/query.py:70-72`), so `base = ct.query().pre_type("T4a"); a = base.post_type("X"); b = base.post_type("Y")` gives `b` both filters. Not the immutable style of pandas/polars.
- **Two-step access to types**: connection tables carry only ids; getting partner types is always a manual join against `circ.neurons` (T4) — there is no `with_types()` helper or a `pre_type/post_type` column option on `collect()`.
- **No matrix anywhere**: T5 and T7 need a hand-written pivot; the package's only aggregate is long-form.
- **Polars only**: the user must know polars expressions (`pl.col`, `group_by().agg`, `row(0, named=True)`) for anything beyond the canned aggregates. That is fine for me but is a real learning cost for a pandas user.
- **Dict-of-arrays returns**: `type_index` is `dict[str, np.ndarray]`, `Compartmentalization.node_mapping` is `dict[int, list[int]]`, `summary()` returns a dict of mixed scalars plus a 100-entry `cell_types` list.

### What was missing
- Adjacency / type-by-type matrix helper (dense or sparse), and an `ids -> index` mapping utility.
- Per-partner aggregation before thresholding (the neuropil-split rows make `min_weight` semantically surprising; see bugs 1).
- Any public synapse accessor on `Connectome`, a lazy/streaming synapse loader, or spatial helpers (`synapses_of(neuron_id)`, xyz -> numpy).
- Caching of the global per-type totals used by `aggregate_by_type_normalized` (0.5 s recomputed each call).
- The other three segmentation strategies, `Compartmentalization.to_hines_matrix/summary/plot` (all `NotImplementedError` stubs), and any merging of zero-length compartments (so the cable solver is unusable out of the box on this Dm9).
- MCNS loader; `scan` mode (falls back to full load).
- Schema enforcement: `validate_schema`/`ensure_schema` exist (`datasets/schema.py:119-178`) but are never called.

### Exceptions / wrong results
- No exceptions from shayan in T1-T10. The only wrong-ish result is T4: `min_weight(5)` gives 155 synapses vs the intended 158 and a different 5th partner type, because thresholds are applied to neuropil-split rows.
- T9 `solve_steady_state` returns an all-NaN vector (with a warning) for a completely ordinary Dm9 — effectively the documented default pipeline in `README.md` does not produce a number.
- T11 (not shayan's fault): polars 1.44.1 cannot read the MCNS annotations feather.

### Where I had to read source
- To learn that connection rows are per-neuropil (nothing in docstrings; needed `datasets/flywire.py:63-74` plus the CSV header) and hence what `min_weight` really filters.
- To learn how to get synapse xyz at all: no README/API mention beyond "only needed for load_synapses()"; found `Connectome._synapses` is dead (`core/connectome.py:59`) and that `ct.dataset.load_synapses()` is the only path (`datasets/flywire.py:76-92`).
- To confirm `load_skeleton` returns nm by default and that `CompartmentalModel` assumes um (`morphology/loader.py:16,24-26`; `models/compartmental_model.py:75`).
- To find that `Compartmentalization` can be re-constructed by hand (positional args `neuron, compartments, node_mapping, strategy`, `morphology/compartmentalization.py:45-50`) in order to patch zero-length compartments.
- To see what `percent_input` normalises against (global type totals, `core/circuit.py:297,320-355`) versus `add_normalized_weights` (per-neuron, min_weight-filtered, `core/circuit.py:227-229`).

## Bugs / surprises (with locations)

1. **Neuropil-split edge rows make weight filters per-neuropil, not per-partner.** `datasets/flywire.py:63-74` keeps one row per (pre, post, neuropil) (22.3M rows for 19.8M pairs); `core/query.py:190-205` `min_weight` and `core/circuit.py:107-125` `filter_by_weight` filter those rows. T4: 155 vs 158 synapses, different 5th partner type. Nothing documents this.
2. **Cable model returns all-NaN on a normal neuron.** `morphology/segmentation.py:154-157` yields length 0 for single-node segments (8/549 here); `models/compartmental_model.py:98-106` only warns; `:201` `spsolve` on a singular matrix -> NaN everywhere. No merge/drop utility exists (README "Known gaps" admits it).
3. **Unit footgun.** `morphology/loader.py:16` `convert_units=False` default returns nanometres (`1 nanometer` verified) but `models/compartmental_model.py:75` assumes um; the README example passes `convert_units=True`, the loader docstring example does not. Silent 1e3 error in Rm/Ra scaling if forgotten.
4. **`load_synapses` is not memory-bounded.** `datasets/flywire.py:83` `pl.read_csv` on the 2.7 GB gzip decompresses 7.4 GB into RAM and parses all 13 columns before projecting: child peak 11.9 GB for a 3.6 GB result; RSS stays 5.4 GB after `del`. A column-projected streaming read (pyarrow) achieves 3.6 GB peak in the same time.
5. **Fragile id reconstruction.** `datasets/flywire.py:85-86` rebuilds root ids with `concat_str("720575940", int_col)`; correct today only because all suffixes are 9 digits (checked); any suffix < 1e8 would produce a wrong 17-digit id. Arithmetic (`720575940_000_000_000 + suffix`) is the safe form.
6. **Dead `_synapses` slot / no accessor.** `core/connectome.py:59` allocates `_synapses`, nothing fills or exposes it; `_load_all` (`:73-77`) explicitly skips it. Users must go through `ct.dataset`.
7. **`scan` mode is a no-op.** `core/connectome.py:79-102`: `_get_connection_scanner` raises `NotImplementedError`, caught and silently downgraded to lazy full load.
8. **Global totals recomputed per call.** `core/circuit.py:320-355` joins the full 22M-row table twice inside `aggregate_by_type_normalized` on every call (0.52 s each for T6, 1.2 s for T7); `add_normalized_weights` (`:257-272`) uses Python-list `is_in` filters over the full table.
9. **Mutable query builder.** `core/query.py:56-255` every filter mutates and returns `self`; sharing a partially built query silently accumulates filters.
10. **Dtype inconsistency.** `core/circuit.py:69-73` `Circuit.empty` uses `weight: Int64`, real circuits carry `Int32` (`datasets/flywire.py:70`); concatenating empty and non-empty circuits will need casts. Docstring `datasets/base.py:121-124` says x,y,z float32 while `datasets/schema.py:78-80` says Int64 (loader emits Int64).
11. **Schema module unused; extra/dead columns.** `datasets/flywire.py:53-60` emits `nt_type_score` (not in `NEURON_SCHEMA`) and a constant-null `side`; `validate_schema/ensure_schema` (`datasets/schema.py:119,137`) are never invoked, so `AbstractDataset` contracts are not enforced.
12. **Neuroglancer overlay URLs are unusably long.** `morphology/visualization.py:467-625` inlines annotations in the URL fragment: 171 kB for 549 compartment centres, 1.8 MB for all nodes; no JSON state-server posting. `color_by` is documented "not yet implemented" (`:488`) and `colors` (`:510`) is computed but unused. The `'flywire'` viewer uses the auth-gated graphene prodv1 source (`:395`).
13. **Quadratic segmentation loop.** `morphology/segmentation.py:46-64`: per segment a pandas `isin` over all nodes (`:54`, `:144`) plus `_find_parent_segment` linear scan over all segments (`:162-177`): 0.61 s for 5.9k nodes / 549 segments; this scales ~O(S*N) and will take minutes on large (100k-node) neurons.
14. **Stubs presented as API.** `morphology/segmentation.py:72-127` (three strategies), `morphology/compartmentalization.py:116-215` (`to_hines_matrix`, `summary`, `plot`) raise `NotImplementedError` but are exported from `shayan.morphology.__all__`.
15. **navis progress bar noise.** Every `load_skeleton` prints a tqdm "Importing" bar to stderr (`morphology/loader.py:43` does not pass `progress=False`).
16. **Untyped neurons become `null` groups.** 745k synapses have a null pre_type in T7 and a `pre_type=null` row appears in T6 (`core/circuit.py:330-343` left joins); `Circuit.neurons` silently drops ids absent from `neurons.csv` (`core/query.py:297-301`), so `n_neurons` undercounts endpoints (139,116 vs the ids present in edges).
17. (Environment, not shayan) polars 1.44.1 `read_ipc`/`scan_ipc` fail on the MCNS annotations feather (`dictionary key must fit in a usize, but -1`); read via pyarrow instead. Relevant for any MCNS loader.
