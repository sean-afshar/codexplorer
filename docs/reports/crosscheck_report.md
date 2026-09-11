# shayan vs cex cross-check on FlyWire FAFB v783

Date 2026-09-10. Script: `scratchpad/crosscheck.py` (`conn` stage = Q1-Q8, 6 s, peak RSS 7.6 GB; `syn` stage = Q9, 19 s, peak RSS 11.6 GB, run only after `ps aux | grep bench_shayan.py` was empty). Raw numbers: `scratchpad/crosscheck_results.json`; the 43 odd pairs: `scratchpad/cex_only_nonautapse_pairs.json`. Nothing under `src/`, `tests/`, `notebooks/`, `~/Code/connect_tome` was modified; no commits.

shayan = `src/shayan` (Polars) reading `/Users/shayanafshar/Code/connect_tome/data/flywire_fafb/` raw CSVs.
cex = `src/cex` (pandas + scipy.sparse) reading `/Users/shayanafshar/Code/codexplorer/data/flywire/preprocessed/` parquet via `get_dataset("flywire", ...)`.

## One-paragraph verdict

The two pipelines describe the **same cell universe (139,255 root_ids, 8,547 types, identical type per cell)** and the **same synapses (80,215,790 rows, identical nm coordinates)**. Every discrepancy in every question reduces to exactly three facts about the raw *connections* export that shayan uses (`connections_princeton_no_threshold.csv.gz`) versus the *synapse table* that cex counts from (`fafb_v783_princeton_synapse_table.csv.gz`):

1. the connections export **drops all autapses** (3,271,078 synapses, 138,971 pre==post pairs);
2. the connections export is **missing one cell entirely**, R7 `720575940623940963` (213 non-autapse synapses in 43 pairs, 215 rows including 2 autapses);
3. the connections export is **per (pre, post, neuropil)**, with the 30,628 synapses that have an empty neuropil in the synapse table relabelled `UNASGD`.

Arithmetic: 80,215,790 − 3,271,078 − 213 = **76,944,499** = shayan's total, exactly. There is **no confidence/cleft-score threshold** difference and **no count threshold** difference (16,041,273 shared pairs of weight 1-4 agree exactly, min weight is 1 in both). Beyond connectivity, the packages differ in metadata semantics: cex has `side` (shayan hard-codes null), cex applies a hand-written neurotransmitter override to some types (`Dm9`, `R7`, ...), shayan carries `neuropil` and a per-edge `nt_type` that cex drops.

---

## Q1 Cell universe

| quantity | shayan (`neurons.csv.gz` ⟕ `consolidated_cell_types.csv.gz`) | cex (`cell_data.parquet`, built from `classification.csv` ⟕ types ⟕ `neurons.csv.gz`) | diff |
|---|---:|---:|---:|
| cells | 139,255 | 139,255 | 0 |
| typed cells (`type` non-null) | 137,677 | 137,677 (= `type_data.num_cells` sum) | 0 |
| distinct types | 8,547 | 8,547 (`cell_data`) = 8,547 (`type_data`) | 0 |
| root_ids only in shayan / only in cex | 0 / 0 | | identical sets |
| type names only in shayan / only in cex | 0 / 0 | | identical sets |
| shared cells whose type differs | 0 | | |
| `nt_type` null | 19,658 | 19,658 (NaN) | 0 |
| `side` | all 139,255 null | L 69,959 / R 69,093 / unknown 203 | see Q7 |

Both `neurons.csv.gz` (139,255 unique root_id) and cex's `classification.csv` describe the same 139,255 cells; `consolidated_cell_types.csv.gz` has 137,677 rows, all present in `neurons.csv.gz`. So the cell universe needs no reconciliation. Only the *representation* differs: shayan keeps FlyWire root_ids as `neuron_id` (Int64), cex adds a dense `id` (uint32, 0..139254, sorted by `(type, rid)`, `raw.py:sort_table_and_add_id`) and keeps the root_id as `rid`.

## Table totals (the "known facts")

| quantity | shayan `connections` | cex `edge_table` | note |
|---|---:|---:|---|
| rows | 22,285,323 (pre, post, neuropil) | 19,912,747 (pre, post) | |
| distinct (pre, post) pairs | 19,773,733 | 19,912,747 | +139,014 in cex |
| synapse sum | 76,944,499 | 80,215,790 (= `synapses.parquet` rows) | +3,271,291 in cex |
| min weight | 1 | 1 | no count threshold either side |
| autapse rows / synapses | 0 / 0 | 138,971 / 3,271,078 | root cause 1 |
| neuropil | 79 labels incl. `UNASGD` (24,790 rows, 30,628 syn), 0 nulls | none | root cause 3 |
| distinct cells appearing | 139,116 | 139,189 (139,117 in non-autapse edges) | 66 cells have no synapses at all; 72 have only autapses; 1 (R7) dropped from the export |

## Q2 Neuron 720575940599755718 (T4a, cex id 97249) — outputs per postsynaptic cell

| quantity | shayan (group_by post_id, sum weight over neuropils) | cex (`Neuron(97249, ds).output`, == `connectivity.connected_ids_and_counts`) | diff |
|---|---:|---:|---|
| neuropil-level rows | 90 | n/a | |
| partners | 87 | 88 | cex has the autapse 720575940599755718→itself (7 syn) |
| total synapses | 282 | 289 | +7 (autapse) |
| top 5 | 720575940632504874:30, 720575940627706398:15, 720575940614051555:12, 720575940639834165:11, 720575940640184398:10 | identical | 0 |
| partners with ≥5 syn (per partner) | 18, 158 syn | 19, 165 syn (`min_syn_per_rid=5`) | +1 partner / +7 syn = autapse |
| shayan native `min_weight(5)` (per **neuropil row**) | 18 partners, **155** syn | | 3 syn lost: 3 partners are split across LOP_R/ME_R and a sub-5 fragment is dropped |

All 87 shared partners have identical counts. cex additionally reports partner types (LPi14 35, T4a 28, Y1 20, Y11 17, LLPC1 17).

## Q3 Same neuron — inputs per presynaptic cell

| quantity | shayan | cex | diff |
|---|---:|---:|---|
| partners | 69 | 70 | autapse |
| total synapses | 264 | 271 | +7 (autapse) |
| top 5 | 720575940643266583:29, 720575940605262880:28, 720575940626979621:19, 720575940627706398:16, 720575940633138330:15 | identical | 0 |
| ≥5 per partner | 12 partners, 169 syn | 13 partners, 176 syn | autapse |
| shayan native `min_weight(5)` per row | 12 partners, 167 syn | | 2 syn lost to neuropil splitting |

## Q4 Type-to-type totals

| pair | shayan Σweight (pairs) | cex `num_syn_between_type_groups` | cex `counts_between_types(...)["matrix"].sum()` | diff | cells pre/post shayan | cells pre/post cex |
|---|---:|---:|---:|---:|---|---|
| T4a→LPi14 | 49,069 (2,525) | 49,069 | 49,069 (2,525 nnz) | 0 | 1457 / 4 | 1457 / 4 |
| Mi1→T4a | 92,542 (8,890) | 92,542 | 92,542 | 0 | 1581 / 1457 | 1581 / 1457 |
| L1→Mi1 | 192,286 (1,862) | 192,286 | 192,286 | 0 | 1596 / 1581 | 1596 / 1581 |
| R7→Dm9 | 28,662 (1,801 pairs, 1,804 rows) | 28,699 | 28,699 (1,805 nnz) | **+37 in cex** | 1333 / 206 | 1333 / 206 |

The R7→Dm9 gap is exactly the 4 pairs from the dropped R7 cell `720575940623940963` to Dm9 cells 720575940642650331 (32), 720575940606097068 (2), 720575940635029273 (2), 720575940629371978 (1) = 37. Per-type cell sets are identical for all six types (no cell only on one side). cex's two type-level APIs agree with each other (the type matrix is built from the edge table, `raw.py:build_type_connectivity_data`, and excludes the 1,578 untyped cells; `remove_autapse_Q` is irrelevant for distinct types).

## Q5 Full (pre, post, total) pair comparison

| quantity | value |
|---|---:|
| pairs in shayan | 19,773,733 |
| pairs in cex | 19,912,747 |
| pairs in both | 19,773,733 — **all 19,773,733 with identical totals** |
| pairs only in shayan | **0** |
| pairs only in cex | **139,014** carrying 3,271,291 synapses |
| shared pairs with differing totals / Σ|diff| | 0 / 0 |
| shared pairs by cex weight bin 1-4 / 5-19 / 20-99 / 100+ | 16,041,273 / 3,154,638 / 549,204 / 28,618 pairs, diff 0 in every bin |

Characterization of the 139,014 cex-only pairs (all have both endpoints in both cell tables, so "cells absent from the other table" is *not* a cause):
- **138,971 are autapses** (pre == post), 3,271,078 synapses; e.g. 720575940640039040→itself 155, 720575940623807672→itself 144. Weight histogram peaks at 8-10 synapses per autapse pair (649 pairs have weight 1), i.e. these are not rare noise.
- **43 are non-autapse pairs, every one involving R7 `720575940623940963`** (213 synapses): 720575940642650331 (Dm9)→R7 39, R7→Dm9 720575940642650331 32, R7→Dm8b 720575940629825027 21, R7→Dm11 12, R7→L3 9, ... This cell exists in `neurons.csv.gz` (type R7, group `NO_CONS`, nt_type null, score 0.0) and in `cell_data.parquet`, and has 215 rows in the raw synapse table (167 as pre, 50 as post, 2 autapses; neuropil ME_R 212, LA_R 2, null 1), but has **zero rows** in `connections_princeton_no_threshold.csv.gz`. (Its `group` label `NO_CONS` is not the explanation: 5,008 of 5,074 `NO_CONS` cells do have connections.)
- Systematic offset consistent with a confidence threshold: **none**. If the export had applied a cleft-score filter, low-weight shared pairs would differ; they do not (16.0 M weight-1..4 pairs identical).

## Q6 Neurotransmitter

| type | shayan per-cell `nt_type` counts → majority | cex `type_data.nt` | agree? | why |
|---|---|---|---|---|
| T4a | ACH 1365, null 82, GABA 5, GLUT 5 → **ACH** (93.7%) | ACH | yes | |
| Mi1 | ACH 1484, null 94, GLUT 3 → **ACH** (93.9%) | ACH | yes | |
| L1 | GLUT 835, GABA 322, null 430, ACH 9 → **GLUT** (52.3%) | GLUT | yes | |
| Dm9 | ACH 179, null 26, GABA 1 → **ACH** (86.9%) | **GLUT** | **no** | `schema.TYPE_NEUROTRANSMITTER_GT["Dm9"]="GLUT"` override (src/cex/dataset/schema.py) applied in `raw.py:build_type_data` |
| R7 | null 864, GLUT 206, GABA 185, ACH 66, OCT 7, DA 3, SER 2 → **GLUT** (15.5%) | **HIS** | **no** | `TYPE_NEUROTRANSMITTER_GT["R7"]="HIS"` override; HIS does not exist in the FlyWire prediction at all |

The underlying per-cell `nt_type` values are identical in both packages (both come from `neurons.csv.gz`). The disagreement is purely cex's deliberate literature override for 13 types (R1-6, R7, R8 → HIS; Dm1/6/8/9/12/16/17/19/20 and T1 → GLUT). Note also that shayan's *connections* table carries a per-edge `nt_type` that is never null and constant per pre-cell; for the 19,406 pre-cells whose `neurons.csv` `nt_type` is null (nt_type_score 0.0, often all six `*_avg` = 0) it still says ACH 13,246 / GLUT 3,130 / GABA 2,375 / SER 414 / DA 207 / OCT 34 — a fill value, not information.

## Q7 Side

| | shayan | cex |
|---|---|---|
| `side` | null for all 139,255 cells (hard-coded `pl.lit(None).alias('side')`, src/shayan/datasets/flywire.py `load_neurons`) | int8 from `classification.csv`: L(-1) 69,959, R(1) 69,093, unknown(0) 203 |
| T4a L/R split | n/a | **L 720 / R 737 / unknown 0** (1,457); `ds.get_cell_type_id("T4a", side="left"/"right")` gives the same |
| the T4a neuron 720575940599755718 | `region`(=`group`) "ME.LOP" | side = 1 (right) — consistent with all its synapses being in LOP_R / ME_R (Q8) |

shayan's `region` column is FlyWire's `group` (e.g. T4a: ME.LOP 1,422, ME 30, LOP 3, NO_CONS 2), which is a neuropil grouping and not a side.

## Q8 Neuropil

shayan (per-edge `neuropil`, from the connections export) for 720575940599755718:

| direction | neuropil | rows | synapses |
|---|---|---:|---:|
| outputs | LOP_R | 56 | 208 |
| outputs | ME_R | 34 | 74 |
| inputs | ME_R | 41 | 194 |
| inputs | LOP_R | 29 | 70 |

3 output partners are split across both neuropils (hence 90 rows for 87 partners). Synapse-level check (Q9) agrees: as presynaptic cell it has LOP_R 214 + ME_R 75 = 289 rows, of which 7 are autapses → 282 non-autapse = shayan total. Per-edge `nt_type` is ACH on all 90 rows. cex has **no neuropil anywhere**: `edge_table` columns are `pre_id, post_id, num_syn`; `synapses.parquet` columns are `pre_id, post_id, x_nm, y_nm, z_nm`, because `paths.FLYWIRE_SYNAPSE_COLUMNS` reads only `pre_root_id_720575940, post_root_id_720575940, ctr_x, ctr_y, ctr_z` from the raw table (which does have a `neuropil` column).

## Q9 Synapse locations (T4a neuron as presynaptic cell)

| quantity | shayan `load_synapses()` (`x,y,z` = raw `ctr_x/y/z`) | cex `synapse_table` (`x_nm,y_nm,z_nm`) | diff |
|---|---|---|---|
| rows with pre == neuron | 289 (7 autapses) | 289 (7 autapses) | 0 |
| bbox min | (692,800, 317,392, 156,760) | (692,800, 317,392, 156,760) | identical |
| bbox max | (728,208, 335,856, 219,000) | (728,208, 335,856, 219,000) | identical |
| set of (post, x, y, z) rows | 289 | 289 | **identical under scale (1,1,1)**; scale (4,4,40) or (8,8,40) gives 0 overlap |
| per-partner counts | == cex (88 partners incl. self) | | |
| dtypes | Int64 ×5 + String neuropil | uint32 ×5 | |

Units: **both are nanometers; the raw Princeton `ctr_x/y/z` are already nm, not voxels.** Evidence: the global bbox is x 82,041-912,416, y 33,808-458,448, z 760-281,720 in both tables (≈0.83 × 0.42 × 0.28 mm, the FAFB volume); ~95 % of x and y values are multiples of 4 nm and ~93 % of z values are multiples of 40 nm (with half-voxel offsets because `ctr` is the pre/post midpoint), i.e. nm on the 4×4×40 nm voxel grid. shayan's `SYNAPSE_SCHEMA` docstring gives no unit ("X coordinate"); cex's column name does.

Global synapse-table facts established in this stage: 80,215,790 rows in both; 3,271,078 autapse rows; 126,477 rows with empty neuropil (of which 95,848 autapses, 1 belonging to the dropped R7 cell, and **30,628 non-autapse = exactly shayan's `UNASGD` synapse total**); 139,189 distinct root_ids, all present in both cell tables (so 66 cells have no synapses at all); non-autapse rows excluding the dropped R7 cell = **76,944,499**. Also: the raw synapse CSV has CRLF line endings (both readers cope; naive shell tools do not).

---

## Root causes (every discrepancy, with evidence)

1. **Autapses dropped from the connections export.** `connections_princeton_no_threshold.csv.gz` has 0 rows with pre_root_id == post_root_id (table totals), while the synapse table has 3,271,078 such rows over 138,971 pairs (Q5 `only_cex.n_autapse`, Q9 `global.shayan_autapse_rows`). This accounts for 138,971 of the 139,014 cex-only pairs, the +7 in Q2/Q3 (720575940599755718→itself), and 3,271,078 of the 3,271,291 synapse gap. cex keeps autapses in `edge_table` and `synapse_table` and exposes them via `remove_autapse_Q` (default `False` in `Neuron`/`connected_ids_and_counts`, default `True` in `counts_between_types` and the synapse xyz helpers, src/cex/dataset/connectivity.py and synapse.py). shayan has no autapse concept at all.

2. **One cell missing from the connections export.** R7 `720575940623940963` has 215 synapse rows but 0 connection rows (verified directly: `conn.filter(pre_id==r7)` and `post_id==r7` both empty). Its 213 non-autapse synapses (43 pairs, listed in `cex_only_nonautapse_pairs.json`) are the remaining 139,014 − 138,971 = 43 cex-only pairs and the remaining 3,271,291 − 3,271,078 = 213 synapses, and explain the R7→Dm9 +37 in Q4. Why the upstream export omitted it is not determinable from these files (plausibly a proofreading edit between the synapse-table and connections-table snapshots); it is a data-provenance defect in the export, not a code bug in either package.

3. **Edge granularity and the `UNASGD` sentinel.** shayan edges are (pre, post, neuropil) rows (22,285,323 rows for 19,773,733 pairs; 3 of the T4a neuron's 87 partners are split over LOP_R/ME_R). Consequences: (a) `ConnectionQuery.min_weight(k)` (src/shayan/core/query.py) thresholds each neuropil fragment, not the partner — Q2 gives 155 vs 158 synapses, Q3 167 vs 169; (b) `Circuit.n_connections`, `node_stats().in_degree/out_degree` (`pl.len()` per pre/post, src/shayan/core/circuit.py) count neuropil rows, not partners; (c) synapses with an empty neuropil in the synapse table (30,628 non-autapse) are labelled `UNASGD` in the export and thus stay in shayan's totals — so this is *not* a source of count difference, only a sentinel-vs-null difference.

4. **No thresholds.** Neither table applies a synapse-count threshold (min 1 in both) and the export applies no synapse-level confidence filter (all 19,773,733 shared pairs agree exactly, including 16.0 M pairs of weight 1-4). "no_threshold" in the file name is accurate.

5. **Cell universe: identical.** Both packages ultimately enumerate the same 139,255 root_ids and 8,547 type names (Q1), so nothing in Q2-Q5 is caused by cells missing from a cell table. 66 cells have no synapses at all, 72 have only autapses; they exist in both cell tables and simply have zero (non-autapse) edges.

6. **Neurotransmitter override.** cex's `type_data.nt` applies `schema.TYPE_NEUROTRANSMITTER_GT` (13 types) on top of the per-type majority; shayan reports the raw per-cell prediction. Hence Dm9 (ACH vs GLUT) and R7 (GLUT vs HIS) disagree while per-cell values are identical. Separately, the connections export's per-edge `nt_type` is a never-null fill (ACH for most cells whose prediction is null / score 0), so it is less trustworthy than `neurons.csv`.

7. **Side.** shayan hard-codes `side = None` (src/shayan/datasets/flywire.py, `pl.lit(None).cast(pl.Utf8).alias('side')`) although FlyWire provides side in `classification.csv`, which the shayan data directory does not contain; cex reads it (L/R/unknown as −1/1/0).

8. **Neuropil.** cex discards the raw synapse table's `neuropil` column at preprocessing (`paths.FLYWIRE_SYNAPSE_COLUMNS`) and never had it for edges; shayan has it per edge (79 labels incl. `UNASGD`) and per synapse (null for 126,477 rows).

9. **Units.** Same nm values in both; only naming differs (`x,y,z` Int64 vs `x_nm,y_nm,z_nm` uint32).

10. **Unknown encodings.** shayan: null (`type`, `nt_type`, `side`), plus the upstream sentinels `UNASGD` (neuropil) and `NO_CONS` (group). cex: NaN in `cell_data.parquet` (`type`, `nt_type`) but the string `"NAN"` in `type_data.parquet` (`schema.DEFAULT_UNKNOWN`; 191 types), `0` for unknown side, `"Unknown"` for untyped cells in `Cell.cell_types`/`rid_to_type`, `-1` for not-found ids in `ScalarDict.get_value`.

## Implications for the merged package (recommendations)

1. **Derive connectivity from the synapse table, not from the connections export.** The synapse-table-derived edge table (cex semantics) is a superset that is exactly reproducible (Σ = number of synapse rows) and free of the two export defects (dropped autapses, dropped R7 cell). Keep the per-(pre, post) `num_syn` table as the canonical edge list and treat any Princeton connections CSV as a derived/legacy input. Document that FlyWire's own connections export = synapse table minus autapses minus (at least) one cell.

2. **Edges are per (pre, post) pair; neuropil is an optional secondary breakdown.** Store the canonical edge as one row per pair (cex). Provide neuropil via a separate `(pre, post, neuropil) → num_syn` table or a per-synapse `neuropil` column (which cex should stop discarding: add `neuropil` to `FLYWIRE_SYNAPSE_COLUMNS` and persist it as a categorical). All thresholds (`min_weight` / `min_syn_per_rid`), partner counts and degrees must be computed on the pair total, never on neuropil fragments — this fixes shayan's `min_weight`, `n_connections` and `node_stats` semantics (Q2/Q3: 155 vs 158).

3. **Autapses: keep them in the data, make removal an explicit flag, default it consistently.** Adopt cex's `remove_autapse_Q` but pick one default across the whole API (today `Neuron` defaults to keep, `counts_between_types` to remove). Recommended: keep by default in raw tables/edge queries, and expose `remove_autapse=True` as the default for per-neuron I/O summaries where users expect "partners". Never silently drop them at preprocessing.

4. **IDs: keep the external root_id as the public identifier and the dense id as an internal index.** cex's dense `id` is what makes the 139,255×139,255 CSR matrices and `bincount`-style aggregations fast, but it is dataset-version-specific and meaningless to users; shayan's root_id is what appears in Codex/Neuroglancer. Public API should accept/return root_ids (Int64) and convert internally via a `ScalarDict`-style lookup; never expose `id` in user-facing frames (or name it `idx` and document it as positional).

5. **Cell table: use the same source and keep `side`.** Both universes are identical, so either source works, but the merged loader must ingest `classification.csv` (for `side`, `super_class`, `class`, `flow`, ...) — shayan's `side = None` is a data-directory omission, not a dataset limitation. Encode side as a small categorical/string (`"L"/"R"/None`), not as −1/1/0 sentinel ints in user-facing tables.

6. **Units: standardize on nanometers and say so in the column name.** Store `x_nm, y_nm, z_nm` (uint32 is sufficient: max 912,416). Any loader receiving voxel coordinates must convert with the dataset's voxel size at ingest. The Princeton `ctr_*` columns are already nm; do not apply a (4,4,40) factor.

7. **Unknowns: use real nulls everywhere in stored tables; convert sentinels at ingest.** Map `UNASGD` → null neuropil (or keep as an explicit category but never let it masquerade as a region in aggregations), `"NAN"` → null in `type_data`, and do not mint `"Unknown"`/`"NAN"` strings in stored data — produce them only at display time. Keep `"NO_CONS"` only if its meaning is documented (it is a `group` label, not "no connections").

8. **Neurotransmitter: store the per-cell prediction as data, keep type-level overrides as a separate, versioned, documented layer.** Keep `nt_type` + `nt_type_score` (and the six `*_avg` columns) from `neurons.csv`; compute a type majority from non-null cells only, and apply `TYPE_NEUROTRANSMITTER_GT`-style corrections through an explicit `nt_source` column (`"prediction"` vs `"literature"`) so Dm9/R7 disagreements are visible rather than silent. Drop the connections export's per-edge `nt_type` (it is a never-null fill of the pre-cell label).

9. **Preprocessing must assert reconciliation invariants.** Add checks like the ones this crosscheck used: Σ edge num_syn == synapse rows; every synapse endpoint in the cell table; number of cells absent from edges reported (66 no-synapse + 72 autapse-only here); and, if a vendor connections table is also ingested, an explicit report of its deviation from the synapse-derived table (autapses, missing cells), so a future export with different defects is caught immediately.

10. **Backends.** The two backends give numerically identical answers once semantics are aligned; the choice (Polars vs pandas+scipy.sparse) is an engineering decision. The sparse CSR/CSC count matrix (cex) is the right structure for per-neuron/per-type lookups; a columnar (Polars/Arrow) edge table is the right structure for filtered joins and neuropil breakdowns. The merged package can keep both views over one canonical parquet edge table.
