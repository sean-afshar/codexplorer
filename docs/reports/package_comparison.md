# codexplorer: shayan vs cex — pros and cons (2026-09-10)

Scope: src/shayan (condensed connect_tome, ~4.0k LOC, polars) and src/cex (~3.1k LOC + 3 notebooks, pandas/scipy.sparse).
Both import cleanly from the shared .venv; pytest: 11 cex tests pass on synthetic data, 24 shayan tests skip without FlyWire data.

## shayan — pros
- Clear layering: AbstractDataset -> Connectome -> ConnectionQuery -> Circuit; canonical typed Polars schemas + validate_schema.
- Pure-Polars lazy query builder; semi-joins for type filters; count() without materializing.
- Rich Circuit analytics: per-edge and per-type %-input/%-output normalization with geometric mean, honoring the query's min_weight; node/type stats, hubs, reciprocal edges, degree distributions, neuropil aggregation.
- Only package with morphology + biophysics: navis SWC loading from zip, segment_natural, Compartmentalization, CompartmentalModel (steady-state + implicit Euler, Borst & Meier defaults, dimensionally correct units), parameter scan.
- Visualization: plotly 3D, matplotlib 2D, neuroglancer URLs for CAVE and FlyWire viewers with compartment annotations.
- click+rich CLI (connect-tome info/query/types).
- Lazy top-level imports: `import shayan` ~0.1 s.
- Extensive Google-style docstrings; graceful empty-circuit handling; neuropil + per-edge nt_type preserved.
- Ground-truth integration tests against a known T4a neuron.

## shayan — cons
- No preprocessed/on-disk layer: every process re-parses gz CSVs; CLI reloads the full table per invocation. "scan" mode is a stub.
- side is always null for FlyWire; region is populated from the cell-type `group` column (semantic mismatch).
- Synapse IDs reconstructed by scraping digits from the CSV header (brittle).
- Many advertised stubs: segment_fixed_length/uniform_radius/electrotonic, Compartmentalization.to_hines_matrix/summary/plot, MaleCNSDataset (referenced by CLI, never defined).
- Performance: O(S^2) parent-segment lookup in segment_natural; per-compartment DataFrame filters in loops; spsolve refactorized every timestep; aggregate_by_type_normalized recomputes global totals per call.
- Unit footgun: skeletons default to nm, model/neuroglancer assume um; nothing records units.
- Tests require real 15 GB data; function-scoped eager fixture reloads everything per test; zero coverage of morphology/models/viz/CLI.
- Dtype/shape drift: Circuit.empty weight Int64 vs schema Int32; empty results sometimes schemaless.
- Duplication (pre_type/post_type, Circuit filters mirror query filters, CLI blocks, two neuroglancer configs); dead neuron_index; mutable query builder.
- No type x type matrix, no side-aware queries, no per-neuron synapse xyz API.

## cex — pros
- Explicit preprocessing -> normalized parquet/npz layer, documented in docs/cex_data_layout.md; fast startup, no re-parsing.
- Dense assigned `id` (0..N-1) indexing sparse CSR/CSC matrices directly; ScalarDict vectorized rid<->id lookup.
- Precomputed type x type CSR matrix with order integrity check.
- Real multi-dataset support: FlyWire + Male CNS v0.9 + v1.0 via alias registry and factory; MCNS<->FlyWire type-name conversion; R7/R8 merging.
- Side encoded (-1/0/1) and used in queries; NT vocabulary + sign map; hex column (p,q) assignments.
- Synapse xyz table with per-id index maps; per-pair synapse rows.
- Neuron object: I/O parsing, top-N partner types, Neuroglancer (spelunker) views with segmentation + synapse annotation layers; Codex URLs.
- Offline per-type IO-stat CSVs with vectorized aggregation and multiprocessing.
- Fast synthetic-fixture tests (11 pass in 0.3 s, no data needed); dtype-minimization guard test.
- Download helper for preprocessed archives; small clean dependency footprint (numpy/pandas/scipy/pyarrow).

## cex — cons
- pandas + Python loops in hot spots (Column._get_columns per-key .loc, Synapse._collect_rows per-id iloc + concat, _group_root_ids nested dict comprehensions); polars declared but never imported.
- Memory: entire synapse parquet in pandas + two per-id index dicts; cell table loaded twice; CSR and CSC copies.
- 22 bare asserts as validation (vanish under -O); inconsistent unknown sentinel ("NAN" vs "Unknown").
- Column-type features (column_type_list, col_pq_type_to_rid) depend on a `type` column that build_column_data never writes.
- get_dataset() creates five directories as a side effect; default root is cwd-relative "./data".
- FlyWire-centric viz: annotation voxel size (4,4,40) hard-coded, MCNS has no viewer path (NotImplementedError); FlyWire NT ground truth applied to MCNS in notebook.
- No neuropil anywhere; no per-edge nt_type; no morphology, no plotting, no CLI.
- MaleCNSDataset entirely untested; ~30-40 lines of MCNS preprocessing exist only in notebook 01.
- Dead constants and legacy residue (unused MCNS maps, _conn_to_str helpers exported with underscores, unreachable else branch); Dropbox rlkey tokens embedded in notebooks.
- One-line docstrings; some docstrings mismatch signatures; builtins `id`/`dir` shadowed; `side` means left/right in Cell but pre/post in Synapse.
- Return-type instability: load_cell_type_io_stat returns DataFrame or dict depending on files present.

## Overlaps and conflicts
- Both define FlyWireDataset with incompatible semantics (raw-CSV loader vs normalized facade).
- IDs: shayan uses raw int64 root_id everywhere; cex uses dense id with rid alongside. Every merged API must pick one.
- Dataframes: polars (shayan) vs pandas (cex). navis returns pandas, so shayan already converts at the morphology boundary.
- Neuroglancer URL generation duplicated (shayan: ngl.flywire.ai + ngl.cave-explorer.org; cex: spelunker.cave-explorer.org + codex).
- Type-level aggregation duplicated: Circuit.aggregate_by_type(_normalized) vs Connectivity.num_syn_between_type_groups / IO stats.
- Unknown-type handling: shayan null, cex "NAN"/"Unknown".
- Data layout: shayan expects data/flywire_fafb raw exports; cex expects data/flywire/{download_data,preprocessed}.
- pyproject: single root file; cex uses none of navis/plotly/matplotlib/click/rich; shayan uses none of pyarrow directly.
