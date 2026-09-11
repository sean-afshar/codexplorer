# Merged connectome package: design plan

Date: 2026-09-10. Inputs: static review of `src/shayan` and `src/cex`, hands-on
benchmarks of both against FlyWire FAFB v783 and Male CNS v0.9, a cross-package
consistency check, and a backend micro-benchmark on the normalized tables.
Raw reports are in `docs/reports/` (`bench_shayan_report.md`,
`bench_cex_report.md`, `crosscheck_report.md`, `backend_bench_report.md`,
`cex_flywire_data_report.md`, `cex_mcns_data_report.md`).

Priorities, in order (from the authors): speed of loading and manipulation;
an API familiar to pandas/numpy users; first-class connectivity, synapse
locations and morphology; FlyWire and Male CNS today, more datasets later;
clean code.

Package name below is the placeholder `cx`. Pick a real name in phase 0.

---

## 1. What the evidence says

### 1.1 Both pipelines are numerically the same data

The cross-check found identical cell universes (139,255 cells, 8,547 types,
same type per cell), identical synapse rows and identical nanometer
coordinates. Every connectivity difference traced to the vendor
`connections_princeton_no_threshold.csv.gz` that `shayan` reads:

| defect in the connections export | effect |
|---|---|
| all autapses dropped | 138,971 pairs / 3,271,078 synapses missing |
| one R7 cell absent entirely | 43 pairs / 213 synapses missing |
| rows are (pre, post, neuropil) | `min_weight` and degrees act on neuropil fragments (155 vs 158 syn for the reference T4a) |

80,215,790 - 3,271,078 - 213 = 76,944,499 exactly. There is no threshold
difference. Conclusion: derive connectivity from the synapse table, keep
autapses in the data, remove them by flag.

### 1.2 Load and query timings (FlyWire, same machine)

| operation | shayan (raw gz CSV, polars) | cex (parquet, pandas+scipy) |
|---|---:|---:|
| import | 0.07 s | 0.20 s |
| cell table | 0.02 s | 0.02 s |
| edge table | 1.4 s (22.3M neuropil rows) | 0.03 s (19.9M pair rows) |
| sparse matrix build | n/a (no matrix) | 0.11 s CSR + 0.09 s CSC |
| one neuron's outputs | 9 ms first, 6 ms repeat | 0.3 ms first, 0.06 ms repeat |
| T4a x LPi14 block | 29 ms (matrix hand-built) | 0.13 s first, 0.3 ms repeat |
| top-10 input types to Dm9, normalized | 0.51 s every call | 5 ms (precomputed CSV) / 6 ms live |
| full type x type | 0.70 s, long form only | 8 ms (precomputed npz) |
| synapse table load | 18 s, 11.9 GB peak | 0.22 s, 3.8 GB RSS |
| one skeleton + segment + cable solve | 1.2 s | not supported |

### 1.3 Backend micro-benchmark (same normalized parquet, fresh interpreters, medians)

| operation | polars | pandas | pandas[pyarrow] | scipy.sparse + numpy |
|---|---:|---:|---:|---:|
| load 19.9M-row edge table | 16 ms / 0.35 GB | 52 ms / 0.64 GB | 46 ms / 0.43 GB | +120 ms to build CSR+CSC |
| one neuron's partners, per call | 550 us (100 us with search_sorted) | 3.3 ms | 21 ms | 9 us (0.2 us on raw indptr) |
| T4a x LPi14 block | 9 ms | 87 ms | 140 ms | 0.3 ms |
| top-10 input types, normalized | 21 ms | 122 ms | 158 ms | 0.9 ms |
| full 8547 x 8547 type matrix | 0.32 s | 0.75 s / 5.1 GB | 1.3 s | 0.35 s (or 8 ms from npz) |
| synapse xyz for one neuron, cold | 102 ms full read | 257 ms | 324 ms | lazy scan: 2.6 ms, 0.13 GB |
| rid -> id, 1e6 lookups | 3-5 ms | 8 ms | - | 12 ms (searchsorted) |
| polars -> pandas, 20k rows | 0.2 ms (zero-copy with arrow dtypes) | | | |

Two footguns surfaced: pandas keeps `uint16` through `groupby().sum()` and
silently overflows on `100 * s / total`; `np.searchsorted` with a Python int
key casts the whole 20M array per call (1.9 ms instead of 1.9 us).

The lazy synapse path is fast only because `synapses.parquet` is sorted by
`pre_id` in 1M-row row groups and polars prunes to 1 of 77 groups. Queries by
`post_id` fall back to a 50 ms full scan.

### 1.4 Decisions the evidence settles

1. **On disk: Parquet, normalized, preprocessed once** (cex model). Raw vendor
   files are ingestion inputs only.
2. **Connectivity derived from synapses**, one row per (pre, post), autapses
   kept, neuropil preserved in a secondary table.
3. **In memory: three layers.** Polars DataFrames hold loaded tables (fastest
   load, lowest RSS). scipy CSR/CSC plus numpy serve every hot path (100 to
   1000x faster than any DataFrame filter). pandas is what the user receives.
4. **Public IDs are root IDs (int64).** Dense 0..N-1 ids exist internally
   for sparse indexing and are exposed only as documented array positions.
5. **Synapses are never loaded wholesale.** `scan_parquet` with predicate
   pushdown, 0.5 ms per neuron, 0.13 GB.
6. **Precomputed I/O-stat CSVs are dropped.** Live computation from CSC plus
   `bincount` takes 1 ms; the 17,094 CSVs (784 MB) buy nothing.
7. **Precomputed type x type npz is kept** (8 ms load versus 0.35 s recompute).
8. **pandas with pyarrow dtypes is rejected**: slower than plain pandas on every
   filter, isin and pivot path.

---

## 2. Data layer

### 2.1 Directory layout

```
data/<dataset>/            e.g. data/flywire_783/, data/mcns_v0.9/
  raw/                     vendor downloads or symlinks; never read at runtime
  tables/                  normalized Parquet + npz; the only runtime input
    manifest.json          dataset id, version, source file hashes, row counts,
                           invariants checked, cx schema version
```

Runtime never creates directories. A missing table raises with the exact
path and the preprocessing command that produces it.

### 2.2 Normalized tables (schema v1)

| file | columns | notes |
|---|---|---|
| `cells.parquet` | `id u32` (row position), `root_id i64`, `type str?`, `side str?` (`"L"`,`"R"`,`"M"`, null), `nt str?`, `nt_score f32?`, `superclass str?`, `class str?`, `subclass str?`, `hemilineage str?`, `flow str?`, dataset extras | real nulls, no `"NAN"`/`"Unknown"` sentinels |
| `types.parquet` | `type str`, `type_idx u32` (row position), `n_cells u32`, `nt str?`, `nt_source str?` (`"prediction"`/`"literature"`), majority-vote metadata | row order defines the type matrix axes |
| `edges.parquet` | `pre u32`, `post u32`, `n_syn u32` | sorted by (`pre`, `post`); derived from synapses; autapses included |
| `edges_by_neuropil.parquet` | `pre u32`, `post u32`, `neuropil str?`, `n_syn u32` | secondary; absent for datasets without neuropil labels |
| `synapses.parquet` | `pre u32`, `post u32`, `x_nm u32`, `y_nm u32`, `z_nm u32`, `neuropil str?`, dataset extras | sorted by `pre`, 1,048,576-row row groups; **invariant** |
| `synapses_by_post.parquet` | same columns | optional copy sorted by `post`, for input-side synapse queries |
| `type_matrix.npz` | CSR `data u64`, `indices i32`, `indptr i32`, `shape`, `type_list` | axes = `types.parquet` order |
| `columns.parquet` | `id u32`, `p i16`, `q i16`, `side str`, dataset extras | optical-lobe hex columns; optional |
| `skeletons/` | dataset-native skeleton store (FlyWire: the SWC zip; MCNS: TBD) plus `skeleton_index.parquet` (`root_id`, `member`, `n_nodes`, `units`) | optional |

`n_syn` is `u32`, not `u16`, to remove the overflow footgun at the source.

### 2.3 Invariants checked at preprocessing and recorded in the manifest

- `cells.id == arange(N)`; `root_id` unique.
- Every `pre`/`post` in edges and synapses is a valid `id`.
- `edges.n_syn.sum() == len(synapses)`.
- `type_matrix.sum()` equals the edge sum restricted to typed cells; the
  difference (untyped cells) is reported.
- Synapse row groups have disjoint, monotone `pre` ranges.
- Cells absent from all edges are counted and listed.
- Vendor-table deviations are reported (for FlyWire: autapse count and the
  cells the connections export omits).

### 2.4 Preprocessing as library code

`cx.ingest` holds one `Source` per dataset. A `Source` maps raw vendor files to
the normalized tables and nothing else. Notebooks become thin callers.

```python
cx.ingest.flywire.build("data/flywire_783", raw="~/Code/connect_tome/data/flywire_fafb")
cx.ingest.mcns.build("data/mcns_v0.9", raw=..., release="v0.9")
```

Requirements taken from the hands-on runs: stream the 6.8 GB MCNS synapse
feather in row-group batches (the notebook's full pandas load would exceed 35
GB; streaming ran in 57 s at 10 GB); read FlyWire `classification.csv` for
`side`; keep the raw `neuropil` column on FlyWire synapses; apply the
literature neurotransmitter overrides as a labelled layer (`nt_source`), not a
silent overwrite; validate the four raw MCNS files that are actually used and
do not require the three that nothing reads.

A `cx download <dataset>` command fetches a prebuilt `tables/` archive when one
exists (the FlyWire archive downloads in 30 s).

---

## 3. Runtime API

Design rule: nouns are attributes, verbs are methods, results are pandas
DataFrames, numpy arrays, or scipy sparse matrices. Brackets index by root ID
or type name the way a pandas user expects. No fluent builder, no
dict-of-arrays returns, no `_Q` suffixes, no browser or filesystem side
effects.

### 3.1 Opening a dataset

```python
import cx

ds = cx.open("flywire")                 # resolves data/flywire_783 via config or CX_DATA env
ds = cx.open("mcns", version="v0.9")
ds = cx.open("/abs/path/to/data/mcns_v0.9")

ds.name, ds.version, ds.n_cells, ds.n_synapses
ds.info()                               # manifest summary
```

Opening reads the manifest only. Tables load on first attribute access and
stay cached on the object. Everything is cheap except the CSR/CSC build
(~0.3 s, once) and skeletons.

### 3.2 Tables (pandas by default)

```python
ds.cells                    # DataFrame indexed by root_id: type, side, nt, nt_score, class, ...
ds.cells.loc[720575940599755718]
ds.cells[ds.cells.type == "T4a"]
ds.types                    # DataFrame indexed by type: n_cells, nt, nt_source, ...
ds.edges                    # DataFrame: pre, post (root ids), n_syn   (19.9M rows, 13 ms to convert)
ds.edges_by_neuropil        # DataFrame or None
ds.columns                  # DataFrame indexed by root_id: p, q, side  (or None)

cx.config.frame = "polars"  # switch all table returns to polars; or per call: ds.cells_(frame="polars")
```

Internally these are polars frames; conversion happens at the boundary
(0.2 ms for typical results, zero-copy where possible). Integer results are
upcast to int64 before they leave the package.

### 3.3 Selecting neurons

```python
n = ds[720575940599755718]              # Neuron
grp = ds["T4a"]                         # NeuronSet, all cells of a type
grp = ds[["T4a", "T4b"]]                # NeuronSet, union of types
grp = ds[[rid1, rid2, rid3]]            # NeuronSet by ids
grp = ds.select(type="T4a", side="R")   # keyword filters over ds.cells columns
grp = ds.select(ds.cells.nt == "GABA")  # any boolean mask over ds.cells

grp.ids            # numpy int64 root ids
grp.cells          # DataFrame slice of ds.cells
len(grp); grp & other; grp | other; grp - other
```

`Neuron` is a `NeuronSet` of size one; every method below works on both.

### 3.4 Connectivity

```python
# partner tables
n.outputs()                              # DataFrame: post, type, side, n_syn  (sorted desc)
n.outputs(min_syn=5)                     # threshold applies to the pair total, never to fragments
n.inputs(by="type")                      # DataFrame: type, n_syn, n_partners, frac_input
grp.inputs(by="type", normalize="input") # frac of total input to the set
n.outputs(by="neuropil")                 # uses edges_by_neuropil when present
n.partners()                             # both directions

# matrices: numpy-style indexing on the connectivity object
ds.connectivity["T4a", "LPi14"]          # ConnBlock: rows T4a cells, cols LPi14 cells
ds.connectivity[grp_a, grp_b]            # by NeuronSets
ds.connectivity[ids_a, ids_b]            # by root-id arrays
blk = ds.connectivity["T4a", "LPi14"]
blk.values                               # numpy (n_pre, n_post) dense; includes zero rows/cols
blk.sparse                               # scipy csr, same shape
blk.frame                                # pandas DataFrame, index=pre root ids, columns=post root ids
blk.sum(), blk.row_ids, blk.col_ids, blk.row_types, blk.col_types

# type-level matrix
ds.connectivity.types["T4a", "LPi14"]                  # int
ds.connectivity.types.loc[["T4a","T4b"], ["LPi14"]]    # DataFrame block
ds.connectivity.types.frame                            # full 8547 x 8547 DataFrame (292 MB) on demand
ds.connectivity.types.sparse                           # csr

# whole-brain sparse matrix for power users
ds.connectivity.sparse                    # scipy csr (N x N), rows/cols in ds.cells order
ds.connectivity.sparse_t                  # csc view for input-side slicing
ds.connectivity.autapses                  # bool flag, default False -> autapses masked in every result
```

Normalization ported from `shayan`: `frac_input`, `frac_output`, and the
geometric-mean `weight_norm`, with denominators computed over the full
dataset once and cached (the 0.5 s recompute per call disappears).

Graph statistics ported from `shayan.Circuit` onto `NeuronSet`:
`grp.subgraph()` returns edges within the set; `grp.degree()`,
`grp.hubs(k)`, `grp.reciprocal()`, `grp.degree_distribution()`.

### 3.5 Synapse locations

```python
n.synapses()                          # DataFrame: pre, post, x_nm, y_nm, z_nm, neuropil  (0.5 ms)
n.synapses(direction="in")            # uses synapses_by_post when present, else scans
n.synapses(partner="LPi14")
ds.synapses.between(pre, post)        # any Neuron/NeuronSet/ids
ds.synapses.in_box(xmin, xmax, ...)   # lazy filter, collected on return
ds.synapses.scan()                    # raw polars LazyFrame for anything else
n.synapses().to_numpy()               # (k, 3) uint32 via a helper .xyz
```

Never a full load unless the user asks: `ds.synapses.load()` returns the whole
table (0.1 s, 1.6 GB in polars) for people who want it.

### 3.6 Morphology and models (ported from `shayan`)

```python
sk = n.skeleton()                     # navis TreeNeuron, ALWAYS micrometers; sk.units set
sk = n.skeleton(units="nm")
grp.skeletons(max_n=5)                # NeuronList; progress bar off by default

comp = cx.morph.segment(sk, method="natural", min_length_um=0.5)
                                      # zero-length compartments merged, not warned about
comp.table                            # DataFrame of compartments (length, radius, parent, ...)
comp.hines()                          # scipy sparse conductance structure
comp.summary()

m = cx.models.Cable(comp, Rm=8000, Ra=400, Cm=0.6)
V = m.steady_state(inject={0: 10e-12})          # mV, numpy
Vt = m.transient(inject={0: pulse}, dt=1e-3)     # factorize once with splu, reuse per step
m.input_resistance(0)
cx.models.scan(comp, Ra=[...], Rm=[...])
```

Fixes carried in from the benchmark: default micrometers with the unit stored
on the object; zero-length compartments merged (the reference Dm9 solve
returned all-NaN otherwise); parent-segment lookup made linear; the three
unimplemented segmentation strategies removed from the public surface until
they exist.

### 3.7 Viewing

```python
n.view()                               # returns a Neuroglancer URL string; never opens a browser
n.view(viewer="spelunker" | "cave" | "flywire" | "codex")
n.view(partners="in", top=5, synapses=True)   # cex's layered view: segments + synapse points
comp.view()                            # compartment annotations (state kept short via the state server when available)
cx.viz.plot3d(comp); cx.viz.plot2d(comp)      # plotly / matplotlib, imported lazily
```

Viewer configuration (segmentation source, voxel size, host) is a per-dataset
table in the `Source`, not hard-coded in functions, so MCNS gets a viewer too.

### 3.8 Cross-dataset

```python
fw, mc = cx.open("flywire"), cx.open("mcns")
mc.type_map                            # DataFrame: mcns_type <-> flywire_type (from the vendor table)
mc.types_like("LPi14")                 # -> ["LPi12"]
cx.compare(fw["T4a"], mc["T4a"]).inputs(by="type")   # aligned DataFrame with both datasets as columns
```

---

## 4. Package layout

```
src/cx/
  __init__.py          open(), config, lazy attribute loading for heavy submodules
  schema.py            table schemas, dtypes, invariants, schema version
  dataset.py           Dataset: manifest, table cache, __getitem__, select()
  neurons.py           Neuron, NeuronSet
  connectivity.py      Connectivity, ConnBlock, TypeMatrix (scipy + numpy hot paths)
  synapses.py          lazy synapse access
  frames.py            polars <-> pandas boundary, int upcasting, frame config
  ingest/
    base.py            Source protocol, manifest writer, invariant checks
    flywire.py         FlyWire FAFB source
    mcns.py            Male CNS source (streamed synapse build)
    download.py        prebuilt-archive fetch
  morph/               loader (navis), segmentation, compartmentalization
  models/              cable model, parameter scan
  viz/                 neuroglancer/codex URL builders, plotly, matplotlib
  cli.py               cx download | ingest | info | query | types
tests/
  conftest.py          synthetic 4-cell fixture written to tmp_path (from cex)
  test_*.py            unit tests on the fixture; integration tests gated on CX_DATA with the T4a ground truth (from shayan)
docs/
  schema.md, quickstart.md, migrating_from_shayan.md, migrating_from_cex.md
```

Dependencies: `polars`, `numpy`, `scipy`, `pyarrow`, `pandas`; optional
extras `morph` (navis), `viz` (plotly, matplotlib), `cli` (click, rich). Import
of `cx` stays under 0.15 s by loading pandas lazily and morph/viz on first use.

---

## 5. What comes from where

| capability | take from | change |
|---|---|---|
| normalized Parquet layout, dense ids, manifest | cex | real nulls, u32 counts, side as strings, neuropil kept, manifest added |
| MCNS ingestion, type-name conversion, R7/R8 merge | cex notebook + dataset.py | move notebook logic into `ingest/mcns.py`; stream synapses |
| CSR/CSC connectivity, type matrix npz, id<->rid vectorized lookup | cex | int32 indices, lazy build, autapse flag with one default |
| Codex + spelunker URLs with annotation layers | cex | per-dataset viewer config; no browser side effect |
| synthetic-fixture tests, download helper | cex | as is |
| polars loaders, fast CSV/IPC reading | shayan | become ingestion utilities |
| %input / %output / geometric-mean normalization, graph stats | shayan Circuit | onto NeuronSet; denominators cached |
| skeleton loading, natural segmentation, Compartmentalization, cable model, parameter scan | shayan | units fixed, zero-length merge, linear parent lookup, splu |
| plotly / matplotlib compartment plots, CAVE/FlyWire viewer URLs | shayan | merged into `viz/` with cex's builders |
| CLI, docstring style, T4a ground-truth tests | shayan | CLI grows `download`/`ingest`; tests gated on `CX_DATA` |
| dropped | shayan: raw-CSV runtime loaders, `ConnectionQuery` builder, `Circuit` as a public type, scan mode stub, unimplemented strategies | |
| dropped | cex: dict-of-arrays returns, `_Q` suffixes, I/O-stat CSV tree, directory-creating constructor, assert-based validation, `"NAN"` sentinels, notebooks as the preprocessing entry point | |

---

## 6. Phases

**Phase 0, decisions and scaffold (1 to 2 days).** Package name. Confirm
pandas-by-default returns. Create `src/cx` with `schema.py`, `frames.py`,
`conftest.py` fixture and CI running the fixture tests. Add
`docs/schema.md`.

**Phase 1, ingestion (3 to 5 days).** `ingest/base.py` with manifest and
invariants; `ingest/flywire.py` producing every table from the raw exports
(including `side`, `neuropil`, `edges_by_neuropil`, sorted synapses,
optional `synapses_by_post`); `ingest/mcns.py` from the notebook logic with
streamed synapses; `cx download`. Acceptance: FlyWire tables match the
downloaded archive on shared columns; MCNS edges equal the vendor weights
table (already verified once); all invariants pass.

**Phase 2, core runtime (3 to 5 days).** `Dataset`, tables with lazy cache,
`Neuron`/`NeuronSet`, `Connectivity` with `ConnBlock` and `TypeMatrix`,
`synapses` lazy access. Acceptance: T4a ground-truth tests pass on both the
per-pair semantics (18 partners, 158 synapses at >=5, 165 with autapses);
type totals match the cross-check (49,069; 92,542; 192,286; 28,699); per-call
latencies within 2x of the micro-benchmark.

**Phase 3, analytics (2 to 3 days).** Normalization and graph statistics on
`NeuronSet`; cross-dataset type mapping and `cx.compare`. Acceptance:
normalized Dm9 input profile reproduces `shayan`'s percentages after autapse
alignment.

**Phase 4, morphology, models, viz (3 to 5 days).** Port with the listed
fixes; unify viewer URL builders behind per-dataset config; lazy imports.
Acceptance: reference Dm9 gives a finite steady-state solution (6.2 mV at
10 pA with the merge fix); one-skeleton load stays under 1 s.

**Phase 5, CLI, docs, migration (2 to 3 days).** Port `connect-tome`
commands; write the two migration guides; convert the three notebooks to
the new API; mark `src/shayan` and `src/cex` deprecated, then delete once
both authors have migrated their working code.

---

## 7. Open questions for the two authors

1. Package name (`cx` is a placeholder).
2. Return pandas by default with a polars switch (recommended here), or
   polars by default. The cost difference is negligible; it is purely about
   familiarity.
3. Keep `Circuit` as a named concept, or is `NeuronSet.subgraph()` enough?
4. Ship a `synapses_by_post` copy (doubles synapse storage to ~1.3 GB per
   dataset) or accept 50 ms input-side synapse queries?
5. Where should MCNS skeletons come from, and in what units?
6. Literature neurotransmitter overrides: keep cex's 13-type table as a
   versioned layer, extend it, or drop it in favor of predictions only?
