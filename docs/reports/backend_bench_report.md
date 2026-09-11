# Backend micro-benchmark for the merged fly-connectome package

Scripts/outputs: `backend_bench.py` (driver + workers), `backend_bench.json` (all runs + medians), this report.
Machine: macOS (Darwin 25.3), 14 polars threads. Python 3.13, polars 1.44.1, pandas 3.0.5, pyarrow 25.0.1, numpy 2.5.2, scipy 1.18.1.

## Method

- Every task runs in a **fresh interpreter subprocess**, 3 repetitions, **median** reported. Cold-load numbers therefore include a cold page cache only on the first rep; the spread across reps was small (e.g. polars edge load 15/16/17 ms, pandas synapse load 253/258/257 ms), so the OS cache was warm for all reps in practice.
- Peak RSS is `getrusage(RUSAGE_SELF).ru_maxrss` reported by the worker itself (bytes on macOS); the driver's `getrusage(RUSAGE_CHILDREN).ru_maxrss` (5.09 GB, the pandas O5 run) matches the largest worker. Everything stayed under 10 GB.
- "First" = first call after data is loaded in the process; "per call/repeat" = mean over N different random cell ids (N given per row; smaller N for backends where one call takes >10 ms).
- Data facts that matter, verified with pyarrow row-group statistics: `cell_to_cell_syn_count.parquet` (19 row groups) and `synapses.parquet` (77 row groups of 1,048,576 rows) are both **sorted by `pre_id`** with disjoint, monotone per-row-group min/max; `cell_data.id == arange(N)` (row position = id); `npz.type_list` equals `type_data.type` in order. Both files are Snappy + dictionary encoded. Reference neuron rid 720575940599755718 -> id 97249 (T4a). T4a has 1,457 cells, LPi14 4, Dm9 206, Mi1 1,581.
- T4a -> Dm9 has **zero** synapses (T4a outputs are in the lobula plate), so it is reported as the empty-result case and Mi1 -> T4a (1,581 x 1,457, 92,542 synapses) was added as the real "larger" case.

## Import time (fresh interpreter)

| library | import (median of 3) | peak RSS |
| --- | --- | --- |
| numpy | 20.8 ms | 0.04 GB |
| pyarrow | 43.5 ms | 0.06 GB |
| polars | 38.8 ms | 0.05 GB |
| pandas | 139.6 ms | 0.10 GB |
| scipy_sparse | 64.4 ms | 0.05 GB |

## O1 cold load of the 19.9M-row edge table

| backend | cold load | in-memory size | peak RSS (process) | notes |
| --- | --- | --- | --- | --- |
| A polars eager | 15.9 ms | 0.20 GB | 0.35 GB |  |
| B polars lazy .collect() | 16.2 ms | 0.20 GB | 0.35 GB |  |
| C pandas (numpy dtypes) | 52.0 ms | 0.20 GB | 0.64 GB | uint32/uint32/uint16 preserved |
| D pandas (pyarrow dtypes) | 46.0 ms | 0.20 GB | 0.43 GB |  |
| E scipy CSR+CSC | 314.5 ms | 0.48 GB | 1.44 GB | read 197.9 ms + CSR 73.1 ms (direct; via COO 92.6 ms) + CSC 44.4 ms; int64 data/indices |
| F numpy 3 arrays (pyarrow) | 197.0 ms | 0.20 GB | 0.63 GB |  |

## O2 single-neuron outputs (post partners + counts)

| backend | first call (warm data) | per call, random cells | peak RSS | notes |
| --- | --- | --- | --- | --- |
| A polars filter(pre_id == id) | 0.9 ms | 553 us | 0.35 GB | rid->id lookup 0.3 ms |
| A' polars set_sorted + filter |  | 167 us |  | same result, sortedness hint |
| A'' polars search_sorted + slice |  | 100 us |  | same result, exploits sort order |
| B polars scan_parquet+filter+collect | 2.6 ms | 457 us | 0.13 GB | Predicate pushdown: reading 1 / 19 row groups (n=200) |
| C pandas boolean mask | 4.1 ms | 3,282 us | 0.65 GB | n=300; rid->id 0.4 ms |
| C' pandas set_index('pre_id').loc[[id]] | 10.3 ms (index build) | 159 us |  | sorted non-unique index |
| D pandas[pyarrow] boolean mask | 28.5 ms | 21,069 us | 0.60 GB | n=300 |
| E scipy csr[id] (outputs) | 0.1 ms | 9.36 us | 1.12 GB | sparse row -> .indices/.data |
| E' numpy indptr slice on CSR arrays |  | 0.21 us |  | views, no allocation |
| E csc[:, id] (inputs) |  | 9.28 us |  | raw indptr slice 0.20 us |
| F numpy searchsorted + slice | 0.0 ms | 1.94 us | 0.63 GB | keys must match dtype (uint32) or numpy casts the whole array per call (1.9 ms) |

Agreement: all backends return n=88 partners, sum=289 synapses for id 97249 (rid 720575940599755718); CSC inputs n=70, sum=271.

## O3 type x type cell-level submatrix

| backend | T4a x LPi14 first / repeat | Mi1 x T4a first / repeat | T4a x Dm9 (empty) first / repeat | peak RSS |
| --- | --- | --- | --- | --- |
| A polars is_in + pivot | 11.4 ms / 8.6 ms | 27.2 ms / 27.2 ms | 19.6 ms / 22.5 ms | 0.35 GB |
| C pandas isin + pivot | 92.1 ms / 86.6 ms | 279.9 ms / 281.9 ms | 191.6 ms / 191.9 ms | 0.74 GB |
| D pandas[pyarrow] isin + pivot | 143.5 ms / 139.7 ms | 226.4 ms / 228.7 ms | 210.0 ms / 211.7 ms | 0.59 GB |
| E scipy csr[a][:, b].toarray() | 0.5 ms / 0.3 ms | 1.6 ms / 1.5 ms | 0.4 ms / 0.3 ms | 1.09 GB |

Agreement: totals T4a->LPi14 = 49069, Mi1->T4a = 92542, T4a->Dm9 = 0 in every backend. DataFrame pivots return only cells with >=1 connection ([1453, 4] and [1573, 1457]); scipy returns the full [1457, 4] / [1581, 1457] block with zeros, whose nonzero rows/cols match the pivots ([1453, 4], [1573, 1457]).

## O4 top-10 input types onto Dm9 (with % of total input)

| backend | first | repeat | peak RSS |
| --- | --- | --- | --- |
| A polars is_in + join + group_by | 29.4 ms | 20.8 ms | 0.35 GB |
| C pandas isin + merge + groupby | 124.5 ms | 122.5 ms | 0.63 GB |
| D pandas[pyarrow] isin + merge + groupby | 162.6 ms | 157.7 ms | 0.55 GB |
| E scipy csc[:, targets].sum(1) + np.bincount | 1.3 ms | 0.9 ms | 1.13 GB |

Agreement: identical top-10 in all four backends: [['R8', 49524, 31.5528], ['R7', 28699, 18.2847], ['L3', 27045, 17.2309], ['Dm9', 10779, 6.8675], ['Dm8b', 10454, 6.6605], ['Dm8a', 6105, 3.8896], ['C2', 3720, 2.3701], ['T1', 1533, 0.9767], ['L1', 1278, 0.8142], ['aMe4', 1087, 0.6926]]

## O5 full 8547 x 8547 type x type matrix

| backend | long form (join+group_by) | long -> dense numpy | total | native pivot (for reference) | peak RSS | checksum / nnz |
| --- | --- | --- | --- | --- | --- | --- |
| A polars | 220.9 ms | 102.2 ms | 324.0 ms | 1906.9 ms | 2.33 GB | 78,899,678 / 2,798,101 |
| C pandas (positional type index) | 709.6 ms | 43.8 ms | 754.5 ms | 166.2 ms | 5.09 GB | 78,899,678 / 2,798,101 |
| D pandas[pyarrow] (merge) | 1299.3 ms | 42.7 ms | 1342.4 ms | n/a | 3.40 GB | 78,899,678 / 2,798,101 |
| E scipy T.T @ C @ T (sparse) | 276.6 ms | 71.2 ms (toarray) | 347.8 ms | n/a | 2.61 GB | 78,899,678 / 2,798,101 |
| precomputed npz -> CSR -> dense | 8.3 ms (load) | 27.6 ms | 37.1 ms | n/a | 0.70 GB | 78,899,678 / 2,798,101 |

pandas merge-based long form (what a pandas user would write without the positional trick): 989.2 ms.

## O6 synapse xyz for the reference neuron as presynaptic (80.2M-row table)

| backend | cold (load + first query) | per query after warm-up | peak RSS | notes |
| --- | --- | --- | --- | --- |
| A polars read_parquet + filter | 102.2 ms | 1,821 us | 2.35 GB | load 99.3 ms, table 1.60 GB in RAM; n=100 |
| B polars scan_parquet + filter + collect | 2.6 ms | 518 us | 0.13 GB | Predicate pushdown: reading 1 / 77 row groups; n=50 |
| C pandas read_parquet + mask | 257.0 ms | 13,371 us | 4.16 GB | load 233.8 ms; n=30 |
| D pandas[pyarrow] read_parquet + mask | 323.5 ms | 94,164 us | 3.15 GB | load 199.5 ms; n=30 |
| pyarrow read_table(filters=...) | 127.8 ms | 3,163 us | 0.32 GB | row-group pruning by statistics; n=50 |
| F numpy: pre_id column + searchsorted -> read row group(s) | 173.1 ms | 1,417 us | 0.81 GB | pre_id column 0.32 GB loaded in 171.5 ms; n=200 |
| F' numpy: all columns in RAM + searchsorted slice | 368.0 ms | 2.54 us | 2.13 GB | 1.28 GB of arrays; n=1000 |

Agreement: 289 synapse rows for id 97249 in every backend.

## O7 1e6 random rid -> id lookups

| method | build | 1e6 lookups | per lookup |
| --- | --- | --- | --- |
| dict | 6.4 ms | 37.2 ms | 37.179 us |
| dict_to_numpy | 6.4 ms | 60.6 ms | 60.584 us |
| pandas_Index.get_indexer | 0.1 ms | 7.6 ms | 7.558 us |
| pandas_Series.loc | 0.1 ms | 18.8 ms | 18.828 us |
| polars_join | 0.0 ms | 5.0 ms | 5.024 us |
| polars_join_maintain_order | 0.0 ms | 3.3 ms | 3.327 us |
| polars_replace_strict | 0.0 ms | 4.6 ms | 4.563 us |
| numpy_searchsorted | 5.0 ms | 11.7 ms | 11.713 us |

All methods agree: True

## O8 interop

| conversion | edge table (19.9M rows) | result (20k rows) |
| --- | --- | --- |
| pl to pandas | 12.9 ms | 198 us |
| pl to pandas arrow dtypes | 264 us | 127 us |
| pl to numpy | 10.5 ms | 96 us |
| pl col to numpy | 22 us | 6 us |
| pl to arrow | 52 us | 26 us |
| pandas to polars | 463 us | 267 us |
| pandas to numpy | 12.6 ms | 36 us |
| arrow to polars | 132 us | 32 us |
| arrow to pandas | 6.0 ms | 105 us |
| pl to dict of numpy | 19 us | 6 us |
## Code snippets for O2-O4 (exactly what was timed; LOC = statements inside the function)

### O2 - outputs of one neuron (post partners + counts)

```python
# A  polars eager                                                  (1 LOC)
def outputs(cid):
    return edges.filter(pl.col("pre_id") == cid).select("post_id", "num_syn")

# A'' polars exploiting the sort order                             (2 LOC)
def outputs(cid):
    a, b = pre.search_sorted(cid, "left"), pre.search_sorted(cid, "right")   # pre = edges["pre_id"]
    return edges.slice(a, b - a).select("post_id", "num_syn")

# B  polars lazy                                                   (1 LOC)
def outputs(cid):
    return lf.filter(pl.col("pre_id") == cid).select("post_id", "num_syn").collect()   # lf = pl.scan_parquet(EDGES)

# C/D pandas (numpy or pyarrow dtypes)                             (1 LOC)
def outputs(cid):
    return edges.loc[edges.pre_id == cid, ["post_id", "num_syn"]]

# C' pandas with a sorted index                                    (1 LOC + ei = edges.set_index("pre_id").sort_index())
def outputs(cid):
    return ei.loc[[cid], ["post_id", "num_syn"]]

# E  scipy CSR (outputs) / CSC (inputs)                            (2 LOC)
def outputs(cid):
    r = csr[cid]
    return r.indices, r.data           # post ids, counts  (inputs: c = csc[:, cid]; c.indices, c.data)

# F  numpy searchsorted on the sorted pre_id column                (2 LOC)
def outputs(cid):
    a, b = np.searchsorted(pre, np.array([cid, cid + 1], dtype=pre.dtype))   # dtype must match!
    return post[a:b], w[a:b]
```

### O3 - type x type cell-level submatrix (T4a rows x LPi14 columns) and total

```python
# A  polars                                                        (5 LOC)
def submatrix(pre_type, post_type):
    a = cells.filter(pl.col("type") == pre_type)["id"]
    b = cells.filter(pl.col("type") == post_type)["id"]
    sub = edges.filter(pl.col("pre_id").is_in(a.implode()) & pl.col("post_id").is_in(b.implode()))
    mat = sub.pivot(on="post_id", index="pre_id", values="num_syn").fill_null(0)
    return mat, int(sub["num_syn"].sum())

# C/D pandas                                                       (5 LOC)
def submatrix(pre_type, post_type):
    a = cells.id[cells.type == pre_type]
    b = cells.id[cells.type == post_type]
    sub = edges[edges.pre_id.isin(a) & edges.post_id.isin(b)]
    mat = sub.pivot(index="pre_id", columns="post_id", values="num_syn").fillna(0)
    return mat, int(sub.num_syn.sum())

# E  scipy CSR + numpy   (tidx = per-cell type index array, tpos = {type: index})   (4 LOC)
def submatrix(pre_type, post_type):
    a = np.flatnonzero(tidx == tpos[pre_type])
    b = np.flatnonzero(tidx == tpos[post_type])
    mat = csr[a][:, b].toarray()          # dense (len(a), len(b)) block incl. zero rows/cols
    return mat, int(mat.sum())
```

### O4 - top-10 input types onto Dm9 with percent of total input

```python
# A  polars                                                        (7 LOC)
def top_inputs(post_type, k=10):
    targets = cells.filter(pl.col("type") == post_type)["id"]
    inputs = edges.filter(pl.col("post_id").is_in(targets.implode()))
    total = inputs["num_syn"].sum()
    return (inputs.join(cells, left_on="pre_id", right_on="id")
            .group_by("type").agg(pl.col("num_syn").sum())
            .with_columns(pct=100 * pl.col("num_syn") / total)
            .sort("num_syn", descending=True).head(k))

# C/D pandas                                                       (7 LOC)
def top_inputs(post_type, k=10):
    targets = cells.id[cells.type == post_type]
    inputs = edges[edges.post_id.isin(targets)]
    total = inputs.num_syn.sum()
    s = (inputs.merge(cells, left_on="pre_id", right_on="id")
         .groupby("type").num_syn.sum().astype("int64")      # uint16 stays uint16 -> 100*s silently overflows!
         .sort_values(ascending=False))
    return pd.DataFrame({"num_syn": s, "pct": 100 * s / total}).head(k)

# E  scipy CSC + numpy                                             (6 LOC)
def top_inputs(post_type, k=10):
    targets = np.flatnonzero(tidx == tpos[post_type])
    per_pre = np.asarray(csc[:, targets].sum(axis=1)).ravel()      # input synapses per presynaptic cell
    ok = tidx >= 0                                                  # drop untyped cells
    per_type = np.bincount(tidx[ok], weights=per_pre[ok], minlength=len(type_list))
    order = np.argsort(-per_type)[:k]
    return [(type_list[i], int(per_type[i]), 100 * per_type[i] / per_pre.sum()) for i in order]
```

### O5 - full type x type matrix (for completeness)

```python
# A polars: two joins + group_by -> long form, then scatter into a dense numpy array
long = (edges.join(c2t, left_on="pre_id", right_on="id").rename({"tidx": "pre_t"})
             .join(c2t, left_on="post_id", right_on="id").rename({"tidx": "post_t"})
             .group_by("pre_t", "post_t").agg(pl.col("num_syn").sum()))
m = np.zeros((K, K), np.int64); np.add.at(m, (long["pre_t"].to_numpy(), long["post_t"].to_numpy()), long["num_syn"].to_numpy())

# E scipy: indicator matrix product
Tm = sp.csr_matrix((np.ones(len(ok)), (ok, tidx[ok])), shape=(N, K))   # cell x type indicator
m = (Tm.T @ csr @ Tm).toarray()
```

## Agreement checks (same numbers from every backend)

| operation | quantity | value (all backends agree) |
| --- | --- | --- |
| O2 | outputs of id 97249: partners / synapses | 88 / 289 (A, A', A'', B, C, C', D, E, F) |
| O2 | inputs of id 97249 via CSC: partners / synapses | 70 / 271 (E); consistent with the 271 synapse rows found by `post_id == 97249` in the synapse table |
| O3 | T4a->LPi14 total; Mi1->T4a total; T4a->Dm9 total | 49,069; 92,542; 0 (A, C, D, E). Pivot shapes agree once zero rows are accounted for |
| O4 | top-10 input types onto Dm9 | identical lists incl. percentages to 4 decimals (A, C, D, E) after making the denominator "all input synapses" everywhere |
| O5 | type x type matrix sum / nnz | 78,899,678 / 2,798,101 for A, C, D, E **and the precomputed npz** (sum < 80,215,790 total synapses because 1,578 untyped cells are excluded) |
| O6 | synapse rows for id 97249 | 289 in A, B, C, D, pyarrow, F, F' |
| O7 | 1e6 rid->id results | all 8 methods return identical id arrays |

Two things had to be fixed to get agreement, and both are relevant to the design:
1. **pandas kept `uint16` through `groupby().sum()`**, so `100 * s / total` silently overflowed (R8 showed 0.237 % instead of 31.55 %). Polars and pandas[pyarrow] upcast. Any pandas path must cast `num_syn` to int64 first (or store the edge weights as int32 from the start).
2. **`np.searchsorted(uint32_array, [python_int, ...])`** casts the whole 20M-element array to int64 on every call (1.9 ms instead of 1.9 us). Keys must be built with the array's dtype.

## What the numbers say

- **Loading**: polars reads the 20M-row edge table in 16 ms (3x faster than pandas, 12x faster than pyarrow->numpy), with the lowest peak RSS (0.35 GB vs 0.64 GB pandas). Building CSR+CSC from the sorted table costs 0.12 s on top of the read (CSR needs no COO pass because the table is sorted: `indptr = searchsorted(pre_id, arange(N+1))`). Import: polars 39 ms, pandas 140 ms, scipy.sparse 64 ms.
- **Per-neuron lookups** (O2): the CSR/CSC row/column slice is 9 us (0.2 us on the raw `indptr` arrays), numpy searchsorted 2 us. Every DataFrame filter is 100-2000x slower: polars 550 us (100 us with `search_sorted`+`slice`, 170 us with `set_sorted`), pandas mask 3.3 ms, pandas[pyarrow] mask 21 ms. Pandas with a sorted index and `.loc[[id]]` gets to 160 us but costs a 10 ms index build and a copy of the table.
- **Type-level analytics** (O3/O4): sparse fancy indexing + `bincount` is 1 ms; polars `is_in`/`join`/`group_by` is 10-30 ms; pandas is 90-280 ms (pandas[pyarrow] is not faster, and is slower on `isin`).
- **Full type x type** (O5): polars join+group_by and the scipy indicator product are equal (0.33-0.35 s, 2.3-2.6 GB peak); pandas is 2-4x slower and peaks at 5.1 GB; loading the precomputed npz is 8 ms (37 ms to dense). Polars' native `pivot` to 8,547 columns is slow (1.9 s) - scatter the long form into a numpy array instead (0.1 s). pandas `pivot` is fine (0.17 s) but its long-form step is the slow part.
- **Synapses** (O6): because the file is sorted by `pre_id` with 1M-row row groups, `pl.scan_parquet(...).filter(pre_id == id).collect()` reads **1 of 77 row groups** and returns in 0.5 ms with 0.13 GB RSS, with zero load time. Loading the whole table costs 0.1 s / 2.4 GB (polars) or 0.25 s / 4.2 GB (pandas) and then each filter is slower than the lazy scan (1.8 ms polars, 13 ms pandas, 94 ms pandas[pyarrow]) except for the pure-numpy searchsorted slice (2.5 us, but 1.3 GB of arrays resident). pyarrow `filters=` prunes row groups too but is 6x slower per query (3.2 ms). Caveat measured separately: for the **unsorted** column (`post_id == id`, i.e. a neuron's input synapses) pushdown cannot prune, and the lazy scan takes 50 ms per query (pyarrow filters 124 ms). A second copy of the file sorted by `post_id` (or an in-memory CSC-style index) would be needed if input-synapse queries are hot.
- **rid->id** (O7): every vectorised method is 3-12 ns per lookup for a 1e6 batch (polars join 3.3 ms, pandas `Index.get_indexer` 7.6 ms, numpy searchsorted 11.7 ms); the Python dict is 37-60 ms. id->rid is a plain array index (`rid[id]`) because ids are dense.
- **Interop** (O8): polars -> pandas is 13 ms for 20M rows and 0.2 ms for a 20k-row result; with `use_pyarrow_extension_array=True` it is zero-copy (0.26 ms for 20M rows). polars column -> numpy is zero-copy (22 us). pandas -> polars is 0.5 ms. Converting results at the API boundary is therefore free relative to any query; converting the whole edge table on every call is not (13 ms ~ a full polars join).

## Recommendation

The evidence supports a **three-layer design**: parquet on disk -> polars DataFrames as the loaded tables -> scipy CSR/CSC + numpy for every hot path, with pandas/numpy objects handed back to the user.

| data | structure to hold it | why | trade-off |
| --- | --- | --- | --- |
| (a) cell metadata (139k rows) | polars DataFrame, row position == `id`; plus `rid` and `type_idx` as numpy arrays | tiny; loads in ms; `filter(type == ...)["id"]` is the entry point for every type query; row position == id lets per-cell arrays index directly | none material - at this size pandas would be fine too; choose polars for consistency with the edge table |
| (b) cell-cell edge list (19.9M rows) | polars DataFrame sorted by `pre_id`, dtypes `u32,u32,u16` (200 MB, 16 ms) | 3x faster load and 6-40x faster filter/join than pandas at half the RSS; is the source for building (c) | users who want it as pandas pay 13 ms `to_pandas()` (0.3 ms zero-copy with arrow dtypes); pandas[pyarrow] is dominated on every measure and should not be used |
| (c) connectivity matrix | `scipy.sparse.csr_matrix` (outputs) and its `.tocsc()` (inputs), int32 indices and int32 data (160 MB each), built lazily from (b) in 0.12 s and cached | 9 us per-neuron lookup, 1 ms type-block extraction, 1 ms top-input-types; everything DataFrame-based is 100-1000x slower; `csr[a][:, b].toarray()` is exactly the numpy-user mental model | second copy of the edges in memory (CSR+CSC ~0.32 GB with int32; the benchmark used int64 = 0.48 GB); requires ids to be dense (they are) |
| (d) type x type matrix (8547^2) | load the precomputed npz into a `csr_matrix` (8 ms); `toarray()` on demand (28 ms, 584 MB int64 / 292 MB int32); recompute from (c) with `T.T @ C @ T` in 0.35 s when a filtered variant is needed | fastest by 10x; exactly matches recompute (checksum 78,899,678) | dense form is large - return dense only for requested sub-blocks; keep type labels as a numpy array / `pd.Index` alongside |
| (e) synapses (80M rows) | **do not load**; `pl.scan_parquet` + `filter(pre_id == id)` + `collect()` (0.5 ms, 0.13 GB) | predicate pushdown reads 1/77 row groups because the file is sorted by pre_id; zero start-up cost | queries on `post_id` fall back to a 50 ms full scan - add a post_id-sorted copy if needed; bulk analytics over all synapses should use polars lazy expressions, not per-neuron loops |
| (f) id <-> rid | id->rid: `rid` numpy array (`rid[id]`); rid->id: `pd.Index(rid).get_indexer(batch)` or polars join for batches (3-8 ms per 1e6), a dict only for scalar lookups | all vectorised methods are equivalent (3-12 ns/lookup); no build cost worth optimising | a dict costs 37 us/1000 lookups and 6 ms to build - fine for scalars, poor for batches |

**Public return types for a pandas/numpy user:**
- Partner tables (outputs/inputs of a neuron, top-k input types, type-pair tables): **`pandas.DataFrame`** with explicit columns (`post_rid`/`post_id`, `type`, `num_syn`, `pct`), produced from numpy arrays or via `polars.to_pandas()` (0.1-0.2 ms for typical result sizes). A `.to_polars()` escape hatch costs nothing.
- Matrices (cell-block or type x type): **`numpy.ndarray`** with the row/column labels returned alongside (or a `pandas.DataFrame` indexed by rid/type for blocks smaller than ~1e6 elements, since labelled 2-D data is what pandas users reach for). Keep the full 139k x 139k matrix sparse (`scipy.sparse` CSR) - never densify.
- Synapse coordinates: **`numpy.ndarray` (n, 3) uint32** or a 3-column DataFrame.
- Ids: expose `rid` (int64) in all user-facing tables; keep the dense `id` internal, or expose it as a documented "index" column since it is what indexes the arrays.

**Trade-offs stated plainly:**
- Two frame libraries in the dependency tree (polars inside, pandas at the boundary). The alternative - pandas everywhere - costs 3x on load, 2-4x on group-bys, 2x RSS, and inherits the uint16-overflow footgun; but because the hot paths live in scipy/numpy regardless, a pandas-only implementation would still be *usable*, just not fast on the analytics paths (O3-O5: 90 ms - 1.3 s instead of 1-30 ms) and unable to do the 0.5 ms lazy synapse lookup (pyarrow `filters=` gets 3 ms with no pandas involvement, which is an acceptable fallback).
- The scipy layer is the only one that needs dense ids and a build step; it is ~0.3 s cold and ~0.3 GB, so it should be built lazily on first use and cached.
- Polars lazy scanning of synapses is fast only because the file is sorted by `pre_id` and written in 1M-row row groups; the preprocessing pipeline must preserve that invariant (and add a `post_id`-sorted copy if input-synapse queries matter).
- pandas with `dtype_backend="pyarrow"` was slower than plain pandas on every filter/isin/pivot path here (20-90 ms per boolean-mask filter) and offers nothing polars does not; skip it.
