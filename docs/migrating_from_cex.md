# Migrating from `cex`

`connexplorer` keeps `cex`'s design: normalized Parquet tables, dense ids
behind the scenes, CSR/CSC matrices for the hot paths, per-type matrices, lazy
synapses and Neuroglancer links. What changed:

- **Root ids are the public ids.** Every method takes and returns root ids
  (`bodyId` for Male CNS); dense ids are `ds.cells["id"]`, used only for array positions.
- **DataFrames, not dicts.** Partner tables, synapse queries and matrix blocks
  come back as polars DataFrames or objects with `.values` / `.sparse` / `.long`,
  never as dict-of-arrays; there are no `_Q` flags.
- **No side effects.** Opening a dataset creates no folders; viewer methods
  return URL strings and never open a browser.
- **Autapses are kept in the data** and masked by a single flag
  (`ds.connectivity.autapses`, default off) in every result, matrices included.
- **Real nulls** replace `"NAN"` / `"Unknown"` sentinels; side is `L`/`R`/`M`/null.
- **The per-type I/O-stat CSV tree is gone.** `grp.inputs(by="type", normalize=True)`
  computes the same quantities live in about a millisecond.
- **Literature transmitter overrides are labelled**: `types.nt_source` is
  `"literature"` or `"prediction"`, and `nt_predicted` keeps the raw majority.

## API map

| cex | connexplorer |
|---|---|
| `get_dataset("flywire", root)` | `cnx.open("flywire")` or `cnx.open(path)` |
| `ds.cell.cell_table` / `cell_type_info` | `ds.cells` / `ds.types` |
| `ds.cell.rid_to_id`, `id_to_rid` | `ds.idx_of(rids)`, `ds.root_ids[idx]` |
| `ds.get_cell_type_id("Mi1", side="right")` | `ds.select(type="Mi1", side="R").ids` |
| `ds.cell.type_to_rid["Mi1"]` | `ds["Mi1"].ids` |
| `ds.connectivity.count_matrix_csr` / `_csc` | `ds.connectivity.sparse` / `sparse_t` |
| `ds.connectivity.type_count_matrix_csr` | `ds.connectivity.types.sparse` |
| `counts_between_types(pre_types, post_types, pre_side, ...)["matrix"]` | `ds.connectivity[ds.select(type=..., side="R"), ds.select(...)].sparse` |
| `num_syn_between_type_groups(a, b)` | `ds.connectivity.types[a, b].values` |
| `num_syn_between_rid_pairs(pre, post)` | `ds.connectivity[pre, post].values` (block) |
| `connected_ids_and_counts(id, direction)` | `ds[rid].outputs()` / `.inputs()` |
| `Neuron(id, ds, remove_autapse_Q=True, load_syn_Q=True)` | `ds[rid]` (autapses masked by default; synapses are lazy) |
| `neuron.get_top_n_input_type_stat(n)` | `n.inputs(by="type").head(n)` |
| `neuron.get_type_counts_combined_stat("output")` | `n.outputs(by="type", normalize=True)` |
| `ds.show_cell_type_io_stat("Mi1", "input")` | `ds["Mi1"].inputs(by="type", normalize=True)` |
| `ds.synapse.get_pair_synapses(pre, post)` | `ds.synapses.between([pre], [post])` |
| `ds.synapse.get_synapse_xyz_for_ids(ids, "pre")` | `cnx.xyz(ds[ids].synapses())` |
| `ds.column.column_table` | `ds.columns` |
| `ds.ct_mcns2fw("R7d")` / `ct_fw2mcns("R7")` | `mc.type_map` / `mc.types_like("R7")` |
| `MaleCNSDataset.combined_photoreceptor_types` | `mc[mc.types_like("R7")]` |
| `ds.ng_url(rids)` / `ng_url_for_ids(ids)` | `ds[rids].view()` |
| `neuron.vis_self_w_top_n_conn_type_in_ng("input", n=5, vis_syn_Q=True)` | `n.view(partners="in", top=5, synapses=True)` |
| `ds.codex_url(rids)` | `ds[rids].view(viewer="codex")` |
| preprocessing notebooks | `connexplorer build flywire --raw ... --out ...` (see `docs/schema.md`) |
| `download_and_extract(url, ...)` | `connexplorer download flywire --raw ...` |

## Behaviour notes

- Male CNS defaults to release v1.0 and carries the v1.0 transmitter columns
  (`nt`, `nt_score`, `nt_consensus`, `nt_ground_truth`) and a per-synapse neuropil.
- The `stat/` folder and `visual_type_data.parquet` are not produced; the visual
  type family/subsystem/category columns live on `ds.types`.
- Everything under `src/cex` still imports (with a deprecation warning) until
  both authors have moved their working code.
