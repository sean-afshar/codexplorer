# Migrating from `shayan`

`shayan` read the vendor CSV exports at runtime and built circuits with a fluent
query. `connexplorer` reads prebuilt Parquet tables, keeps the whole graph in a
sparse matrix, and answers the same questions from cell or type handles. Three
things changed in the answers themselves:

- **Autapses exist.** The vendor pair export that `shayan` read had none; the
  synapse table has 3.27 M of them on FlyWire. They are masked by default
  (`ds.connectivity.autapses = True` shows them).
- **Thresholds apply to the pair total, not to neuropil fragments.**
  `min_weight(5)` on `shayan`'s neuropil-split rows dropped sub-threshold
  fragments of strong partners (155 vs 158 synapses for the reference T4a).
- **One R7 cell** missing from the vendor export is present (37 extra synapses onto Dm9).

## API map

| shayan | connexplorer |
|---|---|
| `Connectome(FlyWireDataset(dir), mode=...)` | `ds = cnx.open("flywire")` (manifest only; tables load lazily) |
| `ctome.neurons` | `ds.cells` (`root_id`, not `neuron_id`; `side` is `L`/`R`/`M`) |
| `ctome.connections` (pre, post, weight, neuropil rows) | `ds.edges` (one row per pair) and `ds.edges_by_neuropil` |
| `ctome.get_cell_type("T4a")`, `get_cell_type_ids` | `ds["T4a"].cells`, `ds["T4a"].ids` |
| `ctome.query().pre_type("T4a").min_weight(5).as_circuit()` | `ds["T4a"].outputs(min_syn=5)` or `ds["T4a"].subgraph()` |
| `query().post_type(t)` / `.pre_id(i)` / `.post_id(i)` | `ds[t].inputs()` / `ds[i].outputs()` / `ds[i].inputs()` |
| `query().neuropil("ME_R")` | `n.outputs(by="neuropil")` or filter `ds.edges_by_neuropil` |
| `query().nt_type("GABA")` | `ds.select(nt="GABA")` then any connectivity call |
| `circuit.aggregate_by_type()` | `ds.connectivity.types.long` or `grp.outputs(by="type")` |
| `circuit.aggregate_by_type_normalized()` | `ds.connectivity.types.normalized` or `grp.inputs(by="type", normalize=True)` |
| `percent_input`, `percent_output`, `normalized_weight` | `frac_input`, `frac_partner_output`, `weight_norm` (fractions, not percent) |
| `circuit.add_normalized_weights()` | `ds.connectivity[pre, post].normalized` or `n.outputs(normalize=True)` |
| `circuit.node_stats()` / `find_hubs()` | `grp.degree()` / `grp.hubs(k, by=...)` |
| `circuit.reciprocal_connections()` | `grp.reciprocal()` |
| `circuit.degree_distribution()` | `grp.degree_distribution("in")`, `("out")` |
| `circuit.summary()` | `grp.summary()` |
| `load_skeleton(zip, [rid], convert_units=True)` | `ds[rid].skeleton()` (always micrometres) |
| `load_skeletons_by_type(zip, ctome, "L1", max_num=5)` | `ds["L1"].skeletons(max_n=5)` |
| `segment_natural(neuron)` | `cnx.morph.segment(sk)` (short compartments merged; lengths sum to cable length) |
| `Compartmentalization.compartments` | `comp.table` (polars), `comp.nodes(i)`, `comp.parents` |
| `CompartmentalModel(comp, Rm, Ra, Cm)` | `cnx.models.Cable(comp, Rm, Ra, Cm)` |
| `solve_steady_state([0], [10e-12])` | `m.steady_state({0: 10e-12})` |
| `solve_time_dependent([0], trace, dt, t_max)` | `m.transient({0: trace}, dt)` |
| `get_input_resistance(i)` / `analyze_compartmentalization(i)` | `m.input_resistance(i)` / `m.attenuation(i)` |
| `scan_parameter_space(comp, Ra, Rm)` | `cnx.models.scan(comp, Ra, Rm)` |
| `plot_compartmentalization(comp)` / `_2d` | `comp.plot3d()` / `comp.plot2d()` |
| `neuron_to_neuroglancer_url(rid, viewer="cave")` | `ds[rid].view(viewer="cave")` |
| `to_neuroglancer_url(comp)` | `comp.view(ds)` |
| `connect-tome info/query/types` | `connexplorer info/query/types` (plus `neuron`, `view`, `build`) |

## Behaviour notes

- Nothing is returned as pandas. Use `.to_pandas()` on any frame if you need it.
- `n.outputs()` sorts strongest first and includes partner `type` and `side`.
- The unimplemented segmentation strategies and `Compartmentalization.to_hines_matrix`
  stubs are gone; `comp.hines(Ra)` is implemented.
- `import connexplorer` is fast; navis, plotly and matplotlib load on first use.
