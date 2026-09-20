# connexplorer quickstart

```bash
uv sync                                    # or: pip install -e .
connexplorer datasets                      # built datasets found under ./data or $CONNEXPLORER_DATA
```

If nothing is built yet, see [Building datasets](#building-datasets) at the end.

## Open, select, look

```python
import connexplorer as cnx
import polars as pl

ds = cnx.open("flywire")            # or "mcns", or a directory; reads only the manifest
ds.info()
ds.cells                            # polars DataFrame: root_id, type, side, nt, nt_score, ...
ds.types                            # type, n_cells, nt, nt_source, nt_predicted, ...

n = ds[720575940599755718]          # Neuron; n.type, n.side, n.nt, n.cell
grp = ds["T4a"]                     # NeuronSet, all cells of a type
grp = ds[["T4a", "T4b"]]            # union of types;  ds[[rid1, rid2]] by root ids
grp = ds.select(type="T4a", side="R")
grp = ds.select(pl.col("nt") == "GABA")
grp.ids, grp.cells, grp.types, len(grp), grp[0], grp & other, grp | other, grp - other
```

Every table is a polars DataFrame. Root ids are the public identifiers; the
dense ids the package uses internally are `ds.cells["id"]`.

## Connectivity

```python
n.outputs()                                  # post, type, side, n_syn  (strongest first)
n.outputs(min_syn=5)                         # threshold on the per-partner total
n.inputs(by="type")                          # type, n_syn, n_partners, frac_input
n.inputs(by=("type", "neuropil"))            # FlyWire and Male CNS both have neuropils
grp.inputs(by="type", normalize=True)        # + frac_partner_output, weight_norm
n.partners()                                 # both directions side by side

ds.connectivity["T4a", "LPi14"]              # ConnBlock: .values (dense), .sparse, .long, .frame, .sum()
ds.connectivity[grp_a, grp_b]; ds.connectivity[[rid, ...], "LPi14"]
ds.connectivity.types["Mi1", "T4a"]          # type-level total
ds.connectivity.types[["T4a", "T4b"], ["LPi14"]].values
ds.connectivity.types.normalized             # every type pair with frac_output, frac_input, weight_norm
ds.connectivity.sparse                       # whole-brain CSR (rows pre, cols post, ds.cells order)
ds.connectivity.autapses = True              # self-connections are masked by default
```

Normalization: `frac_input` is the share of the set's own input; `frac_partner_output`
is the share of the partner's whole-dataset output that lands on the set;
`weight_norm` is the geometric mean (shayan's `normalized_weight / 100`).

## Graph statistics

```python
grp.degree()                     # per cell: in/out degree and synapses (whole dataset)
grp.degree(within=True)          # restricted to the set
grp.hubs(k=10, by="out_syn"); grp.degree_distribution("in"); grp.reciprocal(); grp.summary()
grp.subgraph()                   # ConnBlock of edges inside the set
```

## Synapses

```python
n.synapses()                     # pre, post, x_nm, y_nm, z_nm, neuropil   (about 2 ms)
n.synapses("in"); n.synapses(partner="LPi14")
ds.synapses.between("T4a", "LPi14"); ds.synapses.in_box(x=(300_000, 400_000))
cnx.xyz(df)                      # (k, 3) uint32 nanometres
ds.synapses.scan()               # LazyFrame over the whole table (dense ids) for bulk work
```

## Morphology and cable models

```python
sk = n.skeleton()                                   # navis TreeNeuron, micrometres
comp = cnx.morph.segment(sk, min_length_um=0.5)     # Compartments; comp.table, comp.summary()
m = cnx.models.Cable(comp, Rm=8000, Ra=400, Cm=0.6)
V = m.steady_state({0: 10e-12})                     # mV per compartment
Vt = m.transient({0: pulse}, dt=1e-3)               # (n_comp, n_steps) mV
m.input_resistance(0); m.attenuation(0); cnx.models.scan(comp, Ra=[...], Rm=[...])
comp.plot3d(); comp.plot2d(); cnx.viz.plot_voltage(comp, V)
```

Skeletons are read from `data/<dataset>/tables/skeletons/` (a zip or a folder of
`<root_id>.swc`), or from `cnx.config.skeletons["flywire"]`. The Male CNS has no
local set: `mc[body_id].skeleton()` fetches that body's precomputed skeleton from
the vendor store on first use, stores it as SWC in that folder, and assigns
`radius_um` (default 0.25) to every node because the store carries no radii.

## Viewers

```python
n.view()                                      # Neuroglancer URL (spelunker); never opens a browser
n.view(viewer="cave" | "flywire" | "codex")
n.view(partners="in", top=5, synapses=True)   # partner types as layers + synapse points
comp.view(ds)                                 # compartment centres as annotations
```

## Two datasets

```python
mc = cnx.open("mcns")
mc.type_map                                   # Male CNS type -> FlyWire type
mc.types_like("R7")                           # ['R7_unclear', 'R7d', 'R7p', 'R7y']
cnx.compare(ds["T4a"], mc["T4a"]).inputs()    # aligned by FlyWire type names
```

## Command line

```bash
connexplorer info flywire
connexplorer types mcns --like "^Mi" -l 10
connexplorer query flywire T4a -t 5
connexplorer query flywire 720575940599755718 -m inputs --by cell
connexplorer neuron mcns 12473
connexplorer view flywire T4a --partners in --top 3 --synapses
```

## Building datasets

```bash
connexplorer download flywire --raw data/flywire/raw         # 3 GB Codex bundle
connexplorer build flywire --raw data/flywire/raw --out data/flywire_783
connexplorer build mcns --raw data/mcns/raw --out data/mcns_v1.0 --version v1.0
```

The Male CNS raw folder needs `body-annotations`, `body-neurotransmitters` and
`syn-partners` for the release from
`gs://flyem-male-cns/<release>/connectome-data/flat-connectome/`. Layout and
invariants of the output are in [schema.md](schema.md).
