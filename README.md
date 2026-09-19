# codexplorer

Shared workspace for merging two independently developed fly-connectome
analysis packages. Each author's condensed code lives in its own subfolder
under `src/`, installed together from the single root `pyproject.toml`.

- `src/shayan/` — connectome querying (Polars), skeleton loading (navis),
  compartmentalization, cable modeling, and visualization. See its README.
- `src/cex/` — normalized-table dataset access, connectivity matrices,
  preprocessing, I/O statistics, and FlyWire links. 

## Setup

```bash
uv sync            # or: pip install -e .
python -c "from shayan import Connectome, FlyWireDataset"
python -c "from cex import get_dataset"
```

Data files are not stored in the repo. See `src/shayan/README.md` for the
Shayan layout and [the CEX data layout](docs/cex_data_layout.md) for the CEX
preprocessing notebooks. After preparing a dataset with the FlyWire `00` or
Male CNS `01` notebook, use
[`00_basic_package_usage.ipynb`](notebooks/cex/00_basic_package_usage.ipynb)
for documented examples of type metadata, connectivity, single-neuron I/O,
and FlyWire viewer URLs. Its neuron example displays the five strongest input
types and their synapse locations in Neuroglancer. Its optional setup section
can download a ZIP of preprocessed FlyWire tables directly into
`data/flywire/preprocessed/`.

## connexplorer (merged package, in progress)

`src/connexplorer/` is the merged package described in
[docs/merge_plan.md](docs/merge_plan.md). Phase 1 (ingestion) is in place:

```bash
# FlyWire FAFB v783: fetch the Codex raw bundle (3 GB), then build the tables (~30 s, 11 GB RAM)
uv run python -m connexplorer.ingest download flywire --raw data/flywire/raw
uv run python -m connexplorer.ingest build flywire --raw data/flywire/raw --out data/flywire_783

# Male CNS v1.0: body-annotations, body-neurotransmitters and syn-partners from
# gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/ in a raw folder, then (~45 s, 20 GB RAM)
uv run python -m connexplorer.ingest build mcns --raw data/mcns/raw --out data/mcns_v1.0 --version v1.0
```

The output layout and invariants are documented in [docs/schema.md](docs/schema.md).

Phase 2 (runtime) is in place; every table it returns is a polars DataFrame:

```python
import connexplorer as cnx
import polars as pl

ds = cnx.open("flywire")                    # finds data/flywire_783 (or set CONNEXPLORER_DATA)
n = ds[720575940599755718]                  # Neuron;  ds["T4a"], ds[[rid, ...]] -> NeuronSet
n.outputs(min_syn=5)                        # post, type, side, n_syn
n.inputs(by="type")                         # type, n_syn, n_partners, frac_input
n.inputs(by=("type", "neuropil"))           # needs edges_by_neuropil
ds.connectivity["T4a", "LPi14"].values      # dense (1457, 4) block; .sparse / .long / .frame
ds.connectivity.types["Mi1", "T4a"]         # type-level total
ds.connectivity.autapses = True             # self-connections are masked by default
n.synapses(direction="in")                  # pre, post, x_nm, y_nm, z_nm, neuropil (about 2 ms)
ds.select(type="T4a", side="R")             # or any polars expression over ds.cells

# phase 3: normalization, graph statistics, cross-dataset comparison
ds["Dm9"].inputs(by="type", normalize=True) # + frac_input, frac_partner_output, weight_norm
ds.connectivity.types.normalized            # every type pair with frac_output / frac_input / weight_norm
ds["T4a"].degree(); ds["T4a"].hubs(k=10); ds["T4a"].reciprocal(); ds["T4a"].summary()
mc = cnx.open("mcns"); mc.types_like("R7")  # -> ['R7_unclear', 'R7d', 'R7p', 'R7y']
cnx.compare(ds["T4a"], mc["T4a"]).inputs()  # type, n_syn_flywire, n_syn_mcns, ..., frac_input_mcns

# phase 4: morphology, cable models, viewers (skeleton zip under data/<dataset>/tables/skeletons/)
sk = n.skeleton()                           # navis TreeNeuron, micrometres, read straight from the zip
comp = cnx.morph.segment(sk, min_length_um=0.5)   # one compartment per navis segment, short ones merged
m = cnx.models.Cable(comp, Rm=8000, Ra=400, Cm=0.6)
m.steady_state({0: 10e-12}); m.input_resistance(0); cnx.models.scan(comp, Ra=[100, 400], Rm=[4000, 8000])
n.view(partners="in", top=5, synapses=True) # Neuroglancer URL string (viewer="spelunker" | "cave" | "flywire" | "codex")
comp.view(ds); cnx.viz.plot3d(comp); comp.plot2d()
```
