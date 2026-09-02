# shayan

Condensed version of the `connect_tome` package: Polars-based querying of the
FlyWire FAFB connectome, navis-based skeleton loading, neuron
compartmentalization, a passive cable-equation solver, and visualization
helpers (plotly, matplotlib, neuroglancer URLs).

## Layout

| Module | Contents |
|---|---|
| `datasets/` | Canonical Polars schemas, `AbstractDataset` base class, `FlyWireDataset` loader |
| `core/` | `Connectome` (cached tables + type index), `ConnectionQuery` fluent builder, `Circuit` subgraph with normalization and graph stats |
| `morphology/` | `load_skeleton` from an SWC zip, `Compartmentalization`, `segment_natural`, plotly / matplotlib / neuroglancer visualizers |
| `models/` | `CompartmentalModel` passive cable solver (Borst & Meier 2019 defaults), `scan_parameter_space` |
| `cli/` | `connect-tome info|query|types` command-line interface (click + rich) |

## Data

Data files are **not** stored in this repo (the FlyWire set alone is over 15 GB).
`FlyWireDataset(data_dir)` expects a directory containing:

```
data/flywire_fafb/
  neurons.csv.gz                              # root_id, nt_type, nt_type_score, ...
  consolidated_cell_types.csv.gz              # root_id, primary_type, group
  connections_princeton_no_threshold.csv.gz   # pre_root_id, post_root_id, syn_count, neuropil, nt_type
  fafb_v783_princeton_synapse_table.csv.gz    # only needed for load_synapses()
  sk_lod1_783_healed.zip                      # <root_id>.swc skeletons, only needed for morphology
```

These are the FlyWire FAFB v783 Princeton exports. Point `data_dir` anywhere;
the `data/` folder at the repo root is gitignored if you want to keep them there.

## Example

```python
from shayan import Connectome, FlyWireDataset, load_skeleton, segment_natural
from shayan.models import CompartmentalModel

ctome = Connectome(FlyWireDataset("data/flywire_fafb"), mode="lazy")

# Connectivity
circuit = ctome.query().pre_type("T4a").min_weight(5).as_circuit()
print(circuit.summary())
print(circuit.aggregate_by_type_normalized().head())

# Morphology + cable model
nid = int(ctome.get_cell_type_ids("Dm9")[0])
neuron = load_skeleton("data/flywire_fafb/sk_lod1_783_healed.zip", [nid], convert_units=True)[0]
comp = segment_natural(neuron)
model = CompartmentalModel(comp)
V = model.solve_steady_state(injection_comps=[0], currents=[10e-12])
```

## Known gaps

- Only `segment_natural` is implemented; the other segmentation strategies and
  `Compartmentalization.to_hines_matrix / summary / plot` are stubs.
- Only the FlyWire loader exists. A Male CNS loader is referenced by the CLI
  but not implemented.
- `Connectome(mode="scan")` falls back to lazy loading for connections.
- `segment_natural` can produce zero-length compartments (e.g. single-node
  segments). `CompartmentalModel` warns about them; its solvers return NaN
  until they are merged or dropped.
