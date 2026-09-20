# connexplorer

Fast, polars-native access to fly connectomes. One package for FlyWire FAFB
(v783) and the Male CNS (v1.0): cells and types, connectivity at cell and type
level, synapse locations, skeletons and passive cable models, and Neuroglancer
links.

```bash
uv sync                         # or: pip install -e .
connexplorer datasets           # what is built under ./data (or $CONNEXPLORER_DATA)
```

```python
import connexplorer as cnx

ds = cnx.open("flywire")
n = ds[720575940599755718]                  # a T4a
n.inputs(by="type", normalize=True).head()  # polars DataFrame
ds.connectivity["T4a", "LPi14"].values      # dense block, 1457 x 4
n.synapses("in")                            # locations, ~2 ms
comp = cnx.morph.segment(n.skeleton())      # compartments from the SWC skeleton
cnx.models.Cable(comp).steady_state({0: 10e-12})
n.view(partners="in", top=5, synapses=True) # Neuroglancer URL
```

- [docs/quickstart.md](docs/quickstart.md): the API in one page, plus the CLI.
- [docs/schema.md](docs/schema.md): on-disk tables, invariants, manifest.
- [docs/migrating_from_shayan.md](docs/migrating_from_shayan.md), [docs/migrating_from_cex.md](docs/migrating_from_cex.md).
- [docs/merge_plan.md](docs/merge_plan.md): the design and its evidence.
- [notebooks/00_tour.ipynb](notebooks/00_tour.ipynb): a tour of everything, with T4/T5 connectivity and CT1 morphology; [notebooks/01_build_datasets.ipynb](notebooks/01_build_datasets.ipynb) builds the tables.

## Data

Datasets are not in the repo. Build them once (30 to 45 s each; 11 to 20 GB
RAM while building) and they are read lazily afterwards:

```bash
connexplorer download flywire --raw data/flywire/raw                          # 3 GB Codex bundle
connexplorer build flywire --raw data/flywire/raw --out data/flywire_783
connexplorer build mcns --raw data/mcns/raw --out data/mcns_v1.0 --version v1.0
```

For the Male CNS put `body-annotations`, `body-neurotransmitters` and
`syn-partners` for the release (from
`gs://flyem-male-cns/<release>/connectome-data/flat-connectome/`) in the raw
folder.

Skeletons live in `data/<dataset>/tables/skeletons/`, either one zip or a
folder of `<root_id>.swc` files. FlyWire ships one 13 GB zip. The Male CNS
skeletons are too large to hold as a set, so `mc[body_id].skeleton()` fetches
that one body from the vendor's public store on first use and keeps it as an
SWC file in that folder.

## Development

```bash
uv run pytest                    # fixture tests (CI)
uv run pytest -m slow            # also the real-data tests, when datasets are built
```
