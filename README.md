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

# Male CNS: put the neuPrint feather exports for a release in a raw folder, then
uv run python -m connexplorer.ingest build mcns --raw data/mcns/raw --out data/mcns_v0.9 --version v0.9
```

The output layout and invariants are documented in [docs/schema.md](docs/schema.md).
