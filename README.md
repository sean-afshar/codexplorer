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
and FlyWire viewer URLs. Its optional setup section can download a ZIP of
preprocessed FlyWire tables directly into `data/flywire/preprocessed/`.
