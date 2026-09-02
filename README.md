# codexplorer

Shared workspace for merging two independently developed fly-connectome
analysis packages. Each author's condensed code lives in its own subfolder
under `src/`, installed together from the single root `pyproject.toml`.

- `src/shayan/` — connectome querying (Polars), skeleton loading (navis),
  compartmentalization, cable modeling, and visualization. See its README.

## Setup

```bash
uv sync            # or: pip install -e .
python -c "from shayan import Connectome, FlyWireDataset"
```

Data files are not stored in the repo; see `src/shayan/README.md` for the
expected layout.
