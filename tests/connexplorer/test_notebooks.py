"""The quickstart notebook's code cells run against the built datasets (skipped without them)."""

import os
from pathlib import Path

import nbformat
import pytest

DATA = Path(os.environ.get("CONNEXPLORER_DATA", "data"))
NB = Path("notebooks/connexplorer/00_quickstart.ipynb")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not ((DATA / "flywire_783" / "tables" / "skeletons").exists() and (DATA / "mcns_v1.0" / "tables" / "manifest.json").exists()),
        reason="needs both datasets and the FlyWire skeleton zip",
    ),
]


def test_quickstart_cells_execute():
    os.environ.setdefault("MPLBACKEND", "Agg")
    nb = nbformat.read(NB, as_version=4)
    ns = {"display": lambda *a, **k: None}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{NB}#{nb.cells.index(cell)}", "exec"), ns)
    assert ns["V"][0] > 0 and "url" in ns

