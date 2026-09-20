"""The quickstart notebook's code cells run against the built datasets (skipped without them)."""

import os
from pathlib import Path

import nbformat
import pytest

DATA = Path(os.environ.get("CONNEXPLORER_DATA", "data"))
NB = Path("notebooks/00_tour.ipynb")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not ((DATA / "flywire_783" / "tables" / "skeletons").exists() and (DATA / "mcns_v1.0" / "tables" / "manifest.json").exists()),
        reason="needs both datasets and the FlyWire skeleton zip",
    ),
]


def test_tour_cells_execute():
    os.environ.setdefault("MPLBACKEND", "Agg")
    nb = nbformat.read(NB, as_version=4)
    import plotly.graph_objects as go

    monkeypatch_show = go.Figure.show
    go.Figure.show = lambda self, *a, **k: None  # headless: fig.show() must not open anything
    ns = {"display": lambda *a, **k: None}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{NB}#{nb.cells.index(cell)}", "exec"), ns)
    go.Figure.show = monkeypatch_show
    assert ns["V"].min() > 0                       # CT1 cable model solved
    assert ns["fields"].height == 128              # one row per right-hemisphere Dm9
    assert ns["N_shared"].nnz > 0 and ns["W"].shape[0] == len(ns["pr"])
    assert ns["mi1"].skeleton().n_nodes > 0        # Male CNS skeleton fetched on demand

