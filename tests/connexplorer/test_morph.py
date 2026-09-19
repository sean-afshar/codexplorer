"""Skeletons, compartments, cable model and viewers on a synthetic SWC.

Skeleton (nanometres in the file, loaded as micrometres): root 1 at the origin,
a 3-edge trunk along x (10 um each), then a branch point at node 4 with a
2-edge limb along y (5 um each) and a 2-edge limb along z (8 um each); node 7
also carries a 0.1 um twig (node 9). navis makes three linear segments: the
trunk plus y limb (40 um), the z limb (16 um) and the twig (0.1 um), which
the default min_length merges into the z limb.
"""

import json
import os
import zipfile
from urllib.parse import unquote

import numpy as np
import polars as pl
import pytest

import connexplorer as cnx
from conftest import write_fixture

os.environ.setdefault("MPLBACKEND", "Agg")

SWC = """# Meta: {"id": "10", "units": "1 nanometer"}
# PointNo Label X Y Z Radius Parent
1 1 0 0 0 1000 -1
2 0 10000 0 0 500 1
3 0 20000 0 0 500 2
4 5 30000 0 0 500 3
5 0 30000 5000 0 300 4
6 6 30000 10000 0 300 5
7 5 30000 0 8000 400 4
8 6 30000 0 16000 400 7
9 6 30000 100 8000 100 7
"""
SWC_NO_HEADER = "\n".join(l for l in SWC.splitlines() if not l.startswith("#")) + "\n"


@pytest.fixture
def ds(tmp_path):
    tables = write_fixture(tmp_path)
    (tables / "skeletons").mkdir()
    with zipfile.ZipFile(tables / "skeletons" / "skeletons.zip", "w") as z:
        z.writestr("10.swc", SWC)
        z.writestr("11.swc", SWC_NO_HEADER)
    return cnx.open(tables)


def test_skeleton_store_and_units(ds):
    store = ds.skeletons
    assert len(store) == 2 and store.has(10) and not store.has(12)
    assert sorted(store.ids().tolist()) == [10, 11]
    sk = ds[10].skeleton()
    assert sk.n_nodes == 9 and str(sk.units.units) == "micrometer" and sk.id == 10
    assert sk.nodes.loc[sk.nodes.node_id == 4, "x"].item() == pytest.approx(30.0)
    assert float(sk.cable_length) == pytest.approx(30 + 10 + 16 + 0.1)
    nm = ds[10].skeleton(units="nm")
    assert str(nm.units.units) == "nanometer" and float(nm.cable_length) == pytest.approx(56100)
    # no units header: the store's source units (nm) apply
    assert float(ds[11].skeleton().cable_length) == pytest.approx(56.1)
    nl = ds["A"].skeletons()
    assert len(nl) == 2 and len(ds["A"].skeletons(max_n=1)) == 1
    with pytest.raises(KeyError, match="no skeleton"):
        ds[12].skeleton()


def test_missing_store_message(fixture_tables, monkeypatch):
    ds = cnx.open(fixture_tables)
    monkeypatch.delenv("CONNEXPLORER_SKELETONS_FIXTURE", raising=False)
    with pytest.raises(FileNotFoundError, match="CONNEXPLORER_SKELETONS_FIXTURE"):
        ds[10].skeleton()


def test_natural_segmentation_geometry(ds):
    sk = ds[10].skeleton()
    comp = cnx.morph.segment(sk, min_length_um=0)
    t = comp.table
    assert comp.method == "natural" and comp.n_merged == 0
    assert t["length"].sum() == pytest.approx(float(sk.cable_length))
    assert np.bincount(comp.comp_of_node).sum() == 9 and (comp.comp_of_node >= 0).all()
    assert t["parent_id"][0] == -1 and (t["parent_id"][1:] >= 0).all()  # root first
    assert (t["parent_id"].to_numpy()[1:] < t["compartment_id"].to_numpy()[1:]).all()  # parents precede children
    assert 1 in comp.nodes(0)  # root node in compartment 0
    assert len(comp) == 3 and sorted(t["length"].round(3).to_list()) == [0.1, 16.0, 40.0]
    assert comp.nodes(0).tolist() == [1, 2, 3, 4, 5, 6] and t["depth"].to_list() == [0, 1, 2]  # twig hangs off the z limb
    h = comp.hines(400.0)
    assert (abs(h - h.T) > 1e-15).nnz == 0 and np.allclose(h.sum(axis=1), 0)


def test_short_compartments_merge_into_parent(ds):
    sk = ds[10].skeleton()
    full = cnx.morph.segment(sk, min_length_um=0)
    merged = cnx.morph.segment(sk, min_length_um=0.5)
    assert len(merged) == len(full) - 1 and merged.n_merged == 1
    assert merged.table["length"].sum() == pytest.approx(float(sk.cable_length))
    assert merged.table["length"].min() >= 0.5
    assert (merged.table["n_nodes"].sum()) == 9 and sorted(merged.table["length"].round(3).to_list()) == [16.1, 40.0]
    with pytest.raises(ValueError):
        cnx.morph.segment(sk, method="fixed_length")
    text = merged.summary()
    assert "merged" in text and repr(merged).startswith("Compartments(")


def test_cable_model_physics(ds):
    comp = cnx.morph.segment(ds[10].skeleton())
    m = cnx.models.Cable(comp, Rm=8000, Ra=400, Cm=0.6)
    V = m.steady_state({0: 10e-12})
    assert V.shape == (len(comp),) and np.isfinite(V).all() and V[0] > 0 and V[0] == V.max()
    # current conservation: injected current leaves through the leaks
    g_leak = np.pi * comp.diameters * comp.lengths / (8000 * 1e8)
    assert float((g_leak * V / 1e3).sum()) == pytest.approx(10e-12, rel=1e-9)
    assert m.input_resistance(0) == pytest.approx(V[0] / 1e3 / 10e-12 / 1e9)
    # a constant pulse converges to the steady state
    steps = 400
    Vt = m.transient({0: np.full(steps, 10e-12)}, dt=1e-3)
    assert Vt.shape == (len(comp), steps) and np.allclose(Vt[:, -1], V, rtol=1e-3)
    a = m.attenuation(0)
    assert 0 < a["attenuation_percent"] < 100 and a["n_neighbours"] == len(comp.children(0))
    grid = cnx.models.scan(comp, Ra=[100, 400], Rm=[2000, 8000, 16000])
    assert grid.shape == (3, 2) and (grid > 0).all() and (grid < 100).all()
    with pytest.raises(ValueError, match="zero length"):
        bad = cnx.morph.segment(ds[10].skeleton(), min_length_um=0)
        bad.table = bad.table.with_columns(pl.Series("length", [0.0] + bad.table["length"].to_list()[1:]))
        cnx.models.Cable(bad)


def _state(url):
    return json.loads(unquote(url.split("#!", 1)[1]))


def test_neuron_and_compartment_views(ds):
    with pytest.raises(ValueError, match="no viewer configuration"):
        ds[10].view()
    from connexplorer.viz.neuroglancer import VIEWERS

    VIEWERS["fixture"] = dict(VIEWERS["flywire"])
    try:
        url = ds[10].view()
        assert url.startswith("https://spelunker.cave-explorer.org/#!")
        st = _state(url)
        seg = [l for l in st["layers"] if l["type"] == "segmentation" and "10" in l.get("segments", [])]
        assert len(seg) == 1 and seg[0]["source"] == "precomputed://gs://flywire_v141_mv0"
        assert st["selectedLayer"]["layer"] == "Self A"
        url = ds[12].view(partners="both", top=2, synapses=True)
        st = _state(url)
        names = [l["name"] for l in st["layers"]]
        assert "A -> self" in names and "self -> A" in names  # inputs from A and outputs to A
        syn = [l for l in st["layers"] if l["type"] == "annotation"]
        assert {l["name"]: len(l["annotations"]) for l in syn} == {"synapses A -> self": 5, "synapses self -> A": 4}
        assert syn[0]["annotations"][0]["point"][2] == pytest.approx(300 / 40)  # z_nm 300 -> 40 nm voxels
        assert ds[10].view(viewer="cave").startswith("https://ngl.cave-explorer.org/")
        assert "graphene://" in _state(ds[10].view(viewer="flywire"))["layers"][-1]["source"]
        assert ds["A"].view(viewer="codex") == "https://codex.flywire.ai/app/search?filter_string=10+11&sort_by=&page_size=10&data_version=v0"
        with pytest.raises(ValueError, match="viewer must be one of"):
            ds[10].view(viewer="nope")
        comp = cnx.morph.segment(ds[10].skeleton())
        st = _state(comp.view(ds))
        ann = [l for l in st["layers"] if l["type"] == "annotation"][0]
        assert len(ann["annotations"]) == len(comp) and "compartment 0" in ann["annotations"][0]["description"]
        assert ann["annotations"][0]["point"][0] == pytest.approx(comp.centers[0, 0] * 1e3 / 4)
    finally:
        del VIEWERS["fixture"]


def test_mcns_viewer_config_formats_version():
    from connexplorer.viz import neuroglancer as ng

    class Fake:
        name, version = "mcns", "v1.0"
        root_ids = np.array([10])

    layer = ng.segmentation_layer(Fake(), [10], "x")
    assert layer["source"][0] == "precomputed://gs://flyem-male-cns/v1.0/segmentation"
    assert ng.base_state(Fake())["dimensions"]["x"] == [8e-9, "m"]


def test_plots_render(ds):
    comp = cnx.morph.segment(ds[10].skeleton())
    fig = cnx.viz.plot3d(comp, color_by="length")
    assert fig.__class__.__name__ == "Figure" and any(tr.name == "compartments" for tr in fig.data)
    fig2 = comp.plot2d(color_by="depth")
    assert fig2.__class__.__name__ == "Figure"
    V = cnx.models.Cable(comp).steady_state({0: 10e-12})
    assert cnx.viz.plot_voltage(comp, V).layout.title.text.startswith(f"{len(comp)} compartments")
    with pytest.raises(ValueError):
        cnx.viz.plot3d(comp, color_by="mood")
