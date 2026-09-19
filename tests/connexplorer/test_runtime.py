"""Runtime API on the four-cell fixture (see conftest.py for the graph)."""

import numpy as np
import polars as pl
import pytest

import connexplorer as cnx
from connexplorer.connectivity import ConnBlock, TypeBlock
from conftest import write_fixture


@pytest.fixture
def ds(fixture_tables):
    return cnx.open(fixture_tables)


def test_open_by_path_name_and_env(tmp_path, monkeypatch):
    root = tmp_path / "root"
    tables = write_fixture(root / "fixture_v0")
    a = cnx.open(tables)
    b = cnx.open(tables.parent)
    assert a.name == b.name == "fixture" and a.version == "v0"
    assert repr(a) == "Dataset(fixture v0: 4 cells, 15 synapses)"
    monkeypatch.setattr(cnx.config, "data_dir", root)
    assert cnx.open("fixture").tables_dir == tables
    assert cnx.open("fixture", version="v0").tables_dir == tables
    with pytest.raises(FileNotFoundError, match="no built dataset 'fixture' version v9"):
        cnx.open("fixture", version="v9")
    monkeypatch.setattr(cnx.config, "data_dir", None)
    monkeypatch.setenv("CONNEXPLORER_DATA", str(root))
    assert cnx.open("fixture").n_cells == 4
    monkeypatch.chdir(tmp_path)  # no ./data here
    monkeypatch.delenv("CONNEXPLORER_DATA")
    with pytest.raises(FileNotFoundError, match="no built dataset 'nothing'"):
        cnx.open("nothing")
    assert "invariant problems: 0" in a.info()


def test_tables_and_lookups(ds):
    assert ds.n_cells == 4 and ds.n_synapses == 15 and ds.n_types == 2
    assert ds.cells.columns[:4] == ["id", "root_id", "type", "side"]
    assert ds.types["type"].to_list() == ["A", "B"]
    assert ds.edges.height == 5 and ds.edges_by_neuropil.height == 6 and ds.columns.height == 2
    np.testing.assert_array_equal(ds.root_ids, [10, 11, 12, 13])
    np.testing.assert_array_equal(ds.idx_of([13, 10]), [3, 0])
    with pytest.raises(KeyError, match="unknown root id"):
        ds.idx_of(99)
    np.testing.assert_array_equal(ds.type_idx("B"), [2, 3])
    with pytest.raises(KeyError, match="unknown type"):
        ds.type_idx("Z")


def test_selection(ds):
    n = ds[10]
    assert isinstance(n, cnx.Neuron) and n.root_id == 10 and n.id == 0 and int(n) == 10
    assert (n.type, n.side, n.nt) == ("A", "L", "ACH")
    assert n.cell["root_id"] == 10 and len(n) == 1
    assert ds["A"].ids.tolist() == [10, 11] and ds["A"].label == "A"
    assert ds[[12, 13]].ids.tolist() == [12, 13]
    assert ds[np.array([13, 12])].ids.tolist() == [12, 13]
    assert ds[["B", "A"]].ids.tolist() == [10, 11, 12, 13]
    assert ds.select(type="B", side="R").ids.tolist() == [13]
    assert ds.select(pl.col("nt") == "GABA").ids.tolist() == [12, 13]
    assert ds.select(side=["L", "R"]).ids.tolist() == [10, 11, 12, 13]
    assert len(ds.select()) == 4
    with pytest.raises(KeyError):
        ds.select(bogus=1)
    with pytest.raises(TypeError):
        ds[[1.5]]
    assert ds["A"].cells["root_id"].to_list() == [10, 11]
    assert ds[["A", "B"]].types == ["A", "B"]
    assert [m.root_id for m in ds["B"]] == [12, 13]
    assert 12 in ds["B"] and ds[12] in ds["B"] and 10 not in ds["B"]


def test_set_algebra(ds):
    a, b = ds["A"], ds["B"]
    assert (a | b).ids.tolist() == [10, 11, 12, 13]
    assert len(a & b) == 0
    assert (a - ds[[10]]).ids.tolist() == [11]
    assert (a & ds[[11, 12]]) == ds[[11]]
    assert ds[10] == ds[[10]] and ds[10] != ds[11]


def test_partner_tables_respect_autapse_flag_and_thresholds(ds):
    n = ds[10]
    out = n.outputs()
    assert out.columns == ["post", "type", "side", "n_syn"]
    assert out["post"].to_list() == [12] and out["n_syn"].to_list() == [3]
    ds.connectivity.autapses = True
    out = n.outputs()
    assert out["post"].to_list() == [12, 10] and out["n_syn"].to_list() == [3, 1]
    assert ds.connectivity.sparse.sum() == 15
    ds.connectivity.autapses = False
    assert ds.connectivity.sparse.sum() == 14 and ds.connectivity.sparse_t.sum() == 14
    assert n.outputs(min_syn=4).height == 0
    assert n.outputs(min_syn=3)["post"].to_list() == [12]

    inp = ds[12].inputs()
    assert inp["pre"].to_list() == [10, 11] and inp["n_syn"].to_list() == [3, 2]
    assert inp["n_syn"].dtype == pl.UInt64

    grp = ds["A"].outputs()
    assert grp["post"].to_list() == [12] and grp["n_syn"].to_list() == [5]
    by_type = ds["B"].inputs(by="type")
    assert by_type.columns == ["type", "n_syn", "n_partners", "frac_input"]
    assert by_type.row(0) == ("A", 5, 2, 1.0)
    assert ds["A"].outputs(by="type").row(0) == ("B", 5, 1, 1.0)
    assert ds[13].outputs().rows() == [(11, "A", "R", 5)] and ds[13].inputs().height == 0  # 13 -> 11 only; no inputs

    both = ds[10].partners()
    assert both.columns == ["partner", "type", "side", "n_out", "n_in", "n_syn"]
    assert both.row(0) == (12, "B", "L", 3, 4, 7)
    assert ds[12].partners(min_syn=4).row(0)[0] == 10  # 4 in from 10
    with pytest.raises(ValueError, match="direction"):
        ds.connectivity.partners_of(np.array([0]), "sideways")


def test_partner_tables_by_neuropil(ds):
    inp = ds[12].inputs(by="neuropil")
    assert inp.columns == ["neuropil", "n_syn", "n_partners", "frac_input"]
    assert inp.rows() == [("ME", 4, 2, 0.8), ("LO", 1, 1, 0.2)]
    out = ds[10].outputs(by=("type", "neuropil")).sort("n_syn", descending=True)
    assert out.select("type", "neuropil", "n_syn").rows() == [("B", "ME", 2), ("B", "LO", 1)]
    ds.connectivity.autapses = True
    out = ds[10].outputs(by="neuropil")
    assert out.select("neuropil", "n_syn").rows() == [("ME", 3), ("LO", 1)]
    ds.connectivity.autapses = False
    assert ds[11].inputs(by="neuropil")["neuropil"].to_list() == [None]  # 3 -> 1 has a null neuropil
    assert ds["B"].inputs(by="neuropil", min_syn=3).select("neuropil", "n_syn").rows() == [("ME", 2), ("LO", 1)]
    with pytest.raises(ValueError, match="cannot group by"):
        ds[10].outputs(by="nt")


def test_blocks(ds):
    blk = ds.connectivity["A", "B"]
    assert isinstance(blk, ConnBlock) and blk.shape == (2, 2)
    np.testing.assert_array_equal(blk.values, [[3, 0], [2, 0]])
    assert blk.sum() == 5
    assert blk.row_ids.tolist() == [10, 11] and blk.col_ids.tolist() == [12, 13]
    assert blk.row_types.to_list() == ["A", "A"]
    assert blk.long.rows() == [(10, 12, 3), (11, 12, 2)]
    assert blk.frame.columns == ["pre", "12", "13"] and blk.frame["12"].to_list() == [3, 2]
    np.testing.assert_array_equal(ds.connectivity["B", "A"].values, [[4, 0], [0, 5]])
    np.testing.assert_array_equal(ds.connectivity[ds["A"], ds["A"]].values, [[0, 0], [0, 0]])
    ds.connectivity.autapses = True
    np.testing.assert_array_equal(ds.connectivity["A", "A"].values, [[1, 0], [0, 0]])
    ds.connectivity.autapses = False
    np.testing.assert_array_equal(ds.connectivity[[10, 11], [12]].values, [[3], [2]])
    np.testing.assert_array_equal(ds.connectivity[10, 12].values, [[3]])
    np.testing.assert_array_equal(ds.connectivity[["A", "B"], "A"].values, [[0, 0], [0, 0], [4, 0], [0, 5]])
    assert ds["A"].subgraph().sum() == 0
    with pytest.raises(TypeError):
        ds.connectivity["A"]


def test_type_matrix(ds):
    tm = ds.connectivity.types
    assert tm.names == ["A", "B"]
    assert tm["A", "B"] == 5 and tm["B", "A"] == 9 and tm["A", "A"] == 0
    ds.connectivity.autapses = True
    assert tm["A", "A"] == 1 and tm.sparse.sum() == 15
    ds.connectivity.autapses = False
    assert tm.sparse.sum() == 14
    blk = tm[["A", "B"], ["A"]]
    assert isinstance(blk, TypeBlock)
    np.testing.assert_array_equal(blk.values, [[0], [9]])
    assert blk.long.rows() == [("B", "A", 9)]
    assert tm.frame.columns == ["pre_type", "A", "B"] and tm.frame["A"].to_list() == [0, 9]
    assert tm.long.rows() == [("A", "B", 5), ("B", "A", 9)]
    with pytest.raises(KeyError, match="unknown type"):
        tm["A", "Z"]


def test_synapses(ds):
    s = ds[10].synapses()
    assert s.columns == ["pre", "post", "x_nm", "y_nm", "z_nm", "neuropil"]
    assert s.height == 3 and s["pre"].to_list() == [10, 10, 10] and s["post"].to_list() == [12, 12, 12]
    ds.connectivity.autapses = True
    assert ds[10].synapses().height == 4
    ds.connectivity.autapses = False
    inp = ds[12].synapses("in")
    assert inp.height == 5 and sorted(inp["pre"].to_list()) == [10, 10, 10, 11, 11]
    assert ds[12].synapses("in", partner="A").height == 5
    assert ds[12].synapses("in", partner=ds[[10]]).height == 3
    assert ds[12].synapses("in", partner=[11]).height == 2
    assert ds[10].synapses("both").height == 3 + 4
    assert ds.synapses.between("A", "B").height == 5
    assert ds.synapses.between([12], ds["A"]).height == 4
    box = ds.synapses.in_box(x=(0, 300))
    assert box.height == 4
    assert cnx.xyz(box).shape == (4, 3) and cnx.xyz(box).dtype == np.uint32
    assert ds.synapses.load().height == 15
    assert ds.synapses.scan("post").select(pl.col("post").is_sorted()).collect().item()
    with pytest.raises(ValueError, match="direction"):
        ds[10].synapses("up")


def test_synapses_without_post_sorted_copy(tmp_path):
    ds = cnx.open(write_fixture(tmp_path, by_post=False))
    assert not ds.synapses.has_by_post
    inp = ds[12].synapses("in")
    assert inp.height == 5 and inp["post"].to_list() == [12] * 5
    assert ds[12].inputs()["pre"].to_list() == [10, 11]
