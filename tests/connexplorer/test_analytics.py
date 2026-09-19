"""Normalization, graph statistics and cross-dataset comparison on the four-cell fixture.

Without autapses: 0->2:3, 1->2:2, 2->0:4, 3->1:5.
Cell out totals (c0..c3) = 3, 2, 4, 5; in totals = 4, 5, 5, 0.
Type A = {c0, c1}: out 5, in 9.  Type B = {c2, c3}: out 9, in 5.
"""

import math

import numpy as np
import polars as pl
import pytest

import connexplorer as cnx
from connexplorer import cross as cmp


@pytest.fixture
def ds(fixture_tables):
    return cnx.open(fixture_tables)


def test_cell_and_type_totals(ds):
    t = ds.connectivity.cell_totals
    assert t.out_syn.tolist() == [3, 2, 4, 5] and t.in_syn.tolist() == [4, 5, 5, 0]
    assert t.out_deg.tolist() == [1, 1, 1, 1] and t.in_deg.tolist() == [1, 1, 2, 0]
    tt = ds.connectivity.type_totals
    assert tt.out_syn.tolist() == [5, 9] and tt.in_syn.tolist() == [9, 5]
    ds.connectivity.autapses = True
    assert ds.connectivity.cell_totals.out_syn.tolist() == [4, 2, 4, 5]
    assert ds.connectivity.type_totals.in_syn.tolist() == [10, 5]
    ds.connectivity.autapses = False
    assert ds.connectivity.cell_totals.out_syn.tolist() == [3, 2, 4, 5]  # cache is per flag


def test_partner_level_normalization(ds):
    inp = ds[12].inputs(normalize=True)
    assert inp.columns == ["pre", "type", "side", "n_syn", "frac_input", "frac_partner_output", "weight_norm"]
    assert inp["pre"].to_list() == [10, 11]
    assert inp["frac_input"].to_list() == pytest.approx([0.6, 0.4])
    assert inp["frac_partner_output"].to_list() == pytest.approx([1.0, 1.0])
    assert inp["weight_norm"].to_list() == pytest.approx([math.sqrt(0.6), math.sqrt(0.4)])
    out = ds[10].outputs(normalize=True)
    assert out.columns == ["post", "type", "side", "n_syn", "frac_output", "frac_partner_input", "weight_norm"]
    assert out.row(0)[3:] == pytest.approx((3, 1.0, 0.6, math.sqrt(0.6)))
    assert "_idx" not in ds[10].outputs().columns


def test_type_level_normalization(ds):
    t = ds["B"].inputs(by="type", normalize=True)
    assert t.columns == ["type", "n_syn", "n_partners", "frac_input", "frac_partner_output", "weight_norm"]
    assert t.row(0) == pytest.approx(("A", 5, 2, 1.0, 1.0, 1.0))
    o = ds["A"].outputs(by="type", normalize=True)
    assert o.columns == ["type", "n_syn", "n_partners", "frac_output", "frac_partner_input", "weight_norm"]
    assert o.row(0) == pytest.approx(("B", 5, 1, 1.0, 1.0, 1.0))
    ds.connectivity.autapses = True
    o = ds["A"].outputs(by="type", normalize=True).sort("type")
    # A -> A: 1 of A's 6 outputs; 1 of A's 10 inputs
    assert o.filter(pl.col("type") == "A").row(0) == pytest.approx(("A", 1, 1, 1 / 6, 1 / 10, math.sqrt(1 / 60)))
    ds.connectivity.autapses = False
    with pytest.raises(ValueError, match="only defined for by='type'"):
        ds["A"].outputs(by="neuropil", normalize=True)


def test_block_and_type_matrix_normalized(ds):
    n = ds.connectivity["A", "B"].normalized
    assert n.columns == ["pre", "post", "n_syn", "frac_output", "frac_input", "weight_norm"]
    assert n.rows() == pytest.approx([(10, 12, 3, 1.0, 0.6, math.sqrt(0.6)), (11, 12, 2, 1.0, 0.4, math.sqrt(0.4))])
    tm = ds.connectivity.types.normalized
    assert tm.columns == ["pre_type", "post_type", "n_syn", "frac_output", "frac_input", "weight_norm"]
    assert tm.rows() == pytest.approx([("A", "B", 5, 1.0, 1.0, 1.0), ("B", "A", 9, 1.0, 1.0, 1.0)])
    ds.connectivity.autapses = True
    tm = ds.connectivity.types.normalized.filter(pl.col("pre_type") == "A").sort("post_type")
    assert tm.row(0) == pytest.approx(("A", "A", 1, 1 / 6, 1 / 10, math.sqrt(1 / 60)))
    ds.connectivity.autapses = False
    assert ds.connectivity.types[["A"], ["B"]].normalized.row(0)[2:] == pytest.approx((5, 1.0, 1.0, 1.0))


def test_degree_hubs_and_distribution(ds):
    d = ds[["A", "B"]].degree()
    assert d.columns == ["root_id", "type", "in_degree", "out_degree", "total_degree", "in_syn", "out_syn", "total_syn"]
    assert d["in_degree"].to_list() == [1, 1, 2, 0] and d["out_syn"].to_list() == [3, 2, 4, 5]
    assert d["total_syn"].to_list() == [7, 7, 9, 5]
    assert ds["A"].degree(within=True)["total_degree"].to_list() == [0, 0]
    assert ds[[10, 12]].degree(within=True).select("in_degree", "out_degree", "in_syn", "out_syn").rows() == [(1, 1, 4, 3), (1, 1, 3, 4)]
    assert ds[["A", "B"]].hubs(k=1, by="in_syn")["root_id"].to_list() == [11]
    assert ds[["A", "B"]].hubs(k=2, by="total_syn")["root_id"].to_list() == [12, 10]
    with pytest.raises(ValueError):
        ds["A"].hubs(by="fame")
    assert ds[["A", "B"]].degree_distribution("out").rows() == [(1, 4)]
    assert ds[["A", "B"]].degree_distribution("in").rows() == [(0, 1), (1, 2), (2, 1)]
    ds.connectivity.autapses = True
    assert ds[10].degree().row(0)[2:5] == (2, 2, 4)
    ds.connectivity.autapses = False


def test_reciprocal_and_summary(ds):
    r = ds[["A", "B"]].reciprocal()
    assert r.columns == ["a", "b", "type_a", "type_b", "n_ab", "n_ba"]
    assert r.rows() == [(10, 12, "A", "B", 3, 4)]
    assert ds["A"].reciprocal().height == 0
    assert ds["A"].reciprocal(within=False).rows() == [(10, 12, "A", "B", 3, 4)]
    assert ds[12].reciprocal(within=False).rows() == [(10, 12, "A", "B", 3, 4)]  # still a < b
    assert ds[13].reciprocal(within=False).height == 0
    ds.connectivity.autapses = True
    assert ds[10].reciprocal(within=False).height == 1  # the autapse is not a reciprocal pair
    ds.connectivity.autapses = False
    s = ds[["A", "B"]].summary()
    assert s == {"n_cells": 4, "n_types": 2, "n_edges_within": 4, "n_syn_within": 14, "n_syn_in": 14, "n_syn_out": 14, "n_partners_in": 4, "n_partners_out": 3}
    assert ds["B"].summary()["n_edges_within"] == 0


def test_compare_same_dataset_aligns_on_type(ds):
    c = cnx.compare(ds["A"], ds["B"])
    assert c.names == ("a", "b")
    inp = c.inputs()
    assert inp.columns == ["type", "n_syn_a", "n_syn_b", "n_partners_a", "n_partners_b", "frac_input_a", "frac_input_b"]
    assert inp.sort("type").rows() == [("A", 0, 5, 0, 2, 0.0, 1.0), ("B", 9, 0, 2, 0, 1.0, 0.0)]
    out = c.outputs(min_syn=4)
    assert out.filter(pl.col("type") == "A").row(0)[1:3] == (0, 9)  # 2->0:4 and 3->1:5 survive; A->B (3, 2) do not
    assert ds.type_map is None and ds.types_like("A") == ["A"] and ds.types_like("Q") == []


def test_translation_folds_vendor_subtypes():
    a = pl.DataFrame({"type": ["Mi1", "R7d", "R7p", "Xx"], "flywire_type": ["Mi1", "R7", "R7", None]})

    class Fake:
        name = "mcns"
        types = a
        _type_ranges = {"Mi1": (0, 1), "R7d": (1, 2), "R7p": (2, 3), "Xx": (3, 4)}

    class FakeFw:
        name = "flywire"
        types = pl.DataFrame({"type": ["Mi1", "R7"]})
        _type_ranges = {"Mi1": (0, 1), "R7": (1, 2)}

    assert cmp.type_map(Fake()).height == 3
    assert cmp.types_like(Fake(), "R7") == ["R7d", "R7p"] and cmp.types_like(Fake(), "Xx") == ["Xx"]
    table = pl.DataFrame({"type": ["R7d", "R7p", "Mi1", "Xx"], "n_syn": [1, 2, 3, 4], "n_partners": [1, 1, 1, 1]})
    t = cmp._translate(Fake(), table, FakeFw())
    assert t["type"].to_list() == ["R7", "R7", "Mi1", "Xx"]  # unmapped names pass through
    back = cmp._translate(FakeFw(), pl.DataFrame({"type": ["Mi1", "R7"], "n_syn": [1, 2]}), Fake())
    assert back["type"].to_list() == ["Mi1", "R7"]  # R7 is one-to-many, so it keeps the FlyWire name
