"""Integration tests on the real FlyWire v783 tables (skipped when they are not built).

Ground truth for neuron 720575940599755718 (T4a) from docs/reports/crosscheck_report.md.
"""

import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import connexplorer as cnx

DATA = Path(os.environ.get("CONNEXPLORER_DATA", "data"))
TABLES = DATA / "flywire_783" / "tables"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not (TABLES / "manifest.json").exists(), reason="FlyWire tables not built")]

T4A = 720575940599755718


@pytest.fixture(scope="module")
def ds():
    return cnx.open(TABLES)


def test_open_and_facts(ds):
    assert (ds.name, ds.version, ds.n_cells, ds.n_synapses, ds.n_types) == ("flywire", "783", 139255, 80215790, 8547)
    n = ds[T4A]
    assert n.type == "T4a" and n.side == "R" and n.nt == "ACH"


def test_t4a_outputs_match_ground_truth(ds):
    n = ds[T4A]
    out = n.outputs()
    assert out.height == 87 and out["n_syn"].sum() == 282
    top = out.head(5)
    assert top["post"].to_list() == [720575940632504874, 720575940627706398, 720575940614051555, 720575940639834165, 720575940640184398]
    assert top["n_syn"].to_list() == [30, 15, 12, 11, 10]
    strong = n.outputs(min_syn=5)
    assert strong.height == 18 and strong["n_syn"].sum() == 158
    by_type = n.outputs(by="type", min_syn=5)
    assert by_type.row(0)[:2] == ("LPi14", 35)
    ds.connectivity.autapses = True
    try:
        strong = n.outputs(min_syn=5)
        assert strong.height == 19 and strong["n_syn"].sum() == 165
        assert n.outputs().height == 88 and n.outputs()["n_syn"].sum() == 289
    finally:
        ds.connectivity.autapses = False


def test_t4a_inputs_match_ground_truth(ds):
    n = ds[T4A]
    inp = n.inputs()
    assert inp.height == 69 and inp["n_syn"].sum() == 264
    assert inp.head(5)["pre"].to_list() == [720575940643266583, 720575940605262880, 720575940626979621, 720575940627706398, 720575940633138330]
    strong = n.inputs(min_syn=5)
    assert strong.height == 12 and strong["n_syn"].sum() == 169
    # neuropil breakdown never changes the per-partner threshold (shayan's 167 was the fragment bug)
    frag = n.inputs(by="neuropil", min_syn=5)
    assert frag["n_syn"].sum() == 169


def test_type_totals(ds):
    tm = ds.connectivity.types
    assert tm["T4a", "LPi14"] == 49069
    assert tm["Mi1", "T4a"] == 92542
    assert tm["L1", "Mi1"] == 192286
    assert tm["R7", "Dm9"] == 28699
    blk = ds.connectivity["T4a", "LPi14"]
    assert blk.shape == (1457, 4) and blk.sum() == 49069 and blk.sparse.nnz == 2525
    assert ds.connectivity["Mi1", "T4a"].sum() == 92542
    assert ds.connectivity[ds["T4a"], ds["LPi14"]].sum() == blk.sum()
    ds.connectivity.autapses = True
    try:
        assert tm["T4a", "LPi14"] == 49069  # distinct types: unaffected
        assert tm.sparse.sum() == 78899678
    finally:
        ds.connectivity.autapses = False
    assert tm.sparse.sum() == 78899678 - ds.connectivity._autapse_syn[ds.cells["type"].is_not_null().to_numpy()].sum()


def test_synapse_queries(ds):
    n = ds[T4A]
    out = n.synapses()
    assert out.height == 282 and (out["pre"] == T4A).all()
    inp = n.synapses("in")
    assert inp.height == 264 and (inp["post"] == T4A).all()
    assert n.synapses(partner="LPi14").height == 35
    assert ds.synapses.between(n, [720575940632504874]).height == 30
    assert cnx.xyz(out).shape == (282, 3)


def test_latencies_are_interactive(ds):
    n = ds[T4A]
    n.outputs(); ds.connectivity["T4a", "LPi14"].sum(); n.synapses(); ds["Dm9"].inputs(by="type")  # warm caches

    def ms(fn, reps=20):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1000

    timings = {
        "one neuron outputs": ms(lambda: n.outputs()),
        "one neuron inputs by type": ms(lambda: n.inputs(by="type")),
        "T4a x LPi14 block sum": ms(lambda: ds.connectivity["T4a", "LPi14"].sum()),
        "Dm9 inputs by type (206 cells)": ms(lambda: ds["Dm9"].inputs(by="type")),
        "type matrix lookup": ms(lambda: ds.connectivity.types["Mi1", "T4a"]),
        "one neuron output synapses": ms(lambda: n.synapses(), reps=5),
        "one neuron input synapses": ms(lambda: n.synapses("in"), reps=5),
    }
    print("\n" + "\n".join(f"  {k:36s} {v:8.2f} ms" for k, v in timings.items()))
    assert timings["one neuron outputs"] < 5
    assert timings["T4a x LPi14 block sum"] < 5
    assert timings["Dm9 inputs by type (206 cells)"] < 50
    assert timings["one neuron output synapses"] < 50
