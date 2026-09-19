"""Cross-dataset comparison on the real FlyWire v783 and Male CNS v1.0 tables (skipped if not built)."""

import os
from pathlib import Path

import polars as pl
import pytest

import connexplorer as cnx

DATA = Path(os.environ.get("CONNEXPLORER_DATA", "data"))
FW, MC = DATA / "flywire_783" / "tables", DATA / "mcns_v1.0" / "tables"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not ((FW / "manifest.json").exists() and (MC / "manifest.json").exists()), reason="both datasets needed")]


@pytest.fixture(scope="module")
def pair():
    return cnx.open(FW), cnx.open(MC)


def test_type_map_and_types_like(pair):
    fw, mc = pair
    assert fw.type_map is None and mc.type_map.columns == ["type", "flywire_type"]
    assert mc.types_like("R7") == ["R7_unclear", "R7d", "R7p", "R7y"]
    assert mc.types_like("Mi1") == ["Mi1"] and fw.types_like("Mi1") == ["Mi1"]
    assert mc.types_like("R1-6") == ["R1-R6"]


def test_t4a_input_profiles_agree_across_datasets(pair):
    fw, mc = pair
    c = cnx.compare(fw["T4a"], mc["T4a"])
    inp = c.inputs()
    assert inp.columns[:3] == ["type", "n_syn_flywire", "n_syn_mcns"]
    top_fw = inp.sort("frac_input_flywire", descending=True).head(5)["type"].to_list()
    top_mc = inp.sort("frac_input_mcns", descending=True).head(5)["type"].to_list()
    assert {"Mi1", "Tm3", "Mi9"} <= set(top_fw) and {"Mi1", "Tm3", "Mi9"} <= set(top_mc)
    mi1 = inp.filter(pl.col("type") == "Mi1").row(0, named=True)
    assert mi1["n_syn_flywire"] == 92542 and mi1["n_syn_mcns"] == 115153
    assert abs(mi1["frac_input_flywire"] - mi1["frac_input_mcns"]) < 0.1
    # Male CNS photoreceptor subtypes fold into the FlyWire name
    dm9 = cnx.compare(fw["Dm9"], mc["Dm9"]).inputs()
    r7 = dm9.filter(pl.col("type") == "R7").row(0, named=True)
    assert r7["n_syn_flywire"] == 28699 and r7["n_syn_mcns"] > 0
    assert dm9.filter(pl.col("type").is_in(mc.types_like("R7"))).height == 0  # R7d/R7p/R7y/R7_unclear folded into R7
    assert "R7R8_unclear" in dm9["type"].to_list()  # no vendor mapping -> passes through under its own name
