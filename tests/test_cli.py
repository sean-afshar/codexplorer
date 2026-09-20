"""The connexplorer command line on the fixture dataset."""

import json
from urllib.parse import unquote

import pytest
from click.testing import CliRunner

from connexplorer.cli import cli
from connexplorer.viz.neuroglancer import VIEWERS


@pytest.fixture
def run(fixture_tables, monkeypatch):
    monkeypatch.setenv("CONNEXPLORER_DATA", str(fixture_tables.parent.parent))
    monkeypatch.setenv("COLUMNS", "200")
    runner = CliRunner()

    def _run(*args):
        res = runner.invoke(cli, [str(a) for a in args], catch_exceptions=False)
        return res

    _run.tables = fixture_tables
    return _run


def test_datasets_info_types(run):
    r = run("datasets")
    assert r.exit_code == 0 and "fixture" in r.output and "v0" in r.output
    r = run("info", run.tables)
    assert r.exit_code == 0 and "cells 4" in r.output and "invariant problems: 0" in r.output
    r = run("types", "fixture", "-l", "5")
    assert r.exit_code == 0 and "literature" in r.output and "GLUT" in r.output
    r = run("types", "fixture", "--like", "^B")
    assert "1 types" in r.output


def test_query_neuron_and_errors(run):
    r = run("query", "fixture", "B", "-m", "inputs")
    assert r.exit_code == 0 and "inputs to B" in r.output and "100.0%" in r.output
    r = run("query", "fixture", "12", "--by", "cell", "-t", "3", "-m", "inputs")
    assert r.exit_code == 0 and "10" in r.output and "11" not in r.output.split("inputs to 12")[1]
    r = run("query", "fixture", "10,11", "--by", "neuropil", "-m", "outputs")
    assert r.exit_code == 0 and "ME" in r.output
    r = run("neuron", "fixture", "10")
    assert r.exit_code == 0 and "out_degree" in r.output and "top input types" in r.output
    r = run("query", "fixture", "Zzz")
    assert r.exit_code != 0 and "unknown type" in r.output
    r = run("info", "nothing")
    assert r.exit_code != 0 and "no built dataset" in r.output


def test_view_prints_a_url(run):
    r = run("view", "fixture", "10")
    assert r.exit_code != 0 and "no viewer configuration" in r.output
    VIEWERS["fixture"] = dict(VIEWERS["flywire"])
    try:
        r = run("view", "fixture", "12", "--partners", "in", "--top", "1", "--synapses")
        assert r.exit_code == 0
        state = json.loads(unquote(r.output.strip().split("#!", 1)[1]))
        assert any(l["type"] == "annotation" for l in state["layers"])
        r = run("view", "fixture", "A", "--viewer", "codex")
        assert r.output.strip().startswith("https://codex.flywire.ai/")
    finally:
        del VIEWERS["fixture"]


def test_build_command_on_synthetic_raw(tmp_path, monkeypatch):
    from test_ingest import write_flywire_raw

    raw = write_flywire_raw(tmp_path / "raw")
    r = CliRunner().invoke(cli, ["build", "flywire", "--raw", str(raw), "--out", str(tmp_path / "fw"), "--no-hash"], catch_exceptions=False)
    assert r.exit_code == 0 and "ok" in r.output and (tmp_path / "fw" / "tables" / "manifest.json").exists()
