from pathlib import Path

import numpy as np
import polars as pl
import pytest

from connexplorer import schema
from conftest import write_fixture


def _rewrite(tables: Path, name: str, df: pl.DataFrame, **kw) -> None:
    schema.write_table(df, tables / schema.TABLES[name].file, **kw)


def test_every_fixture_table_satisfies_its_spec(fixture_tables):
    for name, spec in schema.TABLES.items():
        df = schema.read_table(fixture_tables, name)
        assert df is not None, name
        assert schema.validate_table(spec, df) == []


def test_fixture_passes_all_invariants(fixture_tables):
    rep = schema.check_invariants(fixture_tables)
    assert rep.problems == []
    assert rep.counts["n_cells"] == 4
    assert rep.counts["synapses_rows"] == 15
    assert rep.counts["n_synapses_from_edges"] == 15
    assert rep.counts["n_autapses"] == 1
    assert rep.counts["n_autapse_synapses"] == 1
    assert rep.counts["synapse_row_groups"] == 4  # 15 rows / 4 per group
    assert rep.counts["has_synapses_by_post"] == 1
    assert rep.counts["n_cells_without_edges"] == 0
    assert rep.counts["n_synapses_involving_untyped_cells"] == 0


def test_missing_optional_tables_are_allowed(tmp_path):
    tables = write_fixture(tmp_path, by_post=False)
    for name in ("edges_by_neuropil", "columns"):
        (tables / schema.TABLES[name].file).unlink()
    rep = schema.check_invariants(tables)
    assert rep.problems == []
    assert rep.counts["has_synapses_by_post"] == 0
    assert schema.read_table(tables, "columns") is None
    with pytest.raises(FileNotFoundError):
        (tables / "cells.parquet").unlink()
        schema.read_table(tables, "cells")


def test_validate_table_rejects_wrong_dtype_nulls_sentinels_and_order():
    spec = schema.TABLES["edges"]
    ok = pl.DataFrame({"pre": [0, 1], "post": [1, 0], "n_syn": [2, 3]}).cast(pl.UInt32)
    assert schema.validate_table(spec, ok) == []

    u16 = ok.with_columns(pl.col("n_syn").cast(pl.UInt16))
    assert any("n_syn: dtype" in p for p in schema.validate_table(spec, u16))

    with_null = ok.with_columns(pl.Series("n_syn", [2, None], dtype=pl.UInt32))
    assert any("nulls" in p for p in schema.validate_table(spec, with_null))

    unsorted = ok.reverse()
    assert any("not sorted" in p for p in schema.validate_table(spec, unsorted))

    cells = pl.DataFrame(
        {"id": pl.Series([0], dtype=pl.UInt32), "root_id": pl.Series([10], dtype=pl.Int64), "type": ["NAN"], "side": ["left"], "nt": ["Ach"]}
    )
    problems = schema.validate_table(schema.TABLES["cells"], cells)
    assert any("sentinel" in p for p in problems)
    assert any("side" in p for p in problems)
    assert any(".nt:" in p for p in problems)


def test_invariants_catch_edge_synapse_disagreement(fixture_tables):
    edges = schema.read_table(fixture_tables, "edges")
    _rewrite(fixture_tables, "edges", edges.with_columns(pl.col("n_syn") + 1))
    rep = schema.check_invariants(fixture_tables)
    assert any("edges.n_syn.sum()" in p for p in rep.problems)
    assert any("grouped by (pre, post)" in p for p in rep.problems)
    assert any("type_matrix.sum()" in p for p in rep.problems)
    assert any("edges_by_neuropil" in p for p in rep.problems)


def test_invariants_catch_unsorted_synapses_and_bad_ids(fixture_tables):
    syn = pl.read_parquet(fixture_tables / "synapses.parquet")
    _rewrite(fixture_tables, "synapses", syn.reverse(), row_group_size=4)
    rep = schema.check_invariants(fixture_tables)
    assert any("not sorted by pre" in p for p in rep.problems)
    assert any("row-group pre ranges" in p for p in rep.problems)

    _rewrite(fixture_tables, "synapses", syn.with_columns(pl.col("post") + 10), row_group_size=4)
    rep = schema.check_invariants(fixture_tables)
    assert any("post ids >= 4" in p for p in rep.problems)


def test_invariants_catch_inconsistent_nt_layer_and_type_counts(fixture_tables):
    types = schema.read_table(fixture_tables, "types")
    _rewrite(fixture_tables, "types", types.with_columns(pl.Series("nt", ["GABA", "GLUT"])))
    rep = schema.check_invariants(fixture_tables)
    assert any("labelled 'prediction'" in p for p in rep.problems)

    _rewrite(fixture_tables, "types", types.with_columns(pl.Series("n_cells", [3, 1], dtype=pl.UInt32)))
    rep = schema.check_invariants(fixture_tables)
    assert any("n_cells disagrees" in p for p in rep.problems)


def test_invariants_catch_id_gaps_and_missing_matrix(fixture_tables):
    cells = schema.read_table(fixture_tables, "cells")
    _rewrite(fixture_tables, "cells", cells.with_columns(pl.Series("id", [0, 1, 2, 5], dtype=pl.UInt32)))
    (fixture_tables / schema.TYPE_MATRIX_FILE).unlink()
    rep = schema.check_invariants(fixture_tables)
    assert "cells.id is not arange(N)" in rep.problems
    assert any("type_matrix.npz missing" in p for p in rep.problems)


def test_type_matrix_round_trip(tmp_path):
    import scipy.sparse as sp

    m = sp.csr_matrix(np.array([[0, 7], [3, 0]], dtype=np.uint64))
    schema.save_type_matrix(tmp_path / "tm.npz", m, ["X", "Y"])
    back, names = schema.load_type_matrix(tmp_path / "tm.npz")
    assert back.dtype == np.uint64
    assert back.indices.dtype == np.int32
    np.testing.assert_array_equal(back.toarray(), m.toarray())
    assert list(names) == ["X", "Y"]


def test_sum_u64_does_not_overflow():
    df = pl.DataFrame({"n": pl.Series([2**32 - 1, 2**32 - 1], dtype=pl.UInt32)})
    assert schema.sum_u64(df, "n") == 2 * (2**32 - 1)
    assert schema.sum_u64(df.clear(), "n") == 0


def test_manifest_round_trip_and_version_check(fixture_tables):
    m = schema.Manifest.read(fixture_tables)
    assert m.dataset == "fixture"
    assert m.schema_version == schema.SCHEMA_VERSION
    assert m.tables["synapses"]["rows"] == 15
    assert m.tables["synapses_by_post"]["rows"] == 15
    assert m.invariants["problems"] == []
    assert m.nt_overrides_version == "fixture-1"

    m.schema_version = 99
    m.write(fixture_tables)
    with pytest.raises(ValueError, match="schema version 99"):
        schema.Manifest.read(fixture_tables)


def test_lazy_scan_prunes_to_one_row_group(fixture_tables):
    """The whole synapse design rests on this: a filter on the sort column reads one group."""
    lazy = pl.scan_parquet(fixture_tables / "synapses.parquet")
    out = lazy.filter(pl.col("pre") == 3).collect()
    assert out.height == 5
    assert out["post"].to_list() == [1] * 5

    lazy_bp = pl.scan_parquet(fixture_tables / "synapses_by_post.parquet")
    inputs = lazy_bp.filter(pl.col("post") == 2).collect()
    assert inputs.height == 5
    assert sorted(inputs["pre"].to_list()) == [0, 0, 0, 1, 1]
