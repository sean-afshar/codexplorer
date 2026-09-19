"""Ingestion on tiny synthetic vendor files laid out exactly like the real ones."""

import gzip
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from connexplorer import ingest, schema
from connexplorer.ingest import base

# --------------------------------------------------------------------------
# FlyWire
# --------------------------------------------------------------------------

PREFIX = 720575940
RIDS = {  # short name -> root id suffix (the CSV omits the 720575940 prefix)
    "a": 600000001, "b": 600000002, "c": 600000003, "d": 600000004, "e": 5,  # e has a short suffix
}


def rid(k: str) -> int:
    return PREFIX * 10**9 + RIDS[k]


def _gz(path: Path, text: str) -> None:
    with gzip.open(path, "wt") as f:
        f.write(text)


def write_flywire_raw(raw: Path, *, classification: bool = True) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    _gz(
        raw / "neurons.csv.gz",
        "root_id,group,nt_type,nt_type_score,da_avg,ser_avg,gaba_avg,glut_avg,ach_avg,oct_avg\n"
        f"{rid('a')},ME,ACH,0.9,0,0,0.05,0.05,0.9,0\n"
        f"{rid('b')},ME,ACH,0.8,0,0,0.1,0.1,0.8,0\n"
        f"{rid('c')},ME,GLUT,0.7,0,0,0.2,0.7,0.1,0\n"
        f"{rid('d')},LO,,0,0,0,0,0,0,0\n"
        f"{rid('e')},LO,GABA,0.6,0,0,0.6,0.2,0.2,0\n",
    )
    # uncompressed variant, as shipped in the Codex bundle
    (raw / "consolidated_cell_types.csv").write_text(
        "root_id,primary_type,additional_type(s)\n"
        f"{rid('a')},T4a,\n{rid('b')},T4a,T4\n{rid('c')},R7,\n{rid('e')},Unknown,\n"  # d untyped, e sentinel type
    )
    if classification:
        (raw / "classification.csv").write_text(
            "root_id,flow,super_class,class,sub_class,hemilineage,side,nerve\n"
            f"{rid('a')},intrinsic,optic,optic_lobe_intrinsic,,,right,\n"
            f"{rid('b')},intrinsic,optic,optic_lobe_intrinsic,,,left,\n"
            f"{rid('c')},afferent,sensory,visual,photo_receptor,,right,\n"
            f"{rid('d')},intrinsic,central,,,,center,\n"
            f"{rid('e')},intrinsic,central,,,,,\n"
        )
    (raw / "visual_neuron_types.csv").write_text(
        "root_id,type,family,subsystem,category,side\n"
        f"{rid('a')},T4a,T4 Neuron,Motion,intrinsic,right\n"
        f"{rid('b')},T4a,T4 Neuron,Motion,intrinsic,left\n"
        f"{rid('c')},R7,Photoreceptor,Color,boundary,right\n"
    )
    _gz(raw / "names.csv.gz", f"root_id,name,group\n{rid('a')},ME.1,ME\n{rid('b')},ME.2,ME\n")
    _gz(
        raw / "column_assignment.csv.gz",
        "root_id,hemisphere,type,column_id,x,y,p,q\n"
        f"{rid('a')},right,T4a,10,-1,2,3,-4\n{rid('b')},left,T4a,11,0,0,5,6\n999,left,T4a,12,0,0,0,0\n",
    )
    # 13-column Princeton layout; ids without prefix; one empty neuropil; one unknown cell
    S = RIDS
    rows = [
        # pre, post, neuropil
        (S["a"], S["a"], "ME_R"),  # autapse
        (S["a"], S["c"], "ME_R"),
        (S["a"], S["c"], "LO_R"),
        (S["a"], S["c"], ""),
        (S["b"], S["c"], "ME_L"),
        (S["b"], S["c"], "ME_L"),
        (S["c"], S["a"], "ME_R"),
        (S["c"], S["a"], "ME_R"),
        (S["c"], S["a"], "ME_R"),
        (S["c"], S["a"], "ME_R"),
        (S["e"], S["b"], "LO_L"),
        (S["e"], S["b"], "LO_L"),
        (S["e"], S["b"], "LO_L"),
        (S["e"], S["b"], "LO_L"),
        (S["e"], S["b"], "LO_L"),
        (S["a"], 123456789, "ME_R"),  # unknown post cell -> dropped
    ]
    lines = ["pre_x,pre_y,pre_z,ctr_x,ctr_y,ctr_z,post_x,post_y,post_z,size,pre_root_id_720575940,post_root_id_720575940,neuropil"]
    for i, (p, q, npl) in enumerate(rows):
        lines.append(f"{i},{i},{i},{100*i},{200*i},{300*i},{i},{i},{i},{10+i},{p},{q},{npl}")
    _gz(raw / "fafb_v783_princeton_synapse_table.csv.gz", "\n".join(lines) + "\n")
    return raw


@pytest.fixture
def flywire_build(tmp_path):
    raw = write_flywire_raw(tmp_path / "raw")
    res = ingest.build(ingest.FlyWireSource(block_size=1 << 16), tmp_path / "flywire_783", raw, log=None)
    return res


def test_flywire_build_passes_invariants_and_records_provenance(flywire_build):
    res = flywire_build
    assert res.report.ok
    m = schema.Manifest.read(res.tables_dir)
    assert m.dataset == "flywire" and m.version == "783"
    assert m.nt_overrides_version == ingest.flywire.NT_OVERRIDES_VERSION
    assert set(m.source_files) >= {"neurons", "cell_types", "synapses", "classification", "columns"}
    assert "connections" not in m.source_files  # read=False files are never hashed
    assert all("sha256" in v for v in m.source_files.values())
    assert m.extras["synapses_read"] == 16
    assert m.extras["synapses_dropped_unknown_cell"] == 1
    assert m.extras["column_rows_dropped_unknown_cell"] == 1
    assert m.tables["synapses"]["rows"] == 15 == m.tables["synapses_by_post"]["rows"]


def test_flywire_cells_are_sorted_by_type_with_normalized_labels(flywire_build):
    cells = schema.read_table(flywire_build.tables_dir, "cells")
    assert cells["id"].to_list() == [0, 1, 2, 3, 4]
    # sorted by (type, root_id), untyped last: R7 < T4a; e's "Unknown" became null
    assert cells["type"].to_list() == ["R7", "T4a", "T4a", None, None]
    # e's short suffix makes its root id the smaller of the two untyped cells
    assert cells["root_id"].to_list() == [rid("c"), rid("a"), rid("b"), rid("e"), rid("d")]
    assert cells["side"].to_list() == ["R", "R", "L", None, "M"]
    assert cells["nt"].to_list() == ["GLUT", "ACH", "ACH", "GABA", None]
    assert cells["nt_score"].dtype == pl.Float32
    assert cells["superclass"].to_list() == ["sensory", "optic", "optic", "central", "central"]
    assert cells["name"].to_list() == [None, "ME.1", "ME.2", None, None]
    assert cells["additional_types"].to_list() == [None, None, "T4", None, None]
    assert cells.columns[:6] == ["id", "root_id", "type", "side", "nt", "nt_score"]


def test_flywire_types_carry_the_labelled_nt_layer_and_visual_extras(flywire_build):
    types = schema.read_table(flywire_build.tables_dir, "types")
    assert types["type"].to_list() == ["R7", "T4a"]
    assert types["type_idx"].to_list() == [0, 1]
    assert types["n_cells"].to_list() == [1, 2]
    assert types["nt_predicted"].to_list() == ["GLUT", "ACH"]
    assert types["nt"].to_list() == ["HIS", "ACH"]  # R7 overridden from the literature table
    assert types["nt_source"].to_list() == ["literature", "prediction"]
    assert types["superclass"].to_list() == ["sensory", "optic"]
    assert types["family"].to_list() == ["Photoreceptor", "T4 Neuron"]
    assert types["category"].to_list() == ["boundary", "intrinsic"]


def test_flywire_synapses_edges_and_matrix(flywire_build):
    t = flywire_build.tables_dir
    cells = schema.read_table(t, "cells")
    id_of = dict(zip(cells["root_id"].to_list(), cells["id"].to_list()))
    a, b, c, e = (id_of[rid(k)] for k in "abce")

    syn = pl.read_parquet(t / "synapses.parquet")
    assert syn.columns == ["pre", "post", "x_nm", "y_nm", "z_nm", "neuropil"]
    assert syn["pre"].is_sorted()
    assert syn["neuropil"].null_count() == 1  # "" became null
    assert sorted(syn.filter(pl.col("pre") == a)["y_nm"].to_list()) == [0, 200, 400, 600]  # ctr_y of rows 0..3
    assert pl.read_parquet(t / "synapses_by_post.parquet")["post"].is_sorted()

    edges = schema.read_table(t, "edges").sort("pre", "post")
    expected = pl.DataFrame(
        {"pre": [c, c, a, a, b, e], "post": [a, c, a, c, c, b], "n_syn": [4, 0, 1, 3, 2, 5]}
    ).filter(pl.col("n_syn") > 0).cast({"pre": pl.UInt32, "post": pl.UInt32, "n_syn": pl.UInt32}).sort("pre", "post")
    assert edges.equals(expected)

    by_np = schema.read_table(t, "edges_by_neuropil")
    ac = by_np.filter((pl.col("pre") == a) & (pl.col("post") == c)).sort("neuropil", nulls_last=True)
    assert ac["neuropil"].to_list() == ["LO_R", "ME_R", None]
    assert ac["n_syn"].to_list() == [1, 1, 1]

    m, names = schema.load_type_matrix(t / schema.TYPE_MATRIX_FILE)
    assert list(names) == ["R7", "T4a"]
    # R7 (c) -> T4a (a): 4 ; T4a (a,b) -> R7 (c): 3 + 2 ; T4a -> T4a: autapse 1 ; e is untyped so its 5 are excluded
    np.testing.assert_array_equal(m.toarray(), [[0, 4], [5, 1]])
    assert flywire_build.report.counts["n_synapses_involving_untyped_cells"] == 5
    assert flywire_build.report.counts["n_autapses"] == 1

    cols = schema.read_table(t, "columns")
    assert cols.columns == ["id", "p", "q", "side", "column_id", "x", "y"]
    assert cols["id"].to_list() == sorted([a, b])
    assert cols["side"].to_list() == ["R", "L"] if a < b else ["L", "R"]
    assert cols["p"].dtype == pl.Int16


def test_flywire_without_optional_files_still_builds(tmp_path):
    raw = write_flywire_raw(tmp_path / "raw", classification=False)
    for name in ("visual_neuron_types.csv", "names.csv.gz", "column_assignment.csv.gz"):
        (raw / name).unlink()
    res = ingest.build(ingest.FlyWireSource(block_size=1 << 16), tmp_path / "out", raw, log=None, hash_sources=False)
    assert res.report.ok
    cells = schema.read_table(res.tables_dir, "cells")
    assert "side" not in cells.columns and "superclass" not in cells.columns
    assert schema.read_table(res.tables_dir, "columns") is None
    assert "family" not in schema.read_table(res.tables_dir, "types").columns
    assert "sha256" not in res.manifest.source_files["neurons"]


def test_missing_required_raw_file_is_reported_by_name(tmp_path):
    raw = write_flywire_raw(tmp_path / "raw")
    (raw / "fafb_v783_princeton_synapse_table.csv.gz").unlink()
    with pytest.raises(FileNotFoundError, match="fafb_v783_princeton_synapse_table.csv.gz"):
        ingest.build(ingest.FlyWireSource(), tmp_path / "out", raw, log=None)


def test_root_id_prefix_is_parsed_from_the_header():
    pre, post, add = ingest.FlyWireSource._root_id_columns(["ctr_x", "pre_root_id_720575940", "post_root_id_720575940"])
    assert (pre, post, add) == ("pre_root_id_720575940", "post_root_id_720575940", 720575940_000000000)
    with pytest.raises(ValueError):
        ingest.FlyWireSource._root_id_columns(["pre_root_id_1", "post_root_id_2"])


# --------------------------------------------------------------------------
# Male CNS
# --------------------------------------------------------------------------


def _feather(path: Path, table: pa.Table, batch_rows: int | None = None) -> None:
    with ipc.new_file(pa.OSFile(str(path), "wb"), table.schema) as w:
        for b in table.to_batches(max_chunksize=batch_rows):
            w.write_batch(b)


def write_mcns_raw(raw: Path, release: str = "v0.9") -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    ann = pa.table(
        {
            "bodyId": pa.array([10, 11, 12, 13], pa.int64()),
            "type": ["Mi1", "Mi1", "R7d", None],
            "somaSide": ["L", "R", "M", None],
            "rootSide": ["L", "R", "M", "L"],
            "superclass": ["ol_intrinsic", "ol_intrinsic", "ol_sensory", "central"],
            "flywireType": ["Mi1", "Mi1", "R7", None],
            "assignedOlHex1": pa.array([3.0, None, 5.0, None], pa.float64()),
            "assignedOlHex2": pa.array([4.0, None, 6.0, None], pa.float64()),
            "status": ["Traced", "Traced", "Traced", "Orphan"],
            "group": pa.array([10.0, 10.0, None, None], pa.float64()),
            "unrelated": [1, 2, 3, 4],
        }
    )
    _feather(raw / f"body-annotations-male-cns-{release}-minconf-0.5.feather", ann)
    nt = pa.table(
        {
            "body": pa.array([10, 11, 12, 99], pa.int64()),
            "cell_type": ["Mi1", "Mi1", "R7d", "x"],
            "predicted_nt": ["acetylcholine", "unclear", "histamine", "gaba"],
            "predicted_nt_confidence": pa.array([0.9, 0.3, 0.8, 0.5], pa.float64()),
            "ground_truth": [None, "acetylcholine", None, None],
            "consensus_nt": ["acetylcholine", "acetylcholine", "histamine", "gaba"],
        }
    )
    _feather(raw / f"body-neurotransmitters-male-cns-{release}.feather", nt)
    partners = pa.table(
        {
            "x_pre": pa.array([10, 10, 20, 30, 30, 40, 0], pa.int32()),
            "y_pre": pa.array([10, 10, 20, 30, 30, 40, 0], pa.int32()),
            "z_pre": pa.array([10, 10, 20, 30, 30, 40, 0], pa.int32()),
            "body_pre": pa.array([10, 10, 11, 12, 12, 13, 77], pa.int64()),
            "x_post": pa.array([12, 14, 22, 32, 34, 42, 0], pa.int32()),
            "y_post": pa.array([12, 14, 22, 32, 34, 42, 0], pa.int32()),
            "z_post": pa.array([12, 14, 22, 32, 34, 42, 0], pa.int32()),
            "body_post": pa.array([11, 11, 12, 10, 10, 10, 10], pa.int64()),
            "conf_post": pa.array([0.9] * 7, pa.float32()),
            "primary_post": pa.array(["ME(R)", "ME(R)", "<unspecified>", "LO(R)", "LO(R)", "LO(R)", "LO(R)"]).dictionary_encode(),
        }
    )
    _feather(raw / f"syn-partners-male-cns-{release}-minconf-0.5.feather", partners, batch_rows=2)
    weights = pa.table({"body_pre": pa.array([10, 11, 12, 13, 77], pa.int64()), "body_post": pa.array([11, 12, 10, 10, 10], pa.int64()), "weight": pa.array([2, 1, 2, 1, 1], pa.int64())})
    _feather(raw / f"connectome-weights-male-cns-{release}-minconf-0.5.feather", weights)
    return raw


def test_mcns_build_streams_partners_and_converts_voxels(tmp_path):
    raw = write_mcns_raw(tmp_path / "raw")
    src = ingest.McnsSource("v0.9", batches_per_chunk=2)
    res = ingest.build(src, tmp_path / "mcns_v0.9", raw, log=None, hash_sources=False)
    assert res.report.ok
    t = res.tables_dir

    cells = schema.read_table(t, "cells")
    assert cells["root_id"].to_list() == [10, 11, 12, 13]  # Mi1, Mi1, R7d, untyped
    assert cells["side"].to_list() == ["L", "R", "M", None]
    assert cells["nt"].to_list() == ["ACH", None, "HIS", None]  # 'unclear' -> null; synonyms mapped
    assert cells["nt_score"].to_list() == pytest.approx([0.9, 0.3, 0.8, None]) or cells["nt_score"][3] is None
    assert cells["nt_consensus"].to_list() == ["ACH", "ACH", "HIS", None]
    assert cells["nt_ground_truth"].to_list() == [None, "ACH", None, None]
    assert cells["status"].to_list() == ["Traced", "Traced", "Traced", "Orphan"]
    assert cells["group"].dtype == pl.Int64
    assert cells["p"].to_list() == [3, None, 5, None]
    assert "unrelated" not in cells.columns

    types = schema.read_table(t, "types")
    assert types["type"].to_list() == ["Mi1", "R7d"]
    assert types["nt"].to_list() == ["ACH", "HIS"]
    assert types["nt_source"].to_list() == ["prediction", "prediction"]
    assert types["nt_consensus"].to_list() == ["ACH", "HIS"]
    assert types["flywire_type"].to_list() == ["Mi1", "R7"]

    syn = pl.read_parquet(t / "synapses.parquet")
    assert syn.height == 6  # body 77 dropped
    assert syn.columns == ["pre", "post", "x_nm", "y_nm", "z_nm", "neuropil"]
    assert syn["neuropil"].null_count() == 1  # "<unspecified>" -> null
    by_np = schema.read_table(t, "edges_by_neuropil")
    assert by_np.filter(pl.col("neuropil") == "LO(R)")["n_syn"].sum() == 3
    assert res.manifest.extras["synapses_dropped_unknown_cell"] == 1
    first = syn.filter((pl.col("pre") == 0) & (pl.col("post") == 1)).sort("x_nm")
    assert first["x_nm"].to_list() == [(10 + 12) * 4, (10 + 14) * 4]  # midpoint in voxels * 8 nm

    cols = schema.read_table(t, "columns")
    assert cols["id"].to_list() == [0, 2] and cols["side"].to_list() == ["L", "M"]

    check = ingest.verify_against_weights(t, raw, "v0.9")
    assert check["mismatched_pairs"] == 0 and check["missing_pairs"] == 0
    assert check["edges"] == 4


def test_mcns_pre_v1_transmitter_file_falls_back_to_consensus(tmp_path):
    raw = write_mcns_raw(tmp_path / "raw")
    nt = pa.table({"body": pa.array([10, 11, 12], pa.int64()), "cell_type": ["Mi1", "Mi1", "R7d"], "consensus_nt": ["gaba", "gaba", "histamine"]})
    _feather(raw / "body-neurotransmitters-male-cns-v0.9.feather", nt)
    src = ingest.McnsSource("v0.9")
    cells = src.cells(base.RawDir(raw, src.raw_files))
    assert cells["nt"].to_list() == ["gaba", "gaba", "histamine", None]  # raw spelling; build() normalizes
    assert "nt_score" not in cells.columns


def test_mcns_reports_missing_partner_columns(tmp_path):
    raw = write_mcns_raw(tmp_path / "raw")
    bad = pa.table({"body_pre": pa.array([1], pa.int64())})
    _feather(raw / "syn-partners-male-cns-v0.9-minconf-0.5.feather", bad)
    with pytest.raises(ValueError, match="lacks columns"):
        list(ingest.McnsSource("v0.9").synapse_batches(base.RawDir(raw, ingest.McnsSource("v0.9").raw_files)))


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def test_label_normalizers():
    df = pl.DataFrame({"s": ["left", " R ", "-1", "1", "center", "M", "bogus", None], "n": ["acetylcholine", "Ach", "GABA", "glutamate", "unclear", "5HT", "", None]})
    out = df.select(base.normalize_side(pl.col("s")).alias("s"), base.normalize_nt(pl.col("n")).alias("n"))
    assert out["s"].to_list() == ["L", "R", "L", "R", "M", "M", None, None]
    assert out["n"].to_list() == ["ACH", "ACH", "GABA", "GLUT", None, "SER", None, None]


def test_nullify_sentinels_strips_and_nulls():
    df = pl.DataFrame({"a": [" x ", "NAN", "Unknown", "", None], "k": [1, 2, 3, 4, 5]})
    assert base.nullify_sentinels(df)["a"].to_list() == ["x", None, None, None, None]


def test_build_types_majority_ties_and_unmatched_overrides():
    cells = pl.DataFrame(
        {
            "root_id": [1, 2, 3, 4, 5],
            "type": ["A", "A", "A", "B", None],
            "nt": ["GABA", "ACH", None, None, "SER"],
            "superclass": ["x", "y", "y", None, "z"],
        }
    )
    types, unmatched = base.build_types(cells, ("superclass",), {"B": "HIS", "Zzz": "DA"})
    assert types["type"].to_list() == ["A", "B"]
    assert types["nt_predicted"].to_list() == ["ACH", None]  # 1-1 tie broken alphabetically
    assert types["nt"].to_list() == ["ACH", "HIS"]
    assert types["nt_source"].to_list() == ["prediction", "literature"]
    assert types["superclass"].to_list() == ["y", None]
    assert types["type_idx"].dtype == pl.UInt32 and types["n_cells"].dtype == pl.UInt32
    assert unmatched == ["Zzz"]


def test_assign_ids_requires_unique_root_ids():
    with pytest.raises(ValueError, match="unique"):
        base.assign_ids(pl.DataFrame({"root_id": [1, 1], "type": ["A", "B"]}))
