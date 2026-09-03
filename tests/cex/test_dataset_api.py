from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from cex import FlyWireDataset, get_dataset
import cex.dataset.schema as schema
from cex.neurons import Neuron, NeuronBase
from cex.preprocessing import build_column_data, build_connectivity_edges, build_mcns_synapse_source_data, build_type_connectivity_data, build_type_data, download_and_extract, normalize_io_stat_table, normalize_side_values, sort_table_and_add_id
from cex.preprocessing import io_stat
from cex.util import write_table


def write_fixture(root):
    preprocessed = root / "preprocessed"
    preprocessed.mkdir()
    cells = pd.DataFrame({"id": [0, 1, 2, 3], "rid": [10, 11, 12, 13], "type": ["A", "A", "B", "B"], "side": [-1, 1, -1, 1]})
    types = pd.DataFrame({"type": ["A", "B"], "num_cells": [2, 2], "nt": ["ACH", "GABA"]})
    edges = pd.DataFrame({"pre_id": [0, 0, 1, 2, 3], "post_id": [0, 2, 2, 0, 1], "num_syn": [1, 3, 2, 4, 5]})
    synapses = pd.DataFrame({"pre_id": [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3], "post_id": [0, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], "x_nm": np.arange(15), "y_nm": np.arange(15), "z_nm": np.arange(15)})
    cells.to_parquet(preprocessed / "cell_data.parquet", index=False)
    types.to_parquet(preprocessed / "type_data.parquet", index=False)
    edges.to_parquet(preprocessed / "cell_to_cell_syn_count.parquet", index=False)
    synapses.to_parquet(preprocessed / "synapses.parquet", index=False)
    write_table({"id": np.array([0, 2]), "p": np.array([7, 8]), "q": np.array([9, 10]), "side": np.array([-1, -1])}, preprocessed / "columns_data.npz")
    write_table(build_type_connectivity_data(edges, cells, types["type"]), preprocessed / "type_to_type_syn_count.npz")


def test_preprocessing_ids_and_counts_never_use_boolean_dtype():
    cells = sort_table_and_add_id(
        pd.DataFrame({"rid": [10, 11], "type": ["A", "B"]}),
        sort_cols=["type", "rid"],
    )
    connectivity = build_connectivity_edges(
        pd.DataFrame({"pre_id": [0, 1], "post_id": [1, 0]})
    )

    assert cells["id"].dtype != np.dtype(bool)
    assert all(connectivity[col].dtype != np.dtype(bool) for col in connectivity)


def test_io_stat_normalization_drops_all_unknown_type_labels():
    table = pd.DataFrame({"type": ["A", "Unknown", "NAN", "uncertain"], "N_s": [3, 1, 2, 4]})

    normalized = normalize_io_stat_table(table)

    assert normalized["type"].tolist() == ["A"]


def test_type_connectivity_and_metadata(tmp_path):
    write_fixture(tmp_path)
    dataset = get_dataset("flywire", tmp_path)

    result = dataset.connectivity.counts_between_types(["A", "B"], pre_side="left", include_column_Q=True)

    np.testing.assert_array_equal(result["pre_ids"], [0, 2])
    np.testing.assert_array_equal(result["matrix"].toarray(), [[0, 3], [4, 0]])
    np.testing.assert_array_equal(result["pre_type_sign"], [1, -1])
    np.testing.assert_array_equal(result["pre_column"]["pq"], [[7, 9], [8, 10]])
    np.testing.assert_array_equal(dataset.connectivity.num_syn_between_type_groups(["A"], ["A", "B"]).toarray(), [[1, 5]])
    np.testing.assert_array_equal(dataset.connectivity.num_syn_between_id_pairs([0, 2], [2, 0]), [3, 4])
    np.testing.assert_array_equal(dataset.connectivity.num_syn_between_rid_pairs([10], [12]), [3])
    assert dataset.cell.rid_to_type[10] == "A"
    assert dataset.synapse.get_pair_synapses(0, 2).shape[0] == 2


def test_neuron_summary_and_flywire_url(tmp_path):
    write_fixture(tmp_path)
    dataset = FlyWireDataset(tmp_path)
    neuron = Neuron(0, dataset, remove_autapse_Q=True)

    assert neuron.root_id == 10
    assert neuron.get_top_n_output_type_stat().iloc[0]["type"] == "B"
    assert "spelunker.cave-explorer.org" in dataset.ng_url_for_ids([0])
    assert isinstance(NeuronBase([0, 2], dataset).root_id, np.ndarray)


def test_visual_types_reads_only_normalized_table(tmp_path):
    write_fixture(tmp_path)
    dataset = get_dataset("flywire", tmp_path)
    raw_table = pd.DataFrame({"type": ["raw"]})
    raw_fp = tmp_path / "download_data" / "visual_neuron_types.csv"
    raw_table.to_csv(raw_fp, index=False)

    with pytest.raises(FileNotFoundError):
        _ = dataset.visual_types

    normalized_table = pd.DataFrame({"type": ["normalized"], "category": ["visual"]})
    normalized_table.to_parquet(tmp_path / "preprocessed" / schema.VISUAL_TYPE_DATA_FILE, index=False)

    pd.testing.assert_frame_equal(dataset.visual_types, normalized_table)


def test_preprocessing_normalizes_type_and_column_metadata():
    cells = pd.DataFrame({"id": [0, 1], "rid": [10, 11], "type": ["A", "A"], "side": normalize_side_values(["left", "right"]), "nt_type": ["ACH", "ACH"]})
    type_data = build_type_data(cells, value_cols=["nt_type"], col_name_map={"nt_type": "nt"})
    column_data = build_column_data(pd.DataFrame({"root_id": [10], "p": [3], "q": [4], "side": ["left"]}), cells)

    assert type_data.loc[0, "nt"] == "ACH"
    np.testing.assert_array_equal(column_data["id"], [0])
    np.testing.assert_array_equal(column_data["side"], [-1])


def test_mcns_preprocessing_converts_voxels_to_nanometers():
    partners = pd.DataFrame({"body_pre": [20], "body_post": [20], "x_pre": [1], "y_pre": [2], "z_pre": [3], "x_post": [3], "y_post": [4], "z_post": [5]})

    synapses = build_mcns_synapse_source_data(partners, voxel_size_nm=8)

    np.testing.assert_array_equal(synapses[["ctr_x", "ctr_y", "ctr_z"]].to_numpy(), [[16, 24, 32]])


def test_download_and_extract_removes_successful_archive(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.txt").write_text("ok")
    archive = shutil.make_archive(str(tmp_path / "fixture"), "zip", source)

    output = download_and_extract(Path(archive).as_uri(), tmp_path / "output")

    assert output == tmp_path / "output"
    assert (output / "data.txt").read_text() == "ok"
    assert not (output / "fixture.zip").exists()


def test_download_and_extract_strips_one_archive_root(tmp_path):
    source = tmp_path / "source" / "preprocessed"
    source.mkdir(parents=True)
    (source / "cell_data.parquet").write_text("fixture")
    archive = shutil.make_archive(str(tmp_path / "flywire_preprocessed"), "zip", source.parent)

    output = download_and_extract(
        Path(archive).as_uri(),
        tmp_path / "data" / "flywire" / "preprocessed",
        strip_single_root_Q=True,
    )

    assert (output / "cell_data.parquet").read_text() == "fixture"
    assert not (output / "preprocessed").exists()
    assert not (output / "flywire_preprocessed.zip").exists()


def test_io_stat_handles_missing_connected_cell_types(tmp_path):
    write_fixture(tmp_path)
    fp = tmp_path / "preprocessed" / "cell_data.parquet"
    cell_data = pd.read_parquet(fp)
    cell_data.loc[cell_data["id"] == 2, "type"] = np.nan
    cell_data.to_parquet(fp, index=False)
    dataset = get_dataset("flywire", tmp_path)

    result = io_stat.compute_all_io_stats(dataset, cell_types=["NAN", "A"], write_Q=False)

    assert isinstance(dataset.cell.cell_types, np.ndarray)
    assert "Unknown" in dataset.cell.cell_types
    assert list(result.keys()) == ["A"]
    assert "Unknown" not in set(result["A"]["output"]["type"])


def test_release_dataset_only_copies_required_legacy_grouping(tmp_path):
    write_fixture(tmp_path)
    dataset = get_dataset("flywire", tmp_path)

    grouped = dataset._group_root_ids([10, 11], rid_num_syn=np.asarray([3, 2]))

    assert np.isclose(grouped["t_enum_c"][0], 5 / 3)
    for legacy_name in [
        "cell_type_t", "cell_type_info", "cell_type", "cell_type_to_rid",
        "rid_to_cell_type", "rid_to_idx", "dataset_fp", "analysis_fp",
        "syn_data", "get_cell_type_rid", "rid_to_side",
        "get_num_syn_betwee_two_rid_groups", "get_raw_cell_synapse_data",
        "load_single_neuron_skeleton", "rid_to_column",
    ]:
        assert not hasattr(dataset, legacy_name)
