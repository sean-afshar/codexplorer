"""Hex geometry and the column map plot."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest

from connexplorer.viz.hexmap import HEX_NEIGHBORS, hex_distance, hex_neighbors, hexmap, pq_to_xy, xy_to_pq


def test_neighbours_are_equidistant_and_second_shell_is_farther():
    origin = pq_to_xy([[0, 0]])[0]
    for dp, dq in HEX_NEIGHBORS:
        assert np.isclose(np.linalg.norm(pq_to_xy([[dp, dq]])[0] - origin), np.sqrt(3))
    assert np.linalg.norm(pq_to_xy([[1, -1]])[0] - origin) > np.sqrt(3) + 1e-6


def test_hex_distance_matches_lattice():
    for dp, dq in HEX_NEIGHBORS:
        assert hex_distance([0, 0], [dp, dq]) == 1
    assert hex_distance([0, 0], [1, -1]) == 2
    assert hex_distance([0, 0], [2, 2]) == 2
    assert hex_distance([0, 0], [2, 1]) == 2
    d = hex_distance(np.array([[0, 0], [3, 1]]), np.array([[1, 1], [0, 0]]))
    assert d.tolist() == [1, 3]


def test_roundtrip_and_mirror():
    pq = np.array([[3, -2], [0, 5], [-4, -4]])
    assert np.allclose(xy_to_pq(pq_to_xy(pq, size=2.5), size=2.5), pq)
    assert np.allclose(xy_to_pq(pq_to_xy(pq, mirror=True), mirror=True), pq)
    assert len(hex_neighbors(0, 0)) == 6


@pytest.fixture
def cols():
    return pl.DataFrame(
        {
            "column_id": pl.Series([1, 2, 3, 4], dtype=pl.UInt16),
            "p": pl.Series([0, 1, 0, 1], dtype=pl.Int16),
            "q": pl.Series([0, 0, 1, 1], dtype=pl.Int16),
            "score": [0.1, 0.5, None, 1.0],
            "kind": ["pale", "yellow", None, "pale"],
            "flag": [True, False, True, True],
        }
    )


def test_hexmap_numeric_categorical_mapping_and_array(cols):
    ax = hexmap(cols, "score", title="t")
    assert len(ax.collections) == 1 and ax.get_title() == "t"
    ax = hexmap(cols, "kind", colors={"pale": "purple"})
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["pale (2)", "yellow (1)", "missing (1)"]
    ax = hexmap(cols, "flag")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["False (1)", "True (3)"]
    ax = hexmap(cols, {1: 2.0, 4: 3.0}, annotate=True, highlight=[(0, 0)])
    assert len(ax.texts) == 4 and len(ax.collections) == 2
    ax = hexmap(cols.to_pandas(), np.arange(4))
    assert len(ax.collections) == 1
    ax = hexmap(cols.select("p", "q").to_numpy(), colors="k", colorbar=False)
    assert len(ax.collections) == 1


def test_hexmap_collapses_duplicate_rows(cols):
    dup = pl.concat([cols, cols])
    ax = hexmap(dup, "kind", legend=False)
    assert len(ax.collections[0].get_paths()) == 4


def test_hexmap_rejects_misaligned_values(cols):
    with pytest.raises(ValueError):
        hexmap(cols, [1, 2, 3])
