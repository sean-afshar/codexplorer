"""Cross-dataset comparison of partner profiles, aligned on a shared type naming.

The Male CNS types table carries the vendor's ``flywire_type`` column, so a
comparison between a FlyWire set and a Male CNS set translates the Male CNS
side into FlyWire names (several Male CNS subtypes may fold into one FlyWire
type and are summed). Two sets from the same dataset align on raw names.
"""

from __future__ import annotations

import polars as pl

from connexplorer.dataset import Dataset
from connexplorer.neurons import NeuronSet


def type_map(ds: Dataset) -> pl.DataFrame | None:
    """``type`` -> ``flywire_type`` for datasets whose vendor annotations carry one, else None."""
    if "flywire_type" not in ds.types.columns:
        return None
    return ds.types.select("type", "flywire_type").drop_nulls().sort("type")


def types_like(ds: Dataset, name: str) -> list[str]:
    """Types of ``ds`` that map to (or are named) ``name`` in FlyWire naming."""
    m = type_map(ds)
    if m is None:
        return [name] if name in ds._type_ranges else []
    hits = m.filter(pl.col("flywire_type") == name)["type"].to_list()
    if not hits and name in ds._type_ranges:
        hits = [name]
    return sorted(hits)


def _translate(ds: Dataset, table: pl.DataFrame, to: Dataset) -> pl.DataFrame:
    """Rename ``table.type`` from ``ds`` naming into ``to`` naming where a vendor map exists."""
    if ds is to or ds.name == to.name:
        return table
    m = type_map(ds)
    if m is not None and type_map(to) is None:  # e.g. mcns -> flywire names
        return (
            table.join(m, on="type", how="left")
            .with_columns(pl.coalesce("flywire_type", "type").alias("type"))
            .drop("flywire_type")
        )
    m_to = type_map(to)
    if m_to is not None and m is None:  # e.g. flywire -> mcns names: only one-to-one names survive
        inv = m_to.group_by("flywire_type").agg(pl.col("type").alias("targets"), pl.len())
        one = inv.filter(pl.col("len") == 1).select(pl.col("flywire_type").alias("type"), pl.col("targets").list.first().alias("_t"))
        return table.join(one, on="type", how="left").with_columns(pl.coalesce("_t", "type").alias("type")).drop("_t")
    return table


class Comparison:
    """Side-by-side partner profiles of two NeuronSets, aligned by type."""

    def __init__(self, a: NeuronSet, b: NeuronSet):
        self.a, self.b = a, b
        self.names = (a.ds.name, b.ds.name) if a.ds.name != b.ds.name else ("a", "b")

    def _aligned(self, direction: str, min_syn: int | None) -> pl.DataFrame:
        frac = "frac_input" if direction == "in" else "frac_output"
        sides = []
        for s, name in zip((self.a, self.b), self.names):
            t = s.ds.connectivity.partners_of(s.idx, direction, by="type", min_syn=min_syn)
            t = _translate(s.ds, t, self.a.ds if s is self.b else self.b.ds) if s is self.b else t
            t = t.group_by("type").agg(pl.col("n_syn").sum(), pl.col("n_partners").sum())
            total = int(t["n_syn"].sum())
            t = t.with_columns((pl.col("n_syn") / max(total, 1)).alias(frac))
            sides.append(t.rename({"n_syn": f"n_syn_{name}", "n_partners": f"n_partners_{name}", frac: f"{frac}_{name}"}))
        out = sides[0].join(sides[1], on="type", how="full", coalesce=True)
        cols = [f"n_syn_{n}" for n in self.names] + [f"n_partners_{n}" for n in self.names]
        out = out.with_columns([pl.col(c).fill_null(0) for c in cols] + [pl.col(f"{frac}_{n}").fill_null(0.0) for n in self.names])
        ordered = ["type"] + cols + [f"{frac}_{n}" for n in self.names]
        return out.with_columns(pl.max_horizontal([f"{frac}_{n}" for n in self.names]).alias("_m")).sort("_m", descending=True).select(ordered)

    def inputs(self, min_syn: int | None = None) -> pl.DataFrame:
        """type, n_syn_<a>, n_syn_<b>, n_partners_<a>, n_partners_<b>, frac_input_<a>, frac_input_<b>."""
        return self._aligned("in", min_syn)

    def outputs(self, min_syn: int | None = None) -> pl.DataFrame:
        return self._aligned("out", min_syn)

    def __repr__(self) -> str:
        return f"Comparison({self.a!r} vs {self.b!r})"


def compare(a: NeuronSet, b: NeuronSet) -> Comparison:
    return Comparison(a, b)
