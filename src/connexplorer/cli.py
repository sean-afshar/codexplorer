"""``connexplorer`` command line: build datasets, inspect them, query connectivity, make viewer links."""

from __future__ import annotations

import sys
from pathlib import Path

import click
import polars as pl
from rich import box
from rich.console import Console
from rich.table import Table

import connexplorer as cnx

console = Console()


def _table(title: str, columns: list[str]) -> Table:
    t = Table(title=title, box=box.ROUNDED, header_style="bold cyan", title_style="bold magenta")
    for c in columns:
        t.add_column(c, justify="right" if c not in ("type", "root id", "partner", "side", "nt", "neuropil", "dataset", "version", "path") else "left")
    return t


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{100 * v:.1f}%" if 0 <= v <= 1 else f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _print(df: pl.DataFrame, title: str, limit: int) -> None:
    t = _table(title, [c.replace("_", " ") for c in df.columns])
    for row in df.head(limit).iter_rows():
        t.add_row(*[_fmt(v) for v in row])
    console.print(t)
    if df.height > limit:
        console.print(f"[dim]{df.height - limit:,} more rows; use -l to show more[/dim]")


def _open(dataset: str, version: str | None = None) -> cnx.Dataset:
    try:
        return cnx.open(dataset, version)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))


def _target(ds: cnx.Dataset, spec: str):
    """A root id, a type name, or comma-separated list of either -> NeuronSet."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    keys = [int(p) if p.isdigit() else p for p in parts]
    try:
        return ds[keys[0]] if len(keys) == 1 else ds[keys]
    except KeyError as e:
        raise click.ClickException(str(e))


@click.group()
@click.version_option(cnx.__version__, prog_name="connexplorer")
def cli():
    """connexplorer: fast, polars-native access to fly connectomes."""


@cli.command()
@click.argument("dataset", type=click.Choice(sorted(cnx.ingest.RAW_ARCHIVES)))
@click.option("--raw", required=True, type=click.Path(), help="raw download directory")
def download(dataset: str, raw: str):
    """Fetch a vendor raw bundle into RAW."""
    files = cnx.ingest.download_raw(dataset, raw)
    console.print(f"{len(files)} files in {raw}")


@cli.command()
@click.argument("dataset", type=click.Choice(sorted(cnx.ingest.SOURCES)))
@click.option("--raw", required=True, type=click.Path(exists=True), help="raw download directory")
@click.option("--out", required=True, type=click.Path(), help="dataset directory; tables go in OUT/tables")
@click.option("--version", "version", default=None, help="FlyWire snapshot (783) or Male CNS release (v1.0)")
@click.option("--no-by-post", is_flag=True, help="skip the post-sorted synapse copy")
@click.option("--no-hash", is_flag=True, help="do not sha256 the raw files")
def build(dataset: str, raw: str, out: str, version: str | None, no_by_post: bool, no_hash: bool):
    """Build the normalized tables for DATASET from RAW into OUT."""
    src = cnx.ingest.SOURCES[dataset](version) if version else cnx.ingest.SOURCES[dataset]()
    try:
        res = cnx.ingest.build(src, out, raw, by_post=not no_by_post, hash_sources=not no_hash, log=lambda s: console.print(f"[dim]{s}[/dim]"))
    except cnx.ingest.BuildError as e:
        raise click.ClickException(str(e))
    console.print(f"[green]ok[/green] {res.tables_dir} ({sum(res.timings.values()):.0f} s)")


@cli.command()
def datasets():
    """List the built datasets found in the search roots."""
    rows = cnx.dataset.available()
    if not rows:
        raise click.ClickException("no built datasets found; set CONNEXPLORER_DATA or run `connexplorer build`")
    t = _table("built datasets", ["dataset", "version", "cells", "synapses", "path"])
    for name, version, path, m in rows:
        t.add_row(name, version, _fmt(m.tables["cells"]["rows"]), _fmt(m.tables["synapses"]["rows"]), str(path))
    console.print(t)


@cli.command()
@click.argument("dataset")
@click.option("--version", default=None)
def info(dataset: str, version: str | None):
    """Summarize one dataset (name or directory)."""
    _open(dataset, version).info()


@cli.command()
@click.argument("dataset")
@click.option("-l", "--limit", default=20, show_default=True)
@click.option("--like", default=None, help="substring or regex to filter type names")
@click.option("--sort", type=click.Choice(["n_cells", "type"]), default="n_cells", show_default=True)
def types(dataset: str, limit: int, like: str | None, sort: str):
    """List cell types with counts and transmitters."""
    ds = _open(dataset)
    t = ds.types.select(["type", "n_cells", "nt", "nt_source"] + [c for c in ("flywire_type",) if c in ds.types.columns])
    if like:
        t = t.filter(pl.col("type").str.contains(like))
    t = t.sort(sort, descending=(sort == "n_cells"))
    _print(t, f"{ds.name} {ds.version}: {t.height:,} types", limit)


@cli.command()
@click.argument("dataset")
@click.argument("target")
@click.option("-m", "--mode", type=click.Choice(["inputs", "outputs", "both"]), default="both", show_default=True)
@click.option("-t", "--threshold", default=0, show_default=True, help="minimum synapses per partner")
@click.option("-l", "--limit", default=20, show_default=True)
@click.option("--by", type=click.Choice(["type", "neuropil", "cell"]), default="type", show_default=True, help="group partners by type or neuropil, or list cells")
@click.option("--normalize/--no-normalize", default=True, show_default=True, help="add whole-dataset fractions (type grouping only)")
def query(dataset: str, target: str, mode: str, threshold: int, limit: int, by: str, normalize: bool):
    """Partners of TARGET (a type, a root id, or a comma-separated list of either).

    \b
    connexplorer query flywire T4a -t 5
    connexplorer query flywire 720575940599755718 -m inputs --by cell
    connexplorer query mcns Mi1 --by neuropil -m outputs
    """
    ds = _open(dataset)
    grp = _target(ds, target)
    console.print(f"[bold cyan]{grp!r}[/bold cyan]  types: {', '.join(grp.types[:8])}{'...' if len(grp.types) > 8 else ''}")
    group = None if by == "cell" else by
    norm = normalize and by == "type"
    thr = threshold or None
    if mode in ("inputs", "both"):
        _print(grp.inputs(by=group, min_syn=thr, normalize=norm), f"inputs to {target} (>= {threshold} synapses per partner)", limit)
    if mode in ("outputs", "both"):
        _print(grp.outputs(by=group, min_syn=thr, normalize=norm), f"outputs of {target} (>= {threshold} synapses per partner)", limit)


@cli.command()
@click.argument("dataset")
@click.argument("target")
@click.option("--viewer", default="spelunker", show_default=True, help="spelunker | cave | flywire | codex | neuroglancer | clio")
@click.option("--partners", type=click.Choice(["in", "out", "both"]), default=None, help="add the top partner types as layers")
@click.option("--top", default=5, show_default=True)
@click.option("--synapses", is_flag=True, help="add synapse point layers for those partners")
@click.option("-t", "--threshold", default=0, show_default=True)
def view(dataset: str, target: str, viewer: str, partners: str | None, top: int, synapses: bool, threshold: int):
    """Print a viewer URL for TARGET; nothing is opened."""
    ds = _open(dataset)
    try:
        url = _target(ds, target).view(viewer=viewer, partners=partners, top=top, synapses=synapses, min_syn=threshold or None)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(url)


@cli.command()
@click.argument("dataset")
@click.argument("root_id", type=int)
@click.option("-l", "--limit", default=10, show_default=True)
def neuron(dataset: str, root_id: int, limit: int):
    """One cell: its row, degree, and top partners by type."""
    ds = _open(dataset)
    n = _target(ds, str(root_id))
    t = _table(repr(n), ["field", "value"])
    for k, v in n.cell.items():
        if v is not None and k not in ("id",):
            t.add_row(k, _fmt(v) if not isinstance(v, float) else f"{v:.3g}")
    d = n.degree().row(0, named=True)
    for k in ("in_degree", "out_degree", "in_syn", "out_syn"):
        t.add_row(k, _fmt(d[k]))
    console.print(t)
    _print(n.inputs(by="type"), "top input types", limit)
    _print(n.outputs(by="type"), "top output types", limit)


def main() -> int:
    try:
        cli.main(standalone_mode=False)
    except click.ClickException as e:
        console.print(f"[red]error:[/red] {e.format_message()}")
        return 1
    except click.Abort:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
