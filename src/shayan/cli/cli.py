"""
Command-line interface for querying connectome datasets.

Provides an interactive CLI for exploring connectivity data with beautiful
table formatting and flexible query options.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
import polars as pl

from shayan.datasets.flywire import FlyWireDataset
from shayan.core.connectome import Connectome

# Try to import MaleCNSDataset if available
try:
    from shayan.datasets.male_cns import MaleCNSDataset
    HAS_MALE_CNS = True
except ImportError:
    HAS_MALE_CNS = False
    MaleCNSDataset = None


console = Console()


def create_table(title: str, columns: list[str]) -> Table:
    """Create a rich table with consistent styling."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title_style="bold magenta",
    )

    for col in columns:
        table.add_column(col, style="white")

    return table


def format_number(n: int | float) -> str:
    """Format numbers with commas."""
    if isinstance(n, float):
        if n >= 10:
            return f"{n:,.1f}"
        else:
            return f"{n:.2f}"
    return f"{n:,}"


@click.group()
def cli():
    """Connect-Tome: Query and analyze connectome datasets."""
    pass


@cli.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--dataset-type', '-d', type=click.Choice(['flywire', 'male_cns']),
              default='flywire', help='Dataset type')
def info(dataset_path: str, dataset_type: str):
    """Display basic information about a dataset."""

    # Load dataset
    if dataset_type == 'flywire':
        dataset = FlyWireDataset(dataset_path)
    else:
        if not HAS_MALE_CNS:
            console.print("[red]Error: male_cns dataset not yet implemented.[/red]")
            return
        dataset = MaleCNSDataset(dataset_path)

    ctome = Connectome(dataset, mode="lazy")

    # Create info table
    table = Table(title=f"Dataset: {dataset.name}", box=box.ROUNDED, show_header=False)
    table.add_column("Property", style="cyan bold")
    table.add_column("Value", style="white")

    table.add_row("Organism", dataset.organism)
    table.add_row("Sex", dataset.sex)
    table.add_row("Version", dataset.version)
    table.add_row("Has Neuropil", "Yes" if dataset.has_neuropil else "No")
    table.add_row("Has Physical Synapses", "Yes" if dataset.has_physical_synapses else "No")
    table.add_row("Total Neurons", format_number(len(ctome.neurons)))
    table.add_row("Unique Cell Types", format_number(ctome.neurons["type"].n_unique()))

    console.print(table)


@cli.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.argument('cell_type', type=str)
@click.option('--dataset-type', '-d', type=click.Choice(['flywire', 'male_cns']),
              default='flywire', help='Dataset type')
@click.option('--mode', '-m', type=click.Choice(['inputs', 'outputs', 'both']),
              default='both', help='Query inputs, outputs, or both')
@click.option('--threshold', '-t', type=int, default=0,
              help='Minimum synapse count threshold')
@click.option('--limit', '-l', type=int, default=20,
              help='Number of results to display')
@click.option('--normalize/--no-normalize', default=True,
              help='Show normalized percentages')
def query(dataset_path: str, cell_type: str, dataset_type: str, mode: str,
          threshold: int, limit: int, normalize: bool):
    """Query connectivity for a specific cell type.

    Examples:
        connect-tome query data/flywire_fafb LC33a
        connect-tome query data/flywire_fafb EPG -t 5 -l 15
        connect-tome query data/flywire_fafb T4a -m inputs --no-normalize
    """

    # Load dataset
    if dataset_type == 'flywire':
        dataset = FlyWireDataset(dataset_path)
    else:
        if not HAS_MALE_CNS:
            console.print("[red]Error: male_cns dataset not yet implemented.[/red]")
            return
        dataset = MaleCNSDataset(dataset_path)

    ctome = Connectome(dataset, mode="eager")

    # Check if cell type exists
    cell_type_neurons = ctome.get_cell_type(cell_type)
    if len(cell_type_neurons) == 0:
        console.print(f"[red]Error: Cell type '{cell_type}' not found in dataset.[/red]")
        return

    console.print(f"\n[bold cyan]Cell Type:[/bold cyan] {cell_type}")
    console.print(f"[bold cyan]Number of Neurons:[/bold cyan] {len(cell_type_neurons)}")

    # Query inputs
    if mode in ['inputs', 'both']:
        query_builder = ctome.query().post_type(cell_type)
        if threshold > 0:
            query_builder = query_builder.min_weight(threshold)

        input_circuit = query_builder.as_circuit()

        console.print(f"\n[bold green]═══ INPUTS (threshold ≥ {threshold}) ═══[/bold green]")
        console.print(f"Total connections: {format_number(len(input_circuit.connections))}")
        console.print(f"Total synapses: {format_number(input_circuit.connections['weight'].sum())}")
        console.print(f"Average weight: {input_circuit.connections['weight'].mean():.2f}")

        if normalize:
            type_norm = input_circuit.aggregate_by_type_normalized()

            table = create_table(
                f"Top {limit} Input Sources to {cell_type}",
                ["Source", "Connections", "Synapses", "% Source Out", "% Target In", "Normalized"]
            )

            for row in type_norm.head(limit).iter_rows(named=True):
                table.add_row(
                    row["pre_type"] or "Unknown",
                    format_number(row["n_connections"]),
                    format_number(row["total_weight"]),
                    f"{row['percent_output']:.1f}%" if row['percent_output'] is not None else "N/A",
                    f"{row['percent_input']:.1f}%" if row['percent_input'] is not None else "N/A",
                    f"{row['normalized_weight']:.1f}%" if row['normalized_weight'] is not None else "N/A"
                )
        else:
            type_agg = input_circuit.aggregate_by_type()

            table = create_table(
                f"Top {limit} Input Sources to {cell_type}",
                ["Source", "Synapses"]
            )

            for row in type_agg.head(limit).iter_rows(named=True):
                table.add_row(
                    row["pre_type"],
                    format_number(row["weight"])
                )

        console.print(table)

    # Query outputs
    if mode in ['outputs', 'both']:
        query_builder = ctome.query().pre_type(cell_type)
        if threshold > 0:
            query_builder = query_builder.min_weight(threshold)

        output_circuit = query_builder.as_circuit()

        console.print(f"\n[bold green]═══ OUTPUTS (threshold ≥ {threshold}) ═══[/bold green]")
        console.print(f"Total connections: {format_number(len(output_circuit.connections))}")
        console.print(f"Total synapses: {format_number(output_circuit.connections['weight'].sum())}")
        console.print(f"Average weight: {output_circuit.connections['weight'].mean():.2f}")

        if normalize:
            type_norm = output_circuit.aggregate_by_type_normalized()

            table = create_table(
                f"Top {limit} Output Targets from {cell_type}",
                ["Target", "Connections", "Synapses", "% Source Out", "% Target In", "Normalized"]
            )

            for row in type_norm.head(limit).iter_rows(named=True):
                table.add_row(
                    row["post_type"] or "Unknown",
                    format_number(row["n_connections"]),
                    format_number(row["total_weight"]),
                    f"{row['percent_output']:.1f}%" if row['percent_output'] is not None else "N/A",
                    f"{row['percent_input']:.1f}%" if row['percent_input'] is not None else "N/A",
                    f"{row['normalized_weight']:.1f}%" if row['normalized_weight'] is not None else "N/A"
                )
        else:
            type_agg = output_circuit.aggregate_by_type()

            table = create_table(
                f"Top {limit} Output Targets from {cell_type}",
                ["Target", "Synapses"]
            )

            for row in type_agg.head(limit).iter_rows(named=True):
                table.add_row(
                    row["post_type"],
                    format_number(row["weight"])
                )

        console.print(table)

    console.print()


@cli.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--dataset-type', '-d', type=click.Choice(['flywire', 'male_cns']),
              default='flywire', help='Dataset type')
@click.option('--limit', '-l', type=int, default=20,
              help='Number of cell types to display')
def types(dataset_path: str, dataset_type: str, limit: int):
    """List all cell types in the dataset with neuron counts."""

    # Load dataset
    if dataset_type == 'flywire':
        dataset = FlyWireDataset(dataset_path)
    else:
        if not HAS_MALE_CNS:
            console.print("[red]Error: male_cns dataset not yet implemented.[/red]")
            return
        dataset = MaleCNSDataset(dataset_path)

    ctome = Connectome(dataset, mode="lazy")

    # Get cell type counts
    type_counts = (ctome.neurons
        .group_by("type")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    table = create_table(
        f"Cell Types in {dataset.name} (showing top {limit})",
        ["Cell Type", "Neuron Count"]
    )

    for row in type_counts.head(limit).iter_rows(named=True):
        table.add_row(
            row["type"],
            format_number(row["count"])
        )

    total_types = len(type_counts)
    console.print(table)
    console.print(f"\n[cyan]Total cell types in dataset: {format_number(total_types)}[/cyan]")
    if total_types > limit:
        console.print(f"[dim]Showing top {limit} of {format_number(total_types)} types. Use -l to show more.[/dim]")


if __name__ == '__main__':
    cli()
