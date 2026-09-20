"""
Circuit class representing a subgraph with core functionality.

A Circuit encapsulates connections and neurons with efficient filtering,
graph-theoretic analysis, and normalized connectivity metrics. Uses pure
Polars with lazy evaluation for maximum performance. All operations are
immutable - filtering returns new Circuit objects.
"""

from typing import Optional, Literal, TYPE_CHECKING
import polars as pl
import warnings

if TYPE_CHECKING:
    from .connectome import Connectome


class Circuit:
    """
    Represents a circuit/subgraph with connections and neurons.

    Circuits are immutable - filtering operations return new Circuit objects.
    Uses pure Polars operations with lazy evaluation for optimal performance.

    Attributes:
        connections: DataFrame with connection data
        neurons: DataFrame with neuron metadata
        connectome: Parent Connectome object

    Example:
        >>> circuit = ctome.query().pre_type("T4a").as_circuit()
        >>> print(circuit.summary())
        >>> norm_conn = circuit.add_normalized_weights()
        >>> hubs = circuit.find_hubs(top_n=10)
    """

    def __init__(
        self,
        connections: pl.DataFrame,
        neurons: pl.DataFrame,
        connectome: "Connectome",
        min_weight: Optional[int] = None
    ):
        """
        Initialize a Circuit.

        Args:
            connections: DataFrame with columns [pre_id, post_id, weight, ...]
            neurons: DataFrame with columns [neuron_id, type, ...]
            connectome: Parent Connectome object
            min_weight: Minimum weight threshold used in query (for normalization)
        """
        self.connections = connections
        self.neurons = neurons
        self.connectome = connectome
        self.min_weight = min_weight

    @classmethod
    def empty(cls, connectome: "Connectome") -> "Circuit":
        """
        Create an empty circuit.

        Args:
            connectome: Parent Connectome object

        Returns:
            Empty Circuit with no connections or neurons
        """
        empty_connections = pl.DataFrame({
            "pre_id": pl.Series([], dtype=pl.Int64),
            "post_id": pl.Series([], dtype=pl.Int64),
            "weight": pl.Series([], dtype=pl.Int64),
        })
        empty_neurons = pl.DataFrame({
            "neuron_id": pl.Series([], dtype=pl.Int64),
            "type": pl.Series([], dtype=pl.Utf8),
        })
        return cls(empty_connections, empty_neurons, connectome)

    @property
    def n_neurons(self) -> int:
        """Number of neurons in the circuit."""
        return len(self.neurons)

    @property
    def n_connections(self) -> int:
        """Number of connections in the circuit."""
        return len(self.connections)

    @property
    def total_weight(self) -> int:
        """Total synaptic weight (sum of all connection weights)."""
        if len(self.connections) == 0:
            return 0
        return self.connections["weight"].sum()

    @property
    def cell_types(self) -> list[str]:
        """List of unique cell types in the circuit."""
        return self.neurons["type"].unique().to_list()

    @property
    def n_cell_types(self) -> int:
        """Number of unique cell types."""
        return len(self.cell_types)

    def filter_by_weight(self, min_weight: int) -> "Circuit":
        """
        Filter connections by minimum weight.

        Args:
            min_weight: Minimum connection weight (inclusive)

        Returns:
            New Circuit with filtered connections

        Example:
            >>> strong_circuit = circuit.filter_by_weight(10)
        """
        filtered_conn = self.connections.filter(pl.col("weight") >= min_weight)

        if len(filtered_conn) == 0:
            return Circuit.empty(self.connectome)

        return self._rebuild_with_connections(filtered_conn)

    def filter_by_weight_range(self, min_weight: int, max_weight: int) -> "Circuit":
        """
        Filter connections by weight range.

        Args:
            min_weight: Minimum weight (inclusive)
            max_weight: Maximum weight (inclusive)

        Returns:
            New Circuit with filtered connections
        """
        filtered_conn = self.connections.filter(
            (pl.col("weight") >= min_weight) & (pl.col("weight") <= max_weight)
        )

        if len(filtered_conn) == 0:
            return Circuit.empty(self.connectome)

        return self._rebuild_with_connections(filtered_conn)

    def filter_by_neuropil(self, neuropil: str | list[str]) -> "Circuit":
        """
        Filter to specific neuropil(s).

        Args:
            neuropil: Single region name or list of names

        Returns:
            New Circuit with filtered connections

        Example:
            >>> medulla_circuit = circuit.filter_by_neuropil("ME_R")
        """
        regions = [neuropil] if isinstance(neuropil, str) else neuropil

        # Check if neuropil column exists and has data
        if "neuropil" not in self.connections.columns:
            warnings.warn("Circuit does not have neuropil information")
            return Circuit.empty(self.connectome)

        if self.connections["neuropil"].null_count() == len(self.connections):
            warnings.warn("Circuit neuropil data is all null")
            return Circuit.empty(self.connectome)

        filtered_conn = self.connections.filter(pl.col("neuropil").is_in(regions))

        if len(filtered_conn) == 0:
            return Circuit.empty(self.connectome)

        return self._rebuild_with_connections(filtered_conn)

    def filter_by_nt_type(self, nt_type: str | list[str]) -> "Circuit":
        """
        Filter by neurotransmitter type.

        Args:
            nt_type: Neurotransmitter type(s) (e.g., "ACH", "GABA")

        Returns:
            New Circuit with filtered connections
        """
        nt_types = [nt_type] if isinstance(nt_type, str) else nt_type
        filtered_conn = self.connections.filter(pl.col("nt_type").is_in(nt_types))

        if len(filtered_conn) == 0:
            return Circuit.empty(self.connectome)

        return self._rebuild_with_connections(filtered_conn)

    def _rebuild_with_connections(self, new_connections: pl.DataFrame) -> "Circuit":
        """
        Helper to create new Circuit with updated connections and filtered neurons.

        Uses pure Polars operations - no Python loops.

        Args:
            new_connections: New connections DataFrame

        Returns:
            New Circuit object
        """
        # Pure Polars: get unique neuron IDs from both endpoint columns
        all_ids = (
            pl.concat([
                new_connections.select(pl.col("pre_id").alias("neuron_id")),
                new_connections.select(pl.col("post_id").alias("neuron_id")),
            ])
            .unique()
        )

        # Filter neurons using semi-join (fastest method)
        new_neurons = self.neurons.join(all_ids, on="neuron_id", how="semi")

        return Circuit(new_connections, new_neurons, self.connectome, self.min_weight)


    def add_normalized_weights(self) -> pl.DataFrame:
        """
        Add normalized weight columns to connections.

        Normalization uses context-aware totals:
        - Denominator: All connections from/to neurons in this circuit, filtered by min_weight if specified
        - Numerator: Connections in this circuit (all filters applied)

        Returns:
            DataFrame with original columns plus:
            - total_output: Total output synapses from pre neuron (min_weight filtered)
            - total_input: Total input synapses to post neuron (min_weight filtered)
            - percent_output: % of pre neuron's total output (0-100)
            - percent_input: % of post neuron's total input (0-100)
            - normalized_weight: Geometric mean of percent_output and percent_input

        Example:
            >>> circuit = ctome.query().pre_type("T4a").min_weight(5).as_circuit()
            >>> norm_conn = circuit.add_normalized_weights()
            >>> # percent_output uses totals from all T4a outputs with weight >= 5
        """
        if len(self.connections) == 0:
            return self.connections

        # Get all connections from the dataset
        all_connections = self.connectome.connections
        if isinstance(all_connections, pl.LazyFrame):
            all_connections = all_connections.collect()

        # Apply min_weight filter if specified (for denominator)
        if self.min_weight is not None:
            all_connections = all_connections.filter(pl.col("weight") >= self.min_weight)

        # Get neurons in this circuit
        pre_ids = self.connections["pre_id"].unique().to_list()
        post_ids = self.connections["post_id"].unique().to_list()

        # Total output per neuron (only for neurons in this circuit, min_weight filtered)
        total_outputs = (all_connections
            .filter(pl.col("pre_id").is_in(pre_ids))
            .group_by("pre_id")
            .agg(pl.col("weight").sum().alias("total_output"))
        )

        # Total input per neuron (only for neurons in this circuit, min_weight filtered)
        total_inputs = (all_connections
            .filter(pl.col("post_id").is_in(post_ids))
            .group_by("post_id")
            .agg(pl.col("weight").sum().alias("total_input"))
        )

        # Join with circuit connections and compute percentages
        normalized = (self.connections
            .join(total_outputs, left_on="pre_id", right_on="pre_id", how="left")
            .join(total_inputs, left_on="post_id", right_on="post_id", how="left")
            .with_columns([
                (100.0 * pl.col("weight") / pl.col("total_output")).alias("percent_output"),
                (100.0 * pl.col("weight") / pl.col("total_input")).alias("percent_input"),
            ])
            .with_columns(
                (pl.col("percent_output") * pl.col("percent_input")).sqrt().alias("normalized_weight")
            )
        )

        return normalized

    def aggregate_by_type_normalized(self) -> pl.DataFrame:
        """
        Aggregate connections by cell type with normalization.

        Computes biologically meaningful type-level percentages:
        - percent_output: (total synapses from type A to type B) / (total output from all type A neurons)
        - percent_input: (total synapses from type A to type B) / (total input to all type B neurons)

        Uses GLOBAL normalization (total connectivity across full dataset).

        Returns:
            DataFrame with columns:
            - pre_type, post_type: Cell types
            - n_connections: Number of individual neuron→neuron connections
            - total_weight: Sum of raw synapse counts (type A → type B)
            - percent_output: % of type A's total output going to type B
            - percent_input: % of type B's total input coming from type A
            - normalized_weight: Geometric mean of percent_output and percent_input

        Example:
            >>> type_norm = circuit.aggregate_by_type_normalized()
            >>> # Find T4a→LC4 connections
            >>> t4_lc4 = type_norm.filter(
            ...     (pl.col("pre_type") == "T4a") & (pl.col("post_type") == "LC4")
            ... )
            >>> print(f"T4a sends {t4_lc4['percent_output'][0]:.1f}% of output to LC4")
            >>> print(f"LC4 gets {t4_lc4['percent_input'][0]:.1f}% of input from T4a")
        """
        if len(self.connections) == 0:
            return pl.DataFrame()

        # Get full dataset connections
        all_connections = self.connectome.connections
        if isinstance(all_connections, pl.LazyFrame):
            all_connections = all_connections.collect()

        # Apply min_weight filter if specified (for context-aware normalization)
        if self.min_weight is not None:
            all_connections = all_connections.filter(pl.col("weight") >= self.min_weight)

        # Join all connections with neuron types (for global totals)
        conn_with_types_global = (all_connections
            .join(
                self.connectome.neurons.select([pl.col("neuron_id"), pl.col("type").alias("pre_type")]),
                left_on="pre_id",
                right_on="neuron_id",
                how="left"
            )
            .join(
                self.connectome.neurons.select([pl.col("neuron_id"), pl.col("type").alias("post_type")]),
                left_on="post_id",
                right_on="neuron_id",
                how="left"
            )
        )

        # Total output per PRE type (across entire dataset)
        total_output_by_type = (conn_with_types_global
            .group_by("pre_type")
            .agg(pl.col("weight").sum().alias("type_total_output"))
        )

        # Total input per POST type (across entire dataset)
        total_input_by_type = (conn_with_types_global
            .group_by("post_type")
            .agg(pl.col("weight").sum().alias("type_total_input"))
        )

        # Aggregate THIS circuit's connections by type (raw counts)
        circuit_type_agg = (self.connections
            .join(
                self.neurons.select([pl.col("neuron_id"), pl.col("type").alias("pre_type")]),
                left_on="pre_id",
                right_on="neuron_id",
                how="left"
            )
            .join(
                self.neurons.select([pl.col("neuron_id"), pl.col("type").alias("post_type")]),
                left_on="post_id",
                right_on="neuron_id",
                how="left"
            )
            .group_by(["pre_type", "post_type"])
            .agg([
                pl.len().alias("n_connections"),
                pl.col("weight").sum().alias("total_weight")
            ])
        )

        # Join with type-level totals and compute percentages
        normalized = (circuit_type_agg
            .join(total_output_by_type, left_on="pre_type", right_on="pre_type", how="left")
            .join(total_input_by_type, left_on="post_type", right_on="post_type", how="left")
            .with_columns([
                (100.0 * pl.col("total_weight") / pl.col("type_total_output")).alias("percent_output"),
                (100.0 * pl.col("total_weight") / pl.col("type_total_input")).alias("percent_input")
            ])
            .with_columns(
                (pl.col("percent_output") * pl.col("percent_input")).sqrt().alias("normalized_weight")
            )
            .select([
                "pre_type", "post_type", "n_connections", "total_weight",
                "percent_output", "percent_input", "normalized_weight"
            ])
            .sort("total_weight", descending=True)
        )

        return normalized

    def aggregate_by_type(self) -> pl.DataFrame:
        """
        Aggregate connections by cell type (raw counts only).

        For normalized version, use aggregate_by_type_normalized().

        Returns:
            DataFrame with columns [pre_type, post_type, weight]
            sorted by weight descending

        Example:
            >>> type_conn = circuit.aggregate_by_type()
            >>> print(type_conn.head(10))
        """
        if len(self.connections) == 0:
            return pl.DataFrame({
                "pre_type": pl.Series([], dtype=pl.Utf8),
                "post_type": pl.Series([], dtype=pl.Utf8),
                "weight": pl.Series([], dtype=pl.Int64),
            })

        # Use lazy evaluation for query optimization
        result = (
            self.connections.lazy()
            .join(
                self.neurons.lazy().select([pl.col("neuron_id"), pl.col("type").alias("pre_type")]),
                left_on="pre_id",
                right_on="neuron_id",
                how="left"
            )
            .join(
                self.neurons.lazy().select([pl.col("neuron_id"), pl.col("type").alias("post_type")]),
                left_on="post_id",
                right_on="neuron_id",
                how="left"
            )
            .group_by(["pre_type", "post_type"])
            .agg(pl.col("weight").sum())
            .sort("weight", descending=True)
            .collect()
        )

        return result

    def aggregate_by_neuropil(self) -> pl.DataFrame:
        """
        Aggregate connections by neuropil.

        Returns:
            DataFrame with columns [neuropil, n_connections, total_weight]
        """
        if "neuropil" not in self.connections.columns:
            warnings.warn("Circuit does not have neuropil information")
            return pl.DataFrame()

        aggregated = (self.connections
            .group_by("neuropil")
            .agg([
                pl.len().alias("n_connections"),
                pl.col("weight").sum().alias("total_weight")
            ])
            .sort("total_weight", descending=True)
        )

        return aggregated

    def node_stats(self) -> pl.DataFrame:
        """
        Compute per-neuron graph statistics.

        Returns:
            DataFrame with columns:
            - neuron_id: Neuron ID
            - type: Cell type
            - in_degree: Number of input partners
            - out_degree: Number of output partners
            - in_weight: Total input synapses
            - out_weight: Total output synapses
            - total_degree: in_degree + out_degree
            - total_weight: in_weight + out_weight

        Example:
            >>> stats = circuit.node_stats()
            >>> hubs = stats.sort("total_degree", descending=True).head(10)
            >>> print(hubs)
        """
        if len(self.connections) == 0:
            return pl.DataFrame()

        # Pure Polars groupby - very fast
        in_stats = (self.connections
            .group_by("post_id")
            .agg([
                pl.len().alias("in_degree"),
                pl.col("weight").sum().alias("in_weight")
            ])
            .rename({"post_id": "neuron_id"})
        )

        out_stats = (self.connections
            .group_by("pre_id")
            .agg([
                pl.len().alias("out_degree"),
                pl.col("weight").sum().alias("out_weight")
            ])
            .rename({"pre_id": "neuron_id"})
        )

        # Join with neuron metadata
        stats = (self.neurons
            .select(["neuron_id", "type"])
            .join(in_stats, on="neuron_id", how="left")
            .join(out_stats, on="neuron_id", how="left")
            .fill_null(0)  # Neurons with no connections get 0
            .with_columns([
                (pl.col("in_degree") + pl.col("out_degree")).alias("total_degree"),
                (pl.col("in_weight") + pl.col("out_weight")).alias("total_weight")
            ])
        )

        return stats

    def type_stats(self) -> pl.DataFrame:
        """
        Compute per-cell-type graph statistics.

        Returns:
            DataFrame with columns:
            - type: Cell type
            - n_neurons: Number of neurons of this type
            - avg_in_degree: Average input partners
            - avg_out_degree: Average output partners
            - avg_in_weight: Average input synapses
            - avg_out_weight: Average output synapses
            - total_connections: Total connections involving this type

        Example:
            >>> type_stats = circuit.type_stats()
            >>> print(type_stats.sort("avg_in_degree", descending=True))
        """
        node_stats = self.node_stats()

        if len(node_stats) == 0:
            return pl.DataFrame()

        # Aggregate by type
        stats = (node_stats
            .group_by("type")
            .agg([
                pl.len().alias("n_neurons"),
                pl.col("in_degree").mean().alias("avg_in_degree"),
                pl.col("out_degree").mean().alias("avg_out_degree"),
                pl.col("in_weight").mean().alias("avg_in_weight"),
                pl.col("out_weight").mean().alias("avg_out_weight"),
                pl.col("total_degree").sum().alias("total_connections")
            ])
            .sort("total_connections", descending=True)
        )

        return stats

    def find_hubs(
        self,
        top_n: int = 10,
        by: Literal["in_degree", "out_degree", "total_degree", "in_weight", "out_weight", "total_weight"] = "total_degree"
    ) -> pl.DataFrame:
        """
        Find hub neurons by degree or weight.

        Args:
            top_n: Number of top neurons to return
            by: Metric to rank by (degree or weight, in/out/total)

        Returns:
            DataFrame with top N neurons and their statistics

        Example:
            >>> hubs = circuit.find_hubs(top_n=20, by="total_degree")
            >>> input_hubs = circuit.find_hubs(top_n=10, by="in_degree")
        """
        stats = self.node_stats()

        if len(stats) == 0:
            return pl.DataFrame()

        return stats.sort(by, descending=True).head(top_n)

    def reciprocal_connections(self) -> pl.DataFrame:
        """
        Find bidirectional (reciprocal) connections.

        Returns:
            DataFrame with columns [neuron_a, neuron_b, weight_a_to_b, weight_b_to_a]
            where A→B and B→A both exist.

        Example:
            >>> reciprocal = circuit.reciprocal_connections()
            >>> print(f"Found {len(reciprocal)} reciprocal pairs")
        """
        if len(self.connections) == 0:
            return pl.DataFrame()

        # Self-join to find A→B and B→A
        # Use lazy for optimization
        forward = self.connections.lazy().select(["pre_id", "post_id", "weight"])
        backward = self.connections.lazy().select(["pre_id", "post_id", "weight"])

        reciprocal = (forward
            .join(
                backward,
                left_on=["pre_id", "post_id"],
                right_on=["post_id", "pre_id"],  # Swap pre/post
                how="inner",
                suffix="_rev"
            )
            # Only keep unique pairs (avoid A-B and B-A duplication)
            .filter(pl.col("pre_id") < pl.col("post_id"))
            .select([
                pl.col("pre_id").alias("neuron_a"),
                pl.col("post_id").alias("neuron_b"),
                pl.col("weight").alias("weight_a_to_b"),
                pl.col("weight_rev").alias("weight_b_to_a")
            ])
            .collect()
        )

        return reciprocal

    def degree_distribution(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Compute degree distributions.

        Returns:
            Tuple of (in_degree_dist, out_degree_dist) DataFrames
            Each with columns [degree, count]

        Example:
            >>> in_dist, out_dist = circuit.degree_distribution()
            >>> print("In-degree distribution:")
            >>> print(in_dist)
        """
        stats = self.node_stats()

        if len(stats) == 0:
            empty = pl.DataFrame({"degree": [], "count": []})
            return empty, empty

        in_dist = (stats
            .group_by("in_degree")
            .agg(pl.len().alias("count"))
            .rename({"in_degree": "degree"})
            .sort("degree")
        )

        out_dist = (stats
            .group_by("out_degree")
            .agg(pl.len().alias("count"))
            .rename({"out_degree": "degree"})
            .sort("degree")
        )

        return in_dist, out_dist

    def get_inputs(self, neuron_id: int) -> pl.DataFrame:
        """
        Get all input connections to a specific neuron.

        Args:
            neuron_id: Neuron ID to query

        Returns:
            DataFrame of connections targeting this neuron

        Example:
            >>> inputs = circuit.get_inputs(720575940625363947)
            >>> print(f"Found {len(inputs)} inputs")
        """
        return self.connections.filter(pl.col("post_id") == neuron_id)

    def get_outputs(self, neuron_id: int) -> pl.DataFrame:
        """
        Get all output connections from a specific neuron.

        Args:
            neuron_id: Neuron ID to query

        Returns:
            DataFrame of connections from this neuron
        """
        return self.connections.filter(pl.col("pre_id") == neuron_id)

    def get_neuron(self, neuron_id: int) -> pl.DataFrame:
        """
        Get metadata for a specific neuron in the circuit.

        Args:
            neuron_id: Neuron ID

        Returns:
            Single-row DataFrame with neuron metadata
        """
        return self.neurons.filter(pl.col("neuron_id") == neuron_id)

    def summary(self) -> dict:
        """
        Get summary statistics for the circuit.

        Returns:
            Dictionary with circuit statistics

        Example:
            >>> stats = circuit.summary()
            >>> print(stats)
            {'n_neurons': 752, 'n_connections': 15234, ...}
        """
        return {
            "n_neurons": self.n_neurons,
            "n_connections": self.n_connections,
            "n_cell_types": self.n_cell_types,
            "total_synapses": self.total_weight,
            "avg_weight": self.total_weight / self.n_connections if self.n_connections > 0 else 0,
            "cell_types": self.cell_types,
        }

    def to_pandas(self) -> tuple:
        """
        Convert to pandas DataFrames.

        Returns:
            Tuple of (connections_df, neurons_df) as pandas DataFrames

        Example:
            >>> conn_pd, neurons_pd = circuit.to_pandas()
        """
        return self.connections.to_pandas(), self.neurons.to_pandas()

    def __repr__(self) -> str:
        return (
            f"Circuit(neurons={self.n_neurons}, "
            f"connections={self.n_connections}, "
            f"synapses={self.total_weight})"
        )

    def __len__(self) -> int:
        """Number of connections in circuit."""
        return self.n_connections
