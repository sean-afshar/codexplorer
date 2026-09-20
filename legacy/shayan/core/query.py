"""
ConnectionQuery builder for constructing connectivity queries.

Provides a fluent interface for filtering connectome data with method chaining.
All operations use pure Polars for maximum performance.
"""

from typing import Union, Iterable
import polars as pl
import numpy as np
import warnings
from .circuit import Circuit
from .connectome import Connectome


class ConnectionQuery:
    """
    Query builder for connectivity data with method chaining.

    Builds up filters incrementally and executes them efficiently
    using Polars LazyFrame query optimization. All operations use
    pure Polars - no Python loops for maximum performance.

    Example:
        >>> query = ctome.query()
        >>> circuit = (query
        ...     .pre_type("T4a")
        ...     .post_type(["LC4", "LC6"])
        ...     .min_weight(5)
        ...     .neuropil("ME_R")
        ...     .as_circuit())
    """

    def __init__(self, connectome: Connectome):
        """
        Initialize a ConnectionQuery.

        Args:
            connectome: Parent Connectome object
        """
        self.connectome = connectome
        self._lazy_df = self._get_lazy_connections()
        self._min_weight = None  # Track min_weight for normalization

    def _get_lazy_connections(self) -> pl.LazyFrame:
        """Get connections as LazyFrame regardless of loading mode."""
        conn = self.connectome.connections

        if isinstance(conn, pl.LazyFrame):
            return conn
        elif isinstance(conn, pl.DataFrame):
            return conn.lazy()
        else:
            raise TypeError(f"Unexpected connections type: {type(conn)}")

    def pre_id(self, neuron_ids: Union[int, Iterable[int]]) -> "ConnectionQuery":
        """
        Filter by presynaptic neuron ID(s).

        Args:
            neuron_ids: Single neuron ID or list of IDs

        Returns:
            Self for method chaining

        Example:
            >>> circuit = ctome.query().pre_id(720575940625363947).as_circuit()
            >>> circuit = ctome.query().pre_id([id1, id2, id3]).as_circuit()
        """
        ids = np.atleast_1d(neuron_ids).tolist()
        self._lazy_df = self._lazy_df.filter(pl.col("pre_id").is_in(ids))
        return self

    def post_id(self, neuron_ids: Union[int, Iterable[int]]) -> "ConnectionQuery":
        """
        Filter by postsynaptic neuron ID(s).

        Args:
            neuron_ids: Single neuron ID or list of IDs

        Returns:
            Self for method chaining

        Example:
            >>> circuit = ctome.query().post_id(720575940625363947).as_circuit()
        """
        ids = np.atleast_1d(neuron_ids).tolist()
        self._lazy_df = self._lazy_df.filter(pl.col("post_id").is_in(ids))
        return self

    def pre_type(self, cell_type: Union[str, list[str]]) -> "ConnectionQuery":
        """
        Filter by presynaptic cell type(s).

        Uses pure Polars semi-join for maximum performance - no Python loops.

        Args:
            cell_type: Single cell type name or list of names

        Returns:
            Self for method chaining

        Example:
            >>> circuit = ctome.query().pre_type("T4a").as_circuit()
            >>> circuit = ctome.query().pre_type(["T4a", "T4b"]).as_circuit()
        """
        types = [cell_type] if isinstance(cell_type, str) else cell_type

        # Pure Polars: get neurons matching types
        neurons_lazy = self.connectome.neurons.lazy()
        matching_neurons = neurons_lazy.filter(pl.col("type").is_in(types))

        # Semi-join: keep only connections where pre_id matches
        # This is much faster than collecting IDs and using is_in
        self._lazy_df = self._lazy_df.join(
            matching_neurons.select("neuron_id"),
            left_on="pre_id",
            right_on="neuron_id",
            how="semi"  # Only keep left rows that have a match
        )

        return self

    def post_type(self, cell_type: Union[str, list[str]]) -> "ConnectionQuery":
        """
        Filter by postsynaptic cell type(s).

        Uses pure Polars semi-join for maximum performance.

        Args:
            cell_type: Single cell type name or list of names

        Returns:
            Self for method chaining

        Example:
            >>> circuit = ctome.query().post_type("LC4").as_circuit()
            >>> circuit = ctome.query().post_type(["LC4", "LC6"]).as_circuit()
        """
        types = [cell_type] if isinstance(cell_type, str) else cell_type

        # Pure Polars: get neurons matching types
        neurons_lazy = self.connectome.neurons.lazy()
        matching_neurons = neurons_lazy.filter(pl.col("type").is_in(types))

        # Semi-join on post_id
        self._lazy_df = self._lazy_df.join(
            matching_neurons.select("neuron_id"),
            left_on="post_id",
            right_on="neuron_id",
            how="semi"
        )

        return self

    def neuropil(self, region: Union[str, list[str]]) -> "ConnectionQuery":
        """
        Filter by neuropil/brain region.

        Args:
            region: Single region name or list of names

        Returns:
            Self for method chaining

        Note:
            Not all datasets have neuropil information. If neuropil data
            is not available, a warning is issued and an empty result is returned.

        Example:
            >>> circuit = ctome.query().neuropil("ME_R").as_circuit()
            >>> circuit = ctome.query().neuropil(["ME_R", "ME_L"]).as_circuit()
        """
        regions = [region] if isinstance(region, str) else region

        # Check if dataset has neuropil data
        if not self.connectome.dataset.has_neuropil:
            warnings.warn(
                f"Dataset '{self.connectome.dataset.name}' does not have "
                f"neuropil information. Query will return empty result."
            )
            # Filter to impossible condition
            self._lazy_df = self._lazy_df.filter(pl.lit(False))
            return self

        # Pure Polars filter
        self._lazy_df = self._lazy_df.filter(pl.col("neuropil").is_in(regions))
        return self

    def min_weight(self, threshold: int) -> "ConnectionQuery":
        """
        Filter by minimum connection weight (synapse count).

        Args:
            threshold: Minimum weight (inclusive)

        Returns:
            Self for method chaining

        Example:
            >>> circuit = ctome.query().min_weight(5).as_circuit()
        """
        self._min_weight = threshold  # Store for normalization
        self._lazy_df = self._lazy_df.filter(pl.col("weight") >= threshold)
        return self

    def max_weight(self, threshold: int) -> "ConnectionQuery":
        """
        Filter by maximum connection weight.

        Args:
            threshold: Maximum weight (inclusive)

        Returns:
            Self for method chaining
        """
        self._lazy_df = self._lazy_df.filter(pl.col("weight") <= threshold)
        return self

    def weight_range(self, min_weight: int, max_weight: int) -> "ConnectionQuery":
        """
        Filter by weight range.

        Args:
            min_weight: Minimum weight (inclusive)
            max_weight: Maximum weight (inclusive)

        Returns:
            Self for method chaining

        Example:
            >>> circuit = ctome.query().weight_range(5, 20).as_circuit()
        """
        self._lazy_df = self._lazy_df.filter(
            (pl.col("weight") >= min_weight) & (pl.col("weight") <= max_weight)
        )
        return self

    def nt_type(self, neurotransmitter: Union[str, list[str]]) -> "ConnectionQuery":
        """
        Filter by neurotransmitter type.

        Args:
            neurotransmitter: Single NT type or list (e.g., "ACH", "GABA", "GLUT")

        Returns:
            Self for method chaining

        Example:
            >>> circuit = ctome.query().nt_type("ACH").as_circuit()
            >>> circuit = ctome.query().nt_type(["ACH", "GABA"]).as_circuit()
        """
        nt_types = [neurotransmitter] if isinstance(neurotransmitter, str) else neurotransmitter
        self._lazy_df = self._lazy_df.filter(pl.col("nt_type").is_in(nt_types))
        return self

    def collect(self) -> pl.DataFrame:
        """
        Execute the query and return raw DataFrame.

        Returns:
            DataFrame with filtered connections

        Example:
            >>> connections = ctome.query().pre_type("T4a").collect()
            >>> print(connections.head())
        """
        return self._lazy_df.collect()

    def as_circuit(self) -> Circuit:
        """
        Execute the query and return a Circuit object.

        Returns:
            Circuit containing filtered connections and involved neurons

        Example:
            >>> circuit = ctome.query().pre_type("T4a").as_circuit()
            >>> print(circuit.summary())
        """

        # Execute query to get connections
        connections = self._lazy_df.collect()

        if len(connections) == 0:
            # Empty circuit
            return Circuit.empty(self.connectome)

        # Pure Polars: get unique neuron IDs from both columns
        pre_ids = connections.select(pl.col("pre_id").alias("neuron_id")).unique()
        post_ids = connections.select(pl.col("post_id").alias("neuron_id")).unique()

        # Combine using Polars concat + unique (faster than Python sets)
        all_neuron_ids = pl.concat([pre_ids, post_ids]).unique()

        # Get neuron metadata using semi-join (faster than is_in for large sets)
        neurons = self.connectome.neurons.join(
            all_neuron_ids,
            on="neuron_id",
            how="semi"
        )

        return Circuit(
            connections=connections,
            neurons=neurons,
            connectome=self.connectome,
            min_weight=self._min_weight
        )

    def count(self) -> int:
        """
        Count number of connections matching the query (without loading data).

        Returns:
            Number of matching connections

        Example:
            >>> n = ctome.query().pre_type("T4a").count()
            >>> print(f"Found {n} connections")
        """
        return self._lazy_df.select(pl.len()).collect().item()

    def __repr__(self) -> str:
        return f"ConnectionQuery(connectome={self.connectome.dataset.name})"
