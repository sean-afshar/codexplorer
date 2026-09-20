"""
Main Connectome class with flexible loading strategies.

This module provides the primary entry point for working with connectome datasets.
It supports three loading modes for different use cases:
- eager: Load all data immediately (best for multi-query scripts)
- lazy: Load on first access (balanced default)
- scan: Query files directly without loading (best for one-off queries)
"""

from pathlib import Path
from typing import Literal, Optional
import polars as pl
import numpy as np

from ..datasets.base import AbstractDataset



class Connectome:
    """
    Main interface for querying connectome datasets.

    Provides flexible data loading strategies and a unified query interface
    that works across different connectome datasets (Male CNS, FlyWire, etc.).

    Attributes:
        dataset: The underlying dataset loader
        mode: Loading mode ('eager', 'lazy', or 'scan')

    Example:
        >>> from shayan.datasets import MaleCNSDataset
        >>> dataset = MaleCNSDataset("data/male_cns")
        >>> ctome = Connectome(dataset, mode="lazy")
        >>> circuit = ctome.query().pre_type("T4a").as_circuit()
    """

    def __init__(
        self,
        dataset: AbstractDataset,
        mode: Literal["eager", "lazy", "scan"] = "lazy"
    ):
        """
        Initialize a Connectome.

        Args:
            dataset: Dataset loader (e.g., MaleCNSDataset, FlyWireDataset)
            mode: Loading strategy:
                - 'eager': Load all data immediately (~2GB for Male CNS)
                - 'lazy': Load on first query (default, balanced)
                - 'scan': Never load, query files directly (minimal memory)
        """
        self.dataset = dataset
        self.mode = mode

        # Private data stores (lazy-loaded)
        self._neurons: Optional[pl.DataFrame] = None
        self._connections: Optional[pl.DataFrame | pl.LazyFrame] = None
        self._synapses: Optional[pl.DataFrame | pl.LazyFrame] = None

        # Lookup indices (built on first use)
        self._type_index: Optional[dict[str, np.ndarray]] = None
        self._neuron_index: Optional[dict[int, str]] = None

        # Initialize based on mode
        if mode == "eager":
            self._load_all()
        elif mode == "scan":
            self._setup_scanners()
        elif mode != "lazy":
            raise ValueError(f"Invalid mode: {mode}. Must be 'eager', 'lazy', or 'scan'")

    def _load_all(self) -> None:
        """Eagerly load all data into memory (eager mode)."""
        self._neurons = self.dataset.load_neurons()
        self._connections = self.dataset.load_connections()
        # Don't load synapses - too large even in eager mode

    def _setup_scanners(self) -> None:
        """Setup Polars LazyFrame scanners (scan mode)."""
        # For scan mode, we create lazy scanners that query files directly
        # This requires datasets to support scan operations
        # For now, we'll load neurons (small) but scan connections
        self._neurons = self.dataset.load_neurons()

        # Try to setup lazy scanner for connections
        # If dataset doesn't support it, fall back to lazy loading
        try:
            # This is a placeholder - will be implemented per dataset
            self._connections = self._get_connection_scanner()
        except (NotImplementedError, AttributeError):
            # Fall back to lazy mode for connections
            self._connections = None

    def _get_connection_scanner(self) -> pl.LazyFrame:
        """
        Get a LazyFrame scanner for connections (scan mode).

        This allows querying files without loading into memory.
        Subclasses can override to provide dataset-specific scanning.
        """
        raise NotImplementedError("Scan mode not yet implemented for this dataset")

    @property
    def neurons(self) -> pl.DataFrame:
        """
        Get neuron metadata DataFrame.

        Returns:
            DataFrame with columns: neuron_id, type, side, region, nt_type

        Note:
            Neurons are always loaded (small size ~50MB for Male CNS).
            In lazy mode, loads on first access.
        """
        if self._neurons is None:
            self._neurons = self.dataset.load_neurons()
        return self._neurons

    @property
    def connections(self) -> pl.DataFrame | pl.LazyFrame:
        """
        Get connections DataFrame or LazyFrame.

        Returns:
            DataFrame (eager/lazy mode) or LazyFrame (scan mode)
            with columns: pre_id, post_id, weight, neuropil, nt_type

        Note:
            In lazy mode, loads on first access.
            In scan mode, returns LazyFrame that queries files.
        """
        if self._connections is None:
            self._connections = self.dataset.load_connections()
        return self._connections

    @property
    def type_index(self) -> dict[str, np.ndarray]:
        """
        Get cell type to neuron IDs index.

        Returns:
            Dictionary mapping cell_type → array of neuron_ids

        Note:
            Built on first access for fast type lookups.

        Example:
            >>> idx = ctome.type_index
            >>> t4a_ids = idx["T4a"]  # All T4a neuron IDs
        """
        if self._type_index is None:
            self._build_type_index()
        return self._type_index

    def _build_type_index(self) -> None:
        """Build cell type to neuron IDs index."""
        # Group neurons by type and collect IDs
        groups = (self.neurons
            .group_by("type")
            .agg(pl.col("neuron_id"))
        )

        # Convert to dictionary of numpy arrays for fast lookup
        self._type_index = {
            row["type"]: np.array(row["neuron_id"])
            for row in groups.iter_rows(named=True)
        }

    @property 
    def neuron_index(self) -> dict[int, str]:
        """
        Get neuron ID to cell type index.

        Returns:
            Dictionary mapping neuron_id → cell_type

        Note:
            Built on first access for fast ID lookups.

        Example:
            >>> idx = ctome.neuron_index
            >>> cell_type = idx[720575940625363947]  # Cell type for given neuron ID
        """
        if self._neuron_index is None:
            self._build_neuron_index()
        return self._neuron_index

    def _build_neuron_index(self) -> None:
        """Build neuron ID to cell type index."""
        self._neuron_index = {
            row["neuron_id"]: row["type"]
            for row in self.neurons.iter_rows(named=True)
        }

    def query(self):
        """
        Start a new connectivity query.

        Returns:
            ConnectionQuery builder for chaining filters

        Example:
            >>> circuit = ctome.query().pre_type("T4a").post_type("LC4").as_circuit()
        """
        from .query import ConnectionQuery
        return ConnectionQuery(self)

    def get_neuron(self, neuron_id: int) -> pl.DataFrame:
        """
        Get metadata for a specific neuron.

        Args:
            neuron_id: Neuron ID to look up

        Returns:
            Single-row DataFrame with neuron metadata

        Example:
            >>> neuron = ctome.get_neuron(720575940625363947)
            >>> print(neuron["type"][0])
        """
        return self.neurons.filter(pl.col("neuron_id") == neuron_id)

    def get_cell_type(self, cell_type: str) -> pl.DataFrame:
        """
        Get all neurons of a specific cell type.

        Args:
            cell_type: Cell type name (e.g., "T4a", "LC4")

        Returns:
            DataFrame with all neurons of that type

        Example:
            >>> t4a_neurons = ctome.get_cell_type("T4a")
            >>> print(f"Found {len(t4a_neurons)} T4a neurons")
        """
        return self.neurons.filter(pl.col("type") == cell_type)

    def get_cell_type_ids(self, cell_type: str) -> np.ndarray:
        """
        Get neuron IDs for a specific cell type (fast lookup).

        Args:
            cell_type: Cell type name

        Returns:
            Array of neuron IDs

        Example:
            >>> t4a_ids = ctome.get_cell_type_ids("T4a")
        """
        return self.type_index.get(cell_type, np.array([], dtype=np.int64))

    def __repr__(self) -> str:
        return (
            f"Connectome(\n"
            f"  dataset={self.dataset.name},\n"
            f"  mode='{self.mode}',\n"
            f"  neurons={'loaded' if self._neurons is not None else 'not loaded'},\n"
            f"  connections={'loaded' if self._connections is not None else 'not loaded'}\n"
            f")"
        )
