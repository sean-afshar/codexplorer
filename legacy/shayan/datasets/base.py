"""
Abstract base class for connectome dataset loaders.

This module defines the interface that all dataset loaders must implement.
Each dataset (Male CNS, FlyWire, etc.) will have its own loader class that
inherits from AbstractDataset and implements these methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import polars as pl


class AbstractDataset(ABC):
    """
    Abstract base class for connectome datasets.

    All dataset loaders must inherit from this class and implement
    the abstract methods. This ensures a consistent interface across
    different datasets.

    Attributes:
        data_dir: Path to the dataset directory
        name: Human-readable name of the dataset

    Example:
        >>> class MyCNSDataset(AbstractDataset):
        ...     def load_neurons(self):
        ...         # Implementation here
        ...         pass
    """

    def __init__(self, data_dir: str | Path, name: str):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to directory containing dataset files
            name: Name of the dataset (e.g., "Male CNS v0.9")
        """
        self.data_dir = Path(data_dir)
        self.name = name

        # Validate that directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', dir='{self.data_dir}')"

    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def load_neurons(self) -> pl.DataFrame:
        """
        Load neuron metadata in standardized format.

        Must return a DataFrame conforming to NEURON_SCHEMA with columns:
        - neuron_id: int64
        - type: str
        - side: str (optional)
        - region: str (optional)
        - nt_type: str (optional)

        Returns:
            DataFrame with neuron metadata

        Example:
            >>> dataset = MaleCNSDataset("data/male_cns")
            >>> neurons = dataset.load_neurons()
            >>> print(neurons.columns)
            ['neuron_id', 'type', 'side', 'region', 'nt_type']
        """
        pass

    @abstractmethod
    def load_connections(self) -> pl.DataFrame:
        """
        Load connection data (edge list) in standardized format.

        Must return a DataFrame conforming to CONNECTION_SCHEMA with columns:
        - pre_id: int64
        - post_id: int64
        - weight: int32
        - neuropil: str (optional)
        - nt_type: str (optional)

        Returns:
            DataFrame with connections (edge list)

        Example:
            >>> connections = dataset.load_connections()
            >>> print(connections.head())
            shape: (3, 3)
            ┌────────┬─────────┬────────┐
            │ pre_id ┆ post_id ┆ weight │
            │ ---    ┆ ---     ┆ ---    │
            │ i64    ┆ i64     ┆ i32    │
            ╞════════╪═════════╪════════╡
            │ 10352  ┆ 10351   ┆ 2591   │
            │ 13612  ┆ 16076   ┆ 2385   │
            │ 10663  ┆ 10051   ┆ 2248   │
            └────────┴─────────┴────────┘
        """
        pass

    @abstractmethod
    def load_synapses(self) -> Optional[pl.DataFrame]:
        """
        Load synapse-level data (optional) in standardized format.

        If the dataset provides individual synapse locations, return a DataFrame
        conforming to SYNAPSE_SCHEMA with columns:
        - pre_id: int64
        - post_id: int64
        - x, y, z: float32
        - neuropil: str (optional)
        - confidence: float32 (optional)
        - size: float32 (optional)

        If synapse-level data is not available, return None.

        Returns:
            DataFrame with synapse locations, or None if not available

        Note:
            Synapse tables can be VERY large (>300M rows). Consider lazy loading
            or implementing this as a separate method that loads on demand.
        """
        pass

    # =========================================================================
    # OPTIONAL METHODS - Can be overridden by subclasses
    # =========================================================================

    @property
    def has_synapses(self) -> bool:
        """
        Check if dataset provides synapse-level (physical location) data.

        Returns:
            True if synapse data is available
        """
        # Default implementation - subclasses can override
        return False

    @property
    def has_physical_synapses(self) -> bool:
        """Alias for has_synapses."""
        return self.has_synapses

    @property
    def has_neuropil(self) -> bool:
        """
        Check if connections carry a neuropil / brain-region column.

        Returns:
            True if the neuropil column is populated
        """
        # Default implementation - subclasses can override
        return False

    @property
    def organism(self) -> str:
        """
        Get the organism this dataset is from.

        Returns:
            Organism name (e.g., "Drosophila melanogaster")
        """
        # Default - subclasses should override
        return "Unknown"

    @property
    def sex(self) -> Optional[str]:
        """
        Get the sex of the specimen.

        Returns:
            "male", "female", or None if unknown
        """
        # Default - subclasses should override
        return None

    @property
    def version(self) -> Optional[str]:
        """
        Get the dataset version.

        Returns:
            Version string (e.g., "v0.9")
        """
        # Default - subclasses should override
        return None

    def get_metadata(self) -> dict:
        """
        Get dataset metadata as a dictionary.

        Returns:
            Dictionary with dataset information

        Example:
            >>> dataset.get_metadata()
            {
                'name': 'Male CNS v0.9',
                'organism': 'Drosophila melanogaster',
                'sex': 'male',
                'version': 'v0.9',
                'has_synapses': True,
                'data_dir': '/path/to/data'
            }
        """
        return {
            'name': self.name,
            'organism': self.organism,
            'sex': self.sex,
            'version': self.version,
            'has_synapses': self.has_synapses,
            'data_dir': str(self.data_dir),
        }
