"""
Compartmentalization class for representing segmented neurons.

This module defines the core data structure for compartmental models,
storing compartment properties and connectivity for cable equation simulations.
"""

import navis
import numpy as np
import polars as pl
from typing import Dict, List
from scipy import sparse
from itertools import chain


class Compartmentalization:
    """
    Represents a neuron divided into compartments for cable equation modeling.

    This class stores the result of applying a segmentation strategy to a neuron,
    including compartment properties, connectivity, and the mapping back to the
    original skeleton nodes.

    Attributes:
        neuron: Reference to the original TreeNeuron
        compartments: DataFrame with compartment properties:
            - compartment_id (int): Unique compartment identifier (0 to N-1)
            - parent_id (int): ID of parent compartment (-1 for root)
            - length (float): Cable length in micrometers
            - radius_mean (float): Mean radius in micrometers
            - radius_std (float): Std deviation of radius (uniformity metric)
        node_mapping: Maps compartment_id → list of node_ids in that compartment
        strategy: Name of segmentation strategy used (e.g., 'natural', 'fixed_length')

    Example:
        >>> from shayan.morphology.loader import load_skeleton
        >>> from shayan.morphology.segmentation import segment_natural
        >>>
        >>> neuron = load_skeleton('data/...', [720575940627932541])[0]
        >>> comp = segment_natural(neuron)
        >>> print(comp.summary())
        >>> hines = comp.to_hines_matrix()
    """

    def __init__(
        self,
        neuron: navis.TreeNeuron,
        compartments: pl.DataFrame,
        node_mapping: Dict[int, List[int]],
        strategy: str
    ):
        """
        Initialize a Compartmentalization.

        Args:
            neuron: Original TreeNeuron that was segmented
            compartments: DataFrame with compartment properties
            node_mapping: Dictionary mapping compartment_id → list of node_ids
            strategy: Name of segmentation strategy
        """
        self.neuron = neuron
        self.compartments = compartments
        self.node_mapping = node_mapping
        self.strategy = strategy

        # Validate
        self._validate()

    def _validate(self):
        """
        Validate compartmentalization data structure.

        Checks:
            - All compartment_ids are unique and sequential (0 to N-1)
            - All node_ids in mapping exist in neuron
            - Each node belongs to exactly one compartment
            - Parent relationships are valid (no cycles, single root)

        Raises:
            ValueError: If validation fails
        """
        # 1. Check compartment_ids are unique and sequential (0 to N-1)
        n_comps = len(self.compartments)
        expected_ids = set(range(n_comps))
        actual_ids = set(self.compartments['compartment_id'].to_list())
        if expected_ids != actual_ids:
            raise ValueError("Compartment IDs must be unique and sequential from 0 to N-1.")

        # 2. Check all node_ids in mapping exist in neuron
        all_node_ids = set(chain.from_iterable(self.node_mapping.values()))
        neuron_node_ids = set(self.neuron.nodes['node_id'].to_list())
        if not all_node_ids.issubset(neuron_node_ids):
            raise ValueError('Invalid node IDs found in compartmentalization.')

        # 3. Check each node belongs to exactly one compartment
        total_nodes = sum(len(nodes) for nodes in self.node_mapping.values())
        if total_nodes != len(all_node_ids):
            raise ValueError('Duplicate node found in more than 1 compartment.')

        # 4. Check parent_id references are valid
        valid_parents = self.compartments.select(
            (
                pl.col('parent_id').eq(-1) |
                pl.col('parent_id').is_in(self.compartments['compartment_id'].to_list())
            ).all()
        ).item()
        if not valid_parents:
            raise ValueError("Parent IDs do not reference valid compartment IDs.")

        # 5. Check exactly one root
        n_roots = self.compartments['parent_id'].eq(-1).sum()
        if n_roots != 1:
            raise ValueError(f'Must have exactly 1 root, found {n_roots}.')
        

    def to_hines_matrix(
        self,
        R_a: float = 100.0,  # Ω·cm, axial resistivity
    ) -> sparse.spmatrix:
        """
        Convert compartmentalization to Hines matrix for cable equation.

        The Hines matrix represents the coupling between compartments via
        axial resistance. Entry (i,j) is the conductance between compartment
        i and j.

        Args:
            R_a: Axial resistivity in Ω·cm. Default is 100.0.

        Returns:
            Sparse matrix (N×N) where N is number of compartments.
            Matrix is symmetric with:
                - Off-diagonal (i,j): axial conductance between compartments
                - Diagonal (i,i): negative sum of row (ensures current conservation)

        Notes:
            Axial conductance between compartments i and j:
                g_axial = (π / (2 * R_a)) * (r_i² + r_j²) / (L_i + L_j)

            Where r_i, r_j are radii and L_i, L_j are lengths.

        Example:
            >>> comp = segment_natural(neuron)
            >>> hines = comp.to_hines_matrix(R_a=100.0)
            >>> print(f"Hines matrix shape: {hines.shape}")
            >>> print(f"Sparsity: {hines.nnz / (hines.shape[0]**2):.3f}")
        """
        # TODO: Implement Hines matrix construction
        # 1. Get number of compartments
        # 2. For each compartment, find parent from compartments['parent_id']
        # 3. Calculate axial conductance using radii and lengths
        # 4. Build sparse matrix (use scipy.sparse.lil_matrix for construction)
        # 5. Set off-diagonal entries for connected compartments
        # 6. Set diagonal entries (negative sum of row)
        # 7. Convert to CSR format for efficiency
        raise NotImplementedError("TODO: Implement to_hines_matrix")

    def summary(self) -> str:
        """
        Generate summary statistics of the compartmentalization.

        Returns:
            String with summary information including:
                - Segmentation strategy used
                - Number of compartments
                - Compartment length statistics (min, max, mean, median)
                - Radius uniformity statistics
                - Tree depth (max parent-child distance)

        Example:
            >>> comp = segment_natural(neuron)
            >>> print(comp.summary())
            Compartmentalization Summary
            ============================
            Strategy: natural
            Number of compartments: 175

            Length (μm):
              Min: 0.12
              Max: 24.55
              Mean: 3.08
              Median: 1.42

            Radius uniformity (std):
              Mean: 245.3 nm
              Max: 891.2 nm

            Tree depth: 5 levels
        """
        # TODO: Implement summary
        # 1. Extract statistics from self.compartments DataFrame
        # 2. Calculate tree depth by traversing parent_id relationships
        # 3. Format as readable string
        raise NotImplementedError("TODO: Implement summary")

    def plot(self, color_by: str = 'compartment', **kwargs):
        """
        Plot the compartmentalization in 3D.

        Args:
            color_by: How to color compartments:
                - 'compartment': Each compartment gets a different color
                - 'radius': Color by mean radius (heatmap)
                - 'length': Color by compartment length
            **kwargs: Additional arguments passed to visualization function

        Example:
            >>> comp.plot(color_by='radius')
            >>> plt.show()
        """
        # TODO: Import from visualization module and call
        # Will be implemented after visualization.py is complete
        raise NotImplementedError("Visualization not yet implemented")
        # return plot_compartmentalization(self, color_by=color_by, **kwargs)

    def __repr__(self) -> str:
        """String representation of Compartmentalization."""
        return (
            f"Compartmentalization(\n"
            f"  neuron_id={self.neuron.id},\n"
            f"  strategy='{self.strategy}',\n"
            f"  n_compartments={len(self.compartments)},\n"
            f"  n_nodes={self.neuron.n_nodes}\n"
            f")"
        )

    def __len__(self) -> int:
        """Return number of compartments."""
        return len(self.compartments)
