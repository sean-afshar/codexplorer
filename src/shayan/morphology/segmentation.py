"""
Segmentation strategies for dividing neurons into compartments.

Provides multiple approaches to compartmentalization for cable equation modeling.
Each strategy has different trade-offs between accuracy, computational cost,
and biophysical considerations.
"""

import navis
import numpy as np
import polars as pl
from typing import List, Iterable
from .compartmentalization import Compartmentalization


def segment_natural(neuron: navis.TreeNeuron) -> Compartmentalization:
    """
    Strategy 1: Use navis segments as-is (one compartment per segment).

    This is the simplest approach - each linear path between branch points
    becomes a single compartment. Respects the natural branch structure.

    Args:
        neuron: TreeNeuron to segment

    Returns:
        Compartmentalization with one compartment per natural segment

    Example:
        >>> from shayan.morphology.loader import load_skeleton
        >>> neuron = load_skeleton('data/...', [720575940627932541])[0]
        >>> comp = segment_natural(neuron)
        >>> print(f"Created {len(comp)} compartments")
    """
    compartment_data = {
        'compartment_id': [],
        'parent_id': [],
        'length': [],
        'radius_mean': [],
        'radius_std': []
    }
    node_mapping = {}
    strategy = 'natural'
    visited_nodes = set()
    
    for i, segment in enumerate(neuron.segments):
        # Store the current compartment ID
        compartment_data['compartment_id'].append(i)
        # Update the node mapping dictionary 
        frontier_nodes = [x for x in segment if x not in visited_nodes]
        node_mapping[i] = frontier_nodes
        visited_nodes.update(frontier_nodes)
        # Finding the parent segment - each segment has exactly one exit node
        parent_ids = neuron.nodes.loc[neuron.nodes['node_id'].isin(frontier_nodes), 'parent_id'].values        
        # Find parents that are out of the segment
        out_of_segment = parent_ids[~np.isin(parent_ids, frontier_nodes)][0]
        # Find the compartment ID of the parent node 
        parent_segment = _find_parent_segment(out_of_segment, neuron.segments)
        compartment_data['parent_id'].append(parent_segment)
        # Compute physical properties of the segment
        length, mean_radius, std_radius = _calculate_segment_properties(neuron, frontier_nodes)
        compartment_data['length'].append(length)
        compartment_data['radius_mean'].append(mean_radius)
        compartment_data['radius_std'].append(std_radius)
 
    # Convert compartment data into a dataframe
    compartments = pl.DataFrame(compartment_data)
        
    return Compartmentalization(neuron, compartments, node_mapping, strategy)


def segment_fixed_length(
    neuron: navis.TreeNeuron,
    target_length: float = 25.0
) -> Compartmentalization:
    """
    Strategy 2: Subdivide segments to achieve uniform compartment length.

    Args:
        neuron: TreeNeuron to segment
        target_length: Target compartment length in micrometers

    Returns:
        Compartmentalization with approximately uniform length compartments
    """
    # TODO: Implement after strategy 1
    raise NotImplementedError("Implement after strategy 1")


def segment_uniform_radius(
    neuron: navis.TreeNeuron,
    radius_tolerance: float = 0.2
) -> Compartmentalization:
    """
    Strategy 3: Group nodes with similar radius into compartments.

    Args:
        neuron: TreeNeuron to segment
        radius_tolerance: Maximum fractional change in radius within compartment

    Returns:
        Compartmentalization with uniform radius within each compartment
    """
    # TODO: Implement after strategies 1 and 2
    raise NotImplementedError("Implement after strategies 1 and 2")


def segment_electrotonic(
    neuron: navis.TreeNeuron,
    R_m: float = 10000.0,
    R_a: float = 100.0,
    max_lambda_fraction: float = 0.1
) -> Compartmentalization:
    """
    Strategy 4: Use electrotonic length (d_lambda rule) for segmentation.

    Args:
        neuron: TreeNeuron to segment
        R_m: Membrane resistivity (Ω·cm²)
        R_a: Axial resistivity (Ω·cm)
        max_lambda_fraction: Maximum compartment length as fraction of λ

    Returns:
        Compartmentalization sized for numerical accuracy
    """
    # TODO: Implement after strategies 1-3
    raise NotImplementedError("Implement after strategies 1-3")


# Helper functions

def _calculate_segment_properties(neuron: navis.TreeNeuron, node_ids: List[int]):
    """
    Calculate length, mean radius, and std radius for a segment.

    Args:
        neuron: TreeNeuron containing the nodes
        node_ids: Ordered list of node IDs in the segment

    Returns:
        Tuple of (length, radius_mean, radius_std) in micrometers
    """
    # Grab segment nodes
    segment_nodes = neuron.nodes[neuron.nodes['node_id'].isin(node_ids)]

    # Make sure the ordering is correct 
    segment_nodes = segment_nodes.set_index('node_id').loc[node_ids].reset_index()

    # Extract coordinates and radii 
    coords = segment_nodes[['x', 'y', 'z']].values
    radii = segment_nodes['radius'].values
    
    # Calculating Euclidean distance along segment
    if len(coords) < 2:
        length = 0.0
    else:
        length = np.linalg.norm(np.diff(coords, axis=0), axis=1).sum()
    
    return (length, radii.mean(), radii.std())


def _find_parent_segment(node_id: int, segments: List[List[int]]) -> int:
    """
    Find which segment contains a given node_id.

    Args:
        node_id: Node ID to search for
        segments: List of segments (each segment is a list of node IDs)

    Returns:
        Index of segment containing node_id, or -1 if not found
    """
    for i in range(len(segments)):
        if node_id in segments[i]:
            return i
        
    return -1


