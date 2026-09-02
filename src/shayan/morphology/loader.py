"""
Skeleton loader for SWC files.

Provides functions to load neuron skeletons from compressed SWC archives
and return navis TreeNeuron objects with proper unit conversion.
"""

import navis
from pathlib import Path
from typing import Union, Iterable


def load_skeleton(
    zip_path: Union[str, Path],
    neuron_ids: Iterable[int],
    convert_units: bool = False
) -> navis.NeuronList:
    """
    Load neuron skeletons from an SWC zip archive.

    Args:
        zip_path: Path to the zip file containing SWC files
        neuron_ids: Neuron IDs to load (corresponds to filenames without .swc extension)
        convert_units: If True, convert coordinates from nanometers to micrometers.
            Default is False.

    Returns:
        NeuronList containing loaded skeletons. Each TreeNeuron has:
            - nodes: DataFrame with columns [node_id, x, y, z, radius, parent_id, type]
            - segments: List of node ID sequences (linear paths between branches)
            - cable_length: Total cable length
            - n_nodes, n_branches, n_leafs: Morphology statistics

    Example:
        >>> neurons = load_skeleton('data/flywire_fafb/sk_lod1_783_healed.zip',
        ...                         [720575940627932541, 720575940604525792])
        >>> print(f"Loaded {len(neurons)} neurons")
        >>> print(f"First neuron has {neurons[0].n_nodes} nodes")
    """
    # Constructing a regex pattern for searching
    file_pattern =  '|'.join(rf"^{nid}\.swc$" for nid in neuron_ids)
    # Loading via navis 
    neurons = navis.read_swc(f=zip_path, limit=file_pattern)
    # Optionally rescaling distance
    if convert_units:
        return _convert_units_nm_to_um(neurons)
    return neurons


def load_skeletons_by_type(
    zip_path: Union[str, Path],
    connectome,
    cell_type: str,
    max_num: int = None,
    convert_units: bool = False
) -> navis.NeuronList:
    """
    Load all skeletons for a specific cell type.

    Args:
        zip_path: Path to the zip file containing SWC files
        connectome: Connectome object to query for neuron IDs of this cell type
        cell_type: Cell type name (e.g., 'L1', 'EPG', 'LC33a')
        max_num: Maximum number of neurons to load. If None, loads all neurons
            of this type. Default is None.
        convert_units: If True, convert from nanometers to micrometers.
            Default is False.

    Returns:
        NeuronList of TreeNeuron objects for this cell type

    Example:
        >>> from shayan.core.connectome import Connectome
        >>> from shayan.datasets.flywire import FlyWireDataset
        >>>
        >>> dataset = FlyWireDataset('data/flywire_fafb')
        >>> ctome = Connectome(dataset, mode='lazy')
        >>>
        >>> # Load first 5 L1 neurons
        >>> l1_neurons = load_skeletons_by_type(
        ...     'data/flywire_fafb/sk_lod1_783_healed.zip',
        ...     ctome,
        ...     'L1',
        ...     max_num=5
        ... )
        >>> print(f"Loaded {len(l1_neurons)} L1 neurons")
    """
    # Grabbing the neuron IDs 
    neuron_ids = connectome.get_cell_type_ids(cell_type)
    # Optional filtering 
    if max_num is not None:
        neuron_ids = neuron_ids[:max_num]
    return load_skeleton(zip_path=zip_path, neuron_ids=neuron_ids, convert_units=convert_units)


def _convert_units_nm_to_um(neurons: Union[navis.TreeNeuron, navis.NeuronList]) -> Union[navis.TreeNeuron, navis.NeuronList]:
    """
    Convert neuron coordinates and radius from nanometers to micrometers.

    Args:
        neurons: Neuron(s) with coordinates in nanometers

    Returns:
        Same neuron(s) with coordinates in micrometers (converted in-place)
    """
    return neurons.convert_units('microns')