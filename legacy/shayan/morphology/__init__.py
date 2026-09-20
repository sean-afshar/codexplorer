"""
Morphology module for loading and analyzing neuron skeletons.

This module provides tools for:
- Loading SWC skeleton files from zip archives (navis TreeNeurons)
- Segmenting neurons into compartments
- Visualizing compartmentalizations (plotly, matplotlib, neuroglancer URLs)
"""

from .loader import load_skeleton, load_skeletons_by_type
from .compartmentalization import Compartmentalization
from .segmentation import (
    segment_natural, segment_fixed_length, segment_uniform_radius, segment_electrotonic,
)
from .visualization import (
    plot_compartmentalization, plot_compartmentalization_2d, plot_compartments_subset,
    neuron_to_neuroglancer_url, to_neuroglancer_url,
)

__all__ = [
    "load_skeleton", "load_skeletons_by_type",
    "Compartmentalization",
    "segment_natural", "segment_fixed_length", "segment_uniform_radius", "segment_electrotonic",
    "plot_compartmentalization", "plot_compartmentalization_2d", "plot_compartments_subset",
    "neuron_to_neuroglancer_url", "to_neuroglancer_url",
]
