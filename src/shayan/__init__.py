"""
shayan: Polars-based tools for querying fly connectomes, loading skeletons,
compartmentalizing neurons, and visualizing the results.

Originally developed as the ``connect_tome`` package; condensed here for
merging with a colleague's connectome code.

Quick start::

    from shayan import Connectome, FlyWireDataset
    ctome = Connectome(FlyWireDataset("data/flywire_fafb"), mode="lazy")
    circuit = ctome.query().pre_type("T4a").min_weight(5).as_circuit()
    print(circuit.aggregate_by_type_normalized().head())

The morphology and modeling helpers depend on navis / plotly / scipy, which
are slow to import, so they are loaded lazily on first attribute access.
"""

__version__ = "0.1.0"

from .core import Connectome, ConnectionQuery, Circuit
from .datasets import AbstractDataset, FlyWireDataset

# Heavy (navis / plotly / scipy) names resolved lazily on first access.
_LAZY_IMPORTS = {
    "load_skeleton": ".morphology.loader",
    "load_skeletons_by_type": ".morphology.loader",
    "Compartmentalization": ".morphology.compartmentalization",
    "segment_natural": ".morphology.segmentation",
    "plot_compartmentalization": ".morphology.visualization",
    "plot_compartmentalization_2d": ".morphology.visualization",
    "plot_compartments_subset": ".morphology.visualization",
    "neuron_to_neuroglancer_url": ".morphology.visualization",
    "to_neuroglancer_url": ".morphology.visualization",
    "CompartmentalModel": ".models.compartmental_model",
    "scan_parameter_space": ".models.compartmental_model",
}

__all__ = [
    "Connectome", "ConnectionQuery", "Circuit",
    "AbstractDataset", "FlyWireDataset",
    *_LAZY_IMPORTS,
]


def __getattr__(name: str):
    module = _LAZY_IMPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'shayan' has no attribute '{name}'")
    from importlib import import_module
    return getattr(import_module(module, __name__), name)
