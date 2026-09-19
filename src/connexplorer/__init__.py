"""connexplorer: fast, polars-native access to fly connectomes.

    import connexplorer as cnx
    ds = cnx.open("flywire")
    n = ds[720575940599755718]
    n.outputs(min_syn=5)
    ds.connectivity["T4a", "LPi14"].values
    n.synapses(direction="in")
"""

from connexplorer.cross import Comparison, compare
from connexplorer.dataset import Dataset, config, open, resolve
from connexplorer.neurons import Neuron, NeuronSet
from connexplorer.schema import SCHEMA_VERSION, TABLES, Manifest
from connexplorer.synapses import xyz

__version__ = "0.1.0"
__all__ = ["Comparison", "Dataset", "Manifest", "Neuron", "NeuronSet", "SCHEMA_VERSION", "TABLES", "compare", "config", "open", "resolve", "xyz", "__version__"]
