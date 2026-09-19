"""connexplorer: fast, polars-native access to fly connectomes.

Phase 0 ships only the data-layer contract (:mod:`connexplorer.schema`).
Runtime objects (``open``, ``Dataset``, ``Neuron``, ...) arrive in phase 2;
see ``docs/merge_plan.md``.
"""

from connexplorer.schema import SCHEMA_VERSION, TABLES, Manifest

__version__ = "0.0.1"
__all__ = ["SCHEMA_VERSION", "TABLES", "Manifest", "__version__"]
