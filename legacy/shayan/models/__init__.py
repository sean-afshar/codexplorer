"""Biophysical models built on top of a Compartmentalization."""

from .compartmental_model import CompartmentalModel, scan_parameter_space

__all__ = ["CompartmentalModel", "scan_parameter_space"]
