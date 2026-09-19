"""Skeletons and compartments. navis is imported on first use (about 1.5 s)."""

from connexplorer.morph.loader import SkeletonStore
from connexplorer.morph.segment import Compartments, segment

__all__ = ["Compartments", "SkeletonStore", "segment"]
