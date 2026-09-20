"""
Standardized schemas for connectome data.

This module defines the canonical data format that all datasets must conform to.
Different connectome datasets (Male CNS, FlyWire, etc.) have different column names
and data types. This schema provides a unified interface.

Example:
    Male CNS uses 'bodyId', FlyWire uses 'root_id' -> both become 'neuron_id'
"""

from typing import Dict, Any
import polars as pl


# =============================================================================
# NEURON SCHEMA
# =============================================================================

NEURON_SCHEMA: Dict[str, Any] = {
    # Core identifiers
    'neuron_id': pl.Int64,      # Universal neuron ID (bodyId, root_id, etc.)

    # Classification
    'type': pl.Utf8,            # Primary cell type (e.g., "T4a", "LC4")
    'side': pl.Utf8,            # Laterality: "left", "right", "center", or None
    'region': pl.Utf8,          # Primary brain region/neuropil

    # Neurotransmitter
    'nt_type': pl.Utf8,         # Neurotransmitter type (ACh, GABA, Glut, etc.)

    # Additional metadata stored as nested struct
    # Different datasets have different extra fields, so we keep them separate
    # This could be extended in the future to store dataset-specific info
}

# Required columns that MUST be present
NEURON_REQUIRED_COLS = ['neuron_id', 'type']

# Optional columns that may be None
NEURON_OPTIONAL_COLS = ['side', 'region', 'nt_type']


# =============================================================================
# CONNECTION SCHEMA
# =============================================================================

CONNECTION_SCHEMA: Dict[str, Any] = {
    # Edge definition
    'pre_id': pl.Int64,         # Presynaptic neuron ID
    'post_id': pl.Int64,        # Postsynaptic neuron ID

    # Edge weight
    'weight': pl.Int32,         # Synapse count (number of connections)

    # Optional edge attributes
    'neuropil': pl.Utf8,        # Brain region where connection occurs
    'nt_type': pl.Utf8,         # Neurotransmitter type for this connection
}

# Required columns
CONNECTION_REQUIRED_COLS = ['pre_id', 'post_id', 'weight']

# Optional columns
CONNECTION_OPTIONAL_COLS = ['neuropil', 'nt_type']


# =============================================================================
# SYNAPSE SCHEMA (for spatial queries)
# =============================================================================

SYNAPSE_SCHEMA: Dict[str, Any] = {
    # Connection
    'pre_id': pl.Int64,         # Presynaptic neuron ID
    'post_id': pl.Int64,        # Postsynaptic neuron ID

    # Spatial coordinates
    'x': pl.Int64,            # X coordinate
    'y': pl.Int64,            # Y coordinate
    'z': pl.Int64,            # Z coordinate

    # Optional attributes
    'neuropil': pl.Utf8,        # Brain region
    'confidence': pl.Float64,   # Confidence score (if available)
    'size': pl.Int64,         # Synapse size in voxels (if available)
}

# Required columns
SYNAPSE_REQUIRED_COLS = ['pre_id', 'post_id', 'x', 'y', 'z']

# Optional columns
SYNAPSE_OPTIONAL_COLS = ['neuropil', 'confidence', 'size']


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_schema(df: pl.DataFrame, schema: Dict[str, Any],
                    required_cols: list[str], name: str = "DataFrame") -> None:
    """
    Validate that a DataFrame conforms to the expected schema.

    Args:
        df: DataFrame to validate
        schema: Expected schema (dict of column names to types)
        required_cols: List of required column names
        name: Name of the DataFrame (for error messages)

    Raises:
        ValueError: If schema validation fails

    Example:
        >>> validate_schema(neurons_df, NEURON_SCHEMA, NEURON_REQUIRED_COLS, "neurons")
    """
    # Check required columns are present
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{name} is missing required columns: {missing_cols}"
        )

    # Check data types for present columns
    for col in df.columns:
        if col in schema:
            expected_type = schema[col]
            actual_type = df[col].dtype

            # Polars type checking
            if actual_type != expected_type:
                raise ValueError(
                    f"{name} column '{col}' has type {actual_type}, "
                    f"expected {expected_type}"
                )


def ensure_schema(df: pl.DataFrame, schema: Dict[str, Any],
                 optional_cols: list[str]) -> pl.DataFrame:
    """
    Ensure DataFrame has all schema columns, adding None for missing optional ones.

    Args:
        df: DataFrame to process
        schema: Expected schema
        optional_cols: List of optional column names

    Returns:
        DataFrame with all schema columns present

    Example:
        >>> neurons_df = ensure_schema(neurons_df, NEURON_SCHEMA, NEURON_OPTIONAL_COLS)
    """
    for col in optional_cols:
        if col not in df.columns:
            # Add missing optional column with None values
            df = df.with_columns(pl.lit(None).cast(schema[col]).alias(col))

    return df
