"""
Tests for FlyWire dataset loader.

Integration tests use neuron 720575940599755718 (T4a) as ground truth with known:
- 18 output connections (155 synapses total, min_weight >= 5)
- 12 input connections (167 synapses total, min_weight >= 5)
- Outputs to 13 different cell types, primarily LPi14 (35 synapses)
"""
import os
import pytest
import polars as pl
from pathlib import Path
from shayan.datasets.flywire import FlyWireDataset
from shayan.datasets.schema import validate_schema, NEURON_SCHEMA, CONNECTION_SCHEMA, SYNAPSE_SCHEMA
from shayan.datasets.schema import NEURON_REQUIRED_COLS, SYNAPSE_REQUIRED_COLS, CONNECTION_REQUIRED_COLS
from shayan.core.connectome import Connectome
from shayan.core.query import ConnectionQuery

# =============================================================================
# TEST CONSTANTS - Ground Truth Data
# =============================================================================

TEST_NEURON_ID = 720575940599755718
TEST_NEURON_TYPE = "T4a"
MIN_WEIGHT = 5

# Expected connectivity (with min_weight >= 5)
EXPECTED_N_OUTPUTS = 18
EXPECTED_N_INPUTS = 12
EXPECTED_TOTAL_OUTPUT_SYNAPSES = 155
EXPECTED_TOTAL_INPUT_SYNAPSES = 167

# Strongest connections
STRONGEST_OUTPUT_TARGET = 720575940632504874
STRONGEST_OUTPUT_WEIGHT = 30
STRONGEST_INPUT_SOURCE = 720575940643266583
STRONGEST_INPUT_WEIGHT = 29

# =============================================================================
# FIXTURES
# =============================================================================

# Point FLYWIRE_DATA_DIR at a directory with the FlyWire FAFB v783 files
# (see src/shayan/README.md). Tests are skipped when it is absent.
DATA_DIR = Path(os.environ.get("FLYWIRE_DATA_DIR", "data/flywire_fafb"))

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason=f"FlyWire data directory not found: {DATA_DIR} (set FLYWIRE_DATA_DIR)",
)

@pytest.fixture
def dataset():
    """Create a FlyWire dataset instance for testing."""
    return FlyWireDataset(DATA_DIR)

@pytest.fixture
def connectome(dataset):
    """Create a Connectome instance for integration tests."""
    return Connectome(dataset, mode="eager")

# =============================================================================
# BASIC LOADER TESTS
# =============================================================================

def test_dataset_init(dataset):
    """Test that dataset initializes correctly."""
    assert dataset.name == "FLYWIRE_FAFB"
    assert dataset.organism == "Drosophila melanogaster"
    assert dataset.sex == "female"
    assert dataset.has_neuropil == True
    assert dataset.has_physical_synapses == True

def test_load_neurons(dataset):
    """Test loading neuron data."""
    neurons = dataset.load_neurons()

    # Check it's a DataFrame
    assert isinstance(neurons, pl.DataFrame)

    # Check not empty
    assert len(neurons) > 0

    # Check that it matches the schema
    validate_schema(neurons, NEURON_SCHEMA, NEURON_REQUIRED_COLS)

    # Check extra FlyWire-specific column
    assert "nt_type_score" in neurons.columns

    # Print summary for inspection
    print(f"\nLoaded {len(neurons)} neurons")
    print(neurons.head())

def test_load_connections(dataset):
    """Test loading connection data."""
    connections = dataset.load_connections()

    # Check it's a DataFrame
    assert isinstance(connections, pl.DataFrame)

    # Check that it matches the schema
    validate_schema(connections, CONNECTION_SCHEMA, CONNECTION_REQUIRED_COLS)

    # Check weights are positive
    assert connections["weight"].min() > 0

    # Print summary for inspection
    print(f"\nLoaded {len(connections)} connections")
    print(connections.head())

@pytest.mark.slow
def test_load_synapses(dataset):
    """Test loading synapse-level data (marked slow - 2.5GB file)."""
    synapses = dataset.load_synapses()

    # Check it's a DataFrame
    assert isinstance(synapses, pl.DataFrame)

    # Check that it matches the schema
    validate_schema(synapses, SYNAPSE_SCHEMA, SYNAPSE_REQUIRED_COLS)

    # Print summary for inspection
    print(f"\nLoaded {len(synapses)} synapses")
    print(synapses.head())

# =============================================================================
# QUERY METHOD TESTS - Test Each Filter Type
# =============================================================================

def test_query_pre_id(connectome):
    """Test filtering by presynaptic neuron ID."""
    circuit = connectome.query().pre_id(TEST_NEURON_ID).min_weight(MIN_WEIGHT).as_circuit()

    # All connections should have our neuron as pre
    assert (circuit.connections["pre_id"] == TEST_NEURON_ID).all()

    # Should match expected number
    assert len(circuit.connections) == EXPECTED_N_OUTPUTS

def test_query_post_id(connectome):
    """Test filtering by postsynaptic neuron ID."""
    circuit = connectome.query().post_id(TEST_NEURON_ID).min_weight(MIN_WEIGHT).as_circuit()

    # All connections should have our neuron as post
    assert (circuit.connections["post_id"] == TEST_NEURON_ID).all()

    # Should match expected number
    assert len(circuit.connections) == EXPECTED_N_INPUTS

def test_query_pre_type(connectome):
    """Test filtering by presynaptic cell type."""
    circuit = connectome.query().pre_type(TEST_NEURON_TYPE).as_circuit()

    # Check that we got connections
    assert len(circuit.connections) > 0

    # Verify all pre neurons are T4a
    pre_types = circuit.neurons.filter(
        pl.col("neuron_id").is_in(circuit.connections["pre_id"])
    )["type"].unique()

    assert TEST_NEURON_TYPE in pre_types

def test_query_post_type(connectome):
    """Test filtering by postsynaptic cell type."""
    # Query for T4a -> LPi14 connections (we know these exist)
    circuit = (connectome.query()
        .pre_type(TEST_NEURON_TYPE)
        .post_type("LPi14")
        .min_weight(MIN_WEIGHT)
        .as_circuit()
    )

    # Should have connections
    assert len(circuit.connections) > 0

def test_query_neuropil(connectome):
    """Test filtering by neuropil."""
    circuit = connectome.query().neuropil("LOP_R").min_weight(MIN_WEIGHT).as_circuit()

    # Check all connections are in LOP_R
    assert (circuit.connections["neuropil"] == "LOP_R").all()

    # Should have connections
    assert len(circuit.connections) > 0

def test_query_min_weight(connectome):
    """Test filtering by minimum weight."""
    threshold = 10
    circuit = connectome.query().min_weight(threshold).as_circuit()

    # All weights should be >= threshold
    assert (circuit.connections["weight"] >= threshold).all()

def test_query_max_weight(connectome):
    """Test filtering by maximum weight."""
    threshold = 20
    circuit = connectome.query().max_weight(threshold).as_circuit()

    # All weights should be <= threshold
    assert (circuit.connections["weight"] <= threshold).all()

def test_query_weight_range(connectome):
    """Test filtering by weight range."""
    min_w, max_w = 10, 20
    circuit = connectome.query().weight_range(min_w, max_w).as_circuit()

    # All weights should be in range
    assert (circuit.connections["weight"] >= min_w).all()
    assert (circuit.connections["weight"] <= max_w).all()

def test_query_nt_type(connectome):
    """Test filtering by neurotransmitter type."""
    circuit = connectome.query().nt_type("ACH").min_weight(MIN_WEIGHT).as_circuit()

    # All connections should be ACH
    assert (circuit.connections["nt_type"] == "ACH").all()

    # Should have connections
    assert len(circuit.connections) > 0

def test_query_chaining(connectome):
    """Test chaining multiple filters together."""
    circuit = (connectome.query()
        .pre_id(TEST_NEURON_ID)
        .min_weight(MIN_WEIGHT)
        .neuropil("LOP_R")
        .nt_type("ACH")
        .as_circuit()
    )

    # Verify all filters applied
    assert (circuit.connections["pre_id"] == TEST_NEURON_ID).all()
    assert (circuit.connections["weight"] >= MIN_WEIGHT).all()
    assert (circuit.connections["neuropil"] == "LOP_R").all()
    assert (circuit.connections["nt_type"] == "ACH").all()

def test_query_count(connectome):
    """Test count() method (without loading data)."""
    count = connectome.query().pre_id(TEST_NEURON_ID).min_weight(MIN_WEIGHT).count()

    assert count == EXPECTED_N_OUTPUTS

def test_query_collect(connectome):
    """Test collect() method (returns raw DataFrame)."""
    df = connectome.query().pre_id(TEST_NEURON_ID).min_weight(MIN_WEIGHT).collect()

    assert isinstance(df, pl.DataFrame)
    assert len(df) == EXPECTED_N_OUTPUTS

# =============================================================================
# CIRCUIT ANALYSIS TESTS
# =============================================================================

def test_circuit_summary(connectome):
    """Test Circuit.summary() method."""
    circuit = connectome.query().pre_id(TEST_NEURON_ID).min_weight(MIN_WEIGHT).as_circuit()

    summary = circuit.summary()

    assert summary["n_connections"] == EXPECTED_N_OUTPUTS
    assert summary["total_synapses"] == EXPECTED_TOTAL_OUTPUT_SYNAPSES
    assert summary["n_neurons"] > 0

def test_circuit_node_stats(connectome):
    """Test Circuit.node_stats() method."""
    circuit = connectome.query().pre_id(TEST_NEURON_ID).min_weight(MIN_WEIGHT).as_circuit()

    stats = circuit.node_stats()

    # Find our test neuron's stats
    test_stats = stats.filter(pl.col("neuron_id") == TEST_NEURON_ID)

    assert len(test_stats) == 1
    assert test_stats["out_degree"][0] == EXPECTED_N_OUTPUTS
    assert test_stats["out_weight"][0] == EXPECTED_TOTAL_OUTPUT_SYNAPSES

def test_circuit_aggregate_by_type(connectome):
    """Test Circuit.aggregate_by_type() method."""
    circuit = connectome.query().pre_id(TEST_NEURON_ID).min_weight(MIN_WEIGHT).as_circuit()

    type_agg = circuit.aggregate_by_type()

    # Should have multiple post types
    assert len(type_agg) > 0

    # Total weight should sum to our expected total
    assert type_agg["weight"].sum() == EXPECTED_TOTAL_OUTPUT_SYNAPSES

# =============================================================================
# NORMALIZATION TESTS - Ground Truth Validation
# =============================================================================

def test_normalization_per_connection(connectome):
    """Test per-connection normalization with known values."""
    circuit = connectome.query().pre_id(TEST_NEURON_ID).min_weight(MIN_WEIGHT).as_circuit()

    # Add normalized weights
    norm_conn = circuit.add_normalized_weights()

    # Find the strongest output connection (30 synapses)
    strongest = norm_conn.filter(pl.col("weight") == STRONGEST_OUTPUT_WEIGHT)

    assert len(strongest) > 0

    # Calculate expected percent_output
    # percent_output = (30 / 155) * 100 = 19.35%
    expected_percent_output = (STRONGEST_OUTPUT_WEIGHT / EXPECTED_TOTAL_OUTPUT_SYNAPSES) * 100

    actual_percent_output = strongest["percent_output"][0]

    # Assert close (within 0.01%)
    assert abs(actual_percent_output - expected_percent_output) < 0.01

    print(f"\nNormalization check:")
    print(f"Expected: {expected_percent_output:.2f}%")
    print(f"Actual: {actual_percent_output:.2f}%")

def test_normalization_type_level(connectome):
    """Test type-level normalization."""
    # Get T4a outputs with min weight
    circuit = (connectome.query()
        .pre_type(TEST_NEURON_TYPE)
        .min_weight(MIN_WEIGHT)
        .as_circuit()
    )

    # Aggregate by type with normalization
    type_norm = circuit.aggregate_by_type_normalized()

    # Should have results
    assert len(type_norm) > 0

    # All percentages should be between 0 and 100
    assert (type_norm["percent_output"] >= 0).all()
    assert (type_norm["percent_output"] <= 100).all()
    assert (type_norm["percent_input"] >= 0).all()
    assert (type_norm["percent_input"] <= 100).all()

    # Normalized weight should be geometric mean
    expected_norm_weight = (
        type_norm["percent_output"] * type_norm["percent_input"]
    ).sqrt()

    # Check they match (within tolerance for floating point)
    assert ((type_norm["normalized_weight"] - expected_norm_weight).abs() < 0.01).all()

def test_normalization_uses_global_totals(connectome):
    """Test that normalization uses global totals, not circuit-local."""
    # Create a small circuit
    small_circuit = (connectome.query()
        .pre_id(TEST_NEURON_ID)
        .min_weight(MIN_WEIGHT)
        .as_circuit()
    )

    # Get normalized connections
    norm_conn = small_circuit.add_normalized_weights()

    # The total_output column should be the GLOBAL total (from full dataset)
    # NOT the circuit-local total
    test_neuron_row = norm_conn.filter(pl.col("pre_id") == TEST_NEURON_ID).head(1)

    global_total = test_neuron_row["total_output"][0]
    circuit_total = small_circuit.connections["weight"].sum()

    # Global should be >= circuit total (could be equal if all connections included)
    assert global_total >= circuit_total

    print(f"\nGlobal total: {global_total}")
    print(f"Circuit total: {circuit_total}")

# =============================================================================
# EMPTY QUERY TESTS
# =============================================================================

def test_empty_query(connectome):
    """Test that impossible queries return empty circuits gracefully."""
    # Query for non-existent neuron
    circuit = connectome.query().pre_id(999999999).as_circuit()

    assert len(circuit.connections) == 0
    assert len(circuit.neurons) == 0

def test_empty_circuit_operations(connectome):
    """Test that operations on empty circuits don't crash."""
    circuit = connectome.query().pre_id(999999999).as_circuit()

    # These should all work without crashing
    summary = circuit.summary()
    assert summary["n_connections"] == 0

    stats = circuit.node_stats()
    assert len(stats) == 0

    type_agg = circuit.aggregate_by_type()
    assert len(type_agg) == 0
