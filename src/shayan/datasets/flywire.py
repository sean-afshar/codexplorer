"""
Dataset loader for the flywire fafb dataset.

This module defines the dataset loader for the FlyWire FAFB connectome dataset. 
The dataset itself contains synaptic connections, synapse metadata, neuron data, and neuron metadata.
The dataset is a subclass of the AbstractDataset base class, which defines the interface for all connectome datasets.
"""

from pathlib import Path
import polars as pl
from ..datasets.base import AbstractDataset

class FlyWireDataset(AbstractDataset):
    def __init__(self, data_dir: str | Path):
        """
        Initialize the FlyWire dataset loader.

        Args:
            data_dir: Path to the directory containing FlyWire dataset files.
        """
        super().__init__(data_dir, name='FLYWIRE_FAFB')
        self._has_neuropil = True
        self._has_synapses = True

    @property
    def has_synapses(self) -> bool:
        return True

    @property 
    def organism(self) -> str:
        return 'Drosophila melanogaster'
    
    @property 
    def sex(self) -> str:
        return 'female'
    
    @property 
    def version(self) -> str:
        return "FAFB v783 (downloaded October 2025)."

    @property 
    def has_neuropil(self) -> bool:
        return self._has_neuropil
    
    def load_neurons(self) -> pl.DataFrame:
        # Load neuron / neurotransmitter data
        neurons = pl.read_csv(self.data_dir / 'neurons.csv.gz')
        # Load cell type data 
        cell_types = pl.read_csv(self.data_dir / 'consolidated_cell_types.csv.gz')
        # Join the two datasets 
        result = neurons.join(cell_types, on='root_id', how='left')
        # Select the proper columns 
        result = result.select([
            pl.col('root_id').cast(pl.Int64).alias('neuron_id'),
            pl.col('primary_type').cast(pl.Utf8).alias('type'),
            pl.col('group').cast(pl.Utf8).alias('region'),
            pl.col('nt_type').cast(pl.Utf8),
            pl.col('nt_type_score').cast(pl.Float64),
            pl.lit(None).cast(pl.Utf8).alias('side')
        ])
        return result
    
    def load_connections(self) -> pl.DataFrame:
        # Load connection data
        connections = pl.read_csv(self.data_dir / 'connections_princeton_no_threshold.csv.gz')
        # Rename and cast to proper types
        result = connections.select([
            pl.col('pre_root_id').cast(pl.Int64).alias('pre_id'),
            pl.col('post_root_id').cast(pl.Int64).alias('post_id'),
            pl.col('syn_count').cast(pl.Int32).alias('weight'),
            pl.col('neuropil').cast(pl.Utf8),
            pl.col('nt_type').cast(pl.Utf8)
        ])
        return result
    
    def load_synapses(self) -> pl.DataFrame:
        synapse_table = pl.read_csv(self.data_dir / 'fafb_v783_princeton_synapse_table.csv.gz')
        # Grabbing neuron ID prefixes from the header
        pre_root_col = [x for x in synapse_table.columns if x.startswith('pre_root_id')][0]
        post_root_col = [x for x in synapse_table.columns if x.startswith('post_root_id')][0]
        pre_header = "".join([x for x in pre_root_col if x.isnumeric()])
        post_header = "".join([x for x in post_root_col if x.isnumeric()])
        # Using the center of the synapses as synapse locations
        result = synapse_table.select([
            pl.concat_str([pl.lit(pre_header), pl.col(pre_root_col)]).cast(pl.Int64).alias('pre_id'),
            pl.concat_str([pl.lit(post_header), pl.col(post_root_col)]).cast(pl.Int64).alias('post_id'),
            pl.col('ctr_x').cast(pl.Int64).alias('x'),
            pl.col('ctr_y').cast(pl.Int64).alias('y'),
            pl.col('ctr_z').cast(pl.Int64).alias('z'),
            pl.col('neuropil').cast(pl.Utf8)
        ])
        return result