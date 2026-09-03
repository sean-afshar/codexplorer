"""Cached IO-stat loading for dataset facades."""

import os
import warnings

import numpy as np
import pandas as pd


class IOStatMixin:
    """Load and cache precomputed per-cell-type IO statistics."""

    def fp_cell_type_stat(self, cell_type: str, stat_type: str, dir: str):
        """Return path to a precomputed cell-type IO-stat CSV."""
        return self.paths.fp_cell_type_stat_file(cell_type, stat_type=stat_type, dir=dir)

    def load_cell_type_io_stat(
        self,
        cell_type: str,
        stat_type: str = "basic_pc",
        dir=("input", "output"),
        use_cache_Q: bool = True,
    ):
        """Load cached single-cell type I/O statistics."""
        dir_tuple = (dir,) if isinstance(dir, str) else tuple(dir)
        cache_key = (cell_type, stat_type, dir_tuple)
        if use_cache_Q and cache_key in self._io_stat_cache:
            return self._io_stat_cache[cache_key]

        if cell_type not in self.cell.cell_count and self.get_cell_type_id(cell_type).size == 0:
            warnings.warn(f"{cell_type} not in cell type list.")
            return None

        stat = {}
        for d in dir_tuple:
            fp = self.fp_cell_type_stat(cell_type, stat_type, d)
            if not os.path.exists(fp):
                warnings.warn(f"{fp} does not exist.")
                continue
            table = pd.read_csv(fp)
            if table.empty:
                warnings.warn(f"{fp} exists but the table is empty.")
                return None
            # remove nan
            table = table[~table["type"].isna()]
            stat[d] = _round_io_stat_table(table)

        if not stat:
            return None
        result = stat[dir_tuple[0]] if len(stat) == 1 and dir_tuple[0] in stat else stat
        if use_cache_Q:
            self._io_stat_cache[cache_key] = result
        return result

    def clear_io_stat_cache(self) -> None:
        """Clear cached IO-stat tables."""
        self._io_stat_cache.clear()

    def show_cell_type_io_stat(
        self,
        cell_type: str,
        dir: str,
        num_rows: int = 10,
        stat_type: str = "basic_pc",
        vis_keys=("type", "f_c", "n_c", "en_c", "N_s", "F_s", "n_s", "n_spc", "n_c_spc", "NT"),
    ):
        """Return a display-ready IO-stat table."""
        table = self.load_cell_type_io_stat(cell_type, dir=dir, stat_type=stat_type)
        if table is None:
            return None
        if hasattr(self, "cell"):
            nt = self.cell.type_to_neurotransmitter.get(cell_type, "unknown")
            num_cell = self.cell.cell_count.get(cell_type, self.get_cell_type_id(cell_type).size)
            print(
                f"{cell_type} ({nt}, "
                f"# {num_cell}) {dir} ({stat_type})"
            )
        return self._pretty_print_stat_table(table, num_rows=num_rows, vis_keys=vis_keys)

    @staticmethod
    def _pretty_print_stat_table(table, num_rows: int = 10, vis_keys=None):
        """Return selected IO-stat columns."""
        vis_keys = list(vis_keys or table.columns)
        existing = [k for k in vis_keys if k in table.columns]
        return table.loc[:, existing].head(num_rows)


def _round_io_stat_table(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    for key in table.columns:
        if key not in {"type", "NT"} and pd.api.types.is_numeric_dtype(table[key]):
            if key in ["N_s"]:
                table[key] = table[key].astype(int)
            else:
                table[key] = np.round(table[key], 3)
    return table
