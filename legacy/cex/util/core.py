"""Small array, lookup, and color helpers used by the normalized-table core."""

from __future__ import annotations

import colorsys

import numpy as np


def bin_data_to_idx_list(data, return_type='list'):
    data = np.asarray(data)
    sort_idx = np.argsort(data, kind='stable')
    unique_val, first_ind, counts = np.unique(data[sort_idx], return_index=True, return_counts=True)
    # Initialize first so equal-length groups remain a one-dimensional object array.
    bin_idx_list = np.empty(first_ind.size, dtype=object)
    for n, i0 in enumerate(first_ind):
        c = counts[n]
        tmp_idx = sort_idx[i0:(i0 + c)]
        bin_idx_list[n] = np.asarray(tmp_idx)
    if return_type == 'list':
        return bin_idx_list, unique_val
    elif return_type == 'dict':
        return {k: v for k, v in zip(unique_val, bin_idx_list)}
    else:
        raise ValueError(f"Unrecognized return_type. Options: 'list', 'dict'")


def select_val_by_num_repeat(val, num_repeat=1):
    """Select root ids by the number of repeats.

    Input:
        rid_list: list or array of root ids
        min_syn_per_rid: int, minimum number of synapses per root id
    Output:
        selected_Q: logical array, 1 for selected
    """
    if num_repeat <= 1:
        return np.ones_like(val, dtype=bool)
    else:
        id2idx = bin_data_to_idx_list(val, return_type='dict')
        selected_Q = np.zeros_like(val, dtype=bool)
        for k, v in id2idx.items():
            if v.size >= num_repeat:
                selected_Q[v] = True
        return selected_Q


def minimize_int_dtype(arr: np.ndarray, *, allow_bool: bool = True,
                       copy: bool = False, signed: bool = False) -> np.ndarray:
    """Return an integer array cast to the smallest exact NumPy dtype."""
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a NumPy ndarray")
    if arr.size == 0:
        return arr
    if not (np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_):
        raise TypeError(f"arr must have an integer dtype, got {arr.dtype!r}")

    if allow_bool:
        mn = int(arr.min())
        mx = int(arr.max())
        if mn >= 0 and mx <= 1:
            return arr.astype(np.bool_, copy=copy)

    mn = int(arr.min())
    mx = int(arr.max())
    signed_chain = [np.int8, np.int16, np.int32, np.int64]
    unsigned_chain = [np.uint8, np.uint16, np.uint32, np.uint64]

    def fits(dtype):
        info = np.iinfo(dtype)
        return info.min <= mn and mx <= info.max

    if mn >= 0 and (not signed):
        for dt in unsigned_chain:
            if fits(dt):
                return arr.astype(dt, copy=copy)
        return arr
    else:
        for dt in signed_chain:
            if fits(dt):
                return arr.astype(dt, copy=copy)
        return arr


def reorder_pd_columns(df, leading_cols=None, trailing_cols=None):
    cols = df.columns.tolist()
    if leading_cols is not None:
        for c in reversed(leading_cols):
            cols.remove(c)
            cols.insert(0, c)
    if trailing_cols is not None:
        for c in trailing_cols:
            cols.remove(c)
            cols.append(c)
    return df[cols]


class ScalarDict:
    def __init__(self, key, value=None):
        idx = np.argsort(key, kind='stable')
        self.key = key[idx]
        if value is None:
            self.value = np.arange(key.size)
        else:
            assert np.all(value >= 0), "value should be non-negative integers"
            assert key.shape == value.shape, "ind and label should have the same shape"
            self.value = value[idx]

    def get_value(self, keys, not_found_val=-1):
        keys = np.asarray(keys).astype(dtype=self.key.dtype, copy=False)
        idx = np.searchsorted(self.key, keys, side='left')
        in_range_Q = idx < self.key.size
        found_Q = np.zeros(keys.shape, dtype=bool)
        found_Q[in_range_Q] = self.key[idx[in_range_Q]] == keys[in_range_Q]
        result = np.full(keys.shape, not_found_val, dtype=self.value.dtype)
        result[found_Q] = self.value[idx[found_Q]]
        return result

    def get_label(self, query_idx):
        query_idx = np.asarray(query_idx)
        assert np.all((query_idx >= 0) & (query_idx < self.value.size)), "query_idx out of range"
        return self.key[query_idx]

    def __call__(self, rid):
        return self.get_value(rid)


def distinct_hex_colors(num_colors: int) -> list[str]:
    """Return evenly spaced, high-contrast hexadecimal colors."""
    return ["#{:02X}{:02X}{:02X}".format(*(round(255 * value) for value in colorsys.hsv_to_rgb(i / max(num_colors, 1), 0.7, 0.95))) for i in range(num_colors)]
