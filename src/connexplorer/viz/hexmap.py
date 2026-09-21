"""Hexagonal column maps of the optic lobe.

Columns are addressed by the axial hex coordinates ``(p, q)`` of
``Dataset.columns``. On this lattice the six neighbours of ``(p, q)`` are
``(p±1, q)``, ``(p, q±1)`` and ``(p±1, q±1)`` (same sign), verified on FlyWire
by nearest-neighbour Mi1 positions. Columns are drawn as flat-topped hexagons
with centres

    x = 1.5 * size * (p - q),    y = sqrt(3)/2 * size * (p + q)

so that all six neighbours sit ``sqrt(3) * size`` away.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# Axial (p, q) -> Cartesian (x, y) for unit side length; rows are the p and q basis vectors.
HEX_BASIS = np.array([[1.5, np.sqrt(3) / 2], [-1.5, np.sqrt(3) / 2]])
HEX_NEIGHBORS: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))


def pq_to_xy(pq, size: float = 1.0, mirror: bool = False) -> np.ndarray:
    """Axial hex coordinates ``(..., 2)`` to Cartesian centres ``(..., 2)``.

    ``mirror=True`` flips x, e.g. to draw the left eye as seen from the same viewpoint.
    """
    xy = np.asarray(pq, dtype=float) @ HEX_BASIS * size
    if mirror:
        xy = xy * np.array([-1.0, 1.0])
    return xy


def xy_to_pq(xy, size: float = 1.0, mirror: bool = False) -> np.ndarray:
    """Inverse of :func:`pq_to_xy` (continuous; round to get lattice coordinates)."""
    xy = np.asarray(xy, dtype=float)
    if mirror:
        xy = xy * np.array([-1.0, 1.0])
    return xy @ np.linalg.inv(HEX_BASIS) / size


def hex_distance(pq1, pq2) -> np.ndarray:
    """Lattice (step) distance between axial coordinates; broadcasts over ``(..., 2)``.

    With neighbours at ``±(1, 1)`` this is ``(|dp| + |dq| + |dp - dq|) / 2``.
    """
    a = np.asarray(pq1, dtype=float)
    b = np.asarray(pq2, dtype=float)
    dp = a[..., 0] - b[..., 0]
    dq = a[..., 1] - b[..., 1]
    return (np.abs(dp) + np.abs(dq) + np.abs(dp - dq)) / 2


def hex_neighbors(p: int, q: int) -> list[tuple[int, int]]:
    """The six lattice neighbours of ``(p, q)``."""
    return [(p + dp, q + dq) for dp, dq in HEX_NEIGHBORS]


def _to_pandas_like(columns):
    """Return (p, q, column_id or None, frame-like with __getitem__) from polars/pandas/array input."""
    if hasattr(columns, "to_pandas") and hasattr(columns, "columns"):  # polars
        df = columns
        if "column_id" in df.columns:
            df = df.unique(subset=["p", "q"], keep="first", maintain_order=True)
        p = df["p"].to_numpy()
        q = df["q"].to_numpy()
        cid = df["column_id"].to_numpy() if "column_id" in df.columns else None
        get = lambda name: df[name].to_numpy()  # noqa: E731
        return p, q, cid, get
    if hasattr(columns, "iloc"):  # pandas
        df = columns
        if "column_id" in df.columns:
            df = df.drop_duplicates(subset=["p", "q"])
        p = df["p"].to_numpy()
        q = df["q"].to_numpy()
        cid = df["column_id"].to_numpy() if "column_id" in df.columns else None
        get = lambda name: df[name].to_numpy()  # noqa: E731
        return p, q, cid, get
    arr = np.asarray(columns)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise TypeError("columns must be a polars/pandas DataFrame with p, q columns or an (N, 2) array")
    return arr[:, 0], arr[:, 1], None, None


def _resolve_values(values, n: int, cid, get) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, str):
        if get is None:
            raise ValueError("values given by name requires a DataFrame input")
        return np.asarray(get(values), dtype=object)
    if isinstance(values, Mapping):
        if cid is None:
            raise ValueError("values given as a mapping requires a column_id column")
        return np.asarray([values.get(int(c), None) for c in cid], dtype=object)
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    values = np.asarray(values, dtype=object)
    if values.shape[0] != n:
        raise ValueError(f"values has {values.shape[0]} entries for {n} columns")
    return values


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def _is_numeric(vals: np.ndarray) -> bool:
    present = [v for v in vals if not _is_missing(v)]
    if not present:
        return True
    return all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) for v in present)


def hexmap(
    columns,
    values=None,
    *,
    ax=None,
    size: float = 1.0,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colors: Mapping[Any, Any] | str | None = None,
    missing: str = "0.88",
    edgecolor: str = "white",
    linewidth: float = 0.6,
    colorbar: bool = True,
    legend: bool = True,
    label: str | None = None,
    title: str | None = None,
    mirror: bool = False,
    annotate: bool | str = False,
    fontsize: float = 5,
    highlight: Sequence[tuple[int, int]] | None = None,
    highlight_color: str = "red",
):
    """Draw one hexagon per column, filled by ``values``.

    Parameters
    ----------
    columns
        A polars or pandas DataFrame with ``p`` and ``q`` (and optionally
        ``column_id``) columns, such as ``ds.columns.filter(pl.col("side") == "R")``
        or a per-column table built from it, or an ``(N, 2)`` array of ``(p, q)``.
        Duplicate ``(p, q)`` rows (one row per cell) collapse to one hexagon.
    values
        What to colour by: ``None`` (uniform fill), a column name of ``columns``,
        a mapping ``{column_id: value}``, or an array aligned with the rows.
        Numeric values use ``cmap`` with a colorbar; booleans, strings and other
        categories get discrete colours and a legend. ``None``/NaN entries are
        drawn in ``missing``.
    colors
        For categorical values, a mapping ``{category: color}``; for uniform
        fill, a single colour.
    annotate
        ``True`` writes ``column_id`` (or ``"p,q"`` when unavailable) in each
        hexagon; a column name writes that column's value.
    highlight
        ``(p, q)`` pairs to outline in ``highlight_color``.

    Returns the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize, to_rgba
    from matplotlib.patches import Patch

    p, q, cid, get = _to_pandas_like(columns)
    n = len(p)
    xy = pq_to_xy(np.c_[p, q], size=size, mirror=mirror)
    vals = _resolve_values(values, n, cid, get)

    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # flat-topped
    unit = np.c_[np.cos(angles), np.sin(angles)] * size
    verts = xy[:, None, :] + unit[None, :, :]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    facecolors: list
    legend_handles: list[Patch] = []
    mappable = None
    if vals is None:
        fill = colors if isinstance(colors, str) else "0.55"
        facecolors = [to_rgba(fill)] * n
    elif _is_numeric(vals):
        present = np.array([not _is_missing(v) for v in vals])
        num = np.array([float(v) if ok else np.nan for v, ok in zip(vals, present)])
        lo = np.nanmin(num) if vmin is None else vmin
        hi = np.nanmax(num) if vmax is None else vmax
        norm = Normalize(vmin=lo, vmax=hi)
        cm = colormaps.get_cmap(cmap)
        facecolors = [cm(norm(v)) if ok else to_rgba(missing) for v, ok in zip(num, present)]
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    else:
        present = [v for v in vals if not _is_missing(v)]
        cats = sorted(set(present), key=lambda v: (str(type(v)), v))
        if isinstance(colors, Mapping):
            cmap_cat = {c: colors.get(c, missing) for c in cats}
        else:
            base = colormaps.get_cmap("tab10" if len(cats) <= 10 else "tab20")
            cmap_cat = {c: base(i % base.N) for i, c in enumerate(cats)}
        facecolors = [to_rgba(missing) if _is_missing(v) else to_rgba(cmap_cat[v]) for v in vals]
        counts = {c: sum(1 for v in present if v == c) for c in cats}
        legend_handles = [Patch(facecolor=cmap_cat[c], edgecolor="none", label=f"{c} ({counts[c]})") for c in cats]
        n_missing = n - len(present)
        if n_missing:
            legend_handles.append(Patch(facecolor=missing, edgecolor="none", label=f"missing ({n_missing})"))

    coll = PolyCollection(verts, facecolors=facecolors, edgecolors=edgecolor, linewidths=linewidth)
    ax.add_collection(coll)

    if highlight is not None:
        hset = {(int(a), int(b)) for a, b in highlight}
        idx = [i for i in range(n) if (int(p[i]), int(q[i])) in hset]
        if idx:
            ax.add_collection(
                PolyCollection(verts[idx], facecolors="none", edgecolors=highlight_color, linewidths=linewidth * 3)
            )

    if annotate:
        if isinstance(annotate, str):
            text = [str(v) for v in get(annotate)]
        elif cid is not None:
            text = [str(int(c)) for c in cid]
        else:
            text = [f"{int(a)},{int(b)}" for a, b in zip(p, q)]
        for (x, y), t in zip(xy, text):
            ax.text(x, y, t, ha="center", va="center", fontsize=fontsize)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.02)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if mappable is not None and colorbar:
        ax.figure.colorbar(mappable, ax=ax, shrink=0.7, label=label or (values if isinstance(values, str) else None))
    if legend_handles and legend:
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, title=label)
    return ax


__all__ = ["HEX_BASIS", "HEX_NEIGHBORS", "hex_distance", "hex_neighbors", "hexmap", "pq_to_xy", "xy_to_pq"]
