"""Compartment plots with navis: plotly for 3D, matplotlib for 2D. Both import lazily."""

from __future__ import annotations

import numpy as np

from connexplorer.morph.segment import Compartments


def _values(comp: Compartments, color_by, values=None) -> tuple[np.ndarray, str]:
    if values is not None:
        return np.asarray(values, dtype=float), str(color_by)
    t = comp.table
    if color_by == "radius":
        return t["radius_mean"].to_numpy(), "mean radius (um)"
    if color_by == "length":
        return t["length"].to_numpy(), "length (um)"
    if color_by == "depth":
        return t["depth"].to_numpy().astype(float), "depth"
    if color_by == "compartment":
        return np.arange(len(comp), dtype=float), "compartment"
    raise ValueError("color_by must be radius, length, depth or compartment")


def _node_colors(comp: Compartments, vals: np.ndarray, cmap: str = "viridis") -> tuple[dict, object, object]:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    norm = mcolors.Normalize(vmin=float(np.nanmin(vals)), vmax=float(np.nanmax(vals)) if np.nanmax(vals) > np.nanmin(vals) else float(np.nanmin(vals)) + 1)
    cm = plt.get_cmap(cmap)
    hexes = [mcolors.to_hex(cm(norm(v))) for v in vals]
    node_colors = {int(nid): hexes[c] for nid, c in zip(comp.node_ids, comp.comp_of_node)}
    return node_colors, norm, cm


def plot3d(comp: Compartments, color_by: str = "radius", values=None, show_centers: bool = True, **kw):
    """Interactive plotly figure: skeleton coloured per compartment, optional compartment centres."""
    import navis
    import plotly.graph_objects as go

    vals, label = _values(comp, color_by, values)
    node_colors, _, _ = _node_colors(comp, vals)
    fig = navis.plot3d(comp.neuron, backend="plotly", inline=False, color=node_colors, **kw)
    if show_centers:
        c = comp.centers
        t = comp.table
        fig.add_trace(
            go.Scatter3d(
                x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers",
                marker=dict(size=3, color=vals, colorscale="Viridis", colorbar=dict(title=label)),
                text=[f"compartment {i}<br>{label}: {v:.3g}<br>length {t['length'][i]:.2f} um<br>parent {t['parent_id'][i]}" for i, v in enumerate(vals)],
                hoverinfo="text", name="compartments",
            )
        )
    fig.update_layout(title=f"{len(comp)} compartments ({comp.method}), coloured by {label}", scene=dict(xaxis_title="x (um)", yaxis_title="y (um)", zaxis_title="z (um)", aspectmode="data"))
    return fig


def plot2d(comp: Compartments, color_by: str = "radius", values=None, figsize=(8, 8), **kw):
    """matplotlib figure of the skeleton coloured per compartment, with a colour bar."""
    import matplotlib.pyplot as plt
    import navis

    vals, label = _values(comp, color_by, values)
    node_colors, norm, cm = _node_colors(comp, vals)
    fig, ax = navis.plot2d(comp.neuron, color=node_colors, method="2d", figsize=figsize, connectors=False, **kw)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=label)
    ax.set_title(f"{len(comp)} compartments ({comp.method}), coloured by {label}")
    return fig


def plot_voltage(comp: Compartments, V: np.ndarray, **kw):
    """3D plot coloured by a per-compartment voltage (mV), e.g. from ``Cable.steady_state``."""
    return plot3d(comp, color_by="voltage (mV)", values=np.asarray(V), **kw)
