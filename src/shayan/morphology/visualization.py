"""
Visualization tools for neuron compartmentalization.

Provides functions to:
- Interactive 3D visualization with plotly
- Neuroglancer URL generation for FlyWire data
- Statistical analysis plots
- Strategy comparison
"""

import numpy as np
import navis
import plotly.graph_objects as go
from typing import Dict, List, Optional, TYPE_CHECKING
import copy
import json
import urllib.parse

if TYPE_CHECKING:
    from .compartmentalization import Compartmentalization


def plot_compartmentalization_2d(
    comp: 'Compartmentalization',
    color_by: str = 'radius',
    method: str = '2d',
    figsize: tuple = (10, 10),
    node_size: float = 10,
    linewidth: float = 2.0,
    **kwargs
):
    """
    Visualize compartmentalization in 2D using navis.

    Args:
        comp: Compartmentalization object to visualize
        color_by: How to color compartments:
            - 'radius': Color by mean radius (default, recommended)
            - 'length': Color by compartment length
            - 'compartment': Each compartment gets unique color (discrete)
        method: navis plotting method ('2d', '3d', '3d_plotly')
        figsize: Figure size as (width, height) in inches
        node_size: Size of node scatter points (default: 10)
        linewidth: Width of skeleton lines (default: 2.0)
        **kwargs: Additional arguments passed to navis.plot2d()

    Returns:
        matplotlib figure or plotly figure depending on method

    Example:
        >>> from shayan.morphology.loader import load_skeleton
        >>> from shayan.morphology.segmentation import segment_natural
        >>> neuron = load_skeleton('data/...', [720575940627932541])[0]
        >>> comp = segment_natural(neuron)
        >>>
        >>> # Default settings
        >>> fig = plot_compartmentalization_2d(comp, color_by='radius')
        >>>
        >>> # Custom figure size and appearance
        >>> fig = plot_compartmentalization_2d(comp, color_by='radius',
        ...                                     figsize=(12, 8),
        ...                                     node_size=5,
        ...                                     linewidth=3.0)
        >>>
        >>> # For matplotlib, use plt.show() or plt.savefig()
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Get compartment values for coloring
    if color_by == 'radius':
        values = comp.compartments['radius_mean'].to_numpy()
        label = 'Mean Radius (μm)'
    elif color_by == 'length':
        values = comp.compartments['length'].to_numpy()
        label = 'Length (μm)'
    elif color_by == 'compartment':
        values = np.arange(len(comp.compartments))
        label = 'Compartment ID'
    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    # Normalize values for colormap
    vmin, vmax = values.min(), values.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    # Create color mapping: node_id -> color
    node_colors = {}
    for comp_id, node_ids in comp.node_mapping.items():
        color = cmap(norm(values[comp_id]))
        for node_id in node_ids:
            node_colors[node_id] = color

    # Plot using navis with connectors
    fig, ax = navis.plot2d(
        comp.neuron,
        color=node_colors,
        method=method,
        figsize=figsize,
        connectors=False,  # Don't show synapse connectors
        linewidth=linewidth,  # Control skeleton line thickness
        **kwargs
    )

    # Now add the actual node points with their colors
    # Get all node positions
    nodes_df = comp.neuron.nodes

    # Plot each compartment's nodes
    for comp_id, node_ids in comp.node_mapping.items():
        if len(node_ids) == 0:
            continue

        # Get nodes for this compartment
        comp_nodes = nodes_df[nodes_df['node_id'].isin(node_ids)]

        # Get color for this compartment
        color = cmap(norm(values[comp_id]))

        # Plot nodes as scatter points
        ax.scatter(
            comp_nodes['x'].values,
            comp_nodes['y'].values,
            c=[color] * len(comp_nodes),
            s=node_size,  # Point size (user-configurable)
            edgecolors='none',
            zorder=10  # Draw on top of skeleton lines
        )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(label)

    plt.title(f'Compartmentalization: {comp.strategy}\n{len(comp)} compartments, colored by {color_by}')

    return fig 

def plot_compartmentalization(
    comp: 'Compartmentalization',
    color_by: str = 'radius',
    show_skeleton: bool = True,
    show_nodes: bool = True,
    node_size: float = 3.0,
    node_color: str = '#1f77b4',
    alpha: float = 0.8,
    backend: str = 'plotly',
    **kwargs
) -> go.Figure:
    """
    Visualize compartmentalization in interactive 3D using plotly.

    Compartments (skeleton segments) are colored by the specified property,
    while all individual nodes are displayed with uniform size and color.

    Args:
        comp: Compartmentalization object to visualize
        color_by: How to color compartments (skeleton segments):
            - 'radius': Color by mean radius (default, recommended for distinguishing segments)
            - 'length': Color by compartment length
            - 'compartment': Each compartment gets unique color (discrete)
            - 'depth': Color by tree depth from root
        show_skeleton: If True, show underlying neuron skeleton as wireframe
        show_nodes: If True, show individual nodes as scatter points
        node_size: Size of all node markers (uniform)
        node_color: Color of all node markers (uniform)
        alpha: Transparency for nodes (0-1)
        backend: 3D backend ('plotly' only for now)
        **kwargs: Passed to navis.plot3d()

    Returns:
        Interactive plotly figure

    Example:
        >>> from shayan.morphology.loader import load_skeleton
        >>> from shayan.morphology.segmentation import segment_natural
        >>> neuron = load_skeleton('data/...', [720575940627932541])[0]
        >>> comp = segment_natural(neuron)
        >>>
        >>> # Default: segments colored by radius, nodes uniform
        >>> fig = plot_compartmentalization(comp)
        >>> fig.show()
        >>>
        >>> # Color segments by length
        >>> fig = plot_compartmentalization(comp, color_by='length')
        >>> fig.show()
    """
    if backend != 'plotly':
        raise ValueError(f"Only 'plotly' backend supported, got '{backend}'")

    # Get compartment colors for skeleton segments
    colors = _get_compartment_colors(comp, color_by)

    # Start with navis skeleton plot if requested
    # Color skeleton segments by compartment property
    if show_skeleton:
        # Create color mapping for skeleton: node_id -> color based on compartment
        node_colors = {}
        for comp_id, node_ids in comp.node_mapping.items():
            for node_id in node_ids:
                node_colors[node_id] = colors[comp_id]

        fig = navis.plot3d(
            comp.neuron,
            backend='plotly',
            inline=False,
            color=node_colors,
            **kwargs
        )
    else:
        fig = go.Figure()

    if not show_nodes:
        return fig

    # Collect all nodes with uniform properties
    all_x, all_y, all_z = [], [], []
    all_hover_text = []

    for comp_id, node_ids in comp.node_mapping.items():
        if len(node_ids) == 0:
            continue

        # Get node data
        node_data = comp.neuron.nodes[comp.neuron.nodes['node_id'].isin(node_ids)]

        # Get coordinates
        all_x.extend(node_data['x'].values)
        all_y.extend(node_data['y'].values)
        all_z.extend(node_data['z'].values)

        # Get compartment info for hover text
        comp_row = comp.compartments.filter(
            comp.compartments['compartment_id'] == comp_id
        )
        hover_texts = [
            f"Compartment {comp_id}<br>" +
            f"Node {row['node_id']}<br>" +
            f"Comp length: {comp_row['length'].item():.2f} μm<br>" +
            f"Comp mean radius: {comp_row['radius_mean'].item():.3f} μm<br>" +
            f"Node radius: {row['radius']:.3f} μm<br>" +
            f"Parent: {comp_row['parent_id'].item()}"
            for _, row in node_data.iterrows()
        ]
        all_hover_text.extend(hover_texts)

    # Add single scatter trace with all nodes in uniform color and size
    fig.add_trace(go.Scatter3d(
        x=all_x, y=all_y, z=all_z,
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_color,
            opacity=alpha,
            line=dict(width=0)
        ),
        text=all_hover_text,
        hoverinfo='text',
        name='Nodes',
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title=f'Compartmentalization: {comp.strategy}<br>' +
              f'{len(comp)} compartments, {comp.neuron.n_nodes} nodes<br>' +
              f'Segments colored by {color_by}, nodes uniform',
        scene=dict(
            xaxis_title='X (μm)',
            yaxis_title='Y (μm)',
            zaxis_title='Z (μm)',
            aspectmode='data'
        ),
        hovermode='closest'
    )

    return fig


def plot_compartments_subset(
    comp: 'Compartmentalization',
    compartment_ids: List[int],
    show_skeleton: bool = True,
    node_size: float = 5.0,
    node_color: str = '#1f77b4',
    alpha: float = 0.9,
    **kwargs
) -> go.Figure:
    """
    Plot only a subset of compartments for detailed inspection.

    Useful for zooming in on specific regions of interest.

    Args:
        comp: Compartmentalization object
        compartment_ids: List of compartment IDs to visualize
        show_skeleton: If True, show full skeleton as context
        node_size: Size of node markers (larger for better visibility in subset view)
        node_color: Color of node markers
        alpha: Transparency (higher = more opaque)
        **kwargs: Additional arguments passed to plot_compartmentalization

    Returns:
        Interactive plotly figure showing only selected compartments

    Example:
        >>> # Focus on compartments 10-15
        >>> fig = plot_compartments_subset(comp, list(range(10, 16)))
        >>> fig.show()
        >>>
        >>> # Focus on a single compartment
        >>> fig = plot_compartments_subset(comp, [42], show_skeleton=True)
        >>> fig.show()
    """
    # Filter node_mapping to only include selected compartments
    filtered_mapping = {cid: comp.node_mapping[cid] for cid in compartment_ids
                        if cid in comp.node_mapping}

    # Shallow-copy the object so the caller's node_mapping is left untouched
    # (copy.copy skips __init__, so no re-validation is triggered)
    subset = copy.copy(comp)
    subset.node_mapping = filtered_mapping

    # Plot with modified mapping
    fig = plot_compartmentalization(
        subset,
        color_by='compartment',
        show_skeleton=show_skeleton,
        show_nodes=True,
        node_size=node_size,
        node_color=node_color,
        alpha=alpha,
        **kwargs
    )

    # Update title
    fig.update_layout(
        title=f'Compartments {compartment_ids} of {len(comp)} total<br>' +
              f'Strategy: {comp.strategy}'
    )

    return fig


def neuron_to_neuroglancer_url(
    neuron_id: int,
    viewer: str = 'cave',
    data_version: str = '783'
) -> str:
    """
    Generate neuroglancer URL showing a neuron.

    Based on Codex implementation (murthylab/codex).

    Args:
        neuron_id: FlyWire neuron ID (root ID)
        viewer: Which viewer to use:
            - 'cave': CAVE Explorer viewer with materialization (default, most reliable)
            - 'flywire': FlyWire production viewer (may not work for all neurons)
        data_version: Materialization version (e.g., '783' for flywire_v141_m783)
            Only used for 'cave' viewer.

    Returns:
        URL string that opens neuron in neuroglancer viewer

    Example:
        >>> from shayan.morphology.visualization import neuron_to_neuroglancer_url
        >>> import webbrowser
        >>>
        >>> # CAVE explorer (default - most reliable)
        >>> url = neuron_to_neuroglancer_url(720575940627932541)
        >>> webbrowser.open(url)
        >>>
        >>> # Different materialization version
        >>> url = neuron_to_neuroglancer_url(720575940627932541, data_version='630')
        >>> webbrowser.open(url)
    """
    if viewer == 'flywire':
        # FlyWire production viewer (based on Codex implementation)
        # Use the same structure as CAVE viewer but with FlyWire-specific sources
        config = {
            'dimensions': {'x': [1.6e-8, 'm'], 'y': [1.6e-8, 'm'], 'z': [4e-8, 'm']},
            'projectionScale': 30000,
            'layers': [
                {
                    'type': 'image',
                    'source': 'precomputed://gs://microns-seunglab/drosophila_v0/alignment/vector_fixer30_faster_v01/v4/image_stitch_v02',
                    'tab': 'source',
                    'name': 'EM'
                },
                {
                    'type': 'segmentation',
                    'source': 'graphene://https://prodv1.flywire-daf.com/segmentation/table/fly_v31',
                    'tab': 'segments',
                    'segments': [str(neuron_id)],
                    'name': 'Production segmentation'
                }
            ],
            'showSlices': False,
            'perspectiveViewBackgroundColor': '#ffffff',
            'showDefaultAnnotations': False,
            'selectedLayer': {
                'visible': True,
                'layer': 'Production segmentation'
            },
            'layout': '3d',
            'jsonStateServer': 'https://globalv1.flywire-daf.com/nglstate/post'
        }
        base_url = 'https://ngl.flywire.ai'

    elif viewer == 'cave':
        # CAVE Explorer with materialization (based on Codex implementation)
        config = {
            'dimensions': {'x': [1.6e-8, 'm'], 'y': [1.6e-8, 'm'], 'z': [4e-8, 'm']},
            'projectionScale': 30000,
            'layers': [
                {
                    'type': 'image',
                    'source': 'precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14',
                    'tab': 'source',
                    'name': 'EM'
                },
                {
                    'source': 'precomputed://gs://flywire_neuropil_meshes/whole_neuropil/brain_mesh_v3',
                    'type': 'segmentation',
                    'objectAlpha': 0.05,
                    'hideSegmentZero': False,
                    'segments': ['1'],
                    'segmentColors': {'1': '#b5b5b5'},
                    'skeletonRendering': {'mode2d': 'lines_and_points', 'mode3d': 'lines'},
                    'name': 'brain_mesh_v3'
                },
                {
                    'type': 'segmentation',
                    'source': f'precomputed://gs://flywire_v141_m{data_version}',
                    'tab': 'segments',
                    'segments': [str(neuron_id)],
                    'name': f'flywire_v141_m{data_version}'
                }
            ],
            'showSlices': False,
            'perspectiveViewBackgroundColor': '#ffffff',
            'showDefaultAnnotations': False,
            'selectedLayer': {
                'visible': True,
                'layer': f'flywire_v141_m{data_version}'
            },
            'layout': '3d'
        }
        base_url = 'https://ngl.cave-explorer.org'

    else:
        raise ValueError(f"Unknown viewer: {viewer}. Use 'flywire' or 'cave'.")

    # Encode state as JSON and URL encode
    state_json = json.dumps(config)
    state_encoded = urllib.parse.quote(state_json)

    # Build final URL
    url = f"{base_url}/#!{state_encoded}"

    return url


def to_neuroglancer_url(
    comp: 'Compartmentalization',
    neuron_id: Optional[int] = None,
    viewer: str = 'cave',
    data_version: str = '783',
    color_by: str = 'compartment',
    use_centers_only: bool = True
) -> str:
    """
    Generate neuroglancer URL with compartmentalization overlay.

    Creates a neuroglancer state showing:
    - Brain imagery and segmentation
    - Selected neuron
    - Compartment centers as point annotations

    Args:
        comp: Compartmentalization to visualize
        neuron_id: FlyWire neuron ID (if None, uses comp.neuron.id)
        viewer: Which viewer to use ('cave' or 'flywire'). Default 'cave'.
        data_version: Materialization version for CAVE viewer (e.g., '783')
        color_by: How to color compartment points (not yet implemented)
        use_centers_only: If True, only show compartment centers (shorter URL).
            If False, show all nodes (very long URL, may not work).

    Returns:
        URL string that opens compartmentalization in neuroglancer viewer

    Example:
        >>> from shayan.morphology.loader import load_skeleton
        >>> from shayan.morphology.segmentation import segment_natural
        >>> from shayan.morphology.visualization import to_neuroglancer_url
        >>> import webbrowser
        >>>
        >>> neuron = load_skeleton('data/...', [720575940627932541])[0]
        >>> comp = segment_natural(neuron)
        >>> url = to_neuroglancer_url(comp)
        >>> webbrowser.open(url)
    """
    if neuron_id is None:
        neuron_id = comp.neuron.id

    # Get compartment colors (for future use)
    colors = _get_compartment_colors(comp, color_by)

    # Build point annotations for each compartment
    annotations = []
    for comp_id, node_ids in comp.node_mapping.items():
        if len(node_ids) == 0:
            continue

        # Get node positions
        node_data = comp.neuron.nodes[comp.neuron.nodes['node_id'].isin(node_ids)]

        # Get compartment info
        comp_row = comp.compartments.filter(
            comp.compartments['compartment_id'] == comp_id
        )

        if use_centers_only:
            # Only add compartment center (mean position)
            center_x = float(node_data['x'].mean() * 1000)  # Convert μm to nm
            center_y = float(node_data['y'].mean() * 1000)
            center_z = float(node_data['z'].mean() * 1000)

            # Convert color to hex
            color_hex = _rgb_to_hex(colors[comp_id])

            annotations.append({
                'point': [center_x, center_y, center_z],
                'type': 'point',
                'id': f"comp_{comp_id}",
                'description': (
                    f"Compartment {comp_id}\\n" +
                    f"Nodes: {len(node_ids)}\\n" +
                    f"Length: {comp_row['length'].item():.2f} μm\\n" +
                    f"Radius: {comp_row['radius_mean'].item():.3f} μm\\n" +
                    f"Parent: {comp_row['parent_id'].item()}"
                )
            })
        else:
            # Add annotation for each node (original behavior - creates very long URLs)
            for _, node in node_data.iterrows():
                # Convert color to hex
                color_hex = _rgb_to_hex(colors[comp_id])

                annotations.append({
                    'point': [
                        float(node['x'] * 1000),  # Convert μm to nm for neuroglancer
                        float(node['y'] * 1000),
                        float(node['z'] * 1000)
                    ],
                    'type': 'point',
                    'id': f"{comp_id}_{node['node_id']}",
                    'description': (
                        f"Compartment {comp_id}\\n" +
                        f"Node {node['node_id']}\\n" +
                        f"Length: {comp_row['length'].item():.2f} μm\\n" +
                        f"Radius: {comp_row['radius_mean'].item():.3f} μm"
                    )
                })

    # Build neuroglancer state using CAVE viewer
    if viewer == 'cave':
        state = {
            'dimensions': {'x': [1.6e-8, 'm'], 'y': [1.6e-8, 'm'], 'z': [4e-8, 'm']},
            'projectionScale': 30000,
            'layers': [
                {
                    'type': 'image',
                    'source': 'precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14',
                    'tab': 'source',
                    'name': 'EM'
                },
                {
                    'source': 'precomputed://gs://flywire_neuropil_meshes/whole_neuropil/brain_mesh_v3',
                    'type': 'segmentation',
                    'objectAlpha': 0.05,
                    'hideSegmentZero': False,
                    'segments': ['1'],
                    'segmentColors': {'1': '#b5b5b5'},
                    'skeletonRendering': {'mode2d': 'lines_and_points', 'mode3d': 'lines'},
                    'name': 'brain_mesh_v3'
                },
                {
                    'type': 'segmentation',
                    'source': f'precomputed://gs://flywire_v141_m{data_version}',
                    'tab': 'segments',
                    'segments': [str(neuron_id)],
                    'name': f'flywire_v141_m{data_version}'
                },
                {
                    'type': 'annotation',
                    'name': f'Compartments ({comp.strategy})',
                    'annotations': annotations
                }
            ],
            'showSlices': False,
            'perspectiveViewBackgroundColor': '#ffffff',
            'showDefaultAnnotations': False,
            'selectedLayer': {
                'visible': True,
                'layer': f'Compartments ({comp.strategy})'
            },
            'layout': '3d'
        }
        base_url = 'https://ngl.cave-explorer.org'
    else:
        raise ValueError(f"Only 'cave' viewer supported for compartmentalization. Got: {viewer}")

    # Encode state as JSON and URL encode
    state_json = json.dumps(state)
    state_encoded = urllib.parse.quote(state_json)

    # Build final URL
    url = f"{base_url}/#!{state_encoded}"

    return url


def _get_compartment_colors(comp: 'Compartmentalization', color_by: str) -> Dict[int, str]:
    """
    Get color for each compartment based on coloring scheme.

    Args:
        comp: Compartmentalization object
        color_by: Coloring scheme

    Returns:
        Dictionary mapping compartment_id -> color string (hex or rgb)
    """
    import plotly.express as px

    n_comps = len(comp.compartments)

    if color_by == 'compartment':
        # Discrete colors
        color_scale = px.colors.qualitative.Plotly
        colors = {i: color_scale[i % len(color_scale)] for i in range(n_comps)}

    elif color_by == 'length':
        # Continuous colormap by length
        lengths = comp.compartments['length'].to_numpy()
        color_scale = px.colors.sequential.Viridis
        colors = _values_to_colors(lengths, color_scale)

    elif color_by == 'radius':
        # Continuous colormap by mean radius
        radii = comp.compartments['radius_mean'].to_numpy()
        color_scale = px.colors.sequential.Plasma
        colors = _values_to_colors(radii, color_scale)

    elif color_by == 'depth':
        # Color by tree depth
        depths = _calculate_compartment_depths(comp)
        color_scale = px.colors.sequential.Turbo
        colors = _values_to_colors(depths, color_scale)

    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    return colors


def _values_to_colors(values: np.ndarray, color_scale: list) -> Dict[int, str]:
    """Map values to colors using a colorscale."""
    # Normalize values to [0, 1]
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    if vmax > vmin:
        normalized = (values - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(values)

    # Map to colors
    colors = {}
    for i, val in enumerate(normalized):
        idx = int(val * (len(color_scale) - 1))
        colors[i] = color_scale[idx]

    return colors


def _calculate_compartment_depths(comp: 'Compartmentalization') -> np.ndarray:
    """Calculate tree depth for each compartment."""
    depths = np.zeros(len(comp.compartments), dtype=int)

    # BFS from root
    for i in range(len(comp.compartments)):
        comp_id = i
        depth = 0
        while True:
            parent_id = comp.compartments.filter(
                comp.compartments['compartment_id'] == comp_id
            )['parent_id'].item()

            if parent_id == -1:
                break
            comp_id = parent_id
            depth += 1
        depths[i] = depth

    return depths


def _rgb_to_hex(color: str) -> str:
    """Convert plotly color to hex."""
    # Handle hex colors
    if color.startswith('#'):
        return color

    # Handle rgb(r, g, b) format
    if color.startswith('rgb'):
        # Extract numbers
        nums = color.replace('rgb(', '').replace(')', '').split(',')
        r, g, b = [int(x.strip()) for x in nums]
        return f'#{r:02x}{g:02x}{b:02x}'

    # Default to a color
    return '#0000ff'
