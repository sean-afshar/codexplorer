"""FlyWire external-link helpers that operate on dataset RIDs."""

from __future__ import annotations

import copy
import json
import random
import string
from urllib.parse import quote_plus
import webbrowser

import numpy as np
from ..util import distinct_hex_colors


FLYWIRE_SEGMENT_KEYS = {
    "783": "flywire_v141_m783",
    "630": "flywire_v141_m630",
}

FLYWIRE_FAFBV14_EM_SOURCE = (
    "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14"
)


def codex_url(rids, version: str = "783", page_size: int = 10) -> str:
    """Return a Codex search URL for one or more FlyWire RIDs."""
    query = "+".join(map(str, np.atleast_1d(rids).flatten()))
    return (
        "https://codex.flywire.ai/app/search"
        f"?filter_string={query}&sort_by=&page_size={int(page_size)}&data_version={version}"
    )


def codex_open(rids, version: str = "783", page_size: int = 10) -> str:
    """Open and return a Codex search URL for one or more FlyWire RIDs."""
    url = codex_url(rids, version=version, page_size=page_size)
    open_url(url)
    return url


def neuroglancer_url(
    rids,
    anno_layers=None,
    version: str = "783",
    open_url_Q: bool = False,
    same_color_Q: bool = False,
    return_payload_Q: bool = False,
) -> str | tuple[str, dict]:
    """Return a Neuroglancer URL for FlyWire RIDs."""
    payload = ng_template(version=version)
    key_to_color = {}
    if isinstance(rids, dict):
        colors = distinct_hex_colors(len(rids))
        add_layers = []
        for i, key in enumerate(rids):
            values = np.asarray(rids[key]).flatten()
            key_to_color[key] = colors[i]
            if values.size > 0:
                add_layers.append(
                    segmentation_layer(
                        values,
                        version=version,
                        color=key_to_color[key],
                        layer_name=str(key),
                    )
                )
    else:
        add_layers = [segmentation_layer(rids, version=version)]
    payload["layers"].extend(add_layers)
    _extend_annotation_layers(payload, anno_layers, key_to_color, same_color_Q=same_color_Q)
    encoded = quote_plus(json.dumps(payload, separators=(",", ":")))
    # url = f"https://ngl.cave-explorer.org/#!{encoded}"
    url = f"https://spelunker.cave-explorer.org/#!{encoded}"
    if open_url_Q:
        open_url(url)
    if return_payload_Q:
        return url, payload
    return url


def ng_template(version: str = "783") -> dict:
    """Return the base Neuroglancer state for a FlyWire version."""
    segment_key = _segment_key(version)
    return {
        "dimensions": {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"]},
        "position": [32736.0371 * 4, 14171.4883 * 4, 4156.3672],
        "crossSectionScale": 1,
        "projectionScale": 90000,
        "layers": [
            {
                "type": "image",
                "source": FLYWIRE_FAFBV14_EM_SOURCE,
                "tab": "source",
                "name": "EM",
            },
            {
                "type": "segmentation",
                "source": "precomputed://gs://flywire_neuropil_meshes/whole_neuropil/brain_mesh_v3",
                "tab": "source",
                "objectAlpha": 0.05,
                "hideSegmentZero": False,
                "segments": ["1"],
                "segmentColors": {"1": "#b5b5b5"},
                "name": "brain_mesh_v3",
            },
        ],
        "showDefaultAnnotations": False,
        "showSlices": False,
        "selectedLayer": {"visible": True, "layer": segment_key},
        "projectionBackgroundColor": "#ffffff",
        "layout": "3d",
    }


def segmentation_layer(rids, version: str = "783", color=None, layer_name: str = "Segmentation") -> dict:
    """Return a Neuroglancer segmentation layer for FlyWire RIDs."""
    rid_values = [str(rid) for rid in np.atleast_1d(rids).flatten()]
    layer = {
        "type": "segmentation",
        "source": f"precomputed://gs://{_segment_key(version)}",
        "tab": "segments",
        "name": f". {layer_name}",
        "segments": rid_values,
        "visible": "Self" in layer_name,
    }
    if color is None:
        return layer
    if isinstance(color, str):
        layer["segmentColors"] = {rid: color for rid in rid_values}
    elif isinstance(color, list):
        if len(color) != len(rid_values):
            raise ValueError("Length of color list must match length of RID list")
        layer["segmentColors"] = {rid: c for rid, c in zip(rid_values, color)}
    else:
        raise ValueError("color must be either a string or a list of strings")
    return layer


def annotation_layer(
    points_xyz,
    layer_name: str = "Annotation",
    ng_voxel_size_nm=(4, 4, 40),
    assign_random_color_Q: bool = False,
) -> dict:
    """Return a Neuroglancer point-annotation layer."""
    points_xyz = (np.asarray(points_xyz) / np.asarray([ng_voxel_size_nm])).astype(np.int64)
    layer = {
        "type": "annotation",
        "source": {
            "url": "local://annotations",
            "transform": {
                "outputDimensions": {
                    "x": [1.6e-8, "m"],
                    "y": [1.6e-8, "m"],
                    "z": [4e-8, "m"],
                },
                "inputDimensions": {
                    "0": [4e-9, "m"],
                    "1": [4e-9, "m"],
                    "2": [4e-8, "m"],
                },
            },
        },
        "tab": "source",
        "annotations": annotation_point_list(points_xyz),
        "name": f". {layer_name}",
        "visible": False,
    }
    if assign_random_color_Q:
        layer["annotationColor"] = random.choice(distinct_hex_colors(20))
    return layer


def annotation_point_list(points_xyz) -> list[dict]:
    """Return Neuroglancer point annotations."""
    random_str_length = 40
    return [
        {
            "point": np.asarray(xyz).tolist(),
            "type": "point",
            "id": "".join(random.choices(string.ascii_lowercase + string.digits, k=random_str_length)),
        }
        for xyz in points_xyz
    ]


def open_url(url: str) -> None:
    """Open a URL from notebooks or a regular Python process."""
    try:
        from IPython.display import Javascript, display

        display(Javascript(f'window.open("{url}");'))
    except Exception:
        webbrowser.open_new_tab(url)


def _segment_key(version: str) -> str:
    version = str(version)
    if version not in FLYWIRE_SEGMENT_KEYS:
        raise ValueError(f"Unsupported FlyWire version: {version}")
    return FLYWIRE_SEGMENT_KEYS[version]


def _extend_annotation_layers(payload: dict, anno_layers, key_to_color: dict, same_color_Q: bool) -> None:
    if anno_layers is None:
        return
    if isinstance(anno_layers, dict):
        if "annotations" in anno_layers:
            payload["layers"].append(anno_layers)
            return
        for key in anno_layers:
            layer = copy.deepcopy(anno_layers[key])
            if same_color_Q and key in key_to_color:
                layer["annotationColor"] = key_to_color[key]
            payload["layers"].append(layer)
        return
    if isinstance(anno_layers, list):
        payload["layers"].extend(anno_layers)
