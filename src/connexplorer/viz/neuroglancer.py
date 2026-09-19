"""Neuroglancer state builders. Every function returns a URL string and never opens a browser.

Per-dataset viewer configuration (image source, segmentation source, voxel
size, camera) lives in :data:`VIEWERS`, keyed by dataset name; the
segmentation source is formatted with the dataset version.
"""

from __future__ import annotations

import colorsys
import json
from urllib.parse import quote

import numpy as np

VIEWERS: dict[str, dict] = {
    "flywire": {
        "voxel_nm": (4.0, 4.0, 40.0),
        "position": [130944.0, 56685.0, 4156.0],
        "projection_scale": 90000,
        "image": "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14",
        "segmentation": "precomputed://gs://flywire_v141_m{version}",
        "context_layers": [
            {
                "type": "segmentation",
                "source": "precomputed://gs://flywire_neuropil_meshes/whole_neuropil/brain_mesh_v3",
                "objectAlpha": 0.05,
                "hideSegmentZero": False,
                "segments": ["1"],
                "segmentColors": {"1": "#b5b5b5"},
                "name": "brain_mesh_v3",
            }
        ],
        "hosts": {
            "spelunker": "https://spelunker.cave-explorer.org",
            "cave": "https://ngl.cave-explorer.org",
            "flywire": "https://ngl.flywire.ai",
        },
        # the production viewer needs the live graphene source and a state server
        "flywire_production": {
            "image": "precomputed://gs://microns-seunglab/drosophila_v0/alignment/vector_fixer30_faster_v01/v4/image_stitch_v02",
            "segmentation": "graphene://https://prodv1.flywire-daf.com/segmentation/table/fly_v31",
            "jsonStateServer": "https://globalv1.flywire-daf.com/nglstate/post",
        },
        "codex": "https://codex.flywire.ai/app/search?filter_string={ids}&sort_by=&page_size={n}&data_version={version}",
    },
    "mcns": {
        "voxel_nm": (8.0, 8.0, 8.0),
        "position": [48686.5, 27515.5, 24721.5],
        "projection_scale": 134522,
        "image": "precomputed://gs://flyem-male-cns/em/em-clahe-jpeg",
        "segmentation": [
            "precomputed://gs://flyem-male-cns/{version}/segmentation",
            "precomputed://gs://flyem-male-cns/{version}/segmentation/type_property",
            "precomputed://gs://flyem-male-cns/{version}/segmentation/instance_property",
            "precomputed://gs://flyem-male-cns/{version}/segmentation/flywireType_property",
            "precomputed://gs://flyem-male-cns/{version}/segmentation/meshes-malecns/single-res-meshes",
        ],
        "context_layers": [
            {
                "type": "segmentation",
                "source": "precomputed://gs://flyem-male-cns/rois/brain-shell-v2.2",
                "selectedAlpha": 0.0,
                "objectAlpha": 0.08,
                "segments": ["1"],
                "segmentDefaultColor": "#b5b5b5",
                "name": "brain-shell",
            }
        ],
        "hosts": {
            "spelunker": "https://spelunker.cave-explorer.org",
            "cave": "https://ngl.cave-explorer.org",
            "neuroglancer": "https://neuroglancer-demo.appspot.com",
            "clio": "https://clio-ng.janelia.org",
        },
    },
}


def distinct_colors(n: int) -> list[str]:
    return ["#{:02X}{:02X}{:02X}".format(*(round(255 * v) for v in colorsys.hsv_to_rgb(i / max(n, 1), 0.7, 0.95))) for i in range(n)]


def viewer_config(ds) -> dict:
    try:
        return VIEWERS[ds.name]
    except KeyError:
        raise ValueError(f"no viewer configuration for dataset {ds.name!r}; add one to connexplorer.viz.neuroglancer.VIEWERS") from None


def _seg_source(cfg: dict, version: str):
    src = cfg["segmentation"]
    return [s.format(version=version) for s in src] if isinstance(src, list) else src.format(version=version)


def base_state(ds, viewer: str = "spelunker") -> dict:
    cfg = viewer_config(ds)
    vx, vy, vz = cfg["voxel_nm"]
    layers = [{"type": "image", "source": cfg["image"], "tab": "source", "name": "EM"}] + [dict(l) for l in cfg.get("context_layers", [])]
    state = {
        "dimensions": {"x": [vx * 1e-9, "m"], "y": [vy * 1e-9, "m"], "z": [vz * 1e-9, "m"]},
        "position": list(cfg["position"]),
        "crossSectionScale": 1,
        "projectionScale": cfg["projection_scale"],
        "layers": layers,
        "showDefaultAnnotations": False,
        "showSlices": False,
        "projectionBackgroundColor": "#ffffff",
        "layout": "3d",
    }
    if viewer == "flywire" and "flywire_production" in cfg:
        prod = cfg["flywire_production"]
        state["layers"][0]["source"] = prod["image"]
        state["jsonStateServer"] = prod["jsonStateServer"]
    return state


def segmentation_layer(ds, root_ids, name: str, color: str | None = None, visible: bool = True, viewer: str = "spelunker") -> dict:
    cfg = viewer_config(ds)
    src = _seg_source(cfg, ds.version)
    if viewer == "flywire" and "flywire_production" in cfg:
        src = cfg["flywire_production"]["segmentation"]
    ids = [str(int(r)) for r in np.atleast_1d(root_ids)]
    layer = {"type": "segmentation", "source": src, "tab": "segments", "name": name, "segments": ids, "visible": visible}
    if color:
        layer["segmentColors"] = {i: color for i in ids}
    return layer


def annotation_layer(ds, xyz_nm: np.ndarray, name: str, color: str | None = None, visible: bool = False) -> dict:
    """Point annotations from nanometre coordinates (converted to the viewer's voxel space)."""
    vox = np.asarray(viewer_config(ds)["voxel_nm"])
    pts = np.asarray(xyz_nm, dtype=np.float64).reshape(-1, 3) / vox
    layer = {
        "type": "annotation",
        "source": "local://annotations",
        "tab": "annotations",
        "name": name,
        "visible": visible,
        "annotations": [{"point": p.tolist(), "type": "point", "id": f"{name}:{i}"} for i, p in enumerate(pts)],
    }
    if color:
        layer["annotationColor"] = color
    return layer


def state_url(ds, state: dict, viewer: str = "spelunker") -> str:
    hosts = viewer_config(ds)["hosts"]
    if viewer not in hosts:
        raise ValueError(f"viewer must be one of {sorted(hosts)} for {ds.name}")
    return f"{hosts[viewer]}/#!{quote(json.dumps(state, separators=(',', ':')))}"


def codex_url(ds, root_ids, page_size: int = 10) -> str:
    cfg = viewer_config(ds)
    if "codex" not in cfg:
        raise ValueError(f"Codex has no {ds.name} data")
    ids = "+".join(str(int(r)) for r in np.atleast_1d(root_ids))
    return cfg["codex"].format(ids=ids, n=page_size, version=ds.version)


def neuron_view(ds, idx: np.ndarray, viewer: str = "spelunker", partners: str | None = None, top: int = 5, synapses: bool = False, min_syn: int | None = None, max_points: int = 5000) -> str:
    """URL showing the cells ``idx`` (dense ids), optionally their top partner types and synapse points.

    ``partners``: ``"in"``, ``"out"`` or ``"both"``; the ``top`` types by synapse
    count each get a coloured segmentation layer and, with ``synapses=True``, a
    point layer of the synapses between them and the set (at most ``max_points``).
    """
    if viewer == "codex":
        return codex_url(ds, ds.root_ids[idx])
    idx = np.asarray(idx, dtype=np.int64)
    state = base_state(ds, viewer)
    label = "Self" if len(idx) > 1 else f"Self {ds._type_series[int(idx[0])] or ds.root_ids[int(idx[0])]}"
    state["layers"].append(segmentation_layer(ds, ds.root_ids[idx], label, viewer=viewer))
    selected = label
    if partners:
        dirs = ("in", "out") if partners == "both" else (partners,)
        groups = []
        for d in dirs:
            table = ds.connectivity.partners_of(idx, d, by="type", min_syn=min_syn).drop_nulls("type").head(top)
            for t in table["type"].to_list():
                groups.append((d, t))
        colors = distinct_colors(len(groups))
        for (d, t), color in zip(groups, colors):
            partner_table = ds.connectivity.partners_of(idx, d, min_syn=min_syn).filter(__import__("polars").col("type") == t)
            col = "pre" if d == "in" else "post"
            name = f"{t} -> self" if d == "in" else f"self -> {t}"
            state["layers"].append(segmentation_layer(ds, partner_table[col].to_numpy(), name, color=color, visible=False, viewer=viewer))
            if synapses:
                syn = ds.synapses.of(idx, direction=d, partner=t)
                xyz = syn.select("x_nm", "y_nm", "z_nm").to_numpy()[:max_points]
                state["layers"].append(annotation_layer(ds, xyz, f"synapses {name}", color=color))
    state["selectedLayer"] = {"visible": True, "layer": selected}
    return state_url(ds, state, viewer)


def compartments_view(comp, ds=None, root_id: int | None = None, viewer: str = "spelunker") -> str:
    """URL with the neuron's segmentation plus one point annotation per compartment centre."""
    if ds is None:
        raise ValueError("pass the dataset: comp.view(ds)")
    rid = int(root_id if root_id is not None else comp.neuron.id)
    state = base_state(ds, viewer)
    state["layers"].append(segmentation_layer(ds, [rid], f"neuron {rid}", viewer=viewer))
    centers_nm = comp.centers * 1e3  # compartments are in micrometres
    layer = annotation_layer(ds, centers_nm, f"compartments ({len(comp)})", visible=True)
    t = comp.table
    for i, a in enumerate(layer["annotations"]):
        a["description"] = f"compartment {i}: {t['n_nodes'][i]} nodes, {t['length'][i]:.2f} um, r {t['radius_mean'][i]:.3f} um, parent {t['parent_id'][i]}"
    state["layers"].append(layer)
    state["selectedLayer"] = {"visible": True, "layer": layer["name"]}
    return state_url(ds, state, viewer)
