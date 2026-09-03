"""Dataset roots, source-file names, and normalized-table paths."""

from __future__ import annotations

from dataclasses import dataclass
import os

from . import schema

DATA_ROOT = os.path.join(".", "data")
DATASET_ALIASES = {
    "flywire": "flywire", "flywire_v783": "flywire", "fw": "flywire",
    "mcns": "mcns", "mcns_v1": "mcns_v1", "male_cns": "mcns",
    "male_cns_v1": "mcns_v1", "malecns": "mcns", "malecns_v1": "mcns_v1",
}
DEFAULT_ROOTS = {name: os.path.join(DATA_ROOT, name) for name in ("flywire", "mcns", "mcns_v1")}
DATASET_FAMILIES = {"flywire": "flywire", "mcns": "mcns", "mcns_v1": "mcns"}
STAT_FOLDER = "stat"
DEFAULT_IO_STAT_TYPE = "basic_pc"

FLYWIRE_DOWNLOAD_FILES = {
    "cell_classification": "classification.csv", "cell_data": "consolidated_cell_types.csv",
    "neurotransmitter_data": "neurons.csv.gz", "visual_cell_data": "visual_neuron_types.csv",
    "columns": "column_assignment.csv.gz", "synapses": "fafb_v783_princeton_synapse_table.csv.gz",
}
MCNS_RELEASE_BY_DATASET = {"mcns": "v0.9", "mcns_v1": "v1.0"}


def _male_cns_download_files(release: str) -> dict[str, str]:
    return {
        "body_annotations": f"body-annotations-male-cns-{release}-minconf-0.5.feather",
        "body_neurotransmitters": f"body-neurotransmitters-male-cns-{release}.feather",
        "syn_partners": f"syn-partners-male-cns-{release}-minconf-0.5.feather",
        "syn_points": f"syn-points-male-cns-{release}-minconf-0.5.feather",
        "tbar_neurotransmitters": f"tbar-neurotransmitters-male-cns-{release}.feather",
        "connectome_weights": f"connectome-weights-male-cns-{release}-minconf-0.5.feather",
        "body_stats": f"body-stats-male-cns-{release}-minconf-0.5.feather",
    }


MCNS_DOWNLOAD_FILES = _male_cns_download_files(MCNS_RELEASE_BY_DATASET["mcns"])
MCNS_V1_DOWNLOAD_FILES = _male_cns_download_files(MCNS_RELEASE_BY_DATASET["mcns_v1"])
DOWNLOAD_FILES_BY_DATASET = {
    "flywire": FLYWIRE_DOWNLOAD_FILES, "mcns": MCNS_DOWNLOAD_FILES, "mcns_v1": MCNS_V1_DOWNLOAD_FILES,
}

MCNS_BODY_ANNOTATION_COLUMNS = ("bodyId", "flywireType", "somaSide", "superclass", "type", "assignedOlHex1", "assignedOlHex2")
MCNS_BODY_NEUROTRANSMITTER_COLUMNS = ("body", "cell_type", "consensus_nt")
MCNS_BODY_ANNOTATION_NAME_MAP = {
    "bodyId": "rid", "assignedOlHex1": "p", "assignedOlHex2": "q", "somaSide": "side", "flywireType": "flywire_type",
}
MCNS_BODY_TRANSMITTER_NAME_MAP = {"body": "rid", "consensus_nt": "nt", "cell_type": "type"}
MCNS_TYPE_VALUE_COLUMNS = ("nt", "superclass", "flywire_type")
MCNS_CELL_SAVE_COLUMNS = ("id", "rid", "type", "side")
MCNS_COLUMN_SOURCE_COLUMNS = ("id", "side", "p", "q")
MCNS_VOXEL_SIZE_NM = 8
MCNS_SYNAPSE_PARTNER_COLUMNS = ("x_pre", "y_pre", "z_pre", "body_pre", "x_post", "y_post", "z_post", "body_post")
FLYWIRE_TYPE_COLUMNS = ("nt_type", "flow", "super_class", "class", "sub_class", "hemilineage", "nerve", "group")
FLYWIRE_TYPE_COLUMN_RENAME = {"nt_type": "nt"}
FLYWIRE_VISUAL_TYPE_COLUMNS = ("family", "subsystem", "category")
FLYWIRE_COLUMN_COLUMNS = ("id", "column_id", "x", "y", "p", "q", "side")
FLYWIRE_PRINCETON_RID_ADD = int(720575940 * 1e9)
FLYWIRE_SYNAPSE_COLUMNS = ("pre_root_id_720575940", "post_root_id_720575940", "ctr_x", "ctr_y", "ctr_z")
MCNS_VISUAL_SUPERCLASSES = ("ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifiugal")
MCNS_NT_TO_IDX = {"unclear": 0, "acetylcholine": 1, "dopamine": 2, "gaba": 3, "glutamate": 4, "octopamine": 5, "serotonin": 6, "histamine": 7}
MCNS_VISUAL_TYPE_COLUMNS = ("side", "flywire_type")
MCNS_COLUMN_COLUMNS = ("id", "p", "q", "side")
MCNS_SYNAPSE_COLUMNS = ("pre_root_id", "post_root_id", "ctr_x", "ctr_y", "ctr_z")


def normalize_dataset_name(name: str) -> str:
    """Return the canonical dataset key."""
    key = str(name).strip().lower()
    if key not in DATASET_ALIASES:
        valid = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"Unknown dataset name {name!r}. Expected one of: {valid}")
    return DATASET_ALIASES[key]


def dataset_family(name: str) -> str:
    """Return the shared dataset family for one canonical or alias name."""
    return DATASET_FAMILIES[normalize_dataset_name(name)]


def is_male_cns_name(name: str) -> bool:
    """Return true for MaleCNS dataset releases."""
    return dataset_family(name) == "mcns"


def download_files_for_dataset(name: str) -> dict[str, str]:
    """Return source download file names for one dataset release."""
    return dict(DOWNLOAD_FILES_BY_DATASET[normalize_dataset_name(name)])


@dataclass
class DatasetPaths:
    """Store raw-download, normalized-table, and statistics paths for one dataset."""

    name: str
    fp_root: str
    fp_download: str
    fp_preprocessed: str
    fp_analysis: str

    @classmethod
    def from_name(cls, name: str, fp_root: str | None = None):
        """Build paths rooted at ``fp_root`` or ``./data/<dataset_name>``."""
        dataset_name = normalize_dataset_name(name)
        root = fp_root or DEFAULT_ROOTS[dataset_name]
        paths = cls(dataset_name, root, os.path.join(root, "download_data"), os.path.join(root, "preprocessed"), os.path.join(root, "analysis"))
        paths.ensure_generated_folders()
        return paths

    @property
    def fp_stat(self) -> str:
        """Precomputed statistics folder under preprocessing."""
        return os.path.join(self.fp_preprocessed, STAT_FOLDER)

    def fp_cell_type_stat_dir(
        self,
        stat_type: str = DEFAULT_IO_STAT_TYPE,
        dir: str | None = None,
    ) -> str:
        """Return the per-cell type IO-stat folder."""
        fp = os.path.join(self.fp_stat, stat_type)
        if dir is not None:
            fp = os.path.join(fp, dir)
        return fp

    def fp_cell_type_stat_file(
        self,
        cell_type: str,
        stat_type: str = DEFAULT_IO_STAT_TYPE,
        dir: str = "input",
    ) -> str:
        """Return the per-cell type IO-stat CSV path."""
        safe_type = str(cell_type).replace(os.path.sep, "_")
        return os.path.join(self.fp_cell_type_stat_dir(stat_type, dir), f"{safe_type}.csv")

    def ensure_generated_folders(self) -> None:
        """Create the release-managed dataset directories."""
        for fp in [self.fp_root, self.fp_download, self.fp_preprocessed, self.fp_analysis, self.fp_stat]:
            os.makedirs(fp, exist_ok=True)
