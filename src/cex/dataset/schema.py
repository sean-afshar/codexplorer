"""Normalized dataset schema and file-name constants."""

DEFAULT_UNKNOWN = "NAN"
UNKNOWN_TYPE_NAMES = frozenset({"", "nan", "none", "unknown", "unclassified", "uncertain", "unclear"})

NEUROTRANSMITTERS = ("NAN", "ACH", "DA", "GABA", "GLUT", "OCT", "SER", "HIS")
NEUROTRANSMITTER_TO_IDX = {name: idx for idx, name in enumerate(NEUROTRANSMITTERS)}
NEUROTRANSMITTER_TO_SIGN = {"ACH": 1, "GABA": -1, "GLUT": -1, "HIS": -1}

SIDE_UNKNOWN = 0
SIDE_LEFT = -1
SIDE_RIGHT = 1
SIDE_TO_IDX = {
    "left": SIDE_LEFT,
    "l": SIDE_LEFT,
    "-1": SIDE_LEFT,
    -1: SIDE_LEFT,
    "right": SIDE_RIGHT,
    "r": SIDE_RIGHT,
    "1": SIDE_RIGHT,
    1: SIDE_RIGHT,
    "unknown": SIDE_UNKNOWN,
    "nan": SIDE_UNKNOWN,
    "none": SIDE_UNKNOWN,
    "0": SIDE_UNKNOWN,
    0: SIDE_UNKNOWN,
    "": SIDE_UNKNOWN,
}
SIDE_IDX_TO_STR = {SIDE_UNKNOWN: "unknown", SIDE_LEFT: "L", SIDE_RIGHT: "R"}

CELL_DATA_FILE = "cell_data.parquet"
TYPE_DATA_FILE = "type_data.parquet"
VISUAL_TYPE_DATA_FILE = "visual_type_data.parquet"
COLUMN_DATA_FILE = "columns_data.npz"
SYNAPSE_DATA_FILE = "synapses.parquet"
CELL_TO_CELL_SYN_COUNT_FILE = "cell_to_cell_syn_count.parquet"
TYPE_TO_TYPE_SYN_COUNT_FILE = "type_to_type_syn_count.npz"

NEUROTRANSMITTER_SYNONYMS = {
    "ACETYLCHOLINE": "ACH",
    'acetylcholine': "ACH",
    "DOPAMINE": "DA",
    'dopamine': "DA",
    "GAMMA-AMINOBUTYRIC ACID": "GABA",
    'gamma-aminobutyric acid': "GABA",
    'gaba': "GABA",
    "GLUTAMATE": "GLUT",
    'glutamate': "GLUT",
    "OCTOPAMINE": "OCT",
    'octopamine': "OCT",
    "SEROTONIN": "SER",
    'serotonin': "SER",
    "HISTAMINE": "HIS",
    'histamine': "HIS",
}

TYPE_NEUROTRANSMITTER_GT = {
    # There are other conflicts, mostly GLUT versus GABA, where ground truth is unclear.
    "R1-6": "HIS",
    "R7": "HIS",
    "R8": "HIS",
    "Dm1": "GLUT",
    "Dm6": "GLUT",
    "Dm8": "GLUT",
    "Dm9": "GLUT",
    "Dm12": "GLUT",
    "Dm16": "GLUT",
    "Dm17": "GLUT",
    "Dm19": "GLUT",
    "Dm20": "GLUT",
    "T1": "GLUT",
}
