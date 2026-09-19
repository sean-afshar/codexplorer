"""Build normalized datasets from vendor downloads.

    from connexplorer import ingest
    ingest.build(ingest.FlyWireSource(), "data/flywire_783", raw="data/flywire/raw")
    ingest.build(ingest.McnsSource("v1.0"), "data/mcns_v1.0", raw="data/mcns/raw")
"""

from pathlib import Path

from connexplorer.ingest.base import BuildError, BuildResult, RawDir, RawFile, Source, build as _build
from connexplorer.ingest.download import RAW_ARCHIVES, download_raw
from connexplorer.ingest.flywire import FlyWireSource
from connexplorer.ingest.mcns import McnsSource, verify_against_weights

SOURCES = {"flywire": FlyWireSource, "mcns": McnsSource}


def build(source: Source, out_dir, raw, **kw) -> BuildResult:
    """See :func:`connexplorer.ingest.base.build`; ``raw`` is the raw download directory."""
    return _build(source, Path(out_dir), Path(raw), **kw)


__all__ = [
    "BuildError", "BuildResult", "FlyWireSource", "McnsSource", "RAW_ARCHIVES", "RawDir", "RawFile",
    "SOURCES", "Source", "build", "download_raw", "verify_against_weights",
]
