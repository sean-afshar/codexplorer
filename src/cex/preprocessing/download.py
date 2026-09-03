"""Optional raw-data download helpers for preprocessing notebooks."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import urlretrieve


def download_and_extract(url: str, output_dir, archive_name: str | None = None, overwrite_Q: bool = False) -> Path:
    """Download an archive and extract it into ``output_dir``.

    Convert Dropbox share URLs to direct-download URLs. Delete the archive
    only after extraction succeeds, then return the extraction directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    if parsed.netloc.endswith("dropbox.com"):
        query = dict(parse_qsl(parsed.query))
        query["dl"] = "1"
        url = urlunparse(parsed._replace(query=urlencode(query)))
    archive_name = archive_name or os.path.basename(urlparse(url).path)
    if not archive_name:
        raise ValueError("Provide archive_name when the download URL has no file name")
    archive_path = output_dir / archive_name
    if overwrite_Q or not archive_path.exists():
        urlretrieve(url, archive_path)
    try:
        shutil.unpack_archive(archive_path, output_dir)
    except shutil.ReadError as error:
        raise ValueError(f"{archive_path} is not a supported archive") from error
    archive_path.unlink()
    return output_dir
