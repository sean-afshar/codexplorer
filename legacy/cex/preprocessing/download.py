"""Optional dataset download helpers for public notebooks."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import urlretrieve


def download_and_extract(
    url: str,
    output_dir,
    archive_name: str | None = None,
    overwrite_Q: bool = False,
    strip_single_root_Q: bool = False,
) -> Path:
    """Download an archive and extract it into ``output_dir``.

    Convert Dropbox share URLs to direct-download URLs. Delete the archive
    only after extraction succeeds, then return the extraction directory.
    When requested, remove one enclosing archive directory so its contents
    are placed directly in ``output_dir``.
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
        if strip_single_root_Q:
            with tempfile.TemporaryDirectory(prefix="cex_extract_", dir=output_dir.parent) as temp_dir:
                temp_dir = Path(temp_dir)
                shutil.unpack_archive(archive_path, temp_dir)
                entries = [path for path in temp_dir.iterdir() if path.name != "__MACOSX"]
                source_dir = entries[0] if len(entries) == 1 and entries[0].is_dir() else temp_dir
                for source in source_dir.iterdir():
                    if source.name == "__MACOSX":
                        continue
                    destination = output_dir / source.name
                    if source.is_dir():
                        shutil.copytree(source, destination, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source, destination)
        else:
            shutil.unpack_archive(archive_path, output_dir)
    except shutil.ReadError as error:
        raise ValueError(f"{archive_path} is not a supported archive") from error
    archive_path.unlink()
    return output_dir
