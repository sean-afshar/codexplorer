"""Fetch vendor archives into a raw directory."""

from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import urlopen

FLYWIRE_RAW_URL = (
    "https://www.dropbox.com/scl/fo/a01t6njrz0hs8b7m47nys/AHSSrFfESyg903zDTCtnOx4"
    "?rlkey=gm8k4sakl70ynxoviidw2w56h&dl=1"
)

# dataset -> (url, archive name). The FlyWire bundle is the Codex download set
# (3.0 GB) minus skeletons; Male CNS files come from neuPrint and have no bundle yet.
RAW_ARCHIVES: dict[str, tuple[str, str]] = {
    "flywire": (FLYWIRE_RAW_URL, "Flywire_raw_data.zip"),
}


def _direct_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("dropbox.com"):
        query = dict(parse_qsl(parsed.query))
        query["dl"] = "1"
        return urlunparse(parsed._replace(query=urlencode(query)))
    return url


def fetch(url: str, dest: Path, progress: Callable[[str], None] | None = None, chunk: int = 8 << 20) -> Path:
    """Stream ``url`` to ``dest``; returns ``dest``."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(_direct_url(url)) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while blk := resp.read(chunk):
            out.write(blk)
            done += len(blk)
            if progress and total:
                progress(f"\r{dest.name}: {done / 1e9:.2f} / {total / 1e9:.2f} GB")
    if progress:
        progress("\n")
    return dest


def extract_flat(archive: Path, dest: Path) -> list[Path]:
    """Unzip ``archive`` into ``dest``, dropping a single enclosing folder and macOS junk."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir() and not m.filename.startswith("__MACOSX")]
        tops = {m.filename.split("/")[0] for m in members if "/" in m.filename.strip("/")}
        strip = tops.pop() + "/" if len(tops) == 1 and all(m.filename.startswith(next(iter(tops)) + "/") for m in members) else ""
        for m in members:
            name = m.filename[len(strip):].lstrip("/")
            if not name or name.endswith(".DS_Store"):
                continue
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 8 << 20)
            out.append(target)
    return out


def download_raw(dataset: str, raw_dir: Path, *, keep_archive: bool = False, progress=sys.stderr.write) -> list[Path]:
    """Download and unpack the raw bundle for ``dataset`` into ``raw_dir``."""
    if dataset not in RAW_ARCHIVES:
        raise ValueError(f"no raw archive known for {dataset!r}; known: {sorted(RAW_ARCHIVES)}")
    url, name = RAW_ARCHIVES[dataset]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="connexplorer_dl_", dir=raw_dir) as tmp:
        archive = fetch(url, Path(tmp) / name, progress=progress)
        files = extract_flat(archive, raw_dir)
        if keep_archive:
            shutil.move(str(archive), raw_dir / name)
    return files
