"""``python -m connexplorer.ingest build flywire --raw data/flywire/raw --out data/flywire_783``."""

from __future__ import annotations

import argparse
import sys

from connexplorer import ingest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="connexplorer.ingest")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build normalized tables from raw files")
    b.add_argument("dataset", choices=sorted(ingest.SOURCES))
    b.add_argument("--raw", required=True, help="raw download directory")
    b.add_argument("--out", required=True, help="dataset directory; tables go in <out>/tables")
    b.add_argument("--version", help="FlyWire snapshot (default 783) or MCNS release (default v0.9)")
    b.add_argument("--no-by-post", action="store_true", help="skip the post-sorted synapse copy")
    b.add_argument("--no-hash", action="store_true", help="do not sha256 the raw files")

    d = sub.add_parser("download", help="fetch a raw bundle")
    d.add_argument("dataset", choices=sorted(ingest.RAW_ARCHIVES))
    d.add_argument("--raw", required=True)

    a = p.parse_args(argv)
    if a.cmd == "download":
        files = ingest.download_raw(a.dataset, a.raw)
        print(f"{len(files)} files in {a.raw}")
        return 0
    src = ingest.SOURCES[a.dataset](a.version) if a.version else ingest.SOURCES[a.dataset]()
    try:
        res = ingest.build(src, a.out, a.raw, by_post=not a.no_by_post, hash_sources=not a.no_hash)
    except ingest.BuildError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"ok: {res.tables_dir} ({sum(res.timings.values()):.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
