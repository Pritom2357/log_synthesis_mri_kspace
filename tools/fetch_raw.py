"""
Pull single files out of the remote M4Raw zip with HTTP range requests.

    python tools/fetch_raw.py list [motion]        # smallest files in the archive
    python tools/fetch_raw.py get <name> [...]     # specific files into data/raw/
    python tools/fetch_raw.py auto 3 [motion]      # the n smallest
"""
from __future__ import annotations
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

# M4Raw, Zenodo record 7523691, CC-BY-4.0
ARCHIVES = {
    "val":    "https://zenodo.org/records/7523691/files/M4RawV1.1_multicoil_val.zip",   # clean scans
    "motion": "https://zenodo.org/records/7523691/files/M4RawV1.1_motion.zip",          # real patient motion
    "train":  "https://zenodo.org/records/7523691/files/M4RawV1.1_multicoil_train.zip", # 12.5 GB, rarely needed
}
ARCHIVE = ARCHIVES["val"]
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
UA = {"User-Agent": "mri-kspace-simulator/1.0 (student project)"}


class HttpRangeFile(io.RawIOBase):
    """Read-only file over HTTP range requests; enough for zipfile to read the central directory."""

    def __init__(self, url: str):
        self.url = url
        self._pos = 0
        with urlopen(Request(url, method="HEAD", headers=UA), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges", "").lower() != "bytes":
                print("  note: server did not advertise range support, trying anyway")

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, min(self.size, base + offset))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self._pos
        n = min(n, self.size - self._pos)
        if n <= 0:
            return b""
        first, last = self._pos, self._pos + n - 1
        req = Request(self.url, headers={**UA, "Range": f"bytes={first}-{last}"})
        with urlopen(req, timeout=180) as r:
            data = r.read()
        self._pos += len(data)
        return data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


def open_archive(url: str = ARCHIVE) -> zipfile.ZipFile:
    return zipfile.ZipFile(HttpRangeFile(url))


def list_entries(url: str = ARCHIVE) -> list[zipfile.ZipInfo]:
    with open_archive(url) as z:
        return [i for i in z.infolist() if not i.is_dir() and i.file_size > 0]


def fetch(names: list[str], url: str = ARCHIVE) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    with open_archive(url) as z:
        for name in names:
            info = z.getinfo(name)
            dest = OUT / Path(name).name
            print(f"  fetching {name}  ({info.file_size/1e6:.1f} MB uncompressed)")
            with z.open(info) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"    -> {dest}  ({dest.stat().st_size/1e6:.1f} MB on disk)")
            written.append(dest)
    return written


def _pick(args: list[str]) -> tuple[list[str], str]:
    if args and args[-1] in ARCHIVES:
        return args[:-1], ARCHIVES[args[-1]]
    return args, ARCHIVE


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "list"
    rest, url = _pick(argv[2:])

    if cmd == "list":
        entries = list_entries(url)
        print(f"{len(entries)} files in the archive (showing 15 smallest):\n")
        for i in sorted(entries, key=lambda e: e.file_size)[:15]:
            print(f"  {i.file_size/1e6:7.1f} MB   {i.filename}")
        total = sum(e.file_size for e in entries)
        print(f"\n  whole archive would be {total/1e9:.1f} GB uncompressed")
        print("  fetch one with:  python tools/fetch_raw.py get <filename>")
        return 0

    if cmd == "get":
        if not rest:
            print("give at least one filename; see 'list'")
            return 2
        fetch(rest, url)
        return 0

    if cmd == "auto":
        n = int(rest[0]) if rest else 2
        entries = [e for e in list_entries(url) if e.filename.endswith(".h5")]
        fetch([e.filename for e in sorted(entries, key=lambda e: e.file_size)[:n]], url)
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
