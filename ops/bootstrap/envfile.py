"""Rewrite only the value of named keys in .env, byte-preserving everything else.

python-dotenv's set_key quotes values by default, which would diverge from the
32 hand-written unquoted values already in the file. Hand-rolled instead.
"""

from __future__ import annotations

import difflib
import os
import pathlib
import re

_LINE = re.compile(r"^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z0-9_]*)(\s*=)(.*?)(\r?\n?)$")
_SAFE = re.compile(r"^[A-Za-z0-9_.:+/@=-]*$")


def write_env(updates: dict[str, str], path: str | os.PathLike[str] = ".env",
              dry_run: bool = False) -> list[str]:
    """Set each key to its new value. Returns the diff lines. Raises if a key is absent."""
    p = pathlib.Path(path)
    src = p.read_text()
    out: list[str] = []
    touched: set[str] = set()

    for ln in src.splitlines(keepends=True):
        m = _LINE.match(ln)
        if m and m.group(2) in updates:
            key, val = m.group(2), updates[m.group(2)]
            if not _SAFE.match(val):
                raise ValueError(f"{key}: value would need quoting: {val!r}")
            out.append(f"{m.group(1)}{key}{m.group(3)}{val}{m.group(5) or os.linesep}")
            touched.add(key)
        else:
            out.append(ln)

    if missing := set(updates) - touched:
        raise KeyError(f"keys absent from {p}: {sorted(missing)}")

    new = "".join(out)
    if len(new.splitlines()) != len(src.splitlines()):
        raise AssertionError("line count changed; refusing to write")

    diff = [d for d in difflib.unified_diff(src.splitlines(), new.splitlines(),
                                            "before", "after", lineterm="", n=0)
            if d.startswith(("+", "-")) and not d.startswith(("+++", "---"))]
    if not dry_run:
        # create at 0600 BEFORE any secret is written, then swap atomically
        tmp = str(p) + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(new)
        os.replace(tmp, p)
    return diff
