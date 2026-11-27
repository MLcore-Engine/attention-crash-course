"""Verify downloaded source bytes and source excerpts using the standard library."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    checked = 0
    for repository in manifest["repositories"]:
        base = ROOT / repository["directory"]
        for item in repository["files"]:
            path = base / item["path"]
            data = str(path.readlink()).encode() if item["type"] == "symlink" else path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"Source bytes changed: {path}")
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if blob != item["git_blob"]:
                raise ValueError(f"Git blob mismatch: {path}")
            checked += 1
    excerpts = 0
    pattern = re.compile(
        r"<!-- source: ([^\n]+):(\d+)-(\d+) -->\n"
        r"```[^\n]*\n(.*?)\n```",
        re.DOTALL,
    )
    for lesson in sorted((ROOT / "lessons").glob("*.md")):
        text = lesson.read_text()
        markers = text.count("<!-- source: ")
        matches = list(pattern.finditer(text))
        if len(matches) != markers:
            raise ValueError(f"Malformed source excerpt in {lesson}")
        for match in matches:
            relative, first, last, actual = match.groups()
            lines = (ROOT / relative).read_text().splitlines()
            expected = "\n".join(lines[int(first) - 1:int(last)])
            if actual != expected:
                raise ValueError(f"Excerpt differs from {relative}:{first}-{last}")
            excerpts += 1
    copies = {
        "block_table.py": "vllm/v1/worker/block_table.py",
        "cache_kernels.cu": "csrc/cache_kernels.cu",
        "attention_kernels.cuh": "csrc/attention/attention_kernels.cuh",
    }
    for old, canonical in copies.items():
        if (ROOT / "upstream" / old).read_bytes() != (ROOT / "upstream/source" / canonical).read_bytes():
            raise ValueError(f"Existing source copy differs: {old}")
    print(f"Verified {checked} upstream files, {excerpts} exact source excerpts and 3 existing copies.")


if __name__ == "__main__":
    main()
