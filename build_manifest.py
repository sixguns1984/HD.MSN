#!/usr/bin/env python3
"""Write a deterministic file manifest and SHA-256 checksum list."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDE = {"MANIFEST.csv", "SHA256SUMS.txt", "analysis_code_release.zip"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


files = [p for p in ROOT.rglob("*") if p.is_file() and p.name not in EXCLUDE and "__pycache__" not in p.parts]
files.sort(key=lambda p: p.relative_to(ROOT).as_posix())
rows = []
for path in files:
    rel = path.relative_to(ROOT).as_posix()
    rows.append({"relative_path": rel, "size_bytes": path.stat().st_size, "sha256": digest(path)})

with (ROOT / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
    writer.writeheader()
    writer.writerows(rows)

with (ROOT / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as handle:
    for row in rows:
        handle.write(f"{row['sha256']}  {row['relative_path']}\n")

print(f"Manifested {len(rows)} files")

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Generate a reproducible release manifest and SHA-256 checksum list for the sanitized code package.
# Input source/location: All release files beneath the repository root, excluding generated manifest/checksum files as configured.
# Output location: MANIFEST.csv and SHA256SUMS.txt in the repository root.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Enumerate files; calculate relative paths, byte sizes, and SHA-256 hashes; sort records; write manifest/checksum outputs.
# Log location: No dedicated log file; completion status is written to standard output.
# =============================================================================
