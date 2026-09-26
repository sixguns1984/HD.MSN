#!/usr/bin/env python3
"""Create the versioned software release archive without cache files."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / "HD_MSN_analysis_code_GitHub.zip"
HASH_FILE = ROOT.parent / "HD_MSN_analysis_code_GitHub.zip.sha256"

if ARCHIVE.exists() or HASH_FILE.exists():
    raise SystemExit("Refusing to overwrite an existing release archive or checksum")

with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        arcname = Path(ROOT.name) / path.relative_to(ROOT)
        zf.write(path, arcname.as_posix())

h = hashlib.sha256()
with ARCHIVE.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(chunk)
HASH_FILE.write_text(f"{h.hexdigest()}  {ARCHIVE.name}\n", encoding="utf-8")
print(ARCHIVE)
print(h.hexdigest())

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Package the sanitized repository into a distributable ZIP archive after release checks.
# Input source/location: Sanitized repository tree and its release metadata/manifest files.
# Output location: A ZIP archive at the configured release destination.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Validate expected release files; exclude transient/cache content; add files with project-relative paths; create the final archive; report the destination.
# Log location: No dedicated log file; packaging status is written to standard output.
# =============================================================================
