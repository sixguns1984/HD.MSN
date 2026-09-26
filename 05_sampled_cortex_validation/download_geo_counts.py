#!/usr/bin/env python3
"""Download the small GSE233408 Gene.Counts files listed in a GEO manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def valid_gzip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 100:
        return False
    try:
        with gzip.open(path, "rb") as handle:
            return bool(handle.read(16))
    except OSError:
        return False


def fetch(url: str, destination: Path) -> tuple[str, int, str]:
    if not valid_gzip(destination):
        temporary = destination.with_suffix(destination.suffix + ".part")
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "Codex-reproducibility-audit/1.0"})
                with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                if not valid_gzip(temporary):
                    raise RuntimeError(f"Downloaded file is not a valid gzip stream: {url}")
                temporary.replace(destination)
                break
            except Exception as error:  # GEO intermittently returns 429/503.
                last_error = error
                temporary.unlink(missing_ok=True)
                if attempt == 5:
                    raise
                time.sleep((2 ** attempt) + random.random())
        if not destination.exists():
            raise RuntimeError(f"Download failed: {url}: {last_error}")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return destination.name, destination.stat().st_size, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    jobs = []
    for row in rows:
        url = row.get("count_url", "")
        if url:
            jobs.append((url, args.destination / Path(url).name))
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch, url, path): (url, path) for url, path in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            name, size, digest = future.result()
            results.append({"file": name, "bytes": size, "sha256": digest})
            if index % 20 == 0 or index == len(futures):
                print(f"verified {index}/{len(futures)}")
    results.sort(key=lambda row: row["file"])
    args.checksums.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Download public GEO count matrices and supporting files required by sampled-cortex validation.
# Input source/location: GEO accession identifiers and destination paths defined in the script/configuration.
# Output location: Downloaded GEO files in the configured public-data cache/input directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Resolve GEO URLs; stream/download files; verify presence/size where implemented; retain files for downstream parsing.
# Log location: No dedicated log file; download status is written to standard output.
# =============================================================================
