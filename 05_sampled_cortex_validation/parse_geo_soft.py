#!/usr/bin/env python3
"""Parse GEO family SOFT files into auditable sample-level manifests."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_soft(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    supplementary: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                if current is not None:
                    current["supplementary_files"] = ";".join(supplementary)
                    records.append(current)
                current = {"gsm": line.split(" = ", 1)[1]}
                supplementary = []
            elif current is None:
                continue
            elif line.startswith("!Sample_title = "):
                current["title"] = line.split(" = ", 1)[1]
            elif line.startswith("!Sample_characteristics_ch1 = "):
                value = line.split(" = ", 1)[1]
                if ": " in value:
                    key, val = value.split(": ", 1)
                    key = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
                    current[key] = val
            elif line.startswith("!Sample_supplementary_file"):
                supplementary.append(line.split(" = ", 1)[1].replace("ftp://", "https://"))
    if current is not None:
        current["supplementary_files"] = ";".join(supplementary)
        records.append(current)
    return records


def standardize(records: list[dict[str, str]], accession: str) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for record in records:
        files = record.get("supplementary_files", "").split(";")
        count_url = next((item for item in files if "Gene.Counts.txt.gz" in item), "")
        sample_id = ""
        if count_url:
            match = re.search(r"_(\d+)_", Path(count_url).name)
            sample_id = match.group(1) if match else ""
        else:
            match = re.search(r"_ID_(\d+)_", record.get("supplementary_files", ""))
            sample_id = match.group(1) if match else ""
        output.append(
            {
                "accession": accession,
                "gsm": record.get("gsm", ""),
                "sample_id": sample_id,
                "title": record.get("title", ""),
                "tissue": record.get("tissue", ""),
                "region": record.get("tissue", "").replace("Cortex ", ""),
                "cell_type": record.get("cell_type", ""),
                "sort_gate": record.get("sort_gate", ""),
                "donor": record.get("donor", ""),
                "condition": record.get("donor_category", ""),
                "age": record.get("age", ""),
                "sex": record.get("sex", ""),
                "pmi_hours": record.get("pmi_hrs", ""),
                "genotype": record.get("genotype", ""),
                "count_url": count_url,
                "supplementary_files": record.get("supplementary_files", ""),
                "biological_replicate": record.get("donor", ""),
                "statistical_unit": "donor",
            }
        )
    return output


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "accession", "gsm", "sample_id", "title", "tissue", "region", "cell_type",
        "sort_gate", "donor", "condition", "age", "sex", "pmi_hours", "genotype",
        "biological_replicate", "statistical_unit", "count_url", "supplementary_files",
    ]
    columns = [item for item in preferred if item in columns] + [item for item in columns if item not in preferred]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> dict:
    donors_by_condition: dict[str, set[str]] = defaultdict(set)
    donors_by_region_condition: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    samples_by_region_condition: Counter = Counter()
    for row in rows:
        donors_by_condition[row["condition"]].add(row["donor"])
        donors_by_region_condition[row["region"]][row["condition"]].add(row["donor"])
        samples_by_region_condition[(row["region"], row["condition"])] += 1
    return {
        "n_samples": len(rows),
        "n_unique_donors": len({row["donor"] for row in rows}),
        "donors_by_condition": {key: len(value) for key, value in sorted(donors_by_condition.items())},
        "donors_by_region_condition": {
            region: {condition: len(donors) for condition, donors in sorted(groups.items())}
            for region, groups in sorted(donors_by_region_condition.items())
        },
        "samples_by_region_condition": {
            f"{region}|{condition}": value
            for (region, condition), value in sorted(samples_by_region_condition.items())
        },
        "cell_types": dict(sorted(Counter(row["cell_type"] for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft", type=Path, required=True)
    parser.add_argument("--accession", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()
    rows = standardize(parse_soft(args.soft), args.accession)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_csv)
    args.out_summary.write_text(json.dumps(summarize(rows), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summarize(rows), ensure_ascii=False))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Parse GEO SOFT metadata into structured sample/phenotype tables for downstream validation analyses.
# Input source/location: Downloaded GEO SOFT family files from the configured data/cache directory.
# Output location: Parsed sample metadata tables in the configured project input/output directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Read SOFT records; extract sample identifiers and characteristic fields; normalize metadata keys; assemble tabular records; export parsed metadata.
# Log location: No dedicated log file; parser status is written to standard output.
# =============================================================================
