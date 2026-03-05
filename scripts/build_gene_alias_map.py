"""Download HGNC complete gene set and build alias → canonical symbol map.

Output: text2sql_demo/data/hgnc_alias_map.json

Format:
{
  "HER2": "ERBB2",
  "P53": "TP53",
  "TP53": "TP53",   // canonical maps to itself
  ...
}

Mapping sources (from HGNC TSV):
  - symbol        → canonical (maps to itself)
  - alias_symbol  → canonical (pipe-delimited synonyms)
  - prev_symbol   → canonical (pipe-delimited previous names)

Only "Approved" genes are included. Withdrawn entries are skipped.
Conflicts (alias maps to multiple symbols) are dropped to avoid ambiguity.
"""
from __future__ import annotations

import csv
import io
import json
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import BinaryIO, cast

HGNC_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "hgnc_alias_map.json"


def _parse_pipe_field(value: str) -> list[str]:
    """Parse a pipe-delimited HGNC field like '"FOO|BAR"' into ['FOO', 'BAR']."""
    value = value.strip().strip('"')
    if not value:
        return []
    return [s.strip() for s in value.split("|") if s.strip()]


def download_hgnc() -> str:
    print(f"Downloading HGNC complete set from {HGNC_URL} ...")
    with closing(cast(BinaryIO, urllib.request.urlopen(HGNC_URL))) as resp:
        raw = resp.read()
    text = raw.decode("utf-8")
    print(f"  Downloaded {len(raw):,} bytes, {text.count(chr(10)):,} lines.")
    return text


def build_alias_map(tsv_text: str) -> dict[str, str]:
    """Build uppercase alias → canonical symbol mapping.

    Strategy:
      1. Canonical symbols always map to themselves (highest priority).
      2. alias_symbol and prev_symbol entries map to canonical, unless
         the alias is already claimed by another canonical (conflict → drop).
    """
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")

    # Pass 1: collect all approved canonical symbols
    rows: list[dict[str, str]] = []
    canonical_set: set[str] = set()

    for row in reader:
        status = (row.get("status") or "").strip()
        if status != "Approved":
            continue
        symbol = (row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        canonical_set.add(symbol)
        rows.append(row)

    print(f"  Approved genes: {len(canonical_set):,}")

    # Pass 2: build alias map
    alias_map: dict[str, str] = {}
    conflicts: set[str] = set()

    # Canonical symbols map to themselves (cannot be overridden)
    for sym in canonical_set:
        alias_map[sym] = sym

    # Aliases and previous symbols
    for row in rows:
        symbol = (row.get("symbol") or "").strip().upper()
        aliases = _parse_pipe_field(row.get("alias_symbol", ""))
        prev_symbols = _parse_pipe_field(row.get("prev_symbol", ""))

        for alt in aliases + prev_symbols:
            alt_upper = alt.upper()
            if alt_upper in canonical_set:
                # This alias is itself a canonical symbol for another gene → skip
                continue
            if alt_upper in conflicts:
                continue
            if alt_upper in alias_map and alias_map[alt_upper] != symbol:
                # Conflict: alias points to multiple canonical symbols → drop
                conflicts.add(alt_upper)
                del alias_map[alt_upper]
                continue
            alias_map[alt_upper] = symbol

    print(f"  Total mappings (base): {len(alias_map):,} (including {len(canonical_set):,} self-mappings)")
    print(f"  Dropped conflicts: {len(conflicts):,}")

    # Pass 3: auto-expand hyphenated aliases (KI-67 → KI67)
    dehyphen_added = 0
    for alt, sym in list(alias_map.items()):
        if "-" in alt:
            compact = alt.replace("-", "")
            if compact and compact not in alias_map and compact not in conflicts:
                alias_map[compact] = sym
                dehyphen_added += 1
    print(f"  De-hyphenated expansions: {dehyphen_added:,}")

    # Pass 4: manual common aliases not in HGNC
    _MANUAL: dict[str, str] = {
        "P21": "CDKN1A",
        "P53": "TP53",
        "NFKB": "NFKB1",
        "CMYC": "MYC",
        "IFNGAMMA": "IFNG",
        "TGFBETA": "TGFB1",
        "PD1": "PDCD1",
    }
    manual_added = 0
    for alt, sym in _MANUAL.items():
        key = alt.upper()
        if key not in alias_map and sym in canonical_set:
            alias_map[key] = sym
            manual_added += 1
    print(f"  Manual alias additions: {manual_added:,}")
    return alias_map


def main() -> None:
    tsv_text = download_hgnc()
    alias_map = build_alias_map(tsv_text)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(alias_map, f, indent=0, ensure_ascii=True, sort_keys=True)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"  Written to {OUTPUT_FILE} ({size_mb:.1f} MB)")

    # Spot-check
    checks = [
        ("P53", "TP53"), ("HER2", "ERBB2"), ("KI67", "MKI67"),
        ("PD1", "PDCD1"), ("MYC", "MYC"), ("BRCA1", "BRCA1"),
    ]
    print("\n  Spot-check:")
    for alias, expected in checks:
        actual = alias_map.get(alias.upper(), "NOT FOUND")
        status = "✓" if actual == expected else "✗"
        print(f"    {status} {alias} → {actual} (expected {expected})")


if __name__ == "__main__":
    main()
