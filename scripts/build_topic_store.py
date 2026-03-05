from __future__ import annotations

import json
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import BinaryIO, cast


ENRICHR_LIB = "MSigDB_Hallmark_2020"
ENRICHR_URL = (
    "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=" + ENRICHR_LIB
)

def fetch_hallmark() -> dict[str, list[str]]:
    with closing(cast(BinaryIO, urllib.request.urlopen(ENRICHR_URL))) as resp:
        raw_bytes = resp.read()
    raw = raw_bytes.decode("utf-8")

    entries: list[tuple[str, list[str]]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if not parts:
            continue
        name = parts[0].strip()
        # Enrichr format is: term \t\t gene1 \t gene2 ...
        genes = [p.strip() for p in parts[2:] if p.strip()]
        if not genes:
            continue
        # Deduplicate while preserving order
        dedup: list[str] = []
        seen: set[str] = set()
        for g in genes:
            gu = g.upper()
            if gu in seen:
                continue
            seen.add(gu)
            dedup.append(gu)
        entries.append((name, dedup))

    # Stable anonymized topic IDs: topic_1, topic_2, ...
    entries.sort(key=lambda item: item[0].upper())
    topics: dict[str, list[str]] = {}
    for idx, (_, genes) in enumerate(entries, start=1):
        topics[f"topic_{idx}"] = genes

    return topics


def main() -> int:
    topics = fetch_hallmark()
    out = {
        "topic_count": len(topics),
        "topics": topics,
    }

    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "topic_gene_hallmark_2020.json"
    _ = out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote: {out_path} (topics={len(topics)})")

    if len(topics) != 50:
        print("WARNING: Expected 50 Hallmark topics; got", len(topics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
