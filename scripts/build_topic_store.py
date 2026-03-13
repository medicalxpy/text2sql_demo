from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, cast

from ..gene_normalizer import normalize_gene
from ..pathway_asset_layout import PathwayAssetLayout, load_default_pathway_asset_layout


MSIGDB_HALLMARK_SOURCE = "msigdb_hallmark"
ENRICHR_LIB = "MSigDB_Hallmark_2020"
ENRICHR_URL = (
    "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=" + ENRICHR_LIB
)
MSIGDB_COLLECTION_URL = "https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#H"
MSIGDB_LICENSE_TERMS_URL = "https://www.gsea-msigdb.org/gsea/license_terms_list.jsp"
CATALOG_SCHEMA_VERSION = "pathway_gene_catalog_v1"


def fetch_hallmark_raw() -> bytes:
    with closing(cast(BinaryIO, urllib.request.urlopen(ENRICHR_URL))) as resp:
        return resp.read()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pinned MSigDB Hallmark raw and normalized pathway assets."
    )
    _ = parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "pathway_gene_assets_v1.json"),
        help="Path to the pathway asset config JSON.",
    )
    _ = parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download the raw MSigDB Hallmark export even if cached.",
    )
    return parser.parse_args(argv)


def parse_hallmark_records(raw_text: str, *, source_version: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if not parts:
            continue

        raw_term_name = parts[0].strip()
        term_id = _normalize_term_id(raw_term_name)
        genes = _canonical_hgnc_genes(parts[2:])
        if not term_id or not genes:
            continue

        term_name = _humanize_hallmark_name(term_id)
        record: dict[str, object] = {
            "term_id": term_id,
            "term_name": term_name,
            "aliases": _build_aliases(term_id, term_name),
            "genes": genes,
            "source": MSIGDB_HALLMARK_SOURCE,
            "version": source_version,
        }
        records.append(record)

    records.sort(key=lambda record: str(record["term_id"]))
    return records


def fetch_hallmark(
    *,
    config_path: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, list[str]]:
    layout = (
        load_default_pathway_asset_layout(Path(config_path).resolve())
        if config_path is not None
        else load_default_pathway_asset_layout()
    )
    hallmark = layout.source(MSIGDB_HALLMARK_SOURCE)
    raw_path = hallmark.raw_dir / f"{ENRICHR_LIB}.txt"

    raw_bytes = _ensure_raw_asset(raw_path=raw_path, refresh=refresh)

    raw_text = raw_bytes.decode("utf-8")
    records = parse_hallmark_records(raw_text, source_version=hallmark.source_version)
    _write_catalog(catalog_path=hallmark.runtime_catalog_path, records=records)
    _update_manifest(
        layout=layout,
        manifest_path=layout.manifest_path,
        source_version=hallmark.source_version,
        download_url=hallmark.download_url,
        raw_path=raw_path,
        catalog_path=hallmark.runtime_catalog_path,
    )

    topics = build_legacy_topics(records)
    return topics


def _ensure_raw_asset(*, raw_path: Path, refresh: bool) -> bytes:
    if raw_path.exists() and not refresh:
        return raw_path.read_bytes()

    raw_bytes = fetch_hallmark_raw()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _ = raw_path.write_bytes(raw_bytes)
    return raw_bytes


def build_legacy_topics(records: list[dict[str, object]]) -> dict[str, list[str]]:
    topics: dict[str, list[str]] = {}
    for idx, record in enumerate(records, start=1):
        genes = cast(list[str], record["genes"])
        topics[f"topic_{idx}"] = genes

    return topics


def _canonical_hgnc_genes(raw_genes: list[str]) -> list[str]:
    genes: list[str] = []
    seen: set[str] = set()
    for raw_gene in raw_genes:
        candidate = raw_gene.strip()
        if not candidate:
            continue
        canonical, resolved = normalize_gene(candidate)
        if not resolved:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        genes.append(canonical)
    return genes


def _normalize_term_id(raw_term_name: str) -> str:
    candidate = re.sub(r"\s+", "_", raw_term_name.strip().upper())
    candidate = re.sub(r"[^A-Z0-9_]+", "_", candidate)
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    if candidate and not candidate.startswith("HALLMARK_"):
        candidate = f"HALLMARK_{candidate}"
    return candidate


def _humanize_hallmark_name(term_id: str) -> str:
    pieces = [piece for piece in term_id.split("_") if piece]
    if not pieces:
        return "Hallmark"

    if pieces[0] == "HALLMARK":
        title_pieces = ["Hallmark"] + [piece.capitalize() for piece in pieces[1:]]
    else:
        title_pieces = [piece.capitalize() for piece in pieces]
    return " ".join(title_pieces)


def _build_aliases(term_id: str, term_name: str) -> list[str]:
    aliases: list[str] = []
    for candidate in (term_name, term_id, term_id.replace("_", " ")):
        if candidate and candidate not in aliases:
            aliases.append(candidate)
    return aliases


def _write_catalog(*, catalog_path: Path, records: list[dict[str, object]]) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_obj: dict[str, object] = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "source_name": MSIGDB_HALLMARK_SOURCE,
        "records": records,
    }
    _ = catalog_path.write_text(
        json.dumps(catalog_obj, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _update_manifest(
    *,
    layout: PathwayAssetLayout,
    manifest_path: Path,
    source_version: str,
    download_url: str,
    raw_path: Path,
    catalog_path: Path,
) -> None:
    manifest_obj = _read_json_object(manifest_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    source_record: dict[str, object] = {
        "source_name": MSIGDB_HALLMARK_SOURCE,
        "source_version": source_version,
        "download_url": download_url,
        "build_timestamp_utc": generated_at,
        "hgnc_version": layout.hgnc.version,
        "source_metadata": {
            "collection": "H",
            "collection_name": "Hallmark",
            "collection_url": MSIGDB_COLLECTION_URL,
            "license_terms_url": MSIGDB_LICENSE_TERMS_URL,
            "registration_required": "true",
            "raw_export_library": ENRICHR_LIB,
            "raw_export_url": ENRICHR_URL,
            "raw_export_note": "Mirrored from the Enrichr text export while preserving MSigDB collection and license provenance.",
            "hgnc_alias_map_path": _repo_relative_path(layout.hgnc.alias_map_path),
            "hgnc_alias_map_sha256": _sha256_path(layout.hgnc.alias_map_path),
        },
        "filter_rules": [
            "preserve_hallmark_source_specific_term_ids",
            "preserve_readable_term_names",
            "deduplicate_genes_case_insensitively",
        ],
        "files": [
            {
                "asset_role": "raw_msigdb_hallmark_gene_sets",
                "relative_path": _repo_relative_path(raw_path),
                "checksum_sha256": _sha256_path(raw_path),
            },
            {
                "asset_role": "normalized_runtime_catalog",
                "relative_path": _repo_relative_path(catalog_path),
                "checksum_sha256": _sha256_path(catalog_path),
            },
        ],
    }

    updated_sources: list[dict[str, object]] = []
    replaced = False
    for source_obj in _manifest_sources(manifest_obj):
        if source_obj.get("source_name") == MSIGDB_HALLMARK_SOURCE:
            updated_sources.append(source_record)
            replaced = True
            continue
        updated_sources.append(source_obj)
    if not replaced:
        updated_sources.append(source_record)

    manifest_obj["generated_at_utc"] = generated_at
    manifest_obj["sources"] = updated_sources
    _ = manifest_path.write_text(
        json.dumps(manifest_obj, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _manifest_sources(manifest_obj: dict[str, object]) -> list[dict[str, object]]:
    sources_obj = manifest_obj.get("sources")
    if not isinstance(sources_obj, list):
        return []

    raw_sources = cast(list[object], sources_obj)
    sources: list[dict[str, object]] = []
    for source_obj in raw_sources:
        if isinstance(source_obj, dict):
            sources.append(cast(dict[str, object], source_obj))
    return sources


def _read_json_object(path: Path) -> dict[str, object]:
    raw_obj = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw_obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast(dict[str, object], raw_obj)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative_path(path: Path) -> str:
    package_root = Path(__file__).resolve().parents[1]
    return path.relative_to(package_root).as_posix()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    refresh = cast(bool, args.refresh)
    topics = fetch_hallmark(config_path=cast(str, args.config), refresh=refresh)
    out = {
        "topic_count": len(topics),
        "topics": topics,
    }

    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "topic_gene_hallmark_2020.json"
    _ = out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path} (topics={len(topics)})")

    if len(topics) != 50:
        print("WARNING: Expected 50 Hallmark topics; got", len(topics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
