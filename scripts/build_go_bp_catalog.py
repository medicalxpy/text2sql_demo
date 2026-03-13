from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, cast
from urllib.parse import urlparse

from ..gene_normalizer import normalize_gene
from ..pathway_asset_layout import PathwayAssetLayout, load_default_pathway_asset_layout


GO_BP_SOURCE = "go_bp"
CATALOG_SCHEMA_VERSION = "pathway_gene_catalog_v1"
GO_BP_NAMESPACE = "biological_process"
GO_BP_ASPECT = "P"
MAX_RUNTIME_GENES = 1000
SYNONYM_PATTERN = re.compile(r'^synonym:\s+"([^"]+)"')


@dataclass(frozen=True)
class GoTerm:
    term_id: str
    term_name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class GoBuildMetadata:
    ontology_version: str
    annotation_generated_date: str
    annotation_go_version: str
    source_version: str


@dataclass
class GoBuildStats:
    kept_records: int = 0
    skipped_missing_term: int = 0
    skipped_not_qualifier: int = 0
    skipped_nd_evidence: int = 0
    skipped_unresolved_genes: int = 0
    dropped_empty_terms: int = 0
    dropped_broad_terms: int = 0
    total_annotations: int = 0
    total_term_gene_edges: int = 0
    total_runtime_genes: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pinned GO Biological Process raw and normalized assets."
    )
    _ = parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "pathway_gene_assets_v1.json"),
        help="Path to the pathway asset config JSON.",
    )
    _ = parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download the GO ontology and annotation files even if cached.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refresh = cast(bool, args.refresh)
    layout = load_default_pathway_asset_layout(Path(cast(str, args.config)).resolve())
    go_bp = layout.source(GO_BP_SOURCE)
    if not go_bp.annotation_download_url:
        raise ValueError("GO BP source is missing annotation_download_url in the asset layout")

    ontology_path = go_bp.raw_dir / _url_filename(go_bp.download_url)
    annotation_path = go_bp.raw_dir / _url_filename(go_bp.annotation_download_url)

    ontology_downloaded = _ensure_raw_asset(
        download_url=go_bp.download_url,
        raw_path=ontology_path,
        refresh=refresh,
    )
    annotation_downloaded = _ensure_raw_asset(
        download_url=go_bp.annotation_download_url,
        raw_path=annotation_path,
        refresh=refresh,
    )

    metadata, terms = _parse_go_terms(ontology_path)
    header_metadata, term_to_genes, stats = _parse_goa_human_annotations(
        annotation_path=annotation_path,
        terms=terms,
    )
    metadata = GoBuildMetadata(
        ontology_version=metadata.ontology_version,
        annotation_generated_date=header_metadata.annotation_generated_date,
        annotation_go_version=header_metadata.annotation_go_version,
        source_version=(
            f"go-basic:{metadata.ontology_version}|goa_human:{header_metadata.annotation_generated_date}"
        ),
    )
    records = _build_go_bp_records(
        terms=terms,
        term_to_genes=term_to_genes,
        source_version=metadata.source_version,
        stats=stats,
    )
    _write_catalog(catalog_path=go_bp.runtime_catalog_path, records=records)
    _update_manifest(
        layout=layout,
        manifest_path=layout.manifest_path,
        source_version=metadata.source_version,
        ontology_download_url=go_bp.download_url,
        annotation_download_url=go_bp.annotation_download_url,
        ontology_path=ontology_path,
        annotation_path=annotation_path,
        catalog_path=go_bp.runtime_catalog_path,
        metadata=metadata,
        stats=stats,
    )

    ontology_status = "downloaded" if ontology_downloaded else "cached"
    annotation_status = "downloaded" if annotation_downloaded else "cached"
    print(f"GO ontology raw asset: {ontology_path} ({ontology_status})")
    print(f"GO annotation raw asset: {annotation_path} ({annotation_status})")
    print(f"GO BP records: {len(records)}")
    print(f"Dropped empty GO BP terms: {stats.dropped_empty_terms}")
    print(f"Dropped broad GO BP terms (> {MAX_RUNTIME_GENES} genes): {stats.dropped_broad_terms}")
    print(f"Normalized catalog: {go_bp.runtime_catalog_path}")
    print(f"Manifest: {layout.manifest_path}")
    return 0


def _ensure_raw_asset(*, download_url: str, raw_path: Path, refresh: bool) -> bool:
    if raw_path.exists() and not refresh:
        return False

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
    with closing(cast(BinaryIO, urllib.request.urlopen(request))) as response:
        raw_bytes = response.read()
    _ = raw_path.write_bytes(raw_bytes)
    return True


def _parse_go_terms(ontology_path: Path) -> tuple[GoBuildMetadata, dict[str, GoTerm]]:
    ontology_version = "unknown"
    terms: dict[str, GoTerm] = {}
    current: dict[str, object] | None = None

    for raw_line in ontology_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if current is None and line.startswith("data-version:"):
            ontology_version = line.partition(":")[2].strip()
            continue

        if line == "[Term]":
            _finalize_term(current, terms)
            current = {"aliases": []}
            continue

        if line.startswith("["):
            _finalize_term(current, terms)
            current = None
            continue

        if current is None:
            continue

        if not line:
            _finalize_term(current, terms)
            current = None
            continue

        if line.startswith("id:"):
            current["term_id"] = line.partition(":")[2].strip()
        elif line.startswith("name:"):
            current["term_name"] = line.partition(":")[2].strip()
        elif line.startswith("namespace:"):
            current["namespace"] = line.partition(":")[2].strip()
        elif line.startswith("is_obsolete:"):
            current["is_obsolete"] = line.partition(":")[2].strip() == "true"
        elif line.startswith("synonym:"):
            match = SYNONYM_PATTERN.match(line)
            if match:
                aliases = cast(list[str], current["aliases"])
                _append_alias(aliases, match.group(1).strip())

    _finalize_term(current, terms)
    metadata = GoBuildMetadata(
        ontology_version=ontology_version,
        annotation_generated_date="unknown",
        annotation_go_version="unknown",
        source_version=f"go-basic:{ontology_version}|goa_human:unknown",
    )
    return metadata, terms


def _finalize_term(current: dict[str, object] | None, terms: dict[str, GoTerm]) -> None:
    if current is None:
        return

    term_id = cast(str | None, current.get("term_id"))
    term_name = cast(str | None, current.get("term_name"))
    namespace = cast(str | None, current.get("namespace"))
    is_obsolete = cast(bool, current.get("is_obsolete", False))
    aliases = cast(list[str], current.get("aliases", []))

    if not term_id or not term_name or namespace != GO_BP_NAMESPACE or is_obsolete:
        return

    normalized_aliases: list[str] = []
    _append_alias(normalized_aliases, term_name)
    _append_alias(normalized_aliases, _clean_alias(term_name))
    for alias in aliases:
        _append_alias(normalized_aliases, alias)
        _append_alias(normalized_aliases, _clean_alias(alias))

    terms[term_id] = GoTerm(
        term_id=term_id,
        term_name=term_name,
        aliases=tuple(normalized_aliases),
    )


def _parse_goa_human_annotations(
    *,
    annotation_path: Path,
    terms: dict[str, GoTerm],
) -> tuple[GoBuildMetadata, dict[str, set[str]], GoBuildStats]:
    annotation_generated_date = "unknown"
    annotation_go_version = "unknown"
    term_to_genes: dict[str, set[str]] = defaultdict(set)
    stats = GoBuildStats()

    with gzip.open(annotation_path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("!"):
                if line.startswith("!date-generated:"):
                    annotation_generated_date = line.partition(":")[2].strip()
                elif line.startswith("!go-version:"):
                    annotation_go_version = line.partition(":")[2].strip()
                continue

            parts = line.split("\t")
            if len(parts) < 15:
                continue

            stats.total_annotations += 1
            relation = parts[3].strip()
            go_id = parts[4].strip()
            evidence_code = parts[6].strip()
            aspect = parts[8].strip()
            gene_symbol = parts[2].strip()

            if aspect != GO_BP_ASPECT or go_id not in terms:
                stats.skipped_missing_term += 1
                continue
            if "NOT" in relation.split("|"):
                stats.skipped_not_qualifier += 1
                continue
            if evidence_code == "ND":
                stats.skipped_nd_evidence += 1
                continue

            canonical, resolved = normalize_gene(gene_symbol)
            if not resolved:
                stats.skipped_unresolved_genes += 1
                continue

            if canonical not in term_to_genes[go_id]:
                term_to_genes[go_id].add(canonical)
                stats.total_term_gene_edges += 1

    metadata = GoBuildMetadata(
        ontology_version="unknown",
        annotation_generated_date=annotation_generated_date,
        annotation_go_version=annotation_go_version,
        source_version=f"go-basic:unknown|goa_human:{annotation_generated_date}",
    )
    return metadata, term_to_genes, stats


def _build_go_bp_records(
    *,
    terms: dict[str, GoTerm],
    term_to_genes: dict[str, set[str]],
    source_version: str,
    stats: GoBuildStats,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for term_id, term in terms.items():
        genes = sorted(term_to_genes.get(term_id, set()))
        if not genes:
            stats.dropped_empty_terms += 1
            continue
        if len(genes) > MAX_RUNTIME_GENES:
            stats.dropped_broad_terms += 1
            continue

        record: dict[str, object] = {
            "term_id": term.term_id,
            "term_name": term.term_name,
            "aliases": list(term.aliases),
            "genes": genes,
            "source": GO_BP_SOURCE,
            "version": source_version,
        }
        records.append(record)
        stats.kept_records += 1
        stats.total_runtime_genes += len(genes)

    records.sort(key=lambda record: (str(record["term_name"]).upper(), str(record["term_id"])))
    return records


def _write_catalog(*, catalog_path: Path, records: list[dict[str, object]]) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_obj: dict[str, object] = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "source_name": GO_BP_SOURCE,
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
    ontology_download_url: str,
    annotation_download_url: str,
    ontology_path: Path,
    annotation_path: Path,
    catalog_path: Path,
    metadata: GoBuildMetadata,
    stats: GoBuildStats,
) -> None:
    manifest_obj = _read_json_object(manifest_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    source_record: dict[str, object] = {
        "source_name": GO_BP_SOURCE,
        "source_version": source_version,
        "download_url": annotation_download_url,
        "annotation_download_url": annotation_download_url,
        "ontology_download_url": ontology_download_url,
        "build_timestamp_utc": generated_at,
        "hgnc_version": layout.hgnc.version,
        "source_metadata": {
            "ontology_version": metadata.ontology_version,
            "annotation_generated_date": metadata.annotation_generated_date,
            "annotation_go_version": metadata.annotation_go_version,
            "broad_term_gene_cap": str(MAX_RUNTIME_GENES),
            "kept_records": str(stats.kept_records),
            "hgnc_alias_map_path": _repo_relative_path(layout.hgnc.alias_map_path),
            "hgnc_alias_map_sha256": _sha256_path(layout.hgnc.alias_map_path),
        },
        "filter_rules": [
            "drop_obsolete_go_terms",
            "drop_not_qualifier_annotations",
            "drop_nd_evidence_annotations",
            "drop_terms_without_hgnc_genes",
            f"drop_terms_with_more_than_{MAX_RUNTIME_GENES}_genes",
        ],
        "files": [
            {
                "asset_role": "raw_go_ontology",
                "relative_path": _repo_relative_path(ontology_path),
                "checksum_sha256": _sha256_path(ontology_path),
            },
            {
                "asset_role": "raw_go_annotation_gaf",
                "relative_path": _repo_relative_path(annotation_path),
                "checksum_sha256": _sha256_path(annotation_path),
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
        if source_obj.get("source_name") == GO_BP_SOURCE:
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


def _read_json_object(path: Path) -> dict[str, object]:
    raw_obj = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw_obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast(dict[str, object], raw_obj)


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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative_path(path: Path) -> str:
    package_root = Path(__file__).resolve().parents[1]
    return path.relative_to(package_root).as_posix()


def _url_filename(download_url: str) -> str:
    filename = Path(urlparse(download_url).path).name
    if not filename:
        raise ValueError(f"Could not determine filename from URL: {download_url}")
    return filename


def _append_alias(aliases: list[str], candidate: str) -> None:
    if not candidate:
        return
    if candidate in aliases:
        return
    aliases.append(candidate)


def _clean_alias(value: str) -> str:
    cleaned = re.sub(r"[-_/,:]", " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


if __name__ == "__main__":
    raise SystemExit(main())
