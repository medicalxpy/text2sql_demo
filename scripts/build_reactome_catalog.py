from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, cast
from urllib.parse import urlparse

from ..gene_normalizer import normalize_gene
from ..pathway_asset_layout import PathwayAssetLayout, load_default_pathway_asset_layout


REACTOME_SOURCE = "reactome"
CATALOG_SCHEMA_VERSION = "pathway_gene_catalog_v1"
CATALOG_MEMBER_SUFFIX = ".gmt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pinned Reactome raw and normalized pathway assets."
    )
    _ = parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "pathway_gene_assets_v1.json"),
        help="Path to the pathway asset config JSON.",
    )
    _ = parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download the raw Reactome artifact even if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refresh = cast(bool, args.refresh)
    layout = load_default_pathway_asset_layout(Path(cast(str, args.config)).resolve())
    reactome = layout.source(REACTOME_SOURCE)

    raw_path = reactome.raw_dir / _url_filename(reactome.download_url)
    downloaded = _ensure_raw_asset(
        download_url=reactome.download_url,
        raw_path=raw_path,
        refresh=refresh,
    )
    records = _build_reactome_records(
        raw_zip_path=raw_path,
        source_version=reactome.source_version,
    )
    _write_catalog(
        catalog_path=reactome.runtime_catalog_path,
        records=records,
    )
    _update_manifest(
        layout=layout,
        manifest_path=layout.manifest_path,
        source_version=reactome.source_version,
        download_url=reactome.download_url,
        raw_path=raw_path,
        catalog_path=reactome.runtime_catalog_path,
    )

    status = "downloaded" if downloaded else "cached"
    print(f"Reactome raw asset: {raw_path} ({status})")
    print(f"Reactome records: {len(records)}")
    print(f"Normalized catalog: {reactome.runtime_catalog_path}")
    print(f"Manifest: {layout.manifest_path}")
    return 0


def _ensure_raw_asset(*, download_url: str, raw_path: Path, refresh: bool) -> bool:
    if raw_path.exists() and not refresh:
        return False

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(cast(BinaryIO, urllib.request.urlopen(download_url))) as response:
        raw_bytes = response.read()
    _ = raw_path.write_bytes(raw_bytes)
    return True


def _build_reactome_records(*, raw_zip_path: Path, source_version: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    with zipfile.ZipFile(raw_zip_path) as archive:
        member_name = _select_gmt_member(archive)
        with archive.open(member_name) as handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                parts = [part.strip() for part in line.split("\t") if part.strip()]
                if len(parts) < 3:
                    continue

                term_name, term_id, *raw_genes = parts
                if not term_id.startswith("R-HSA-"):
                    continue

                genes = _canonical_hgnc_genes(raw_genes)
                if not genes:
                    continue

                record: dict[str, object] = {
                    "term_id": term_id,
                    "term_name": term_name,
                    "aliases": _build_aliases(term_name),
                    "genes": genes,
                    "source": REACTOME_SOURCE,
                    "version": source_version,
                }
                records.append(record)

    records.sort(key=lambda record: (str(record["term_name"]).upper(), str(record["term_id"])))
    return records


def _select_gmt_member(archive: zipfile.ZipFile) -> str:
    for member_name in archive.namelist():
        if member_name.endswith(CATALOG_MEMBER_SUFFIX):
            return member_name
    raise ValueError(f"No {CATALOG_MEMBER_SUFFIX} member found in {archive.filename}")


def _canonical_hgnc_genes(raw_genes: list[str]) -> list[str]:
    genes: list[str] = []
    seen: set[str] = set()

    for raw_gene in raw_genes:
        canonical, resolved = normalize_gene(raw_gene)
        if not resolved or canonical in seen:
            continue
        seen.add(canonical)
        genes.append(canonical)

    return genes


def _build_aliases(term_name: str) -> list[str]:
    aliases: list[str] = []
    _append_alias(aliases, re.sub(r"\s*\([^)]*\)", "", term_name).strip())
    cleaned = re.sub(r"[-_/,:]", " ", term_name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    _append_alias(aliases, cleaned)
    return aliases


def _append_alias(aliases: list[str], candidate: str) -> None:
    if not candidate:
        return
    if candidate in aliases:
        return
    aliases.append(candidate)


def _write_catalog(*, catalog_path: Path, records: list[dict[str, object]]) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_obj: dict[str, object] = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "source_name": REACTOME_SOURCE,
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
        "source_name": REACTOME_SOURCE,
        "source_version": source_version,
        "download_url": download_url,
        "build_timestamp_utc": generated_at,
        "hgnc_version": layout.hgnc.version,
        "source_metadata": {
            "hgnc_alias_map_path": _repo_relative_path(layout.hgnc.alias_map_path),
            "hgnc_alias_map_sha256": _sha256_path(layout.hgnc.alias_map_path),
        },
        "files": [
            {
                "asset_role": "raw_reactome_gmt_zip",
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
        if source_obj.get("source_name") == REACTOME_SOURCE:
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


def _url_filename(download_url: str) -> str:
    filename = Path(urlparse(download_url).path).name
    if not filename:
        raise ValueError(f"Could not determine filename from URL: {download_url}")
    return filename


if __name__ == "__main__":
    raise SystemExit(main())
