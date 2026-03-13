from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast


_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _BASE_DIR / "config" / "pathway_gene_assets_v1.json"
_DEFAULT_MANIFEST_PATH = _BASE_DIR / "data" / "pathway_gene_assets_v1_manifest.json"


@dataclass(frozen=True)
class PathwayAssetSourceLayout:
    source_name: str
    source_version: str
    download_url: str
    annotation_download_url: str | None
    raw_dir: Path
    normalized_dir: Path
    runtime_catalog_path: Path


@dataclass(frozen=True)
class PathwayAssetHgncLayout:
    version: str
    alias_map_path: Path


@dataclass(frozen=True)
class PathwayAssetRebuildStep:
    source_name: str
    module_name: str


@dataclass(frozen=True)
class PathwayAssetLayout:
    name: str
    version: str
    manifest_path: Path
    hgnc: PathwayAssetHgncLayout
    raw_root: Path
    normalized_root: Path
    runtime_root: Path
    rebuild_steps: tuple[PathwayAssetRebuildStep, ...]
    sources: dict[str, PathwayAssetSourceLayout]

    def source(self, source_name: str) -> PathwayAssetSourceLayout:
        try:
            return self.sources[source_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.sources))
            raise KeyError(
                f"Unknown pathway asset source '{source_name}'. Known: {available}"
            ) from exc

    def runtime_catalog_paths(self) -> dict[str, Path]:
        return {
            source_name: source.runtime_catalog_path
            for source_name, source in sorted(self.sources.items())
        }


@dataclass(frozen=True)
class PathwayAssetFileRecord:
    asset_role: str
    relative_path: str
    checksum_sha256: str


@dataclass(frozen=True)
class PathwayAssetSourceRecord:
    source_name: str
    source_version: str
    download_url: str
    annotation_download_url: str | None
    ontology_download_url: str | None
    build_timestamp_utc: str
    hgnc_version: str
    source_metadata: dict[str, str]
    filter_rules: tuple[str, ...]
    files: tuple[PathwayAssetFileRecord, ...]


@dataclass(frozen=True)
class PathwayAssetManifest:
    name: str
    version: str
    schema_version: str
    generated_at_utc: str
    runtime_root: Path
    sources: tuple[PathwayAssetSourceRecord, ...]


def load_default_pathway_asset_layout(
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
) -> PathwayAssetLayout:
    config_obj = _read_json_object(Path(config_path))
    hgnc_obj = _required_object(config_obj, "hgnc")
    roots_obj = _required_object(config_obj, "roots")
    rebuild_obj = _required_object(config_obj, "rebuild")

    sources: dict[str, PathwayAssetSourceLayout] = {}
    for source_obj in _required_list_of_objects(config_obj, "sources"):
        source = PathwayAssetSourceLayout(
            source_name=_required_str(source_obj, "source_name"),
            source_version=_required_str(source_obj, "source_version"),
            download_url=_required_str(source_obj, "download_url"),
            annotation_download_url=_optional_str(source_obj, "annotation_download_url"),
            raw_dir=_resolve_repo_path(_required_str(source_obj, "raw_dir")),
            normalized_dir=_resolve_repo_path(_required_str(source_obj, "normalized_dir")),
            runtime_catalog_path=_resolve_repo_path(_required_str(source_obj, "runtime_catalog")),
        )
        sources[source.source_name] = source

    return PathwayAssetLayout(
        name=_required_str(config_obj, "name"),
        version=_required_str(config_obj, "version"),
        manifest_path=_resolve_repo_path(_required_str(config_obj, "manifest")),
        hgnc=PathwayAssetHgncLayout(
            version=_required_str(hgnc_obj, "version"),
            alias_map_path=_resolve_repo_path(_required_str(hgnc_obj, "alias_map_path")),
        ),
        raw_root=_resolve_repo_path(_required_str(roots_obj, "raw")),
        normalized_root=_resolve_repo_path(_required_str(roots_obj, "normalized")),
        runtime_root=_resolve_repo_path(_required_str(roots_obj, "runtime")),
        rebuild_steps=tuple(
            PathwayAssetRebuildStep(
                source_name=_required_str(step_obj, "source_name"),
                module_name=_required_str(step_obj, "module"),
            )
            for step_obj in _required_list_of_objects(rebuild_obj, "steps")
        ),
        sources=sources,
    )


def load_default_pathway_asset_manifest(
    manifest_path: str | Path = _DEFAULT_MANIFEST_PATH,
) -> PathwayAssetManifest:
    manifest_obj = _read_json_object(Path(manifest_path))

    sources: list[PathwayAssetSourceRecord] = []
    for source_obj in _required_list_of_objects(manifest_obj, "sources"):
        files = tuple(
            PathwayAssetFileRecord(
                asset_role=_required_str(file_obj, "asset_role"),
                relative_path=_required_str(file_obj, "relative_path"),
                checksum_sha256=_required_str(file_obj, "checksum_sha256"),
            )
            for file_obj in _required_list_of_objects(source_obj, "files")
        )
        sources.append(
            PathwayAssetSourceRecord(
                source_name=_required_str(source_obj, "source_name"),
                source_version=_required_str(source_obj, "source_version"),
                download_url=_required_str(source_obj, "download_url"),
                annotation_download_url=_optional_str(source_obj, "annotation_download_url"),
                ontology_download_url=_optional_str(source_obj, "ontology_download_url"),
                build_timestamp_utc=_required_str(source_obj, "build_timestamp_utc"),
                hgnc_version=_required_str(source_obj, "hgnc_version"),
                source_metadata=_optional_str_map(source_obj, "source_metadata"),
                filter_rules=tuple(_optional_str_list(source_obj, "filter_rules")),
                files=files,
            )
        )

    return PathwayAssetManifest(
        name=_required_str(manifest_obj, "name"),
        version=_required_str(manifest_obj, "version"),
        schema_version=_required_str(manifest_obj, "schema_version"),
        generated_at_utc=_required_str(manifest_obj, "generated_at_utc"),
        runtime_root=_resolve_repo_path(_required_str(manifest_obj, "runtime_root")),
        sources=tuple(sources),
    )


def default_runtime_catalog_paths() -> dict[str, Path]:
    return load_default_pathway_asset_layout().runtime_catalog_paths()


def _read_json_object(path: Path) -> dict[str, object]:
    raw_obj = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw_obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast(dict[str, object], raw_obj)


def _required_object(obj: dict[str, object], key: str) -> dict[str, object]:
    value = obj.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid or missing object field: {key}")
    return cast(dict[str, object], value)


def _required_list_of_objects(
    obj: dict[str, object],
    key: str,
) -> list[dict[str, object]]:
    value_obj = obj.get(key)
    if not isinstance(value_obj, list):
        raise ValueError(f"Invalid or missing list field: {key}")
    value = cast(list[object], value_obj)

    items: list[dict[str, object]] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid object entry in list field: {key}")
        items.append(cast(dict[str, object], entry))
    return items


def _required_str(obj: dict[str, object], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid or missing string field: {key}")
    return value.strip()


def _optional_str(obj: dict[str, object], key: str) -> str | None:
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid string field: {key}")
    return value.strip()


def _optional_str_map(obj: dict[str, object], key: str) -> dict[str, str]:
    value = obj.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Invalid object field: {key}")
    result: dict[str, str] = {}
    raw_map = cast(dict[object, object], value)
    for raw_key, raw_value in raw_map.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise ValueError(f"Invalid string map field: {key}")
        result[raw_key] = raw_value
    return result


def _optional_str_list(obj: dict[str, object], key: str) -> list[str]:
    value = obj.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Invalid list field: {key}")
    raw_list = cast(list[object], value)
    items: list[str] = []
    for entry in raw_list:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"Invalid string entry in list field: {key}")
        items.append(entry.strip())
    return items


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return _BASE_DIR / path
