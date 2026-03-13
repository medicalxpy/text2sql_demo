from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from ..pathway_asset_layout import (
    PathwayAssetLayout,
    load_default_pathway_asset_layout,
)


@dataclass(frozen=True)
class RebuildStep:
    source_name: str
    module_name: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild all pinned pathway raw and normalized assets from config."
    )
    _ = parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "pathway_gene_assets_v1.json"),
        help="Path to the pathway asset config JSON.",
    )
    _ = parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download all raw source artifacts before rebuilding normalized outputs.",
    )
    return parser.parse_args(argv)


def default_rebuild_steps(layout: PathwayAssetLayout) -> tuple[RebuildStep, ...]:
    return tuple(
        RebuildStep(source_name=step.source_name, module_name=step.module_name)
        for step in layout.rebuild_steps
    )


def synchronize_manifest_hgnc_provenance(
    *,
    manifest_path: Path,
    hgnc_version: str,
    hgnc_alias_map_path: Path,
) -> None:
    manifest_obj = _read_json_object(manifest_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    metadata_path = _repo_relative_path(hgnc_alias_map_path)
    metadata_checksum = _sha256_path(hgnc_alias_map_path)

    sources_obj = manifest_obj.get("sources")
    if not isinstance(sources_obj, list):
        raise ValueError(f"Invalid or missing list field: sources in {manifest_path}")

    for raw_source_obj in cast(list[object], sources_obj):
        if not isinstance(raw_source_obj, dict):
            raise ValueError(f"Invalid source record in {manifest_path}")
        source_obj = cast(dict[str, object], raw_source_obj)
        source_obj["hgnc_version"] = hgnc_version
        source_metadata_obj = source_obj.get("source_metadata")
        source_metadata: dict[str, str]
        if source_metadata_obj is None:
            source_metadata = {}
        elif isinstance(source_metadata_obj, dict):
            source_metadata = {
                str(key): str(value) for key, value in cast(dict[object, object], source_metadata_obj).items()
            }
        else:
            raise ValueError(f"Invalid source_metadata field in {manifest_path}")
        source_metadata["hgnc_alias_map_path"] = metadata_path
        source_metadata["hgnc_alias_map_sha256"] = metadata_checksum
        source_obj["source_metadata"] = source_metadata

    manifest_obj["generated_at_utc"] = generated_at
    _ = manifest_path.write_text(
        json.dumps(manifest_obj, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(cast(str, args.config)).resolve()
    refresh = cast(bool, args.refresh)
    layout = load_default_pathway_asset_layout(config_path)

    env = os.environ.copy()
    repo_parent = str(Path(__file__).resolve().parents[2])
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_parent if not pythonpath else os.pathsep.join([repo_parent, pythonpath])

    for step in default_rebuild_steps(layout):
        command = [sys.executable, "-m", step.module_name, "--config", str(config_path)]
        if refresh:
            command.append("--refresh")
        print(f"Rebuilding {step.source_name}: {' '.join(command)}")
        _ = subprocess.run(command, check=True, env=env)

    synchronize_manifest_hgnc_provenance(
        manifest_path=layout.manifest_path,
        hgnc_version=layout.hgnc.version,
        hgnc_alias_map_path=layout.hgnc.alias_map_path,
    )
    print(f"Synchronized manifest HGNC provenance: {layout.manifest_path}")
    return 0


def _read_json_object(path: Path) -> dict[str, object]:
    raw_obj = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw_obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast(dict[str, object], raw_obj)


def _repo_relative_path(path: Path) -> str:
    package_root = Path(__file__).resolve().parents[1]
    return path.relative_to(package_root).as_posix()


def _sha256_path(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
