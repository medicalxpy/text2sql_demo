from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from ..pathway_asset_layout import load_default_pathway_asset_layout
from ..scripts.build_pathway_gene_assets import (
    default_rebuild_steps,
    synchronize_manifest_hgnc_provenance,
)


class PathwayRebuildEntrypointContractTest(unittest.TestCase):
    def test_layout_exposes_pinned_hgnc_provenance(self) -> None:
        layout = load_default_pathway_asset_layout()

        self.assertNotEqual(layout.hgnc.version, "pending")
        self.assertTrue(layout.hgnc.alias_map_path.exists())
        self.assertEqual(layout.hgnc.alias_map_path.name, "hgnc_alias_map.json")

    def test_rebuild_entrypoint_keeps_manual_explicit_source_order(self) -> None:
        steps = default_rebuild_steps(load_default_pathway_asset_layout())

        self.assertEqual(
            [step.source_name for step in steps],
            ["reactome", "go_bp", "msigdb_hallmark"],
        )
        self.assertEqual(
            [step.module_name for step in steps],
            [
                "text2sql_demo.scripts.build_reactome_catalog",
                "text2sql_demo.scripts.build_go_bp_catalog",
                "text2sql_demo.scripts.build_topic_store",
            ],
        )

    def test_manifest_sync_replaces_pending_hgnc_version_for_all_sources(self) -> None:
        layout = load_default_pathway_asset_layout()
        manifest: dict[str, object] = {
            "name": "pathway_gene_assets",
            "version": "v1",
            "schema_version": "pathway_gene_asset_manifest_v1",
            "generated_at_utc": "2026-03-09T00:00:00+00:00",
            "runtime_root": "data/normalized",
            "sources": [
                {
                    "source_name": "reactome",
                    "source_version": "v1",
                    "download_url": "https://example.test/reactome",
                    "build_timestamp_utc": "2026-03-09T00:00:00+00:00",
                    "hgnc_version": "pending",
                    "files": [],
                },
                {
                    "source_name": "go_bp",
                    "source_version": "v1",
                    "download_url": "https://example.test/go",
                    "build_timestamp_utc": "2026-03-09T00:00:00+00:00",
                    "hgnc_version": "pending",
                    "files": [],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            _ = manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            synchronize_manifest_hgnc_provenance(
                manifest_path=manifest_path,
                hgnc_version=layout.hgnc.version,
                hgnc_alias_map_path=layout.hgnc.alias_map_path,
            )

            updated = cast(object, json.loads(manifest_path.read_text(encoding="utf-8")))

        self.assertIsInstance(updated, dict)
        updated_obj = cast(dict[str, object], updated)
        sources_obj = updated_obj["sources"]
        self.assertIsInstance(sources_obj, list)
        sources = cast(list[object], sources_obj)

        for source_obj in sources:
            self.assertIsInstance(source_obj, dict)
            source_record = cast(dict[str, object], source_obj)
            self.assertEqual(source_record["hgnc_version"], layout.hgnc.version)
            metadata_obj = source_record["source_metadata"]
            self.assertIsInstance(metadata_obj, dict)
            metadata = cast(dict[str, object], metadata_obj)
            self.assertEqual(
                metadata["hgnc_alias_map_path"],
                "data/hgnc_alias_map.json",
            )
            self.assertTrue(metadata["hgnc_alias_map_sha256"])


if __name__ == "__main__":
    _ = unittest.main()
