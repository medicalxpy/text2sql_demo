from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import cast

from ..pathway_asset_layout import (
    load_default_pathway_asset_layout,
    load_default_pathway_asset_manifest,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PACKAGE_ROOT / "data" / "normalized" / "msigdb_hallmark" / "catalog.json"


class MsigdbHallmarkCatalogContractTest(unittest.TestCase):
    def test_msigdb_hallmark_source_uses_real_raw_artifact_and_provenance(self) -> None:
        layout = load_default_pathway_asset_layout()
        manifest = load_default_pathway_asset_manifest()

        hallmark_layout = layout.source("msigdb_hallmark")
        hallmark_manifest = None
        for record in manifest.sources:
            if record.source_name == "msigdb_hallmark":
                hallmark_manifest = record
                break

        self.assertIsNotNone(hallmark_manifest)
        assert hallmark_manifest is not None

        self.assertNotEqual(hallmark_layout.source_version, "pending")
        self.assertIn("msigdb", hallmark_layout.download_url.lower())
        self.assertNotEqual(hallmark_manifest.hgnc_version, "pending")

        raw_paths = {
            file.asset_role: file.relative_path
            for file in hallmark_manifest.files
            if file.asset_role.startswith("raw_")
        }
        self.assertIn("raw_msigdb_hallmark_gene_sets", raw_paths)
        self.assertFalse(any(path.endswith(".gitkeep") for path in raw_paths.values()))

        metadata = hallmark_manifest.source_metadata
        self.assertEqual(metadata.get("collection"), "H")
        self.assertEqual(metadata.get("registration_required"), "true")
        self.assertTrue(metadata.get("license_terms_url", "").endswith("license_terms_list.jsp"))

    def test_msigdb_hallmark_catalog_records_are_readable_and_source_specific(self) -> None:
        catalog = cast(object, json.loads(CATALOG_PATH.read_text(encoding="utf-8")))
        self.assertIsInstance(catalog, dict)
        catalog_obj = cast(dict[str, object], catalog)

        records_obj = catalog_obj.get("records")
        self.assertIsInstance(records_obj, list)
        records = cast(list[dict[str, object]], records_obj)

        self.assertEqual(len(records), 50)
        for record in records[:10]:
            source = record.get("source")
            term_id = record.get("term_id")
            term_name = record.get("term_name")
            aliases = record.get("aliases")
            genes_obj = record.get("genes")

            self.assertEqual(source, "msigdb_hallmark")
            self.assertIsInstance(term_id, str)
            self.assertIsInstance(term_name, str)
            self.assertIsInstance(aliases, list)
            self.assertIsInstance(genes_obj, list)
            assert isinstance(term_id, str)
            assert isinstance(term_name, str)
            assert isinstance(aliases, list)
            assert isinstance(genes_obj, list)
            raw_aliases = cast(list[object], aliases)
            alias_list = [alias for alias in raw_aliases if isinstance(alias, str)]

            self.assertTrue(term_id.startswith("HALLMARK_"))
            self.assertTrue(term_name.startswith("Hallmark "))
            self.assertTrue(genes_obj)
            self.assertIn(term_name, alias_list)


if __name__ == "__main__":
    _ = unittest.main()
