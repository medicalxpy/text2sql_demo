from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import cast

from ..gene_normalizer import normalize_gene
from ..pathway_asset_layout import (
    load_default_pathway_asset_layout,
    load_default_pathway_asset_manifest,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PACKAGE_ROOT / "data" / "normalized" / "go_bp" / "catalog.json"


class GoBpCatalogContractTest(unittest.TestCase):
    def test_go_bp_source_is_pinned_to_real_raw_assets(self) -> None:
        layout = load_default_pathway_asset_layout()
        manifest = load_default_pathway_asset_manifest()

        go_layout = layout.source("go_bp")
        go_manifest = None
        for record in manifest.sources:
            if record.source_name == "go_bp":
                go_manifest = record
                break

        self.assertIsNotNone(go_manifest)
        assert go_manifest is not None

        self.assertNotEqual(go_layout.source_version, "pending")
        self.assertTrue(go_layout.download_url.endswith("go-basic.obo"))
        self.assertIn("annotations", go_manifest.download_url)

        file_paths = {
            file.asset_role: file.relative_path for file in go_manifest.files
        }
        self.assertIn("raw_go_ontology", file_paths)
        self.assertIn("raw_go_annotation_gaf", file_paths)
        self.assertTrue(file_paths["raw_go_ontology"].endswith("go-basic.obo"))
        self.assertTrue(file_paths["raw_go_annotation_gaf"].endswith("goa_human.gaf.gz"))
        self.assertNotEqual(go_manifest.hgnc_version, "pending")

    def test_go_bp_catalog_records_are_present_and_hgnc_only(self) -> None:
        catalog = cast(object, json.loads(CATALOG_PATH.read_text(encoding="utf-8")))
        self.assertIsInstance(catalog, dict)
        catalog_obj = cast(dict[str, object], catalog)

        records_obj = catalog_obj.get("records")
        self.assertIsInstance(records_obj, list)
        records = cast(list[dict[str, object]], records_obj)

        self.assertTrue(records)
        for record in records[:50]:
            source = record.get("source")
            term_id = record.get("term_id")
            term_name = record.get("term_name")
            aliases = record.get("aliases")
            genes_obj = record.get("genes")

            self.assertEqual(source, "go_bp")
            self.assertIsInstance(term_id, str)
            self.assertIsInstance(term_name, str)
            self.assertIsInstance(aliases, list)
            self.assertIsInstance(genes_obj, list)
            assert isinstance(term_id, str)
            assert isinstance(term_name, str)
            assert isinstance(genes_obj, list)
            genes_obj = cast(list[object], genes_obj)

            self.assertTrue(term_id.startswith("GO:"))
            self.assertTrue(term_name.strip())
            self.assertTrue(genes_obj)
            genes = [gene for gene in genes_obj if isinstance(gene, str)]
            self.assertEqual(len(genes), len(genes_obj))
            for gene in genes:
                canonical, resolved = normalize_gene(gene)
                self.assertTrue(resolved, gene)
                self.assertEqual(canonical, gene)


if __name__ == "__main__":
    _ = unittest.main()
