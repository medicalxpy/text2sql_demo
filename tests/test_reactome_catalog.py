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
CATALOG_PATH = PACKAGE_ROOT / "data" / "normalized" / "reactome" / "catalog.json"


class ReactomeCatalogContractTest(unittest.TestCase):
    def test_reactome_source_is_pinned_to_real_raw_artifact(self) -> None:
        layout = load_default_pathway_asset_layout()
        manifest = load_default_pathway_asset_manifest()

        reactome_layout = layout.source("reactome")
        reactome_manifest = None
        for record in manifest.sources:
            if record.source_name == "reactome":
                reactome_manifest = record
                break

        self.assertIsNotNone(reactome_manifest)
        assert reactome_manifest is not None

        self.assertNotEqual(reactome_layout.source_version, "pending")
        self.assertTrue(reactome_layout.download_url.endswith("ReactomePathways.gmt.zip"))
        raw_paths = [file.relative_path for file in reactome_manifest.files if file.asset_role.startswith("raw")]
        self.assertTrue(raw_paths)
        self.assertTrue(any(path.endswith("ReactomePathways.gmt.zip") for path in raw_paths))
        self.assertFalse(any(path.endswith(".gitkeep") for path in raw_paths))

    def test_reactome_catalog_records_are_present_and_hgnc_only(self) -> None:
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

            self.assertEqual(source, "reactome")
            self.assertIsInstance(term_id, str)
            self.assertIsInstance(term_name, str)
            self.assertIsInstance(aliases, list)
            self.assertIsInstance(genes_obj, list)
            assert isinstance(term_id, str)
            assert isinstance(term_name, str)
            assert isinstance(genes_obj, list)
            genes_obj = cast(list[object], genes_obj)

            self.assertTrue(term_id.startswith("R-HSA-"))
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
