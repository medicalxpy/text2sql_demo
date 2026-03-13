from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ..normalized_pathway_catalog import load_normalized_pathway_catalog
from ..topic_store import load_part1_grounding_catalog


class NormalizedPathwayCatalogLoaderTest(unittest.TestCase):
    def test_loader_reads_all_sources_with_unified_record_schema(self) -> None:
        catalog = load_part1_grounding_catalog()

        self.assertEqual(
            set(catalog.records_by_source),
            {"reactome", "go_bp", "msigdb_hallmark"},
        )

        for source_name in ("reactome", "go_bp", "msigdb_hallmark"):
            records = catalog.records_for_source(source_name)
            self.assertTrue(records)
            first = records[0]
            self.assertEqual(first.source, source_name)
            self.assertTrue(first.term_id)
            self.assertTrue(first.term_name)
            self.assertTrue(first.aliases)
            self.assertTrue(first.hgnc_genes)
            self.assertTrue(first.version)

            self.assertIsInstance(first.cross_links, tuple)
            self.assertIsInstance(first.related_terms, tuple)

    def test_loader_rejects_wrong_top_level_schema_version(self) -> None:
        catalog_path = self._write_catalog(
            {
                "schema_version": "wrong_schema",
                "source_name": "reactome",
                "records": [self._minimal_record(source="reactome")],
            }
        )

        with self.assertRaisesRegex(ValueError, "schema_version"):
            _ = load_normalized_pathway_catalog({"reactome": catalog_path})

    def test_loader_rejects_mismatched_top_level_source_name(self) -> None:
        catalog_path = self._write_catalog(
            {
                "schema_version": "pathway_gene_catalog_v1",
                "source_name": "go_bp",
                "records": [self._minimal_record(source="reactome")],
            }
        )

        with self.assertRaisesRegex(ValueError, "source_name"):
            _ = load_normalized_pathway_catalog({"reactome": catalog_path})

    def _write_catalog(self, payload: dict[str, object]) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        catalog_path = Path(tmpdir.name) / "catalog.json"
        _ = catalog_path.write_text(json.dumps(payload), encoding="utf-8")
        return catalog_path

    def _minimal_record(self, *, source: str) -> dict[str, object]:
        return {
            "term_id": "TERM:1",
            "term_name": "Example Term",
            "aliases": ["Example Term"],
            "hgnc_genes": ["TP53"],
            "source": source,
            "version": "v1",
        }


if __name__ == "__main__":
    _ = unittest.main()
