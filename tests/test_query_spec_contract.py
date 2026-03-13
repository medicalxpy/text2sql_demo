from __future__ import annotations

import unittest
from io import StringIO
from collections.abc import Callable
from typing import cast
from unittest.mock import patch

from ..gene_normalizer import normalize_query_spec
from .. import pipeline
from ..pipeline import coerce_query_spec_contract


class QuerySpecContractTest(unittest.TestCase):
    def test_explicit_gene_query_spec_drops_marker_genes_raw_and_adds_grounding_defaults(self) -> None:
        qspec = {
            "top_k": 10,
            "genes_raw": ["tp53", "BRCA1"],
            "original_query": "TP53 BRCA1",
        }

        normalized = normalize_query_spec(qspec)
        contract = coerce_query_spec_contract(normalized, original_query="TP53 BRCA1")

        self.assertNotIn("marker_genes_raw", normalized)
        self.assertNotIn("marker_genes_raw", contract)
        self.assertEqual(contract["genes"], ["TP53", "BRCA1"])
        self.assertEqual(contract["marker_genes"], [])
        self.assertEqual(contract["original_query"], "TP53 BRCA1")
        self.assertEqual(contract["grounding_mode"], "none")
        self.assertEqual(contract["selected_terms"], [])
        self.assertEqual(contract["selected_sources"], [])
        self.assertEqual(contract["expanded_genes"], [])
        self.assertEqual(contract["expansion_provenance"], [])

    def test_grounded_query_spec_preserves_explicit_grounding_metadata(self) -> None:
        qspec = {
            "top_k": 10,
            "genes_raw": [],
            "grounding_mode": "grounded_terms",
            "selected_terms": [
                {
                    "term_id": "GO:0001666",
                    "term_name": "response to hypoxia",
                    "source": "go_bp",
                }
            ],
            "selected_sources": ["go_bp"],
            "expanded_genes": ["HIF1A", "VEGFA"],
            "expansion_provenance": [
                {
                    "gene": "HIF1A",
                    "term_id": "GO:0001666",
                    "source": "go_bp",
                }
            ],
            "original_query": "hypoxia response",
        }

        normalized = normalize_query_spec(qspec)
        contract = coerce_query_spec_contract(normalized, original_query="hypoxia response")

        self.assertNotIn("marker_genes_raw", normalized)
        self.assertNotIn("marker_genes_raw", contract)
        self.assertEqual(contract["grounding_mode"], "grounded_terms")
        self.assertEqual(contract["selected_terms"], qspec["selected_terms"])
        self.assertEqual(contract["selected_sources"], ["go_bp"])
        self.assertEqual(contract["expanded_genes"], ["HIF1A", "VEGFA"])
        self.assertEqual(contract["expansion_provenance"], qspec["expansion_provenance"])
        self.assertEqual(contract["original_query"], "hypoxia response")

    def test_internal_grounding_query_is_removed_from_public_contract(self) -> None:
        qspec: dict[str, object] = {
            "top_k": 10,
            "genes_raw": [],
            "grounding_query": "epithelial mesenchymal transition",
            "grounding_mode": "none",
            "original_query": "Please find datasets related to EMT",
        }

        normalized = normalize_query_spec(qspec)
        contract = coerce_query_spec_contract(
            normalized,
            original_query="Please find datasets related to EMT",
        )

        self.assertEqual(normalized["grounding_query"], "epithelial mesenchymal transition")
        self.assertNotIn("grounding_query", contract)

    def test_validate_spec_output_does_not_require_marker_genes_raw(self) -> None:
        qspec: dict[str, object] = {
            "top_k": 10,
            "genes_raw": [],
        }

        stderr = StringIO()
        validate_spec_output = cast(Callable[[dict[str, object]], None], getattr(pipeline, "_validate_spec_output"))
        with patch("sys.stderr", stderr):
            validate_spec_output(qspec)

        self.assertNotIn("marker_genes_raw", qspec)
        self.assertNotIn("marker_genes_raw", stderr.getvalue())


if __name__ == "__main__":
    _ = unittest.main()
