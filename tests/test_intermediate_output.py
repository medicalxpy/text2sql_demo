from __future__ import annotations

import unittest

from ..pipeline import build_grounded_pathway_intermediate, query_spec_for_intermediate


class IntermediateOutputTest(unittest.TestCase):
    def test_grounded_query_surfaces_compact_provenance(self) -> None:
        qspec = {
            "top_k": 10,
            "original_query": "hypoxia response",
            "genes": [],
            "marker_genes": [],
            "grounding_mode": "grounded_terms",
            "selected_terms": [
                {
                    "term_id": "GO:0001666",
                    "term_name": "response to hypoxia",
                    "source": "go_bp",
                    "matched_alias": "hypoxia response",
                    "match_type": "token_subset",
                },
                {
                    "term_id": "HALLMARK_HYPOXIA",
                    "term_name": "Hypoxia",
                    "source": "msigdb_hallmark",
                    "matched_alias": "hypoxia",
                    "match_type": "exact",
                },
            ],
            "selected_sources": ["go_bp", "msigdb_hallmark"],
            "expanded_genes": ["HIF1A", "VEGFA", "EGLN1"],
            "expansion_provenance": [
                {"gene": "HIF1A", "term_id": "GO:0001666", "source": "go_bp"},
                {"gene": "VEGFA", "term_id": "GO:0001666", "source": "go_bp"},
                {"gene": "EGLN1", "term_id": "HALLMARK_HYPOXIA", "source": "msigdb_hallmark"},
            ],
        }

        block = build_grounded_pathway_intermediate(qspec)
        self.assertIsNotNone(block)
        assert block is not None

        self.assertEqual(block["grounding_mode"], "grounded_terms")
        self.assertEqual(block["selected_sources"], ["go_bp", "msigdb_hallmark"])
        self.assertEqual(block["expansion"], [
            {"term_id": "GO:0001666", "term_name": "response to hypoxia", "source": "go_bp", "new_gene_count": 2},
            {"term_id": "HALLMARK_HYPOXIA", "term_name": "Hypoxia", "source": "msigdb_hallmark", "new_gene_count": 1},
        ])

    def test_explicit_gene_query_omits_grounded_block_and_raw_provenance_dump(self) -> None:
        qspec = {
            "top_k": 10,
            "original_query": "TP53 BRCA1",
            "genes": ["TP53", "BRCA1"],
            "marker_genes": [],
            "grounding_mode": "none",
            "selected_terms": [],
            "selected_sources": [],
            "expanded_genes": [],
            "expansion_provenance": [],
        }

        self.assertIsNone(build_grounded_pathway_intermediate(qspec))
        self.assertNotIn("expansion_provenance", query_spec_for_intermediate(qspec))


if __name__ == "__main__":
    _ = unittest.main()
