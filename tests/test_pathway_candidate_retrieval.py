from __future__ import annotations

import unittest
from typing import ClassVar, override

from ..normalized_pathway_catalog import (
    NormalizedPathwayCatalog,
    load_default_normalized_pathway_catalog,
    retrieve_pathway_candidates,
)


class PathwayCandidateRetrievalTest(unittest.TestCase):
    catalog: ClassVar[NormalizedPathwayCatalog]

    @classmethod
    @override
    def setUpClass(cls) -> None:
        cls.catalog = load_default_normalized_pathway_catalog()

    def test_alias_query_returns_bounded_expected_candidates(self) -> None:
        candidates = retrieve_pathway_candidates(
            "epithelial mesenchymal transition",
            catalog=self.catalog,
            max_candidates=5,
        )

        self.assertLessEqual(len(candidates), 5)
        candidate_ids = [(candidate.source, candidate.term_id) for candidate in candidates]
        self.assertIn(("go_bp", "GO:0001837"), candidate_ids)
        self.assertIn(
            ("msigdb_hallmark", "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"),
            candidate_ids,
        )

    def test_abbreviation_query_is_deterministic_when_supported_locally(self) -> None:
        first = retrieve_pathway_candidates("UPR", catalog=self.catalog, max_candidates=5)
        second = retrieve_pathway_candidates("UPR", catalog=self.catalog, max_candidates=5)

        self.assertEqual(
            [(candidate.source, candidate.term_id) for candidate in first],
            [(candidate.source, candidate.term_id) for candidate in second],
        )
        self.assertTrue(first)
        self.assertEqual(first[0].term_id, "R-HSA-381119")
        self.assertEqual(first[0].source, "reactome")

    def test_reordered_phrase_query_matches_existing_aliases(self) -> None:
        candidates = retrieve_pathway_candidates(
            "hypoxia response",
            catalog=self.catalog,
            max_candidates=5,
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0].term_id, "GO:0001666")
        self.assertEqual(candidates[0].source, "go_bp")
        self.assertEqual(candidates[0].matched_alias, "response to hypoxia")

    def test_sentence_wrapped_query_stays_conservative_until_pipeline_reduces_it(self) -> None:
        candidates = retrieve_pathway_candidates(
            "Please find datasets related to epithelial-mesenchymal transition",
            catalog=self.catalog,
            max_candidates=5,
        )

        self.assertEqual(candidates, ())

    def test_unsupported_query_stays_conservative(self) -> None:
        candidates = retrieve_pathway_candidates(
            "made up pathway concept",
            catalog=self.catalog,
            max_candidates=5,
        )

        self.assertEqual(candidates, ())


if __name__ == "__main__":
    _ = unittest.main()
