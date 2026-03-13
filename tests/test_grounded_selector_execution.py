from __future__ import annotations

import unittest
from collections.abc import Callable
from unittest.mock import patch
from typing import cast

from .. import pipeline
from ..normalized_pathway_catalog import PathwayGroundingCandidate


class GroundedSelectorExecutionTest(unittest.TestCase):
    def test_valid_selector_output_is_candidate_validated_and_capped_to_three(self) -> None:
        run_spec_and_normalize = cast(
            Callable[..., dict[str, object]],
            getattr(pipeline, "_run_spec_and_normalize"),
        )
        candidates = (
            PathwayGroundingCandidate("TERM:1", "Term One", "go_bp", "one", "exact", "v1"),
            PathwayGroundingCandidate("TERM:2", "Term Two", "reactome", "two", "exact", "v1"),
            PathwayGroundingCandidate("TERM:3", "Term Three", "msigdb_hallmark", "three", "exact", "v1"),
            PathwayGroundingCandidate("TERM:4", "Term Four", "go_bp", "four", "exact", "v1"),
        )

        with (
            patch("text2sql_demo.pipeline.retrieve_pathway_candidates", return_value=candidates),
            patch(
                "text2sql_demo.pipeline._load_grounding_genes_for_terms",
                return_value={
                    "TERM:1": ("GENE_A", "GENE_B"),
                    "TERM:3": ("GENE_C", "GENE_A"),
                    "TERM:4": ("GENE_D",),
                },
            ),
            patch(
                "text2sql_demo.pipeline.chat_json",
                side_effect=[
                    {
                        "top_k": 10,
                        "genes_raw": [],
                    },
                    {
                        "grounding_mode": "grounded_terms",
                        "selected_terms": [
                            {"term_id": "TERM:3"},
                            {"term_id": "TERM:999"},
                            {"term_id": "TERM:1"},
                            {"term_id": "TERM:4"},
                            {"term_id": "TERM:2"},
                        ],
                        "selected_sources": ["bad_source"],
                    },
                ],
            ) as chat_mock,
        ): 
            out = run_spec_and_normalize("hypoxia", model=None)

        self.assertEqual(chat_mock.call_count, 2)
        first_call = cast(dict[str, str], chat_mock.call_args_list[0].kwargs)
        self.assertNotIn("marker_genes_raw", first_call["user_prompt"])
        self.assertNotIn("marker_genes_raw", first_call["system_prompt"])
        self.assertEqual(out["grounding_mode"], "grounded_terms")
        selected_terms = cast(list[dict[str, object]], out["selected_terms"])
        self.assertEqual(
            [str(term["term_id"]) for term in selected_terms],
            ["TERM:3", "TERM:1", "TERM:4"],
        )
        self.assertEqual(out["selected_sources"], ["msigdb_hallmark", "go_bp"])
        self.assertEqual(out["expanded_genes"], ["GENE_C", "GENE_A", "GENE_B", "GENE_D"])
        self.assertEqual(
            out["expansion_provenance"],
            [
                {"gene": "GENE_C", "term_id": "TERM:3", "source": "msigdb_hallmark"},
                {"gene": "GENE_A", "term_id": "TERM:3", "source": "msigdb_hallmark"},
                {"gene": "GENE_B", "term_id": "TERM:1", "source": "go_bp"},
                {"gene": "GENE_D", "term_id": "TERM:4", "source": "go_bp"},
            ],
        )
        self.assertEqual(len(selected_terms), 3)

    def test_invalid_selector_output_is_rejected_to_no_match(self) -> None:
        run_spec_and_normalize = cast(
            Callable[..., dict[str, object]],
            getattr(pipeline, "_run_spec_and_normalize"),
        )
        candidates = (
            PathwayGroundingCandidate("TERM:1", "Term One", "go_bp", "one", "exact", "v1"),
        )

        with (
            patch("text2sql_demo.pipeline.retrieve_pathway_candidates", return_value=candidates),
            patch(
                "text2sql_demo.pipeline.chat_json",
                side_effect=[
                    {
                        "top_k": 10,
                        "genes_raw": [],
                    },
                    {
                        "grounding_mode": "grounded_terms",
                        "selected_terms": [{"term_id": "TERM:404"}],
                    },
                ],
            ),
        ):
            out = run_spec_and_normalize("unsupported concept", model=None)

        self.assertEqual(out["grounding_mode"], "no_match")
        self.assertEqual(out["selected_terms"], [])
        self.assertEqual(out["selected_sources"], [])
        self.assertEqual(out["expanded_genes"], [])

    def test_selector_is_not_invoked_when_candidate_set_is_empty(self) -> None:
        run_spec_and_normalize = cast(
            Callable[..., dict[str, object]],
            getattr(pipeline, "_run_spec_and_normalize"),
        )
        with (
            patch("text2sql_demo.pipeline.retrieve_pathway_candidates", return_value=()),
            patch(
                "text2sql_demo.pipeline.chat_json",
                return_value={
                    "top_k": 10,
                    "genes_raw": [],
                },
            ) as chat_mock,
        ):
            out = run_spec_and_normalize("no candidate query", model=None)

        self.assertEqual(chat_mock.call_count, 1)
        self.assertEqual(out["grounding_mode"], "no_match")
        self.assertEqual(out["selected_terms"], [])
        self.assertEqual(out["expanded_genes"], [])


if __name__ == "__main__":
    _ = unittest.main()
