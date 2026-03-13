from __future__ import annotations

import unittest
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import patch

from .. import pipeline
from ..normalized_pathway_catalog import PathwayGroundingCandidate


class ConceptSentenceGroundingTest(unittest.TestCase):
    def test_first_pass_grounding_query_drives_candidate_lookup(self) -> None:
        run_spec_and_normalize = cast(
            Callable[..., dict[str, object]],
            getattr(pipeline, "_run_spec_and_normalize"),
        )
        candidates = (
            PathwayGroundingCandidate(
                "GO:0001837",
                "epithelial to mesenchymal transition",
                "go_bp",
                "epithelial mesenchymal transition",
                "exact",
                "v1",
            ),
        )

        with (
            patch("text2sql_demo.pipeline.retrieve_pathway_candidates", return_value=candidates) as retrieve_mock,
            patch(
                "text2sql_demo.pipeline.chat_json",
                side_effect=[
                    {
                        "top_k": 10,
                        "genes_raw": [],
                        "grounding_query": "epithelial mesenchymal transition",
                    },
                    {
                        "grounding_mode": "grounded_terms",
                        "selected_terms": [{"term_id": "GO:0001837"}],
                    },
                ],
            ),
            patch(
                "text2sql_demo.pipeline._load_grounding_genes_for_terms",
                return_value={"GO:0001837": ("VIM", "FN1")},
            ),
        ):
            out = run_spec_and_normalize(
                "Please find datasets related to epithelial-mesenchymal transition",
                model=None,
            )

        self.assertEqual(retrieve_mock.call_args.args[0], "epithelial mesenchymal transition")
        self.assertEqual(out["grounding_mode"], "grounded_terms")
        self.assertEqual(out["expanded_genes"], ["VIM", "FN1"])

    def test_blank_grounding_query_falls_back_to_original_question_once(self) -> None:
        run_spec_and_normalize = cast(
            Callable[..., dict[str, object]],
            getattr(pipeline, "_run_spec_and_normalize"),
        )

        with patch(
            "text2sql_demo.pipeline.chat_json",
            return_value={
                "top_k": 10,
                "genes_raw": [],
                "grounding_query": "   ",
            },
        ) as chat_mock:
            def retrieve_side_effect(
                query: str,
                *,
                max_candidates: int = 8,
            ) -> tuple[PathwayGroundingCandidate, ...]:
                del max_candidates
                self.assertEqual(chat_mock.call_count, 1)
                self.assertEqual(query, "unsupported concept")
                return ()

            with patch(
                "text2sql_demo.pipeline.retrieve_pathway_candidates",
                side_effect=retrieve_side_effect,
            ) as retrieve_mock:
                out = run_spec_and_normalize("unsupported concept", model=None)

        self.assertEqual(retrieve_mock.call_count, 1)
        self.assertEqual(out["grounding_mode"], "no_match")
        self.assertEqual(out["selected_terms"], [])

    def test_spec_agent_prompt_mentions_internal_grounding_query(self) -> None:
        prompt = Path("text2sql_demo/prompts/spec_agent_system.txt").read_text(encoding="utf-8")

        self.assertIn('"grounding_query"', prompt)
        self.assertIn("shortest phrase that captures the central biological concept", prompt)
        self.assertIn("first-pass output only", prompt)
        self.assertIn("the pipeline will convert the final query state to \"no_match\"", prompt)


if __name__ == "__main__":
    _ = unittest.main()
