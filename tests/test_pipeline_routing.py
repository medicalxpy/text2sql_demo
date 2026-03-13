from __future__ import annotations

import unittest
from unittest.mock import patch
from typing import cast

from ..pipeline import _run_sqlgen, run_part1_part2


class PipelineRoutingTest(unittest.TestCase):
    def test_empty_topic_candidates_bypass_llm_and_use_values_fallback_sql(self) -> None:
        with patch("text2sql_demo.pipeline.chat_json") as chat_mock:
            out = _run_sqlgen(
                qspec_norm={"top_k": 7},
                topic_cands=[],
                topic_desc_context=[],
                model=None,
            )

        candidates = cast(list[dict[str, object]], out["candidates"])
        self.assertEqual(chat_mock.call_count, 0)
        self.assertEqual(len(candidates), 1)
        first = candidates[0]
        self.assertIn("WITH topic_candidates(topic_id, topic_score) AS (VALUES", str(first["sql"]))

    def test_explicit_gene_query_preserves_query_genes(self) -> None:
        qspec = {
            "top_k": 10,
            "original_query": "TP53 BRCA1",
            "genes": ["TP53", "BRCA1"],
            "marker_genes": ["MYC"],
            "grounding_mode": "none",
            "selected_terms": [],
            "selected_sources": [],
            "expanded_genes": [],
            "expansion_provenance": [],
        }

        with (
            patch("text2sql_demo.pipeline._run_spec_and_normalize", return_value=qspec),
            patch("text2sql_demo.pipeline.TopicStore.load_default", return_value=object()),
            patch("text2sql_demo.pipeline.compute_topic_candidates", return_value=[]) as cand_mock,
            patch("text2sql_demo.pipeline._build_topic_desc_context", return_value=[]),
            patch("text2sql_demo.pipeline._run_sqlgen", return_value={"candidates": []}),
        ):
            _ = run_part1_part2("TP53 BRCA1", model=None)

        self.assertEqual(cand_mock.call_args.kwargs["query_genes"], ["TP53", "BRCA1", "MYC"])

    def test_mixed_query_keeps_explicit_genes_without_pathway_expansion(self) -> None:
        qspec = {
            "top_k": 10,
            "original_query": "TP53 hypoxia",
            "genes": ["TP53"],
            "marker_genes": [],
            "grounding_mode": "grounded_terms",
            "selected_terms": [{"term_id": "TERM:1", "term_name": "Hypoxia", "source": "go_bp"}],
            "selected_sources": ["go_bp"],
            "expanded_genes": ["HIF1A", "VEGFA"],
            "expansion_provenance": [
                {"gene": "HIF1A", "term_id": "TERM:1", "source": "go_bp"},
                {"gene": "VEGFA", "term_id": "TERM:1", "source": "go_bp"},
            ],
        }

        with (
            patch("text2sql_demo.pipeline._run_spec_and_normalize", return_value=qspec),
            patch("text2sql_demo.pipeline.TopicStore.load_default", return_value=object()),
            patch("text2sql_demo.pipeline.compute_topic_candidates", return_value=[]) as cand_mock,
            patch("text2sql_demo.pipeline._build_topic_desc_context", return_value=[]),
            patch("text2sql_demo.pipeline._run_sqlgen", return_value={"candidates": []}),
        ):
            _ = run_part1_part2("TP53 hypoxia", model=None)

        self.assertEqual(cand_mock.call_args.kwargs["query_genes"], ["TP53"])

    def test_mechanism_only_query_uses_expanded_genes_for_topic_scoring(self) -> None:
        qspec = {
            "top_k": 10,
            "original_query": "hypoxia response",
            "genes": [],
            "marker_genes": [],
            "grounding_mode": "grounded_terms",
            "selected_terms": [{"term_id": "TERM:1", "term_name": "Hypoxia", "source": "go_bp"}],
            "selected_sources": ["go_bp"],
            "expanded_genes": ["HIF1A", "VEGFA", "EGLN1"],
            "expansion_provenance": [],
        }

        with (
            patch("text2sql_demo.pipeline._run_spec_and_normalize", return_value=qspec),
            patch("text2sql_demo.pipeline.TopicStore.load_default", return_value=object()),
            patch("text2sql_demo.pipeline.compute_topic_candidates", return_value=[]) as cand_mock,
            patch("text2sql_demo.pipeline._build_topic_desc_context", return_value=[]),
            patch("text2sql_demo.pipeline._run_sqlgen", return_value={"candidates": []}),
        ):
            _ = run_part1_part2("hypoxia response", model=None)

        self.assertEqual(cand_mock.call_args.kwargs["query_genes"], ["HIF1A", "VEGFA", "EGLN1"])

    def test_no_match_query_keeps_no_expansion(self) -> None:
        qspec = {
            "top_k": 10,
            "original_query": "unsupported mechanism",
            "genes": [],
            "marker_genes": [],
            "grounding_mode": "no_match",
            "selected_terms": [],
            "selected_sources": [],
            "expanded_genes": [],
            "expansion_provenance": [],
        }

        with (
            patch("text2sql_demo.pipeline._run_spec_and_normalize", return_value=qspec),
            patch("text2sql_demo.pipeline.TopicStore.load_default", return_value=object()),
            patch("text2sql_demo.pipeline.compute_topic_candidates", return_value=[]) as cand_mock,
            patch("text2sql_demo.pipeline._build_topic_desc_context", return_value=[]),
            patch("text2sql_demo.pipeline._run_sqlgen", return_value={"candidates": []}),
        ):
            _ = run_part1_part2("unsupported mechanism", model=None)

        self.assertEqual(cand_mock.call_args.kwargs["query_genes"], [])


if __name__ == "__main__":
    _ = unittest.main()
