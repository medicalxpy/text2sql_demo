from __future__ import annotations

import unittest
from unittest.mock import patch
from typing import Callable, cast

from .. import pipeline
from ..normalized_pathway_catalog import PathwayGroundingCandidate, retrieve_pathway_candidates
from ..topic_store import load_part1_grounding_catalog


_validate_selector_output = cast(
    Callable[..., dict[str, object]],
    getattr(pipeline, "_validate_selector_output"),
)


class PathwayGroundingSmokeTest(unittest.TestCase):
    def test_catalog_load_smoke_contract(self) -> None:
        catalog = load_part1_grounding_catalog()

        self.assertEqual(
            set(catalog.records_by_source),
            {"reactome", "go_bp", "msigdb_hallmark"},
            "catalog load contract broke: expected all normalized runtime sources",
        )

        reactome_record = catalog.records_for_source("reactome")[0]
        self.assertTrue(
            reactome_record.term_id and reactome_record.term_name and reactome_record.hgnc_genes,
            "catalog load contract broke: normalized record schema is incomplete",
        )

    def test_candidate_bound_smoke_contract(self) -> None:
        candidates = retrieve_pathway_candidates(
            "UPR",
            catalog=load_part1_grounding_catalog(),
            max_candidates=1,
        )

        self.assertEqual(
            len(candidates),
            1,
            "candidate bound contract broke: max_candidates=1 must cap the selector input set",
        )
        self.assertEqual(
            (candidates[0].source, candidates[0].term_id),
            ("reactome", "R-HSA-381119"),
            "candidate bound contract broke: stable top candidate changed for the UPR smoke query",
        )

    def test_selector_validation_smoke_contract(self) -> None:
        payload = _validate_selector_output(
            selector_raw={"selected_terms": "TERM:1"},
            candidates=(
                PathwayGroundingCandidate(
                    "TERM:1",
                    "Term One",
                    "go_bp",
                    "term one",
                    "exact",
                    "v1",
                ),
            ),
        )

        self.assertEqual(
            payload["grounding_mode"],
            "no_match",
            "selector validation contract broke: malformed selector payload must collapse to no_match",
        )
        self.assertEqual(payload["selected_terms"], [])
        self.assertEqual(payload["expanded_genes"], [])

    def test_dedup_union_merge_smoke_contract(self) -> None:
        candidates = (
            PathwayGroundingCandidate("TERM:2", "Term Two", "reactome", "term two", "exact", "v1"),
            PathwayGroundingCandidate("TERM:1", "Term One", "go_bp", "term one", "exact", "v1"),
            PathwayGroundingCandidate("TERM:3", "Term Three", "reactome", "term three", "exact", "v1"),
        )

        with patch(
            "text2sql_demo.pipeline._load_grounding_genes_for_terms",
            return_value={
                "TERM:2": ("GENE_B", "GENE_A"),
                "TERM:1": ("GENE_A", "GENE_C"),
                "TERM:3": ("GENE_D",),
            },
        ):
            payload = _validate_selector_output(
                selector_raw={
                    "selected_terms": [
                        {"term_id": "TERM:2"},
                        {"term_id": "TERM:1"},
                        {"term_id": "TERM:3"},
                    ]
                },
                candidates=candidates,
            )

        self.assertEqual(
            payload["expanded_genes"],
            ["GENE_B", "GENE_A", "GENE_C", "GENE_D"],
            "dedup-union merge contract broke: expanded genes must stay first-hit ordered and unique",
        )
        self.assertEqual(
            payload["expansion_provenance"],
            [
                {"gene": "GENE_B", "term_id": "TERM:2", "source": "reactome"},
                {"gene": "GENE_A", "term_id": "TERM:2", "source": "reactome"},
                {"gene": "GENE_C", "term_id": "TERM:1", "source": "go_bp"},
                {"gene": "GENE_D", "term_id": "TERM:3", "source": "reactome"},
            ],
            "dedup-union merge contract broke: provenance must track the first term that contributed each gene",
        )


if __name__ == "__main__":
    _ = unittest.main()
