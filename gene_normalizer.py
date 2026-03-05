"""Deterministic gene alias → canonical HGNC symbol normalization.

Replaces the former EntityLinkingAgent LLM call with a pure dict lookup
against the pre-built hgnc_alias_map.json.
"""
from __future__ import annotations

import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"
_ALIAS_MAP_FILE = _DATA_DIR / "hgnc_alias_map.json"

_alias_map: dict[str, str] | None = None


def _load_alias_map() -> dict[str, str]:
    global _alias_map
    if _alias_map is None:
        with open(_ALIAS_MAP_FILE, encoding="utf-8") as f:
            _alias_map = json.load(f)
    assert _alias_map is not None
    return _alias_map


def normalize_gene(raw: str) -> tuple[str, bool]:
    """Normalize a single gene string to its canonical HGNC symbol.

    Returns (canonical_symbol, resolved).
    If the gene cannot be resolved, returns (original_uppercased, False).
    """
    key = raw.strip().upper()
    if not key:
        return (key, False)
    alias_map = _load_alias_map()
    canonical = alias_map.get(key)
    if canonical is not None:
        return (canonical, True)
    return (key, False)


def normalize_query_spec(qspec: dict[str, object]) -> dict[str, object]:
    """Normalize a SpecAgent QuerySpec into a QuerySpecNormalized.

    Input fields consumed:
      - genes_raw: list[str]
      - marker_genes_raw: list[str]
      - top_k: int

    Output adds:
      - genes: list[str]          (deduplicated, canonical)
      - marker_genes: list[str]   (deduplicated, canonical)
      - unresolved: { genes: list[str] }
    """
    genes_raw = _to_str_list(qspec.get("genes_raw", []))
    marker_genes_raw = _to_str_list(qspec.get("marker_genes_raw", []))

    genes: list[str] = []
    marker_genes: list[str] = []
    unresolved: list[str] = []
    seen: set[str] = set()

    for raw in genes_raw:
        canonical, resolved = normalize_gene(raw)
        if canonical not in seen:
            seen.add(canonical)
            genes.append(canonical)
        if not resolved:
            unresolved.append(raw)

    for raw in marker_genes_raw:
        canonical, resolved = normalize_gene(raw)
        if canonical not in seen:
            seen.add(canonical)
            marker_genes.append(canonical)
        if not resolved:
            unresolved.append(raw)

    return {
        **qspec,
        "genes": genes,
        "marker_genes": marker_genes,
        "unresolved": {"genes": unresolved},
    }


def _to_str_list(obj: object) -> list[str]:
    if not isinstance(obj, list):
        return []
    return [str(item).strip() for item in obj if str(item).strip()]
