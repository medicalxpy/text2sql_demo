"""Deterministic gene alias → canonical HGNC symbol normalization.

Replaces the former EntityLinkingAgent LLM call with a pure dict lookup
against the pre-built hgnc_alias_map.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import cast

_DATA_DIR = Path(__file__).resolve().parent / "data"
_ALIAS_MAP_FILE = _DATA_DIR / "hgnc_alias_map.json"

_alias_map: dict[str, str] | None = None


def _load_alias_map() -> dict[str, str]:
    global _alias_map
    if _alias_map is None:
        with open(_ALIAS_MAP_FILE, encoding="utf-8") as f:
            _alias_map = cast(dict[str, str], json.load(f))
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


def normalize_query_spec(qspec: Mapping[str, object]) -> dict[str, object]:
    """Normalize a SpecAgent QuerySpec into a QuerySpecNormalized."""
    genes_raw = _to_str_list(qspec.get("genes_raw", []))

    genes: list[str] = []
    marker_genes = _dedupe_preserve_order(_to_str_list(qspec.get("marker_genes", [])))
    unresolved: list[str] = []
    seen: set[str] = set()

    for raw in genes_raw:
        canonical, resolved = normalize_gene(raw)
        if canonical not in seen:
            seen.add(canonical)
            genes.append(canonical)
        if not resolved:
            unresolved.append(raw)

    return {
        **{k: v for k, v in dict(qspec).items() if k != "marker_genes_raw"},
        "genes": genes,
        "marker_genes": marker_genes,
        "grounding_mode": _normalize_grounding_mode(qspec.get("grounding_mode")),
        "selected_terms": _to_dict_list(qspec.get("selected_terms", [])),
        "selected_sources": _dedupe_preserve_order(_to_str_list(qspec.get("selected_sources", []))),
        "expanded_genes": _dedupe_preserve_order(_to_str_list(qspec.get("expanded_genes", []))),
        "expansion_provenance": _to_dict_list(qspec.get("expansion_provenance", [])),
        "unresolved": {"genes": unresolved},
    }


def _to_str_list(obj: object) -> list[str]:
    if not isinstance(obj, list):
        return []
    values = cast(list[object], obj)
    return [str(item).strip() for item in values if str(item).strip()]


def _to_dict_list(obj: object) -> list[dict[str, object]]:
    if not isinstance(obj, list):
        return []

    out: list[dict[str, object]] = []
    for item in cast(list[object], obj):
        if isinstance(item, dict):
            out.append(dict(cast(dict[str, object], item)))
    return out


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _normalize_grounding_mode(value: object) -> str:
    if not isinstance(value, str):
        return "none"
    mode = value.strip()
    return mode or "none"
