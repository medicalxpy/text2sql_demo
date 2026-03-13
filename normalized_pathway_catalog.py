from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .pathway_asset_layout import default_runtime_catalog_paths


_CATALOG_SCHEMA_VERSION = "pathway_gene_catalog_v1"


@dataclass(frozen=True)
class NormalizedPathwayRecord:
    term_id: str
    term_name: str
    aliases: tuple[str, ...]
    hgnc_genes: tuple[str, ...]
    source: str
    version: str
    cross_links: tuple[str, ...] = ()
    related_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class PathwayGroundingCandidate:
    term_id: str
    term_name: str
    source: str
    matched_alias: str
    match_type: str
    version: str


@dataclass(frozen=True)
class _LexicalMatch:
    match_rank: int
    extra_token_count: int
    matched_alias: str
    match_type: str


@dataclass(frozen=True)
class NormalizedPathwayCatalog:
    records_by_source: dict[str, tuple[NormalizedPathwayRecord, ...]]

    def records_for_source(self, source_name: str) -> tuple[NormalizedPathwayRecord, ...]:
        try:
            return self.records_by_source[source_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.records_by_source))
            raise KeyError(
                f"Unknown normalized catalog source '{source_name}'. Known: {available}"
            ) from exc

    def iter_records(self) -> tuple[NormalizedPathwayRecord, ...]:
        merged: list[NormalizedPathwayRecord] = []
        for source_name in sorted(self.records_by_source):
            merged.extend(self.records_by_source[source_name])
        return tuple(merged)


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_PAREN_ABBREV_RE = re.compile(r"\(([A-Za-z0-9]{2,10})\)")


def retrieve_pathway_candidates(
    query: str,
    *,
    catalog: NormalizedPathwayCatalog | None = None,
    max_candidates: int = 8,
) -> tuple[PathwayGroundingCandidate, ...]:
    if max_candidates <= 0:
        return ()

    normalized_query = _normalize_lexical_text(query)
    if not normalized_query:
        return ()

    records = catalog if catalog is not None else load_default_normalized_pathway_catalog()
    query_tokens = tuple(normalized_query.split())
    allow_short_token_match = _is_short_abbreviation_query(query_tokens)

    scored: list[tuple[tuple[object, ...], PathwayGroundingCandidate]] = []
    for record in records.iter_records():
        lexical_match = _best_lexical_match(
            record=record,
            normalized_query=normalized_query,
            query_tokens=query_tokens,
            allow_short_token_match=allow_short_token_match,
        )
        if lexical_match is None:
            continue

        candidate = PathwayGroundingCandidate(
            term_id=record.term_id,
            term_name=record.term_name,
            source=record.source,
            matched_alias=lexical_match.matched_alias,
            match_type=lexical_match.match_type,
            version=record.version,
        )
        scored.append(
            (
                (
                    lexical_match.match_rank,
                    lexical_match.extra_token_count,
                    record.source,
                    record.term_id,
                ),
                candidate,
            )
        )

    scored.sort(key=lambda item: item[0])
    limited = scored[:max_candidates]
    return tuple(candidate for _, candidate in limited)


def load_default_normalized_pathway_catalog() -> NormalizedPathwayCatalog:
    return load_normalized_pathway_catalog(default_runtime_catalog_paths())


def load_normalized_pathway_catalog(catalog_paths: dict[str, Path]) -> NormalizedPathwayCatalog:
    records_by_source: dict[str, tuple[NormalizedPathwayRecord, ...]] = {}
    for source_name, catalog_path in sorted(catalog_paths.items()):
        records_by_source[source_name] = _load_source_catalog(
            source_name=source_name,
            catalog_path=catalog_path,
        )
    return NormalizedPathwayCatalog(records_by_source=records_by_source)


def _load_source_catalog(*, source_name: str, catalog_path: Path) -> tuple[NormalizedPathwayRecord, ...]:
    payload_obj = cast(object, json.loads(catalog_path.read_text(encoding="utf-8")))
    if not isinstance(payload_obj, dict):
        raise ValueError(f"Expected JSON object in {catalog_path}")
    payload = cast(dict[str, object], payload_obj)
    _validate_catalog_metadata(payload=payload, source_name=source_name, catalog_path=catalog_path)

    records_obj = payload.get("records")
    if not isinstance(records_obj, list):
        raise ValueError(f"Missing or invalid records list in {catalog_path}")

    out: list[NormalizedPathwayRecord] = []
    for raw_record in cast(list[object], records_obj):
        if not isinstance(raw_record, dict):
            raise ValueError(f"Invalid record entry in {catalog_path}")
        record = cast(dict[str, object], raw_record)
        out.append(_parse_record(record=record, source_name=source_name))

    return tuple(out)


def _validate_catalog_metadata(
    *, payload: dict[str, object], source_name: str, catalog_path: Path
) -> None:
    schema_version = _required_str(payload, "schema_version")
    if schema_version != _CATALOG_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid schema_version in {catalog_path}: expected '{_CATALOG_SCHEMA_VERSION}', got '{schema_version}'"
        )

    payload_source_name = _required_str(payload, "source_name")
    if payload_source_name != source_name:
        raise ValueError(
            f"Invalid source_name in {catalog_path}: expected '{source_name}', got '{payload_source_name}'"
        )


def _parse_record(*, record: dict[str, object], source_name: str) -> NormalizedPathwayRecord:
    term_id = _required_str(record, "term_id")
    term_name = _required_str(record, "term_name")
    aliases = _required_str_tuple(record, "aliases")
    hgnc_genes = _optional_str_tuple(record, "hgnc_genes")
    if not hgnc_genes:
        hgnc_genes = _required_str_tuple(record, "genes")

    source = _optional_str(record, "source") or source_name
    version = _required_str(record, "version")

    return NormalizedPathwayRecord(
        term_id=term_id,
        term_name=term_name,
        aliases=aliases,
        hgnc_genes=hgnc_genes,
        source=source,
        version=version,
        cross_links=_optional_str_tuple(record, "cross_links"),
        related_terms=_optional_str_tuple(record, "related_terms"),
    )


def _best_lexical_match(
    *,
    record: NormalizedPathwayRecord,
    normalized_query: str,
    query_tokens: tuple[str, ...],
    allow_short_token_match: bool,
) -> _LexicalMatch | None:
    best: _LexicalMatch | None = None
    for alias in _record_lexical_aliases(record):
        normalized_alias = _normalize_lexical_text(alias)
        if not normalized_alias:
            continue

        alias_tokens = tuple(normalized_alias.split())
        alias_token_set = set(alias_tokens)
        match: _LexicalMatch | None = None

        if normalized_alias == normalized_query:
            match = _LexicalMatch(
                match_rank=0,
                extra_token_count=0,
                matched_alias=alias,
                match_type="exact",
            )
        elif len(query_tokens) >= 2 and _contains_token_sequence(alias_tokens, query_tokens):
            match = _LexicalMatch(
                match_rank=1,
                extra_token_count=max(0, len(alias_tokens) - len(query_tokens)),
                matched_alias=alias,
                match_type="token_subset",
            )
        elif len(query_tokens) >= 2 and _contains_token_set(alias_tokens, query_tokens):
            match = _LexicalMatch(
                match_rank=2,
                extra_token_count=max(0, len(alias_tokens) - len(query_tokens)),
                matched_alias=alias,
                match_type="token_reorder",
            )
        elif allow_short_token_match and len(query_tokens) == 1 and query_tokens[0] in alias_token_set:
            match = _LexicalMatch(
                match_rank=3,
                extra_token_count=max(0, len(alias_tokens) - 1),
                matched_alias=alias,
                match_type="abbreviation_token",
            )

        if match is None:
            continue
        if best is None or (match.match_rank, match.extra_token_count, match.matched_alias) < (
            best.match_rank,
            best.extra_token_count,
            best.matched_alias,
        ):
            best = match

    return best


def _record_lexical_aliases(record: NormalizedPathwayRecord) -> tuple[str, ...]:
    aliases: list[str] = [record.term_name, *record.aliases]
    for value in tuple(aliases):
        aliases.extend(_derived_abbreviations(value))
    return tuple(dict.fromkeys(alias for alias in aliases if alias.strip()))


def _derived_abbreviations(value: str) -> tuple[str, ...]:
    matches = [match.group(1).strip() for match in _PAREN_ABBREV_RE.finditer(value)]
    return tuple(dict.fromkeys(match for match in matches if match))


def _normalize_lexical_text(value: str) -> str:
    lowered = value.strip().lower()
    if not lowered:
        return ""
    normalized = _NON_ALNUM_RE.sub(" ", lowered)
    return " ".join(normalized.split())


def _contains_token_sequence(tokens: tuple[str, ...], query_tokens: tuple[str, ...]) -> bool:
    if len(query_tokens) > len(tokens):
        return False
    width = len(query_tokens)
    for index in range(len(tokens) - width + 1):
        if tokens[index : index + width] == query_tokens:
            return True
    return False


def _contains_token_set(tokens: tuple[str, ...], query_tokens: tuple[str, ...]) -> bool:
    return set(query_tokens).issubset(tokens)


def _is_short_abbreviation_query(query_tokens: tuple[str, ...]) -> bool:
    if len(query_tokens) != 1:
        return False
    token = query_tokens[0]
    if not token.isalnum():
        return False
    return 2 <= len(token) <= 6


def _required_str(record: dict[str, object], field: str) -> str:
    value = _optional_str(record, field)
    if value is None:
        raise ValueError(f"Missing or invalid '{field}' field")
    return value


def _optional_str(record: dict[str, object], field: str) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid '{field}' field")
    return value.strip()


def _required_str_tuple(record: dict[str, object], field: str) -> tuple[str, ...]:
    values = _optional_str_tuple(record, field)
    if not values:
        raise ValueError(f"Missing or invalid '{field}' field")
    return values


def _optional_str_tuple(record: dict[str, object], field: str) -> tuple[str, ...]:
    value = record.get(field)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Missing or invalid '{field}' field")

    values: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Missing or invalid '{field}' field")
        values.append(item.strip())
    return tuple(values)
