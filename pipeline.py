from __future__ import annotations
import sys
from typing import TypedDict

from collections.abc import Iterable, Mapping

from .gene_normalizer import normalize_query_spec
from .llm_client import chat_json
from .part3 import default_db_path, run_part3
from .part4 import run_part4
from .prompt_store import load_prompt
from .topic_descriptions import load_topic_descriptions
from .topic_store import TopicStore, compute_topic_candidates


def run_part1_part2(question: str, *, top_m: int = 10) -> dict[str, object]:
    qspec_norm = _run_spec_and_normalize(question)
    genes = _to_str_list(qspec_norm.get("genes", []))
    marker_genes = _to_str_list(qspec_norm.get("marker_genes", []))
    query_genes = list(dict.fromkeys([*genes, *marker_genes]))
    store = TopicStore.load_default()
    topic_cands = compute_topic_candidates(query_genes=query_genes, store=store, top_m=top_m)
    topic_desc_context = _build_topic_desc_context(topic_cands)
    sql_out = _run_sqlgen(qspec_norm=qspec_norm, topic_cands=topic_cands, topic_desc_context=topic_desc_context)

    return {
        "method": "workflow",
        "query_spec": qspec_norm,
        "topic_candidates": topic_cands,
        "topic_descriptions": topic_desc_context,
        "sql_candidates": sql_out,
    }


def run_baseline_a_part2(question: str, *, top_m: int = 10) -> dict[str, object]:
    desc_index = load_topic_descriptions()
    topic_catalog = _topic_catalog_for_prompt(desc_index)
    allowed_ids = {str(item.get("topic_id", "")).strip() for item in topic_catalog if isinstance(item, dict)}

    a_system = load_prompt("baseline_a_system.txt")
    a_user = load_prompt("baseline_a_user.txt")
    a_user = a_user.replace("{{question}}", question)
    a_user = a_user.replace("{{topic_catalog_json}}", _json_pretty(topic_catalog))
    a_out = chat_json(system_prompt=a_system, user_prompt=a_user)

    top_k = _coerce_top_k(a_out.get("top_k", 10), default=10)
    topic_cands = _coerce_topic_candidates(
        a_out.get("topic_candidates", []),
        top_m=top_m,
        allowed_ids=allowed_ids,
    )
    topic_desc_context = _build_topic_desc_context(topic_cands, desc_index=desc_index)
    sql_out = _coerce_sql_out(obj=a_out, top_k=top_k)

    return {
        "method": "baseline_a_single_shot",
        "query_spec": {
            "original_query": question,
            "top_k": top_k,
            "genes": [],
            "marker_genes": [],
        },
        "topic_candidates": topic_cands,
        "topic_descriptions": topic_desc_context,
        "sql_candidates": sql_out,
    }


def run_baseline_b_part2(question: str, *, top_m: int = 10) -> dict[str, object]:
    qspec_norm = _run_spec_and_normalize(question)

    desc_index = load_topic_descriptions()
    topic_catalog = _topic_catalog_for_prompt(desc_index)
    allowed_ids = {str(item.get("topic_id", "")).strip() for item in topic_catalog if isinstance(item, dict)}

    pick_system = load_prompt("baseline_b_topic_picker_system.txt")
    pick_user = load_prompt("baseline_b_topic_picker_user.txt")
    pick_user = pick_user.replace("{{query_spec_json}}", _json_pretty(qspec_norm))
    pick_user = pick_user.replace("{{top_m}}", str(max(0, top_m)))
    pick_user = pick_user.replace("{{topic_catalog_json}}", _json_pretty(topic_catalog))
    picked = chat_json(system_prompt=pick_system, user_prompt=pick_user)

    topic_cands = _coerce_topic_candidates(
        picked.get("topic_candidates", []),
        top_m=top_m,
        allowed_ids=allowed_ids,
    )
    topic_desc_context = _build_topic_desc_context(topic_cands, desc_index=desc_index)
    sql_out = _run_sqlgen(qspec_norm=qspec_norm, topic_cands=topic_cands, topic_desc_context=topic_desc_context)

    return {
        "method": "baseline_b_llm_topic_picker",
        "query_spec": qspec_norm,
        "topic_candidates": topic_cands,
        "topic_descriptions": topic_desc_context,
        "sql_candidates": sql_out,
    }


def run_part1_part3(
    question: str,
    *,
    top_m: int = 10,
    db_path: str | None = None,
    query_timeout_seconds: float = 2.0,
) -> dict[str, object]:
    out = run_part1_part2(question, top_m=top_m)
    return _run_with_part3(
        out=out,
        db_path=db_path,
        query_timeout_seconds=query_timeout_seconds,
    )


def run_baseline_a_part3(
    question: str,
    *,
    top_m: int = 10,
    db_path: str | None = None,
    query_timeout_seconds: float = 2.0,
) -> dict[str, object]:
    out = run_baseline_a_part2(question, top_m=top_m)
    return _run_with_part3(
        out=out,
        db_path=db_path,
        query_timeout_seconds=query_timeout_seconds,
    )


def run_baseline_b_part3(
    question: str,
    *,
    top_m: int = 10,
    db_path: str | None = None,
    query_timeout_seconds: float = 2.0,
) -> dict[str, object]:
    out = run_baseline_b_part2(question, top_m=top_m)
    return _run_with_part3(
        out=out,
        db_path=db_path,
        query_timeout_seconds=query_timeout_seconds,
    )


def _run_with_part3(*, out: dict[str, object], db_path: str | None, query_timeout_seconds: float) -> dict[str, object]:
    qspec_obj = out.get("query_spec", {})
    qspec = qspec_obj if isinstance(qspec_obj, dict) else {}
    top_k = _coerce_top_k(qspec.get("top_k", 10), default=10)

    sql_out_obj = out.get("sql_candidates", {})
    sql_out = sql_out_obj if isinstance(sql_out_obj, dict) else {}
    part3 = run_part3(
        sql_out=sql_out,
        db_path=db_path or default_db_path(),
        top_k=top_k,
        query_timeout_seconds=query_timeout_seconds,
    )
    return {
        **out,
        "part3": part3,
    }


def run_part1_part4(
    question: str,
    *,
    top_m: int = 10,
    db_path: str | None = None,
    query_timeout_seconds: float = 2.0,
) -> dict[str, object]:
    out = run_part1_part3(question, top_m=top_m, db_path=db_path, query_timeout_seconds=query_timeout_seconds)
    return _run_with_part4(out, db_path=db_path)


def run_baseline_a_part4(
    question: str,
    *,
    top_m: int = 10,
    db_path: str | None = None,
    query_timeout_seconds: float = 2.0,
) -> dict[str, object]:
    out = run_baseline_a_part3(question, top_m=top_m, db_path=db_path, query_timeout_seconds=query_timeout_seconds)
    return _run_with_part4(out, db_path=db_path)


def run_baseline_b_part4(
    question: str,
    *,
    top_m: int = 10,
    db_path: str | None = None,
    query_timeout_seconds: float = 2.0,
) -> dict[str, object]:
    out = run_baseline_b_part3(question, top_m=top_m, db_path=db_path, query_timeout_seconds=query_timeout_seconds)
    return _run_with_part4(out, db_path=db_path)


class Part4Inputs(TypedDict):
    original_query: str
    datasets: list[dict[str, object]]
    topic_descriptions: list[dict[str, object]]
    genes: list[str]
    evidence_pack: dict[str, object]
    db_path: str


def extract_part4_inputs(out: dict[str, object], *, db_path: str | None = None) -> Part4Inputs:
    qspec = out.get("query_spec", {})
    if not isinstance(qspec, dict):
        qspec = {}

    part3 = out.get("part3", {})
    if not isinstance(part3, dict):
        part3 = {}

    original_query = str(qspec.get("original_query", ""))
    genes = _to_str_list(qspec.get("genes", [])) + _to_str_list(qspec.get("marker_genes", []))
    datasets = part3.get("datasets", [])
    if not isinstance(datasets, list):
        datasets = []
    topic_descs = out.get("topic_descriptions", [])
    if not isinstance(topic_descs, list):
        topic_descs = []
    evidence_pack = part3.get("evidence_pack", {})
    if not isinstance(evidence_pack, dict):
        evidence_pack = {}

    resolved_db_path = db_path or str(default_db_path())

    return Part4Inputs(
        original_query=original_query,
        datasets=datasets,
        topic_descriptions=topic_descs,
        genes=list(dict.fromkeys(genes)),
        evidence_pack=evidence_pack,
        db_path=resolved_db_path,
    )


def _run_with_part4(out: dict[str, object], *, db_path: str | None = None) -> dict[str, object]:
    inputs = extract_part4_inputs(out, db_path=db_path)
    part4_result = run_part4(**inputs)
    return {**out, "part4": part4_result}



def _validate_spec_output(qspec: dict[str, object]) -> None:
    """Validate SpecAgent output has expected fields; warn and fill defaults on missing."""
    for key, default in (("top_k", 10), ("genes_raw", []), ("marker_genes_raw", [])):
        if key not in qspec:
            print(f"[WARN] SpecAgent output missing '{key}', using default={default!r}", file=sys.stderr)
            qspec.setdefault(key, default)


def _run_spec_and_normalize(question: str) -> dict[str, object]:
    spec_system = load_prompt("spec_agent_system.txt")
    spec_user = load_prompt("spec_agent_user.txt").replace("{{question}}", question)
    qspec = chat_json(system_prompt=spec_system, user_prompt=spec_user)

    _validate_spec_output(qspec)

    qspec_norm = normalize_query_spec(qspec)
    qspec_norm["original_query"] = question
    return qspec_norm


def _run_sqlgen(
    *,
    qspec_norm: dict[str, object],
    topic_cands: Iterable[Mapping[str, object]],
    topic_desc_context: list[dict[str, object]],
) -> dict[str, object]:
    topic_cands_list = [dict(tc) for tc in topic_cands]
    sql_system = load_prompt("sqlgen_system.txt")
    sql_user = load_prompt("sqlgen_user.txt")
    sql_user = sql_user.replace("{{query_spec_json}}", _json_pretty(qspec_norm))
    sql_user = sql_user.replace("{{topic_candidates_json}}", _json_pretty(topic_cands_list))
    sql_user = sql_user.replace("{{topic_descriptions_json}}", _json_pretty(topic_desc_context))
    sql_out = chat_json(system_prompt=sql_system, user_prompt=sql_user)
    top_k = _coerce_top_k(qspec_norm.get("top_k", 10), default=10)
    return _coerce_sql_out(obj=sql_out, top_k=top_k)


def _build_topic_desc_context(
    topic_cands: Iterable[Mapping[str, object]], *, desc_index: dict[str, dict[str, object]] | None = None
) -> list[dict[str, object]]:
    index = desc_index if desc_index is not None else load_topic_descriptions()
    topic_desc_context: list[dict[str, object]] = []
    for tc in topic_cands:
        if not isinstance(tc, dict):
            continue
        tid = str(tc.get("topic_id", "")).strip()
        if not tid or tid not in index:
            continue
        topic_desc_context.append(
            {
                "topic_id": tid,
                "description": index[tid].get("description", ""),
                "top_genes": index[tid].get("top_genes", []),
            }
        )
    return topic_desc_context


def _topic_catalog_for_prompt(desc_index: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for topic_id in sorted(desc_index):
        data = desc_index[topic_id]
        genes = _to_str_list(data.get("top_genes", []))[:8]
        out.append(
            {
                "topic_id": topic_id,
                "description": str(data.get("description", "")).strip(),
                "top_genes": genes,
            }
        )
    return out


def _coerce_top_k(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        k = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        k = int(value.strip())
    else:
        k = default
    return k if k > 0 else default


def _coerce_topic_candidates(
    value: object,
    *,
    top_m: int,
    allowed_ids: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    allowed = {s.strip() for s in allowed_ids} if allowed_ids is not None else None
    if not isinstance(value, list):
        return []

    best: dict[str, float] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        tid = str(item.get("topic_id", "")).strip()
        if not tid:
            continue
        if allowed is not None and tid not in allowed:
            continue
        try:
            score = float(item.get("topic_score", 0.0))
        except (TypeError, ValueError):
            continue
        if score <= 0:
            continue
        prev = best.get(tid)
        if prev is None or score > prev:
            best[tid] = score

    ranked = sorted(best.items(), key=lambda x: (-x[1], x[0]))
    limited = ranked[: max(0, top_m)]
    return [{"topic_id": tid, "topic_score": round(score, 6)} for tid, score in limited]


def _coerce_sql_out(*, obj: dict[str, object], top_k: int) -> dict[str, object]:
    cands_obj = obj.get("candidates", [])

    candidates: list[dict[str, object]] = []
    if isinstance(cands_obj, list):
        for idx, item in enumerate(cands_obj[:1], start=1):
            if not isinstance(item, dict):
                continue
            sql = str(item.get("sql", "")).strip()
            if not sql:
                continue
            cand_id = _coerce_top_k(item.get("id", idx), default=idx)
            notes = str(item.get("notes", "")).strip()
            candidates.append({"id": cand_id, "sql": sql, "notes": notes})

    if not candidates:
        print(
            "[WARN] SQLGenAgent returned no valid SQL candidates; using fallback (empty result set).",
            file=sys.stderr,
        )
        candidates = [
            {
                "id": 1,
                "sql": _fallback_sql(top_k),
                "notes": "fallback candidate: model returned no valid SQL",
            }
        ]

    return {
        "candidates": candidates,
    }


def _fallback_sql(top_k: int) -> str:
    safe_top_k = max(1, int(top_k))
    return (
        "WITH topic_candidates(topic_id, topic_score) AS (VALUES ('__none__', 0.0)) "
        "SELECT d.dataset_id AS dataset_id, d.dataset_name AS dataset_name, "
        "SUM(ct.weight * tc.topic_score) AS score "
        "FROM topic_candidates tc "
        "JOIN cell_topic ct ON ct.topic_id = tc.topic_id "
        "JOIN cell c ON c.cell_id = ct.cell_id "
        "JOIN dataset d ON d.dataset_id = c.dataset_id "
        "WHERE 1=0 "
        "GROUP BY d.dataset_id, d.dataset_name "
        "ORDER BY score DESC "
        f"LIMIT {safe_top_k}"
    )


def _json_pretty(obj: object) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=True)


def _json_one_line(obj: object) -> str:
    import json

    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True)


def _to_str_list(obj: object) -> list[str]:
    if not isinstance(obj, list):
        return []
    out: list[str] = []
    for item in obj:
        s = str(item).strip()
        if s:
            out.append(s)
    return out
