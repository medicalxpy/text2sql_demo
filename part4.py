from __future__ import annotations

import json
import sqlite3
from typing import Generator

from .llm_client import chat_text, chat_text_stream
from .prompt_store import load_prompt

_AGENT_NAME = "answer_agent"

_NO_DATASETS_MSG = "No datasets were retrieved. Please refine your query or check gene names."


def _query_gene_topic_mapping(db_path: str, genes: list[str]) -> list[dict[str, object]]:
    """Return gene→topic mapping with weights for the given genes."""
    if not genes:
        return []
    conn = sqlite3.connect(db_path)
    try:
        placeholders = ",".join("?" for _ in genes)
        sql = (
            f"SELECT gene_symbol, topic_id, weight "
            f"FROM topic_gene "
            f"WHERE gene_symbol IN ({placeholders}) "
            f"ORDER BY gene_symbol, weight DESC"
        )
        rows = conn.execute(sql, genes).fetchall()
        return [
            {"gene": row[0], "topic_id": row[1], "weight": round(row[2], 4)}
            for row in rows
        ]
    finally:
        conn.close()


def _query_dataset_score_breakdown(
    db_path: str, genes: list[str], dataset_ids: list[str],
) -> dict[str, list[dict[str, object]]]:
    """Return per-dataset score breakdown by topic.

    For each retrieved dataset, shows how much each topic contributes to
    its final score: SUM(topic_gene.weight * cell_topic.weight).
    """
    if not genes or not dataset_ids:
        return {}
    conn = sqlite3.connect(db_path)
    try:
        gene_ph = ",".join("?" for _ in genes)
        ds_ph = ",".join("?" for _ in dataset_ids)
        sql = (
            f"SELECT d.dataset_id, tg.topic_id, "
            f"  ROUND(SUM(tg.weight * ct.weight), 4) AS topic_score "
            f"FROM topic_gene tg "
            f"JOIN cell_topic ct ON ct.topic_id = tg.topic_id "
            f"JOIN cell c ON c.cell_id = ct.cell_id "
            f"JOIN dataset d ON d.dataset_id = c.dataset_id "
            f"WHERE tg.gene_symbol IN ({gene_ph}) "
            f"  AND d.dataset_id IN ({ds_ph}) "
            f"GROUP BY d.dataset_id, tg.topic_id "
            f"HAVING topic_score > 0 "
            f"ORDER BY d.dataset_id, topic_score DESC"
        )
        rows = conn.execute(sql, [*genes, *dataset_ids]).fetchall()
        result: dict[str, list[dict[str, object]]] = {}
        for ds_id, topic_id, score in rows:
            result.setdefault(ds_id, []).append(
                {"topic_id": topic_id, "score": score}
            )
        return result
    finally:
        conn.close()


def _build_user_prompt(
    original_query: str,
    datasets: list[dict[str, object]],
    topic_descriptions: list[dict[str, object]],
    genes: list[str],
    evidence_pack: dict[str, object],
    db_path: str,
) -> str:
    gene_topic_mapping = _query_gene_topic_mapping(db_path, genes)
    dataset_ids = [str(d.get("dataset_id", "")) for d in datasets if d.get("dataset_id")]
    dataset_score_breakdown = _query_dataset_score_breakdown(db_path, genes, dataset_ids)

    user_prompt = load_prompt("answer_agent_user.txt")
    user_prompt = user_prompt.replace("{{original_query}}", original_query)
    user_prompt = user_prompt.replace("{{datasets_json}}", _json_pretty(datasets))
    user_prompt = user_prompt.replace("{{topic_descriptions_json}}", _json_pretty(topic_descriptions))
    user_prompt = user_prompt.replace("{{genes_json}}", _json_pretty(genes))
    user_prompt = user_prompt.replace("{{gene_topic_mapping_json}}", _json_pretty(gene_topic_mapping))
    user_prompt = user_prompt.replace("{{dataset_score_breakdown_json}}", _json_pretty(dataset_score_breakdown))
    user_prompt = user_prompt.replace("{{passed_count}}", str(evidence_pack.get("passed_count", 0)))
    user_prompt = user_prompt.replace("{{failed_count}}", str(evidence_pack.get("failed_count", 0)))
    user_prompt = user_prompt.replace("{{row_count}}", str(evidence_pack.get("selected_row_count", 0)))
    return user_prompt


def run_part4(
    *,
    original_query: str,
    datasets: list[dict[str, object]],
    topic_descriptions: list[dict[str, object]],
    genes: list[str],
    evidence_pack: dict[str, object],
    db_path: str,
) -> dict[str, object]:
    if not datasets:
        return {"answer": _NO_DATASETS_MSG, "skipped": True}

    system_prompt = load_prompt("answer_agent_system.txt")
    user_prompt = _build_user_prompt(original_query, datasets, topic_descriptions, genes, evidence_pack, db_path)

    answer = chat_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        agent_name=_AGENT_NAME,
    )

    return {"answer": answer, "skipped": False}


def stream_part4(
    *,
    original_query: str,
    datasets: list[dict[str, object]],
    topic_descriptions: list[dict[str, object]],
    genes: list[str],
    evidence_pack: dict[str, object],
    db_path: str,
) -> Generator[str, None, None]:
    if not datasets:
        yield _NO_DATASETS_MSG
        return

    system_prompt = load_prompt("answer_agent_system.txt")
    user_prompt = _build_user_prompt(original_query, datasets, topic_descriptions, genes, evidence_pack, db_path)

    yield from chat_text_stream(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        agent_name=_AGENT_NAME,
    )


def _json_pretty(obj: object) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=True)
