from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

from ..llm_client import chat_json


class TopicDescription(TypedDict):
    topic_id: str
    gene_count: int
    top_genes: list[str]
    description: str


def _make_description_llm(*, topic_id: str, genes: list[str]) -> TopicDescription:
    top = genes[:30]
    system = (
        "You generate short topic descriptions from a topic_gene distribution. "
        "You must only describe the gene distribution. Do not mention pathway names or biological functions. "
        "Output JSON only."
    )
    user = (
        "Given the following topic gene set (unweighted list), generate a concise description.\n"
        "Requirements:\n"
        "- Output a single JSON object only.\n"
        "- Include keys: topic_id, gene_count, top_genes, description.\n"
        "- description: 1-2 sentences, factual, based only on the genes provided.\n\n"
        f"topic_id: {topic_id}\n"
        f"gene_count: {len(genes)}\n"
        f"top_genes: {', '.join(top)}\n"
    )
    raw = chat_json(system_prompt=system, user_prompt=user)

    raw_topic_id: object = raw.get("topic_id", topic_id)
    out_topic_id = str(raw_topic_id)

    gene_count_raw: object = raw.get("gene_count", len(genes))
    if isinstance(gene_count_raw, (int, float)):
        out_gene_count = int(gene_count_raw)
    elif isinstance(gene_count_raw, str) and gene_count_raw.strip().isdigit():
        out_gene_count = int(gene_count_raw.strip())
    else:
        out_gene_count = len(genes)

    top_genes_raw: object = raw.get("top_genes", top[:20])
    if isinstance(top_genes_raw, list):
        top_genes_list = cast(list[object], top_genes_raw)
        out_top_genes = [str(g).strip().upper() for g in top_genes_list if str(g).strip()][:20]
    else:
        out_top_genes = top[:20]

    description: object = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"LLM did not return a valid description for topic_id={topic_id}")
    return {
        "topic_id": out_topic_id,
        "gene_count": out_gene_count,
        "top_genes": out_top_genes,
        "description": description.strip(),
    }


def main() -> int:
    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data"
    in_path = data_dir / "topic_gene_hallmark_2020.json"
    if not in_path.exists():
        raise FileNotFoundError(
            "Missing topic gene set file. Run: python -m text2sql_demo.scripts.build_topic_store"
        )

    data_obj = cast(dict[str, object], json.loads(in_path.read_text(encoding="utf-8")))

    topics_raw = data_obj.get("topics")
    if not isinstance(topics_raw, dict):
        raise ValueError("Invalid topic_gene file format: missing object field 'topics'")
    topics_obj = cast(dict[str, object], topics_raw)

    topics: dict[str, list[str]] = {}
    for topic_id, genes in topics_obj.items():
        if not isinstance(genes, list):
            continue
        genes_list = cast(list[object], genes)
        normalized = [str(g).strip().upper() for g in genes_list if str(g).strip()]
        topics[topic_id] = normalized

    descs: list[TopicDescription] = []
    for tid, genes in sorted(topics.items()):
        descs.append(_make_description_llm(topic_id=tid, genes=genes))
    out = {
        "source": "llm_from_topic_gene",
        "topic_count": len(descs),
        "topics": descs,
    }

    out_json = data_dir / "topic_descriptions_hallmark_2020.json"
    _ = out_json.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote: {out_json} (topics={len(descs)})")

    # Also write a simple SQL seed script for later use.
    out_sql = data_dir / "topic_descriptions_hallmark_2020.sql"
    lines: list[str] = []
    lines.append("CREATE TABLE IF NOT EXISTS topic_description (")
    lines.append("  topic_id TEXT PRIMARY KEY,")
    lines.append("  description TEXT NOT NULL,")
    lines.append("  gene_count INTEGER NOT NULL,")
    lines.append("  top_genes_json TEXT NOT NULL")
    lines.append(");")
    lines.append("")
    for t in descs:
        topic_id = t["topic_id"].replace("'", "''")
        description = t["description"].replace("'", "''")
        gene_count = t["gene_count"]
        top_genes_json = json.dumps(t["top_genes"], ensure_ascii=True).replace("'", "''")
        lines.append(
            f"INSERT OR REPLACE INTO topic_description(topic_id, description, gene_count, top_genes_json) VALUES ('{topic_id}', '{description}', {gene_count}, '{top_genes_json}');"
        )
    _ = out_sql.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_sql}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
