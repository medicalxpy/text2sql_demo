from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TopicStore:
    topics: dict[str, dict[str, float]]  # topic_id -> {gene_symbol: weight}



    @staticmethod
    def load_from_db(db_path: str | Path) -> "TopicStore":
        """Load topic-gene weights directly from the SQLite topic_gene table."""
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        topics: dict[str, dict[str, float]] = {}
        con = sqlite3.connect(str(db_path))
        try:
            rows = con.execute(
                "SELECT topic_id, gene_symbol, weight FROM topic_gene"
            ).fetchall()
            for topic_id, gene_symbol, weight in rows:
                topics.setdefault(topic_id, {})[gene_symbol.upper()] = float(weight)
        finally:
            con.close()

        if not topics:
            raise ValueError(f"No topic_gene rows found in {db_path}")
        return TopicStore(topics=topics)

    @staticmethod
    def load_from_json(json_path: str | Path) -> "TopicStore":
        """Backward-compatible loader from JSON gene-list file (weight=1.0)."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Missing topic gene set file: {json_path}")

        data = json.loads(json_path.read_text(encoding="utf-8"))
        raw_topics: dict[str, list[str]] = data["topics"]
        topics: dict[str, dict[str, float]] = {}
        for topic_id, genes in raw_topics.items():
            topics[topic_id] = {g.upper(): 1.0 for g in genes}
        return TopicStore(topics=topics)

    @staticmethod
    def load_default() -> "TopicStore":
        """Prefer SQLite DB (has weights), fall back to JSON (weight=1.0)."""
        base = Path(__file__).resolve().parent
        db_path = base / "data" / "sim_store_v1.sqlite"
        if db_path.exists():
            return TopicStore.load_from_db(db_path)

        json_path = base / "data" / "topic_gene_hallmark_2020.json"
        if json_path.exists():
            return TopicStore.load_from_json(json_path)

        raise FileNotFoundError(
            "No topic data found. Run:\n"
            "  python -m text2sql_demo.scripts.build_simulated_store   (SQLite, preferred)\n"
            "  python -m text2sql_demo.scripts.build_topic_store        (JSON fallback)"
        )


def compute_topic_candidates(
    *,
    query_genes: list[str],
    store: TopicStore,
    top_m: int = 10,
) -> list[dict[str, str | float]]:
    """Gene-driven topic candidate retrieval.

    Score = sum of topic_gene weights for overlapping genes.
    """
    q = {g.upper() for g in query_genes if g}
    scored: list[tuple[str, float]] = []
    for topic_id, gene_weights in store.topics.items():
        if not gene_weights:
            continue
        overlap = q & gene_weights.keys()
        if overlap:
            score = sum(gene_weights[g] for g in overlap)
            scored.append((topic_id, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    top = scored[: max(0, top_m)]
    return [{"topic_id": tid, "topic_score": round(score, 4)} for tid, score in top]
