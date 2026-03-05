from __future__ import annotations

import json
from pathlib import Path


def load_topic_descriptions() -> dict[str, dict[str, object]]:
    """Load topic descriptions keyed by topic_id."""
    base = Path(__file__).resolve().parent
    p = base / "data" / "topic_descriptions_hallmark_2020.json"
    if not p.exists():
        raise FileNotFoundError(
            "Missing topic descriptions file: text2sql_demo/data/topic_descriptions_hallmark_2020.json. "
            "Run: python -m text2sql_demo.scripts.build_topic_descriptions"
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    topics = data.get("topics", [])
    out: dict[str, dict[str, object]] = {}
    for t in topics:
        if not isinstance(t, dict):
            continue
        topic_id = str(t.get("topic_id", "")).strip()
        if not topic_id:
            continue
        out[topic_id] = dict(t)
    return out
