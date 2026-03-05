from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast


SCHEMA_SQL = """
CREATE TABLE topic_gene (
  topic_id TEXT NOT NULL,
  gene_symbol TEXT NOT NULL,
  weight REAL NOT NULL,
  PRIMARY KEY (topic_id, gene_symbol)
);

CREATE TABLE cell_topic (
  cell_id TEXT NOT NULL,
  topic_id TEXT NOT NULL,
  weight REAL NOT NULL,
  PRIMARY KEY (cell_id, topic_id)
);

CREATE TABLE cell (
  cell_id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL
);

CREATE TABLE dataset (
  dataset_id TEXT PRIMARY KEY,
  dataset_name TEXT NOT NULL
);

CREATE TABLE topic_description (
  topic_id TEXT PRIMARY KEY,
  description TEXT NOT NULL
);

CREATE INDEX idx_topic_gene_symbol ON topic_gene(gene_symbol);
CREATE INDEX idx_topic_gene_topic ON topic_gene(topic_id);
CREATE INDEX idx_cell_topic_topic ON cell_topic(topic_id);
CREATE INDEX idx_cell_topic_cell ON cell_topic(cell_id);
CREATE INDEX idx_cell_dataset ON cell(dataset_id);
"""


@dataclass(frozen=True)
class BuildConfig:
    name: str
    seed: int
    source: str
    version: str
    dataset_count: int
    cells_per_dataset_min: int
    cells_per_dataset_max: int
    topics_per_cell_min: int
    topics_per_cell_max: int
    dominant_topics_per_dataset_min: int
    dominant_topics_per_dataset_max: int
    dominant_topic_sampling_bias: float
    topic_gene_weight_noise: float
    cell_topic_weight_min: float
    cell_topic_weight_max: float


def _load_config(config_path: Path) -> BuildConfig:
    raw = _read_json_object(config_path)

    cfg = BuildConfig(
        name=_required_str(raw, "name"),
        seed=_required_int(raw, "seed"),
        source=_required_str(raw, "source"),
        version=_required_str(raw, "version"),
        dataset_count=_required_int(raw, "dataset_count"),
        cells_per_dataset_min=_required_int(raw, "cells_per_dataset_min"),
        cells_per_dataset_max=_required_int(raw, "cells_per_dataset_max"),
        topics_per_cell_min=_required_int(raw, "topics_per_cell_min"),
        topics_per_cell_max=_required_int(raw, "topics_per_cell_max"),
        dominant_topics_per_dataset_min=_required_int(raw, "dominant_topics_per_dataset_min"),
        dominant_topics_per_dataset_max=_required_int(raw, "dominant_topics_per_dataset_max"),
        dominant_topic_sampling_bias=_required_float(raw, "dominant_topic_sampling_bias"),
        topic_gene_weight_noise=_required_float(raw, "topic_gene_weight_noise"),
        cell_topic_weight_min=_required_float(raw, "cell_topic_weight_min"),
        cell_topic_weight_max=_required_float(raw, "cell_topic_weight_max"),
    )

    if cfg.dataset_count <= 0:
        raise ValueError("dataset_count must be > 0")
    if cfg.cells_per_dataset_min <= 0 or cfg.cells_per_dataset_max <= 0:
        raise ValueError("cells_per_dataset_min/max must be > 0")
    if cfg.cells_per_dataset_min > cfg.cells_per_dataset_max:
        raise ValueError("cells_per_dataset_min cannot be greater than cells_per_dataset_max")
    if cfg.topics_per_cell_min <= 0 or cfg.topics_per_cell_max <= 0:
        raise ValueError("topics_per_cell_min/max must be > 0")
    if cfg.topics_per_cell_min > cfg.topics_per_cell_max:
        raise ValueError("topics_per_cell_min cannot be greater than topics_per_cell_max")
    if cfg.dominant_topics_per_dataset_min <= 0 or cfg.dominant_topics_per_dataset_max <= 0:
        raise ValueError("dominant_topics_per_dataset_min/max must be > 0")
    if cfg.dominant_topics_per_dataset_min > cfg.dominant_topics_per_dataset_max:
        raise ValueError(
            "dominant_topics_per_dataset_min cannot be greater than dominant_topics_per_dataset_max"
        )
    if not (0.0 <= cfg.dominant_topic_sampling_bias <= 1.0):
        raise ValueError("dominant_topic_sampling_bias must be in [0, 1]")
    if cfg.topic_gene_weight_noise < 0:
        raise ValueError("topic_gene_weight_noise must be >= 0")
    if cfg.cell_topic_weight_min <= 0 or cfg.cell_topic_weight_max <= 0:
        raise ValueError("cell_topic_weight_min/max must be > 0")
    if cfg.cell_topic_weight_min > cfg.cell_topic_weight_max:
        raise ValueError("cell_topic_weight_min cannot be greater than cell_topic_weight_max")

    return cfg


def _maybe_refresh_topic_assets(*, refresh: bool) -> None:
    if not refresh:
        return

    from .build_topic_descriptions import main as build_topic_descriptions_main
    from .build_topic_store import main as build_topic_store_main

    rc_topic_store = build_topic_store_main()
    if rc_topic_store != 0:
        raise RuntimeError(f"build_topic_store failed with exit code {rc_topic_store}")

    rc_topic_desc = build_topic_descriptions_main()
    if rc_topic_desc != 0:
        raise RuntimeError(f"build_topic_descriptions failed with exit code {rc_topic_desc}")


def _load_topic_gene(base_dir: Path) -> dict[str, list[str]]:
    path = base_dir / "data" / "topic_gene_hallmark_2020.json"
    if not path.exists():
        raise FileNotFoundError(
            "Missing topic gene file: text2sql_demo/data/topic_gene_hallmark_2020.json. "
            "Run: python -m text2sql_demo.scripts.build_topic_store"
        )

    payload = _read_json_object(path)
    topics_obj = payload.get("topics")
    if not isinstance(topics_obj, dict):
        raise ValueError("Invalid topic gene file: expected object field 'topics'")

    topics_raw = cast(dict[str, object], topics_obj)
    topics: dict[str, list[str]] = {}
    for topic_id, genes_obj in sorted(topics_raw.items()):
        if not isinstance(topic_id, str):
            continue
        if not isinstance(genes_obj, list):
            continue
        seen: set[str] = set()
        genes: list[str] = []
        for g in genes_obj:
            gs = str(g).strip().upper()
            if not gs or gs in seen:
                continue
            seen.add(gs)
            genes.append(gs)
        if genes:
            topics[topic_id] = genes

    if not topics:
        raise ValueError("Topic gene file contains no usable topics")
    return topics


def _load_topic_descriptions(base_dir: Path, topic_ids: list[str]) -> dict[str, dict[str, object]]:
    path = base_dir / "data" / "topic_descriptions_hallmark_2020.json"
    if not path.exists():
        raise FileNotFoundError(
            "Missing topic description file: text2sql_demo/data/topic_descriptions_hallmark_2020.json. "
            "Run: python -m text2sql_demo.scripts.build_topic_descriptions"
        )

    payload = _read_json_object(path)
    topics_obj = payload.get("topics")
    if not isinstance(topics_obj, list):
        raise ValueError("Invalid topic description file: expected list field 'topics'")

    index: dict[str, dict[str, object]] = {}
    for item in topics_obj:
        if not isinstance(item, dict):
            continue
        raw = cast(dict[str, object], item)
        topic_id = str(raw.get("topic_id", "")).strip()
        description = str(raw.get("description", "")).strip()
        if not topic_id or not description:
            continue

        gene_count_obj = raw.get("gene_count", 0)
        if isinstance(gene_count_obj, (int, float)):
            gene_count = int(gene_count_obj)
        else:
            gene_count = 0

        top_genes_obj = raw.get("top_genes", [])
        top_genes: list[str] = []
        if isinstance(top_genes_obj, list):
            top_genes = [str(x).strip().upper() for x in top_genes_obj if str(x).strip()]

        index[topic_id] = {
            "topic_id": topic_id,
            "description": description,
            "gene_count": gene_count,
            "top_genes": top_genes,
        }

    missing = [tid for tid in topic_ids if tid not in index]
    if missing:
        missing_preview = ", ".join(missing[:5])
        raise ValueError(
            "Topic descriptions are incomplete. Missing topic_ids: "
            f"{missing_preview} (total missing={len(missing)}). "
            "Re-run: python -m text2sql_demo.scripts.build_topic_descriptions"
        )

    return index




def _build_topic_gene_and_desc_rows(
    *,
    topics: dict[str, list[str]],
    topic_descriptions: dict[str, dict[str, object]],
    cfg: BuildConfig,
    rng: random.Random,
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str]]]:
    topic_gene_rows: list[tuple[str, str, float]] = []
    topic_description_rows: list[tuple[str, str]] = []

    for topic_id, genes in sorted(topics.items()):
        n = max(1, len(genes))
        for idx, gene_symbol in enumerate(genes):
            rank = 1.0 - (idx / n)
            base = 0.25 + (0.75 * rank)
            noise = rng.uniform(-cfg.topic_gene_weight_noise, cfg.topic_gene_weight_noise)
            weight = min(1.0, max(0.01, base + noise))
            topic_gene_rows.append((topic_id, gene_symbol, round(weight, 6)))

        desc_obj = topic_descriptions[topic_id]
        description = str(desc_obj.get("description", "")).strip()
        if not description:
            raise ValueError(f"Invalid empty description for topic_id={topic_id}")
        topic_description_rows.append((topic_id, description))

    return topic_gene_rows, topic_description_rows


def _sample_topics_for_cell(
    *,
    topic_ids: list[str],
    dominant_topic_ids: list[str],
    k: int,
    dominant_bias: float,
    rng: random.Random,
) -> list[str]:
    chosen: set[str] = set()
    while len(chosen) < k:
        use_dominant = bool(dominant_topic_ids) and rng.random() < dominant_bias
        if use_dominant:
            candidate = rng.choice(dominant_topic_ids)
        else:
            candidate = rng.choice(topic_ids)
        chosen.add(candidate)
    return list(chosen)


def _normalized_weights(*, k: int, min_weight: float, max_weight: float, rng: random.Random) -> list[float]:
    raw = [rng.uniform(min_weight, max_weight) for _ in range(k)]
    total = sum(raw)
    if total <= 0:
        return [round(1.0 / k, 6) for _ in range(k)]

    normalized = [round(v / total, 6) for v in raw]
    drift = round(1.0 - sum(normalized), 6)
    normalized[0] = round(normalized[0] + drift, 6)
    return normalized


def _build_dataset_rows(
    *,
    cfg: BuildConfig,
    topic_ids: list[str],
    rng: random.Random,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str, float]]]:
    dataset_rows: list[tuple[str, str]] = []
    cell_rows: list[tuple[str, str]] = []
    cell_topic_rows: list[tuple[str, str, float]] = []

    next_cell_idx = 1
    topic_count = len(topic_ids)
    if topic_count == 0:
        raise ValueError("Cannot build dataset rows: no topics available")

    max_topics_per_cell = min(cfg.topics_per_cell_max, topic_count)
    dominant_max = min(cfg.dominant_topics_per_dataset_max, topic_count)

    for dataset_idx in range(1, cfg.dataset_count + 1):
        dataset_id = f"ds_{dataset_idx:04d}"
        dataset_name = f"Dataset {dataset_idx:03d}"

        cell_count = rng.randint(cfg.cells_per_dataset_min, cfg.cells_per_dataset_max)
        dominant_n = rng.randint(cfg.dominant_topics_per_dataset_min, dominant_max)
        dominant_topic_ids = rng.sample(topic_ids, dominant_n)

        dataset_rows.append((dataset_id, dataset_name))

        for _ in range(cell_count):
            cell_id = f"cell_{next_cell_idx:07d}"
            next_cell_idx += 1

            cell_rows.append((cell_id, dataset_id))

            k = rng.randint(cfg.topics_per_cell_min, max_topics_per_cell)
            chosen_topics = _sample_topics_for_cell(
                topic_ids=topic_ids,
                dominant_topic_ids=dominant_topic_ids,
                k=k,
                dominant_bias=cfg.dominant_topic_sampling_bias,
                rng=rng,
            )
            weights = _normalized_weights(
                k=k,
                min_weight=cfg.cell_topic_weight_min,
                max_weight=cfg.cell_topic_weight_max,
                rng=rng,
            )

            for topic_id, weight in zip(chosen_topics, weights):
                cell_topic_rows.append((cell_id, topic_id, weight))

    return dataset_rows, cell_rows, cell_topic_rows


def _write_sqlite(
    *,
    sqlite_path: Path,
    topic_gene_rows: list[tuple[str, str, float]],
    topic_description_rows: list[tuple[str, str]],
    dataset_rows: list[tuple[str, str]],
    cell_rows: list[tuple[str, str]],
    cell_topic_rows: list[tuple[str, str, float]],
) -> None:
    if sqlite_path.exists():
        sqlite_path.unlink()

    conn = sqlite3.connect(str(sqlite_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_SQL)

        conn.executemany(
            "INSERT INTO topic_gene(topic_id, gene_symbol, weight) VALUES (?, ?, ?)",
            topic_gene_rows,
        )
        conn.executemany(
            "INSERT INTO topic_description(topic_id, description) VALUES (?, ?)",
            topic_description_rows,
        )
        conn.executemany(
            "INSERT INTO dataset(dataset_id, dataset_name) VALUES (?, ?)",
            dataset_rows,
        )
        conn.executemany("INSERT INTO cell(cell_id, dataset_id) VALUES (?, ?)", cell_rows)
        conn.executemany("INSERT INTO cell_topic(cell_id, topic_id, weight) VALUES (?, ?, ?)", cell_topic_rows)
        conn.commit()
    finally:
        conn.close()


def _write_sql_dump(*, sqlite_path: Path, sql_path: Path) -> None:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        with sql_path.open("w", encoding="utf-8") as f:
            for line in conn.iterdump():
                f.write(line)
                f.write("\n")
    finally:
        conn.close()


def _write_manifest(
    *,
    manifest_path: Path,
    cfg: BuildConfig,
    sqlite_path: Path,
    sql_path: Path,
    topic_gene_rows: list[tuple[str, str, float]],
    topic_description_rows: list[tuple[str, str]],
    dataset_rows: list[tuple[str, str]],
    cell_rows: list[tuple[str, str]],
    cell_topic_rows: list[tuple[str, str, float]],
) -> None:
    row_counts = {
        "topic_gene": len(topic_gene_rows),
        "topic_description": len(topic_description_rows),
        "dataset": len(dataset_rows),
        "cell": len(cell_rows),
        "cell_topic": len(cell_topic_rows),
    }

    avg_cells_per_dataset = _safe_div(len(cell_rows), len(dataset_rows))
    avg_topics_per_cell = _safe_div(len(cell_topic_rows), len(cell_rows))
    topic_count = len(set(r[0] for r in topic_gene_rows))
    cell_topic_density = _safe_div(len(cell_topic_rows), len(cell_rows) * topic_count)

    dominant_preview = _compute_dataset_topic_preview(cell_topic_rows, cell_rows)

    manifest = {
        "name": cfg.name,
        "seed": cfg.seed,
        "source": cfg.source,
        "version": cfg.version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "assets": {
            "sqlite": str(sqlite_path),
            "sql": str(sql_path),
        },
        "row_counts": row_counts,
        "metrics": {
            "avg_cells_per_dataset": round(avg_cells_per_dataset, 4),
            "avg_topics_per_cell": round(avg_topics_per_cell, 4),
            "cell_topic_density": round(cell_topic_density, 8),
        },
        "dataset_topic_preview": dominant_preview,
        "config": {
            "dataset_count": cfg.dataset_count,
            "cells_per_dataset_min": cfg.cells_per_dataset_min,
            "cells_per_dataset_max": cfg.cells_per_dataset_max,
            "topics_per_cell_min": cfg.topics_per_cell_min,
            "topics_per_cell_max": cfg.topics_per_cell_max,
            "dominant_topics_per_dataset_min": cfg.dominant_topics_per_dataset_min,
            "dominant_topics_per_dataset_max": cfg.dominant_topics_per_dataset_max,
            "dominant_topic_sampling_bias": cfg.dominant_topic_sampling_bias,
            "topic_gene_weight_noise": cfg.topic_gene_weight_noise,
            "cell_topic_weight_min": cfg.cell_topic_weight_min,
            "cell_topic_weight_max": cfg.cell_topic_weight_max,
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _compute_dataset_topic_preview(
    cell_topic_rows: list[tuple[str, str, float]],
    cell_rows: list[tuple[str, str]],
) -> dict[str, list[dict[str, object]]]:
    cell_to_dataset = {cell_id: dataset_id for cell_id, dataset_id in cell_rows}
    dataset_topic_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for cell_id, topic_id, weight in cell_topic_rows:
        dataset_id = cell_to_dataset.get(cell_id)
        if not dataset_id:
            continue
        dataset_topic_scores[dataset_id][topic_id] += weight

    preview: dict[str, list[dict[str, object]]] = {}
    for dataset_id in sorted(dataset_topic_scores.keys())[:10]:
        topic_scores = dataset_topic_scores.get(dataset_id, {})
        top3 = sorted(topic_scores.items(), key=lambda x: (-x[1], x[0]))[:3]
        preview[dataset_id] = [
            {"topic_id": topic_id, "score": round(score, 4)} for topic_id, score in top3
        ]
    return preview


def _safe_div(numer: int, denom: int) -> float:
    if denom == 0:
        return 0.0
    return numer / denom


def _read_json_object(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast(dict[str, object], raw)


def _required_str(obj: dict[str, object], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid or missing string config field: {key}")
    return value.strip()


def _required_int(obj: dict[str, object], key: str) -> int:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid or missing integer config field: {key}")
    return int(value)


def _required_float(obj: dict[str, object], key: str) -> float:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid or missing float config field: {key}")
    return float(value)




def _default_config_path(base_dir: Path) -> Path:
    return base_dir / "config" / "sim_store_v1.json"


def _parse_args(base_dir: Path, argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a full 5-table simulated topic store.")
    parser.add_argument(
        "--config",
        default=str(_default_config_path(base_dir)),
        help="Path to JSON config file (default: text2sql_demo/config/sim_store_v1.json)",
    )
    parser.add_argument(
        "--refresh-topic-assets",
        action="store_true",
        help="Rebuild topic_gene and topic_description assets before constructing the 5-table store.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    base_dir = Path(__file__).resolve().parents[1]
    args = _parse_args(base_dir, argv)

    _maybe_refresh_topic_assets(refresh=bool(args.refresh_topic_assets))

    config_path = Path(args.config).resolve()
    cfg = _load_config(config_path)
    rng = random.Random(cfg.seed)

    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = data_dir / f"{cfg.name}.sqlite"
    sql_path = data_dir / f"{cfg.name}.sql"
    manifest_path = data_dir / f"{cfg.name}_manifest.json"

    topics = _load_topic_gene(base_dir)
    topic_ids = sorted(topics.keys())
    topic_descriptions = _load_topic_descriptions(base_dir, topic_ids)

    topic_gene_rows, topic_description_rows = _build_topic_gene_and_desc_rows(
        topics=topics,
        topic_descriptions=topic_descriptions,
        cfg=cfg,
        rng=rng,
    )
    dataset_rows, cell_rows, cell_topic_rows = _build_dataset_rows(cfg=cfg, topic_ids=topic_ids, rng=rng)

    _write_sqlite(
        sqlite_path=sqlite_path,
        topic_gene_rows=topic_gene_rows,
        topic_description_rows=topic_description_rows,
        dataset_rows=dataset_rows,
        cell_rows=cell_rows,
        cell_topic_rows=cell_topic_rows,
    )
    _write_sql_dump(sqlite_path=sqlite_path, sql_path=sql_path)
    _write_manifest(
        manifest_path=manifest_path,
        cfg=cfg,
        sqlite_path=sqlite_path,
        sql_path=sql_path,
        topic_gene_rows=topic_gene_rows,
        topic_description_rows=topic_description_rows,
        dataset_rows=dataset_rows,
        cell_rows=cell_rows,
        cell_topic_rows=cell_topic_rows,
    )

    print(f"Wrote: {sqlite_path}")
    print(f"Wrote: {sql_path}")
    print(f"Wrote: {manifest_path}")
    print(
        "Rows: "
        f"topic_gene={len(topic_gene_rows)}, "
        f"topic_description={len(topic_description_rows)}, "
        f"dataset={len(dataset_rows)}, "
        f"cell={len(cell_rows)}, "
        f"cell_topic={len(cell_topic_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
