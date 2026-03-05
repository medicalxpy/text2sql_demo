from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_TABLE_COLUMNS: dict[str, set[str]] = {
    "topic_gene": {"topic_id", "gene_symbol", "weight"},
    "topic_description": {"topic_id", "description"},
    "dataset": {"dataset_id", "dataset_name"},
    "cell": {"cell_id", "dataset_id"},
    "cell_topic": {"cell_id", "topic_id", "weight"},
}


REQUIRED_INDEXES: dict[str, set[str]] = {
    "topic_gene": {"idx_topic_gene_symbol", "idx_topic_gene_topic"},
    "cell_topic": {"idx_cell_topic_topic", "idx_cell_topic_cell"},
    "cell": {"idx_cell_dataset"},
}


def _parse_args(base_dir: Path, argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a 5-table simulated topic store SQLite DB.")
    parser.add_argument(
        "--db",
        default=str(base_dir / "data" / "sim_store_v1.sqlite"),
        help="Path to SQLite DB file (default: text2sql_demo/data/sim_store_v1.sqlite)",
    )
    parser.add_argument(
        "--report",
        default=str(base_dir / "data" / "sim_store_v1_validation.json"),
        help="Path to JSON validation report",
    )
    return parser.parse_args(argv)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _get_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(r[1]) for r in rows}


def _get_indexes(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA index_list({table_name})").fetchall()
    return {str(r[1]) for r in rows}


def _scalar_int(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return 0
    return int(row[0])


def _has_duplicate_keys(conn: sqlite3.Connection, table_name: str, cols: list[str]) -> bool:
    cols_expr = ", ".join(cols)
    sql = f"SELECT 1 FROM {table_name} GROUP BY {cols_expr} HAVING COUNT(*) > 1 LIMIT 1"
    return conn.execute(sql).fetchone() is not None


def _validate_join_path(conn: sqlite3.Connection) -> tuple[bool, str]:
    tc_rows = conn.execute(
        """
        SELECT topic_id, AVG(weight) AS score
        FROM cell_topic
        GROUP BY topic_id
        ORDER BY score DESC
        LIMIT 2
        """
    ).fetchall()

    if not tc_rows:
        return False, "cell_topic has no data"

    values_clause = ", ".join(["(?, ?)"] * len(tc_rows))
    params: list[Any] = []
    for topic_id, score in tc_rows:
        params.extend([str(topic_id), float(score)])

    sql = (
        "WITH topic_candidates(topic_id, topic_score) AS ("
        f"VALUES {values_clause}"
        ") "
        "SELECT d.dataset_id, SUM(tc.topic_score * ct.weight) AS score "
        "FROM topic_candidates tc "
        "JOIN cell_topic ct ON ct.topic_id = tc.topic_id "
        "JOIN cell c ON c.cell_id = ct.cell_id "
        "JOIN dataset d ON d.dataset_id = c.dataset_id "
        "GROUP BY d.dataset_id "
        "ORDER BY score DESC "
        "LIMIT 5"
    )
    cur = conn.execute(sql, tuple(params))
    rows = cur.fetchall()
    col_names = [str(c[0]) for c in (cur.description or [])]

    if "dataset_id" not in col_names:
        return False, "join-path query does not return dataset_id"
    if "score" not in col_names:
        return False, "join-path query does not return score"
    if not rows:
        return False, "join-path query returned no rows"
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    base_dir = Path(__file__).resolve().parents[1]
    args = _parse_args(base_dir, argv)
    db_path = Path(args.db).resolve()
    report_path = Path(args.report).resolve()

    if not db_path.exists():
        raise FileNotFoundError(f"DB file not found: {db_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    warnings: list[str] = []
    row_counts: dict[str, int] = {}
    metrics: dict[str, float] = {}

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        for table_name, required_cols in REQUIRED_TABLE_COLUMNS.items():
            if not _table_exists(conn, table_name):
                errors.append(f"Missing table: {table_name}")
                continue
            actual_cols = _get_columns(conn, table_name)
            missing_cols = sorted(required_cols - actual_cols)
            if missing_cols:
                errors.append(f"Table {table_name} missing columns: {', '.join(missing_cols)}")

        for table_name, required_indexes in REQUIRED_INDEXES.items():
            if not _table_exists(conn, table_name):
                continue
            actual_indexes = _get_indexes(conn, table_name)
            missing_idx = sorted(required_indexes - actual_indexes)
            if missing_idx:
                errors.append(f"Table {table_name} missing indexes: {', '.join(missing_idx)}")

        for table_name in REQUIRED_TABLE_COLUMNS:
            if not _table_exists(conn, table_name):
                continue
            count = _scalar_int(conn, f"SELECT COUNT(*) FROM {table_name}")
            row_counts[table_name] = count
            if count <= 0:
                errors.append(f"Table {table_name} is empty")

        duplicate_specs = [
            ("topic_gene", ["topic_id", "gene_symbol"]),
            ("topic_description", ["topic_id"]),
            ("dataset", ["dataset_id"]),
            ("cell", ["cell_id"]),
            ("cell_topic", ["cell_id", "topic_id"]),
        ]
        for table_name, cols in duplicate_specs:
            if not _table_exists(conn, table_name):
                continue
            if _has_duplicate_keys(conn, table_name, cols):
                errors.append(f"Duplicate key rows found in {table_name} for ({', '.join(cols)})")

        # FK: topic_gene.topic_id should exist in topic_description or other topic tables
        # Since there's no standalone topic table, check topic_id consistency across tables
        topic_ids_in_gene = set(
            r[0] for r in conn.execute("SELECT DISTINCT topic_id FROM topic_gene").fetchall()
        )
        topic_ids_in_desc = set(
            r[0] for r in conn.execute("SELECT DISTINCT topic_id FROM topic_description").fetchall()
        )
        orphan_desc = topic_ids_in_desc - topic_ids_in_gene
        if orphan_desc:
            errors.append(f"topic_description has {len(orphan_desc)} topic_ids not in topic_gene")

        # FK: cell.dataset_id -> dataset.dataset_id
        fk_cell_dataset = _scalar_int(
            conn,
            """
            SELECT COUNT(*)
            FROM cell c
            LEFT JOIN dataset d ON d.dataset_id = c.dataset_id
            WHERE d.dataset_id IS NULL
            """,
        )
        if fk_cell_dataset != 0:
            errors.append(f"cell has {fk_cell_dataset} orphan dataset_id rows")

        # FK: cell_topic.cell_id -> cell.cell_id
        fk_cell_topic_cell = _scalar_int(
            conn,
            """
            SELECT COUNT(*)
            FROM cell_topic ct
            LEFT JOIN cell c ON c.cell_id = ct.cell_id
            WHERE c.cell_id IS NULL
            """,
        )
        if fk_cell_topic_cell != 0:
            errors.append(f"cell_topic has {fk_cell_topic_cell} orphan cell_id rows")

        # FK: cell_topic.topic_id should exist in topic_gene
        fk_cell_topic_topic = _scalar_int(
            conn,
            """
            SELECT COUNT(DISTINCT ct.topic_id)
            FROM cell_topic ct
            WHERE ct.topic_id NOT IN (SELECT DISTINCT topic_id FROM topic_gene)
            """,
        )
        if fk_cell_topic_topic != 0:
            errors.append(f"cell_topic has {fk_cell_topic_topic} orphan topic_id values")


        join_path_ok, join_path_msg = _validate_join_path(conn)
        if not join_path_ok:
            errors.append(f"Join-path sanity query failed: {join_path_msg}")

        topic_n = len(topic_ids_in_gene) if topic_ids_in_gene else 0
        cell_n = row_counts.get("cell", 0)
        cell_topic_n = row_counts.get("cell_topic", 0)

        avg_topics_per_cell = _safe_div(cell_topic_n, cell_n)
        cell_topic_density = _safe_div(cell_topic_n, cell_n * topic_n)
        avg_cells_per_dataset = _safe_div(row_counts.get("cell", 0), row_counts.get("dataset", 0))

        metrics["avg_topics_per_cell"] = round(avg_topics_per_cell, 6)
        metrics["cell_topic_density"] = round(cell_topic_density, 10)
        metrics["avg_cells_per_dataset"] = round(avg_cells_per_dataset, 6)

        if avg_topics_per_cell < 1.0 or avg_topics_per_cell > 3.1:
            warnings.append("avg_topics_per_cell is outside expected range [1.0, 3.1]")
        if cell_topic_density <= 0.0:
            warnings.append("cell_topic_density is zero")
        if cell_topic_density > 0.5:
            warnings.append("cell_topic_density is unexpectedly high (>0.5)")

    finally:
        conn.close()

    report = {
        "db_path": str(db_path),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "warnings": warnings,
        "row_counts": row_counts,
        "metrics": metrics,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Wrote: {report_path}")
    if errors:
        print(f"Validation FAILED ({len(errors)} error(s), {len(warnings)} warning(s)).")
        for err in errors:
            print(f"- ERROR: {err}")
        for w in warnings:
            print(f"- WARNING: {w}")
        return 1

    print(f"Validation PASSED ({len(warnings)} warning(s)).")
    for w in warnings:
        print(f"- WARNING: {w}")
    return 0


def _safe_div(numer: int, denom: int) -> float:
    if denom == 0:
        return 0.0
    return numer / denom


if __name__ == "__main__":
    raise SystemExit(main())
