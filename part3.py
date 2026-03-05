from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path


_FORBIDDEN_SQL_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "REPLACE",
    "ATTACH",
    "DETACH",
    "VACUUM",
    "PRAGMA",
}


def default_db_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "sim_store_v1.sqlite"


def run_part3(
    *,
    sql_out: dict[str, object],
    db_path: str | Path,
    top_k: int,
    query_timeout_seconds: float = 2.0,
) -> dict[str, object]:
    verifier = run_verifier_executor(
        sql_out=sql_out,
        db_path=db_path,
        query_timeout_seconds=query_timeout_seconds,
    )
    selector = run_selector(verifier=verifier, top_k=top_k)
    return {
        "passed": verifier["passed"],
        "failed": verifier["failed"],
        "selected_id": selector["selected_id"],
        "selected_sql": selector["selected_sql"],
        "datasets": selector["datasets"],
        "rationale": selector["rationale"],
        "evidence_pack": selector["evidence_pack"],
    }


def run_verifier_executor(
    *,
    sql_out: dict[str, object],
    db_path: str | Path,
    query_timeout_seconds: float = 2.0,
) -> dict[str, list[dict[str, object]]]:
    candidates_obj = sql_out.get("candidates", [])
    if not isinstance(candidates_obj, list):
        return {
            "passed": [],
            "failed": [{"id": None, "reason": "sql_out.candidates is not a list"}],
        }

    db_file = Path(db_path)
    if not db_file.exists():
        return {
            "passed": [],
            "failed": [{"id": None, "reason": f"SQLite DB not found: {db_file}"}],
        }

    passed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        _ = conn.execute("PRAGMA query_only = ON;")
        for cand_obj in candidates_obj:
            if not isinstance(cand_obj, dict):
                failed.append({"id": None, "reason": "candidate is not a JSON object"})
                continue

            candidate = cand_obj
            cand_id = _candidate_id(candidate.get("id"))
            sql_text = str(candidate.get("sql", "")).strip()
            notes = str(candidate.get("notes", "")).strip()

            try:
                checks = _run_static_checks(sql_text)
            except ValueError as exc:
                failed.append(
                    {
                        "id": cand_id,
                        "sql": sql_text,
                        "reason": str(exc),
                        "notes": notes,
                    }
                )
                continue

            normalized_sql = str(checks["normalized_sql"])

            try:
                _run_explain_query_plan(conn=conn, sql_text=normalized_sql, timeout_seconds=query_timeout_seconds)
            except TimeoutError as exc:
                failed.append(
                    {
                        "id": cand_id,
                        "sql": normalized_sql,
                        "reason": f"EXPLAIN timeout: {exc}",
                        "notes": notes,
                    }
                )
                continue
            except sqlite3.Error as exc:
                failed.append(
                    {
                        "id": cand_id,
                        "sql": normalized_sql,
                        "reason": f"EXPLAIN failed: {exc}",
                        "notes": notes,
                    }
                )
                continue

            try:
                rows, col_names = _execute_query_with_timeout(
                    conn=conn,
                    sql_text=normalized_sql,
                    timeout_seconds=query_timeout_seconds,
                )
            except TimeoutError as exc:
                failed.append(
                    {
                        "id": cand_id,
                        "sql": normalized_sql,
                        "reason": f"Execution timeout: {exc}",
                        "notes": notes,
                    }
                )
                continue
            except sqlite3.Error as exc:
                failed.append(
                    {
                        "id": cand_id,
                        "sql": normalized_sql,
                        "reason": f"Execution failed: {exc}",
                        "notes": notes,
                    }
                )
                continue

            shape_ok, shape_reason = _validate_result_shape(col_names)
            if not shape_ok:
                failed.append(
                    {
                        "id": cand_id,
                        "sql": normalized_sql,
                        "reason": shape_reason,
                        "notes": notes,
                    }
                )
                continue

            datasets = _format_dataset_rows(rows)
            dataset_ids = [str(d.get("dataset_id", "")) for d in datasets if str(d.get("dataset_id", ""))]
            metadata = _fetch_dataset_metadata(conn=conn, dataset_ids=dataset_ids)
            merged = _merge_dataset_metadata(datasets=datasets, metadata_by_id=metadata)

            passed.append(
                {
                    "id": cand_id,
                    "sql": normalized_sql,
                    "notes": notes,
                    "row_count": len(datasets),
                    "shape_ok": True,
                    "datasets": merged,
                    "checks": {
                        "has_topic_candidates_cte": checks["has_topic_candidates_cte"],
                        "has_required_path_tables": checks["has_required_path_tables"],
                        "has_order_by_score_desc": checks["has_order_by_score_desc"],
                        "has_limit": checks["has_limit"],
                    },
                }
            )
    finally:
        conn.close()

    return {
        "passed": passed,
        "failed": failed,
    }


def run_selector(*, verifier: dict[str, list[dict[str, object]]], top_k: int) -> dict[str, object]:
    passed = verifier.get("passed", [])
    failed = verifier.get("failed", [])
    if not passed:
        return {
            "selected_id": None,
            "selected_sql": None,
            "datasets": [],
            "rationale": _no_pass_rationale(failed),
            "evidence_pack": {
                "passed_count": 0,
                "failed_count": len(failed),
                "failed": failed,
            },
        }

    best = sorted(passed, key=_selector_sort_key)[0]
    datasets_obj = best.get("datasets", [])
    datasets: list[dict[str, object]] = []
    if isinstance(datasets_obj, list):
        datasets = [d for d in datasets_obj if isinstance(d, dict)][: max(0, top_k)]

    rationale = (
        "Selected the SQL candidate after passing all verifier checks "
        "(static analysis, EXPLAIN, execution, shape validation)."
    )

    return {
        "selected_id": best.get("id"),
        "selected_sql": best.get("sql"),
        "datasets": datasets,
        "rationale": rationale,
        "evidence_pack": {
            "passed_count": len(passed),
            "failed_count": len(failed),
            "selected_checks": best.get("checks", {}),
            "selected_row_count": best.get("row_count", 0),
            "failed": failed,
        },
    }


def _selector_sort_key(item: dict[str, object]) -> tuple[int, float, int]:
    datasets_obj = item.get("datasets", [])
    datasets = [d for d in datasets_obj if isinstance(d, dict)] if isinstance(datasets_obj, list) else []
    has_rows = 1 if datasets else 0

    if datasets:
        top_score_obj = datasets[0].get("score", 0.0)
        try:
            top_score = float(top_score_obj)
        except (TypeError, ValueError):
            top_score = 0.0
    else:
        top_score = 0.0

    cid = _candidate_id(item.get("id"))
    return (-has_rows, -top_score, cid if cid >= 0 else 999999)


def _no_pass_rationale(failed: list[dict[str, object]]) -> str:
    if not failed:
        return "No SQL candidates were provided to Part3."
    reasons: list[str] = []
    for item in failed[:3]:
        reason = str(item.get("reason", "unknown failure")).strip()
        if reason:
            reasons.append(reason)
    if not reasons:
        return "All SQL candidates failed verifier checks."
    return "All SQL candidates failed verifier checks: " + "; ".join(reasons)


def _candidate_id(value: object) -> int:
    if isinstance(value, bool):
        return -1
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return -1


def _run_static_checks(sql_text: str) -> dict[str, object]:
    normalized = _normalize_sql(sql_text)
    lower_sql = normalized.lower()

    if not normalized:
        raise ValueError("SQL is empty")

    first_token = _first_token(normalized)
    if first_token not in {"select", "with"}:
        raise ValueError("SQL must start with SELECT or WITH")

    if _contains_multiple_statements(normalized):
        raise ValueError("SQL must contain exactly one statement")

    forbidden = _find_forbidden_keyword(normalized)
    if forbidden:
        raise ValueError(f"SQL contains forbidden keyword: {forbidden}")

    has_topic_candidates_cte = bool(
        re.search(r"\bwith\s+topic_candidates\b", lower_sql)
        and re.search(r"\bvalues\b", lower_sql)
    )
    if not has_topic_candidates_cte:
        raise ValueError("SQL must define topic_candidates CTE with VALUES")

    path_ok = _has_required_path_tables(lower_sql)
    if not path_ok:
        raise ValueError("SQL must include path tables: topic_candidates, cell_topic, cell, dataset")

    has_order = bool(re.search(r"\border\s+by\s+score\s+desc\b", lower_sql))
    if not has_order:
        raise ValueError("SQL must include ORDER BY score DESC")

    has_limit = bool(re.search(r"\blimit\b\s+\d+", lower_sql))
    if not has_limit:
        raise ValueError("SQL must include LIMIT <top_k>")

    return {
        "normalized_sql": normalized,
        "has_topic_candidates_cte": has_topic_candidates_cte,
        "has_required_path_tables": path_ok,
        "has_order_by_score_desc": has_order,
        "has_limit": has_limit,
    }


def _normalize_sql(sql_text: str) -> str:
    s = sql_text.strip()
    if s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def _first_token(sql_text: str) -> str:
    m = re.match(r"^\s*([A-Za-z_]+)", sql_text)
    if not m:
        return ""
    return m.group(1).lower()


def _contains_multiple_statements(sql_text: str) -> bool:
    return ";" in sql_text


def _find_forbidden_keyword(sql_text: str) -> str | None:
    upper = sql_text.upper()
    for kw in sorted(_FORBIDDEN_SQL_KEYWORDS):
        if re.search(rf"\b{kw}\b", upper):
            return kw
    return None


def _has_required_path_tables(lower_sql: str) -> bool:
    required = [
        r"(?:from|join)\s+topic_candidates\b",
        r"(?:from|join)\s+cell_topic\b",
        r"(?:from|join)\s+cell\b",
        r"(?:from|join)\s+dataset\b",
    ]
    return all(re.search(p, lower_sql) is not None for p in required)


def _run_explain_query_plan(*, conn: sqlite3.Connection, sql_text: str, timeout_seconds: float) -> None:
    _execute_statement_with_timeout(conn=conn, statement=f"EXPLAIN QUERY PLAN {sql_text}", timeout_seconds=timeout_seconds)


def _execute_query_with_timeout(
    *, conn: sqlite3.Connection, sql_text: str, timeout_seconds: float
) -> tuple[list[sqlite3.Row], list[str]]:
    cursor = _execute_statement_with_timeout(conn=conn, statement=sql_text, timeout_seconds=timeout_seconds)
    rows = cursor.fetchall()
    col_names = [str(c[0]) for c in (cursor.description or [])]
    return rows, col_names


def _execute_statement_with_timeout(
    *, conn: sqlite3.Connection, statement: str, timeout_seconds: float
) -> sqlite3.Cursor:
    deadline = time.monotonic() + max(0.01, timeout_seconds)

    def _progress_handler() -> int:
        if time.monotonic() > deadline:
            return 1
        return 0

    conn.set_progress_handler(_progress_handler, 10000)
    try:
        return conn.execute(statement)
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower() and time.monotonic() > deadline:
            raise TimeoutError("statement timed out") from exc
        raise
    finally:
        conn.set_progress_handler(None, 0)


def _validate_result_shape(col_names: list[str]) -> tuple[bool, str]:
    lowered = {c.lower() for c in col_names}
    if "dataset_id" not in lowered:
        return False, "Result shape invalid: missing dataset_id column"
    if "score" not in lowered:
        return False, "Result shape invalid: missing score column"
    return True, "ok"


def _format_dataset_rows(rows: list[sqlite3.Row]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        data = {k: row[k] for k in row.keys()}
        dataset_id = str(data.get("dataset_id", "")).strip()
        if not dataset_id:
            continue
        try:
            score = float(data.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        out.append({"dataset_id": dataset_id, "score": round(score, 6)})
    return out


def _fetch_dataset_metadata(
    *, conn: sqlite3.Connection, dataset_ids: list[str]
) -> dict[str, dict[str, object]]:
    if not dataset_ids:
        return {}

    unique_ids = sorted({d for d in dataset_ids if d})
    placeholders = ", ".join("?" for _ in unique_ids)
    sql = (
        "SELECT dataset_id, dataset_name "
        "FROM dataset "
        f"WHERE dataset_id IN ({placeholders})"
    )
    rows = conn.execute(sql, tuple(unique_ids)).fetchall()
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        did = str(row["dataset_id"])
        out[did] = {
            "dataset_name": row["dataset_name"],
        }
    return out


def _merge_dataset_metadata(
    *, datasets: list[dict[str, object]], metadata_by_id: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for item in datasets:
        dataset_id = str(item.get("dataset_id", "")).strip()
        merged = dict(item)
        if dataset_id in metadata_by_id:
            merged.update(metadata_by_id[dataset_id])
        out.append(merged)
    return out
