import argparse
import json
from pathlib import Path

from .part3 import default_db_path
from .pipeline import (
    run_baseline_a_part2,
    run_baseline_a_part3,
    run_baseline_a_part4,
    run_baseline_b_part2,
    run_baseline_b_part3,
    run_baseline_b_part4,
    run_part1_part2,
    run_part1_part3,
    run_part1_part4,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Part1+Part2+Part3 multi-agent NL→SQL demo.")
    parser.add_argument(
        "--mode",
        choices=["workflow", "baseline_a", "baseline_b"],
        default="workflow",
        help=(
            "Run mode: workflow (Spec+gene_normalizer+deterministic topic grounding+SQLGen), "
            "baseline_a (single-shot SQL), baseline_b (LLM topic selection + SQLGen)."
        ),
    )
    parser.add_argument("--show-intermediate", action="store_true", help="Print intermediate JSON (QuerySpec + topic candidates).")
    parser.add_argument("--top-m", type=int, default=10, help="Number of topic candidates to keep (default: 10).")
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Run Part1+Part2 only (do not execute Part3 verifier/selector).",
    )
    parser.add_argument(
        "--db-path",
        default=str(default_db_path()),
        help="SQLite DB path for Part3 execution (default: text2sql_demo/data/sim_store_v1.sqlite).",
    )
    parser.add_argument(
        "--query-timeout-seconds",
        type=float,
        default=2.0,
        help="Per-candidate SQLite timeout in seconds for Part3 (default: 2.0).",
    )
    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="Skip Part4 AnswerAgent (do not generate natural-language answer).",
    )
    args = parser.parse_args(argv)

    if args.no_execute:
        stage_mode = "Part1+Part2"
    elif args.no_answer:
        stage_mode = "Part1+Part2+Part3"
    else:
        stage_mode = "Part1+Part2+Part3+Part4"
    print(f"Text-to-SQL Demo ({stage_mode}, method={args.mode}). Type 'exit' to quit.")
    if not args.no_execute:
        print(f"SQLite: {Path(args.db_path).resolve()}")

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye.")
            return 0

        try:
            if args.mode == "workflow":
                if args.no_execute:
                    out = run_part1_part2(q, top_m=args.top_m)
                elif args.no_answer:
                    out = run_part1_part3(
                        q,
                        top_m=args.top_m,
                        db_path=args.db_path,
                        query_timeout_seconds=max(0.1, args.query_timeout_seconds),
                    )
                else:
                    out = run_part1_part4(
                        q,
                        top_m=args.top_m,
                        db_path=args.db_path,
                        query_timeout_seconds=max(0.1, args.query_timeout_seconds),
                    )
            elif args.mode == "baseline_a":
                if args.no_execute:
                    out = run_baseline_a_part2(q, top_m=args.top_m)
                elif args.no_answer:
                    out = run_baseline_a_part3(
                        q,
                        top_m=args.top_m,
                        db_path=args.db_path,
                        query_timeout_seconds=max(0.1, args.query_timeout_seconds),
                    )
                else:
                    out = run_baseline_a_part4(
                        q,
                        top_m=args.top_m,
                        db_path=args.db_path,
                        query_timeout_seconds=max(0.1, args.query_timeout_seconds),
                    )
            else:
                if args.no_execute:
                    out = run_baseline_b_part2(q, top_m=args.top_m)
                elif args.no_answer:
                    out = run_baseline_b_part3(
                        q,
                        top_m=args.top_m,
                        db_path=args.db_path,
                        query_timeout_seconds=max(0.1, args.query_timeout_seconds),
                    )
                else:
                    out = run_baseline_b_part4(
                        q,
                        top_m=args.top_m,
                        db_path=args.db_path,
                        query_timeout_seconds=max(0.1, args.query_timeout_seconds),
                    )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        print(f"\n[Method]\n{out.get('method', args.mode)}")

        if args.show_intermediate:
            print("\n[QuerySpec]")
            print(json.dumps(out["query_spec"], indent=2, ensure_ascii=True))
            print("\n[Topic Candidates]")
            print(json.dumps(out["topic_candidates"], indent=2, ensure_ascii=True))

        print("\n[SQL Candidate]")
        sql_candidates_obj = out.get("sql_candidates", {})
        sql_candidates = sql_candidates_obj if isinstance(sql_candidates_obj, dict) else {}
        candidate_list_obj = sql_candidates.get("candidates", [])
        candidate_list = candidate_list_obj if isinstance(candidate_list_obj, list) else []
        for cand in candidate_list:
            if not isinstance(cand, dict):
                continue
            cid = cand.get("id")
            sql = cand.get("sql")
            notes = cand.get("notes", "")
            print(f"\n--- SQL #{cid} ---")
            if notes:
                print(f"Notes: {notes}")
            print(sql)

        if not args.no_execute:
            part3 = out.get("part3", {})
            if isinstance(part3, dict):
                print("\n[Part3]")
                print(f"Passed: {len(part3.get('passed', []))} | Failed: {len(part3.get('failed', []))}")
                selected_id = part3.get("selected_id")
                print(f"Selected candidate id: {selected_id}")
                rationale = str(part3.get("rationale", "")).strip()
                if rationale:
                    print(f"Rationale: {rationale}")

                selected_sql = part3.get("selected_sql")
                if isinstance(selected_sql, str) and selected_sql.strip():
                    print("\n[Selected SQL]")
                    print(selected_sql)

                datasets = part3.get("datasets", [])
                if isinstance(datasets, list) and datasets:
                    print("\n[Top Datasets]")
                    for idx, ds in enumerate(datasets, start=1):
                        if not isinstance(ds, dict):
                            continue
                        dataset_id = ds.get("dataset_id", "")
                        score = ds.get("score", "")
                        dataset_name = ds.get("dataset_name", "")
                        print(
                            f"{idx}. dataset_id={dataset_id} score={score} "
                            f"name={dataset_name}"
                        )
                else:
                    print("\n[Top Datasets]\n(no rows)")

        if not args.no_execute and not args.no_answer:
            part4 = out.get("part4", {})
            if isinstance(part4, dict):
                answer = str(part4.get("answer", "")).strip()
                if answer:
                    print("\n[Answer]")
                    print(answer)

    return 0
