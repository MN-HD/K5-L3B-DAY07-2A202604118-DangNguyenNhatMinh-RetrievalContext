"""Compare two improvement configurations on dev; evaluate only the chosen one on test.

Dev hypotheses:
1. dense_top5_context3200: top-5 candidates and a larger context may capture
   gold evidence ranked 4th or 5th.
2. hybrid_top5_context3200: RRF may fuse complementary semantic and lexical
   rankings under the same retrieval depth and context budget.

Run --split dev to compare both. After selection, run --split test --selected
<chosen-name>. Test mode evaluates only the selected configuration.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from baseline import hit, pack_context, validate
from day7_pipeline import DATA, RESULTS, Day6DenseSearch, RetrievalSuite, load_day6_environment

EXPERIMENTS = {
    "dense_top5_context3200": {
        "method": "dense",
        "hypothesis": "Top-5 candidates and a 3,200-character context may include relevant evidence ranked 4th or 5th.",
    },
    "hybrid_top5_context3200": {
        "method": "hybrid",
        "hypothesis": "RRF may combine semantic and lexical rankings to improve evidence coverage; top-5 and context budget remain fixed.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 7 retrieval improvement configurations")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--selected", choices=tuple(EXPERIMENTS), help="Required for test; evaluated as the selected config")
    parser.add_argument(
        "--day6-root", type=Path,
        default=Path.home() / "Documents" / "K5-L3B-DAY06-DangNguyenNhatMinh-2A202604118-DataIngestion",
    )
    args = parser.parse_args()
    if args.split == "dev" and args.selected:
        parser.error("--selected is only for --split test")
    if args.split == "test" and not args.selected:
        parser.error("After choosing on dev, specify --selected for the held-out test run")

    chunks_path = DATA / "chunks.json"
    questions_path = DATA / "questions.json"
    if not chunks_path.exists() or not questions_path.exists():
        raise FileNotFoundError("Run: python starter/day7_pipeline.py prepare")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    all_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = [q for q in all_questions if q.get("split") == args.split]
    minimum = 15 if args.split == "dev" else 5
    if len(questions) < minimum:
        raise RuntimeError(f"Expected at least {minimum} answerable {args.split} questions; found {len(questions)}")
    validate(chunks, questions)

    dense = Day6DenseSearch(load_day6_environment(args.day6_root))
    suite = RetrievalSuite(chunks, dense)
    selected_names = list(EXPERIMENTS) if args.split == "dev" else [args.selected]
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries = []

    for name in selected_names:
        exp = EXPERIMENTS[name]
        traces = []
        for q in questions:
            started = time.perf_counter()
            ranked = suite.search(exp["method"], q["question"], filters=q.get("filter"), top_k=5)
            context, context_ids = pack_context(ranked, budget=3200)
            marked = [{**row, "hit": hit(row, q["gold"])} for row in ranked]
            top3 = marked[:3]
            traces.append({
                "query_id": q["id"], "split": args.split, "question": q["question"],
                "experiment": name, "method": exp["method"],
                "top_k_retrieved": 5, "top_k_ranking_metric": 3,
                "context_budget_chars": 3200, "hypothesis": exp["hypothesis"],
                "filter": q.get("filter"), "gold": q["gold"],
                "results": marked, "top3": top3,
                "context": context, "context_ids": context_ids,
                "context_chars": len(context),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "hit1": bool(top3 and top3[0]["hit"]),
                "hit3": any(row["hit"] for row in top3),
                "context_hit": any(row["hit"] and row["id"] in context_ids for row in marked),
            })
        n = len(traces)
        summary = {
            "experiment": name, "method": exp["method"], "split": args.split,
            "n": n, "hypothesis": exp["hypothesis"],
            "top_k_retrieved": 5, "top_k_ranking_metric": 3,
            "context_budget_chars": 3200,
            "hit1": sum(t["hit1"] for t in traces),
            "hit3": sum(t["hit3"] for t in traces),
            "context_hit": sum(t["context_hit"] for t in traces),
            "hit1_rate": round(sum(t["hit1"] for t in traces) / n, 4),
            "hit3_rate": round(sum(t["hit3"] for t in traces) / n, 4),
            "context_evidence_rate": round(sum(t["context_hit"] for t in traces) / n, 4),
            "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
            "service_cost_usd": 0,
        }
        folder = RESULTS / f"{args.split}-{name}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)

    combined = RESULTS / f"{args.split}-improvement-experiments.json"
    combined.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Saved: {combined}")


if __name__ == "__main__":
    main()
