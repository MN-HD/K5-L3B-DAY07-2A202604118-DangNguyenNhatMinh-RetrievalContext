"""Run two documented development-only retrieval improvement experiments.

Hypothesis 1: retrieving five candidates and allowing a larger context can
increase Context Evidence Rate when relevant evidence ranks 4th or 5th.
Hypothesis 2: hybrid RRF can improve evidence coverage by combining dense and
lexical rankings, while using the same top-5/context budget as the first test.

Run only on dev while selecting the configuration. Never use these outputs to
tune against the held-out test split.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from baseline import hit, pack_context, validate
from day7_pipeline import DATA, RESULTS, Day6DenseSearch, RetrievalSuite, load_day6_environment


def main() -> None:
    chunks_path = DATA / "chunks.json"
    questions_path = DATA / "questions.json"
    if not chunks_path.exists() or not questions_path.exists():
        raise FileNotFoundError("Run: python starter/day7_pipeline.py prepare")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    all_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = [q for q in all_questions if q.get("split") == "dev"]
    if len(questions) < 15:
        raise RuntimeError(f"Expected at least 15 answerable dev questions; found {len(questions)}")
    validate(chunks, questions)

    day6_root = Path.home() / "Documents" / "K5-L3B-DAY06-DangNguyenNhatMinh-2A202604118-DataIngestion"
    dense = Day6DenseSearch(load_day6_environment(day6_root))
    suite = RetrievalSuite(chunks, dense)
    experiments = [
        {
            "name": "dense_top5_context3200",
            "method": "dense",
            "hypothesis": "Top-5 candidates and a 3,200-character context may include relevant evidence ranked 4th or 5th.",
        },
        {
            "name": "hybrid_top5_context3200",
            "method": "hybrid",
            "hypothesis": "RRF may combine semantic and lexical rankings to improve evidence coverage; top-5 and context budget remain fixed.",
        },
    ]

    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries = []
    for exp in experiments:
        traces = []
        for q in questions:
            started = time.perf_counter()
            ranked = suite.search(exp["method"], q["question"], filters=q.get("filter"), top_k=5)
            context, context_ids = pack_context(ranked, budget=3200)
            marked = [{**row, "hit": hit(row, q["gold"])} for row in ranked]
            top3 = marked[:3]
            traces.append({
                "query_id": q["id"], "split": "dev", "question": q["question"],
                "experiment": exp["name"], "method": exp["method"],
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
            "experiment": exp["name"], "method": exp["method"],
            "split": "dev", "n": n, "hypothesis": exp["hypothesis"],
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
        folder = RESULTS / exp["name"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)

    combined = RESULTS / "dev-improvement-experiments.json"
    combined.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Saved: {combined}")


if __name__ == "__main__":
    main()
