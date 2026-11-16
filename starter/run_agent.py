"""Run one live model-driven agent case against the Day 6 retriever.

Examples:
  python starter/run_agent.py --id Q01 --method dense --top-k 3 --budget 2400
  python starter/run_agent.py --id Q21 --method hybrid --top-k 5 --budget 3200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DAY6_ROOT = Path.home() / "Documents" / "K5-L3B-DAY06-DangNguyenNhatMinh-2A202604118-DataIngestion"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "starter"))

from agentic import run_agent
from day7_pipeline import DATA, RESULTS, Day6DenseSearch, RetrievalSuite, load_day6_environment
from gemini_controller import GeminiController


class ConfiguredRetriever:
    """Expose a selected benchmark retrieval method through the agent protocol."""
    def __init__(self, suite: RetrievalSuite, method: str):
        self.suite = suite
        self.method = method

    def search(self, query: str, *, filters=None, top_k: int = 3):
        return self.suite.search(self.method, query, filters=filters, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Question ID, e.g. Q01 or U03")
    parser.add_argument("--method", choices=("dense", "tfidf", "hybrid"), default="dense")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=2400)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--day6-root", type=Path, default=DAY6_ROOT)
    args = parser.parse_args()
    if args.top_k < 1 or args.budget < 1 or args.max_searches < 1:
        parser.error("--top-k, --budget and --max-searches must be positive")

    load_dotenv(ROOT / ".env", override=False)
    url = load_day6_environment(args.day6_root)
    questions_file = DATA / "questions.json"
    negatives_file = DATA / "agent_cases.json"
    chunks_file = DATA / "chunks.json"
    if not all(p.exists() for p in (questions_file, negatives_file, chunks_file)):
        raise SystemExit("Run starter/day7_pipeline.py prepare first.")
    answerable = json.loads(questions_file.read_text(encoding="utf-8"))
    negatives = json.loads(negatives_file.read_text(encoding="utf-8"))
    case = next((x for x in answerable + negatives if x["id"] == args.id), None)
    if case is None:
        raise SystemExit(f"Unknown case ID: {args.id}")

    controller = GeminiController()
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    retriever = ConfiguredRetriever(
        RetrievalSuite(chunks, Day6DenseSearch(url)), args.method
    )
    started = time.perf_counter()
    try:
        result = run_agent(
            case["question"], controller, retriever,
            filters=case.get("filter"), max_searches=args.max_searches,
            top_k=args.top_k, budget=args.budget,
        )
        result["model"] = controller.last_model_used or controller.model
    except Exception as exc:
        result = {
            "status": "error", "answer": None, "citations": [],
            "events": [{"event": "error", "type": type(exc).__name__, "message": str(exc)}],
            "model": controller.last_model_used or controller.model,
        }
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    result.update({
        "query_id": case["id"], "split": case["split"],
        "method": args.method, "top_k": args.top_k, "context_budget_chars": args.budget,
        "max_searches": args.max_searches, "elapsed_ms": elapsed_ms,
    })
    outdir = RESULTS / "agent-runs" / args.method
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{case['id']}.json"
    outfile.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps({
        "query_id": case["id"], "status": result["status"],
        "answer": result["answer"], "citations": result["citations"],
        "model": result["model"], "method": args.method,
        "searches": sum(e.get("event") == "search" for e in result.get("events", [])),
        "elapsed_ms": elapsed_ms, "trace_file": str(outfile),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
