"""Run one live Gemini agent case; provider errors are saved and reported."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DAY6_ROOT = Path.home() / "Documents" / "K5-L3B-DAY06-DangNguyenNhatMinh-2A202604118-DataIngestion"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "starter"))

from agentic import run_agent
from day7_pipeline import Day6DenseSearch, load_day6_environment
from gemini_controller import GeminiController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Question ID, e.g. Q01 or U03")
    parser.add_argument("--day6-root", type=Path, default=DAY6_ROOT)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    url = load_day6_environment(args.day6_root)

    questions_file = ROOT / "starter" / "data" / "questions.json"
    negatives_file = ROOT / "starter" / "data" / "agent_cases.json"
    if not questions_file.exists() or not negatives_file.exists():
        raise SystemExit("Run day7_pipeline.py prepare first.")
    answerable = json.loads(questions_file.read_text(encoding="utf-8"))
    negatives = json.loads(negatives_file.read_text(encoding="utf-8"))
    case = next((x for x in answerable + negatives if x["id"] == args.id), None)
    if case is None:
        raise SystemExit(f"Unknown case ID: {args.id}")

    controller = GeminiController()
    retriever = Day6DenseSearch(url)
    try:
        result = run_agent(
            case["question"], controller, retriever,
            filters=case.get("filter"), max_searches=3, top_k=5, budget=3200,
        )
        result["model"] = controller.last_model_used or controller.model
    except Exception as exc:
        result = {"status": "error", "answer": None, "citations": [],
                  "events": [{"event": "error", "type": type(exc).__name__, "message": str(exc)}],
                  "model": controller.last_model_used or controller.model}
    result["query_id"] = case["id"]
    result["split"] = case["split"]
    outdir = ROOT / "starter" / "results" / "agent-runs"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{case['id']}.json"
    outfile.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "query_id": case["id"], "status": result["status"],
        "answer": result["answer"], "citations": result["citations"],
        "model": result["model"],
        "searches": sum(e.get("event") == "search" for e in result.get("events", [])),
        "trace_file": str(outfile),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
