"""Day 7 RAG retrieval benchmark using the existing Day 6 pgvector index.

Run from the repository root with the Day 6 virtual environment:
  python starter/day7_pipeline.py prepare
  python starter/day7_pipeline.py benchmark --split dev

The Day 6 database is queried read-only. No embeddings are rebuilt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from fastembed import TextEmbedding

from baseline import TfidfIndex, hit, pack_context, validate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "starter" / "data"
RESULTS = ROOT / "starter" / "results"
SOURCE_ID = "rust-programming-language-book"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
FILTER_FIELDS = {
    "chapter_id", "chapter_title", "section", "section_title",
    "unit_id", "page_id", "location_id",
}


def normalize_evidence(value: str) -> str:
    """Match Day 6's NFKC + collapsed-whitespace normalization."""
    return " ".join(unicodedata.normalize("NFKC", value).split())


NEW_TEST_QUESTIONS = [
    {
        "id": "Q21", "split": "test",
        "question": "In a Cargo workspace, what do the member packages share?",
        "expected_answer": "They share the same Cargo.lock file and output directory.",
        "gold": [{"source_id": SOURCE_ID, "page": "rust-book-ch14-03-cargo-workspaces",
                  "evidence": "A _workspace_ is a set of packages that share the same _Cargo.lock_ and output directory."}],
        "filter": {"chapter_id": "ch14"},
    },
    {
        "id": "Q22", "split": "test",
        "question": "Where does RefCell enforce Rust's borrowing rules?",
        "expected_answer": "RefCell enforces the borrowing rules at runtime rather than at compile time.",
        "gold": [{"source_id": SOURCE_ID, "page": "rust-book-ch15-05-interior-mutability",
                  "evidence": "With `RefCell<T>`, these invariants are enforced _at runtime_."}],
    },
    {
        "id": "Q23", "split": "test",
        "question": "What threading model does Rust's standard library use, and how are its threads mapped?",
        "expected_answer": "It uses a 1:1 model: one operating-system thread for each language thread.",
        "gold": [{"source_id": SOURCE_ID, "page": "rust-book-ch16-01-threads",
                  "evidence": "The Rust standard library uses a _1:1_ model of thread implementation, whereby a program uses one operating system thread per one language thread."}],
    },
    {
        "id": "Q24", "split": "test",
        "question": "Which pattern types are allowed in function parameters, let statements, and for loops?",
        "expected_answer": "Only irrefutable patterns are allowed in those constructs.",
        "gold": [{"source_id": SOURCE_ID, "page": "rust-book-ch19-02-refutability",
                  "evidence": "Function parameters, `let` statements, and `for` loops can only accept irrefutable patterns"}],
    },
    {
        "id": "Q25", "split": "test",
        "question": "What is the central idea of metaprogramming as described in the Rust book's macros chapter?",
        "expected_answer": "Metaprogramming means writing code that writes other code.",
        "gold": [{"source_id": SOURCE_ID, "page": "rust-book-ch20-05-macros",
                  "evidence": "macros are a way of writing code that writes other code, which is known as _metaprogramming_."}],
    },
]

UNANSWERABLE_CASES = [
    {"id": "U01", "split": "dev", "answerable": False,
     "question": "What was the total worldwide number of active Rust developers in 2026?",
     "expected_behavior": "search_and_abstain",
     "absence_reason": "The corpus contains no 2026 worldwide developer-count statistics."},
    {"id": "U02", "split": "dev", "answerable": False,
     "question": "What was the average annual salary of a Rust developer in Hanoi in 2026?",
     "expected_behavior": "search_and_abstain",
     "absence_reason": "The corpus contains no current salary survey or Hanoi labor-market data."},
    {"id": "U03", "split": "test", "answerable": False,
     "question": "What is Rust's market share compared with Go in the 2026 programming-language rankings?",
     "expected_behavior": "search_and_abstain",
     "absence_reason": "The corpus contains no 2026 programming-language market-share ranking."},
    {"id": "U04", "split": "test", "answerable": False,
     "question": "How many job vacancies requiring Rust were posted in Vietnam during October 2026?",
     "expected_behavior": "search_and_abstain",
     "absence_reason": "The corpus contains no Vietnam job-posting dataset or October 2026 employment statistics."},
]


def load_day6_environment(day6_root: Path) -> str:
    env_file = day6_root / "starter" / ".env"
    if not env_file.exists():
        raise FileNotFoundError(f"Day 6 .env not found: {env_file}")
    load_dotenv(env_file, override=False)
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is missing in Day 6 starter/.env")
    return url


def export_day6_chunks(database_url: str) -> list[dict[str, Any]]:
    """Export strategy B only. page_id is the stable gold page identifier."""
    sql = """
        SELECT chunk_id, document_id, source, page, content, metadata
        FROM day6_chunks
        WHERE strategy = %s
        ORDER BY page, chunk_id
    """
    with psycopg.connect(database_url, connect_timeout=8) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, ("B",))
            rows = cur.fetchall()

    chunks = []
    for chunk_id, document_id, source, numeric_page, content, metadata in rows:
        metadata = metadata or {}
        page_id = metadata.get("page_id") or metadata.get("unit_id")
        if not page_id:
            raise RuntimeError(f"Chunk {chunk_id} has no metadata.page_id/unit_id")
        chunks.append({
            "id": str(chunk_id), "text": str(content),
            "source_id": str(document_id), "source": str(source),
            "page": str(page_id), "metadata": metadata,
        })
    if len(chunks) != 1309:
        raise RuntimeError(f"Expected 1,309 strategy-B chunks; exported {len(chunks)}")
    return chunks


def build_questions(day6_root: Path, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = json.loads((day6_root / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    old = raw.get("questions", [])
    if len(old) != 20:
        raise RuntimeError(f"Expected 20 Day 6 questions; found {len(old)}")
    questions = []
    for q in old:
        questions.append({
            "id": q["id"], "split": "dev", "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": [{
                "source_id": q["gold_source_id"],
                "page": q["gold_page"],
                "evidence": normalize_evidence(q["evidence_quote"]),
            }],
            **({"filter": q["filter"]} if q.get("filter") else {}),
        })
    # The previous Day 6 test set is not reused as a Day 7 holdout.
    questions.extend(NEW_TEST_QUESTIONS)
    validate(chunks, questions)
    return questions


def prepare(day6_root: Path) -> None:
    url = load_day6_environment(day6_root)
    chunks = export_day6_chunks(url)
    questions = build_questions(day6_root, chunks)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DATA / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DATA / "agent_cases.json").write_text(json.dumps(UNANSWERABLE_CASES, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cases = (
        [{"id": q["id"], "split": q["split"], "answerable": True} for q in questions]
        + [{"id": c["id"], "split": c["split"], "answerable": False} for c in UNANSWERABLE_CASES]
    )
    (DATA / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "prepared", "chunks": len(chunks),
        "answerable": {"dev": 20, "test": 5},
        "unanswerable": {"dev": 2, "test": 2},
        "strategy": "B", "embedding_model": MODEL_NAME,
        "page_mapping": "metadata.page_id",
        "outputs": ["starter/data/chunks.json", "starter/data/questions.json",
                    "starter/data/agent_cases.json", "starter/data/cases.json"],
    }, ensure_ascii=False, indent=2))


def metadata_matches(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    unknown = set(filters) - FILTER_FIELDS
    if unknown:
        raise ValueError(f"Unsupported metadata filters: {sorted(unknown)}")
    metadata = chunk.get("metadata") or {}
    return all(str(metadata.get(k)) == str(v) for k, v in filters.items())


def vector_literal(vector: list[float]) -> str:
    if len(vector) != 384 or not all(math.isfinite(float(x)) for x in vector):
        raise ValueError("Expected 384 finite query-embedding dimensions")
    return "[" + ",".join(format(float(x), ".9g") for x in vector) + "]"


class Day6DenseSearch:
    """Read-only dense search over the existing Day 6 strategy-B rows."""
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.model = TextEmbedding(model_name=MODEL_NAME)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be positive")
        vectors = list(self.model.query_embed(query))
        if len(vectors) != 1:
            raise RuntimeError("Expected one FastEmbed query vector")
        raw = vectors[0].tolist() if hasattr(vectors[0], "tolist") else list(vectors[0])
        literal = vector_literal(raw)
        conditions = ["strategy = %s"]
        params: list[Any] = [literal, "B"]
        for key, value in sorted((filters or {}).items()):
            if key not in FILTER_FIELDS or value is None or isinstance(value, (dict, list)):
                raise ValueError(f"Invalid metadata filter: {key}")
            conditions.append("metadata ->> %s = %s")
            params.extend([key, str(value)])
        params.append(top_k)
        sql = f"""
            SELECT chunk_id, document_id, source, page, content, metadata,
                   embedding <=> %s::vector AS distance
            FROM day6_chunks
            WHERE {' AND '.join(conditions)}
            ORDER BY distance ASC, chunk_id ASC
            LIMIT %s
        """
        with psycopg.connect(self.database_url, connect_timeout=8) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        results = []
        for cid, doc, source, numeric_page, content, metadata, distance in rows:
            metadata = metadata or {}
            page = metadata.get("page_id") or metadata.get("unit_id") or str(numeric_page)
            results.append({
                "id": str(cid), "text": str(content), "source_id": str(doc),
                "source": str(source), "page": str(page), "metadata": metadata,
                "score": 1.0 - float(distance), "strategy": "dense_pgvector_B",
            })
        return results


class RetrievalSuite:
    def __init__(self, chunks: list[dict[str, Any]], dense: Day6DenseSearch):
        self.chunks = chunks
        self.dense = dense
        self.tfidf = TfidfIndex(chunks)

    def lexical(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        candidates = self.chunks if not filters else [c for c in self.chunks if metadata_matches(c, filters)]
        if not candidates:
            return []
        index = self.tfidf if len(candidates) == len(self.chunks) else TfidfIndex(candidates)
        return index.search(query, top_k=top_k)

    def search(self, method: str, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if method == "dense":
            return self.dense.search(query, filters=filters, top_k=top_k)
        if method == "tfidf":
            return self.lexical(query, filters=filters, top_k=top_k)
        if method != "hybrid":
            raise ValueError(f"Unknown retrieval method: {method}")
        dense_rows = self.dense.search(query, filters=filters, top_k=max(20, top_k))
        lexical_rows = self.lexical(query, filters=filters, top_k=max(20, top_k))
        fused: dict[str, dict[str, Any]] = {}
        k = 60
        for label, ranked in (("dense", dense_rows), ("lexical", lexical_rows)):
            for rank, row in enumerate(ranked, 1):
                item = fused.setdefault(row["id"], {**row, "score": 0.0})
                item["score"] += 1.0 / (k + rank)
                item[f"{label}_rank"] = rank
        final = sorted(fused.values(), key=lambda x: (-x["score"], x["id"]))[:top_k]
        for row in final:
            row["strategy"] = "hybrid_rrf"
        return final


def benchmark(split: str, day6_root: Path) -> None:
    chunks = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))
    questions = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
    selected = [q for q in questions if q["split"] == split]
    if not selected:
        raise RuntimeError(f"No questions found for split {split}; run prepare first")
    validate(chunks, selected)
    suite = RetrievalSuite(chunks, Day6DenseSearch(load_day6_environment(day6_root)))
    summaries = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    for method in ("dense", "tfidf", "hybrid"):
        traces = []
        for q in selected:
            start = time.perf_counter()
            rows = suite.search(method, q["question"], filters=q.get("filter"), top_k=3)
            context, context_ids = pack_context(rows, budget=2400)
            marked = [{**row, "hit": hit(row, q["gold"])} for row in rows]
            traces.append({
                "query_id": q["id"], "split": split, "question": q["question"],
                "strategy": method, "top_k": 3, "budget_chars": 2400,
                "gold": q["gold"], "results": marked, "context": context,
                "context_ids": context_ids, "context_chars": len(context),
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
                "hit1": bool(marked and marked[0]["hit"]),
                "hit3": any(x["hit"] for x in marked),
                "context_hit": any(x["hit"] and x["id"] in context_ids for x in marked),
                "filter": q.get("filter"),
            })
        n = len(traces)
        summary = {
            "strategy": method, "split": split, "n": n,
            "hit1": sum(t["hit1"] for t in traces),
            "hit3": sum(t["hit3"] for t in traces),
            "context_hit": sum(t["context_hit"] for t in traces),
            "hit1_rate": round(sum(t["hit1"] for t in traces) / n, 4),
            "hit3_rate": round(sum(t["hit3"] for t in traces) / n, 4),
            "context_evidence_rate": round(sum(t["context_hit"] for t in traces) / n, 4),
            "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
            "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
            "service_cost_usd": 0,
            "corpus": "Day 6 strategy B, 1,309 chunks, BAAI/bge-small-en-v1.5 (384d)",
        }
        folder = RESULTS / f"{split}-{method}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
    out = RESULTS / f"{split}-retrieval-comparison.json"
    out.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Saved comparison: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 7 retrieval on the existing Day 6 index")
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--day6-root", type=Path, default=Path.home() / "Documents" / "K5-L3B-DAY06-DangNguyenNhatMinh-2A202604118-DataIngestion")
    b = subs.add_parser("benchmark")
    b.add_argument("--split", choices=("dev", "test"), required=True)
    b.add_argument("--day6-root", type=Path, default=Path.home() / "Documents" / "K5-L3B-DAY06-DangNguyenNhatMinh-2A202604118-DataIngestion")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.day6_root)
    else:
        benchmark(args.split, args.day6_root)


if __name__ == "__main__":
    main()
