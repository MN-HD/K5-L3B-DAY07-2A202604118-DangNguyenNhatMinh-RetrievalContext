# Day 7 Individual Report — Dang Nguyen Nhat Minh

## 1. Data and benchmark

- **Corpus:** The Rust Programming Language book, sourced from the Day 6 repository's pinned corpus revision `2771df165f092a4fbb9f46b5b2e7450f19010adb`.
- **Corpus coverage:** 83 Markdown sections (Chapters 3–21) in the Day 6 ingestion pipeline.
- **Indexed data used:** Day 6 strategy B only; 1,309 chunks.
- **Vector store:** PostgreSQL 16 with pgvector 0.8.7, queried read-only by the Day 7 pipeline.
- **Embedding model:** `BAAI/bge-small-en-v1.5`, 384 dimensions, via FastEmbed.
- **Gold page mapping:** Day 6 `metadata.page_id` (stable Markdown section ID); Day 6 numeric page values were not reused as section identifiers.
- **Benchmark:** 20 answerable development questions (the previously exposed Day 6 questions, therefore not used as Day 7 test) and 5 newly authored answerable test questions. Four unanswerable cases were added: two dev and two test.
- **Validation:** 22 unit tests passed. Preparation exported 1,309 chunks and validated all 25 answerable questions against source evidence. The database was not rebuilt or modified.
- **Question split:** Q01–Q20 are dev; Q21–Q25 are fresh test. U01–U02 are unanswerable dev; U03–U04 are unanswerable test.

## 2. Retrieval methods and metrics

All methods use the same 1,309-chunk Day 6 strategy-B corpus, the same answerable questions, and the same source/page/evidence labels. The metrics are:
- Hit@1: fraction of answerable queries with gold evidence in rank 1.
- Evidence Hit@3: fraction with gold evidence in the first three ranked results.
- Context Evidence Rate: fraction with gold evidence included in the packed context sent to the model.

The baseline comparison uses top-3 retrieval and a 2,400-character context budget. Dense search is cosine-distance search in pgvector; TF-IDF is lexical retrieval; hybrid retrieval fuses dense and lexical rankings with Reciprocal Rank Fusion (RRF).

### Baseline retrieval comparison

| Method | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median latency |
|---|---|---:|---:|---:|---:|
| Dense | Dev (20) | 8/20 (40%) | 12/20 (60%) | 10/20 (50%) | 32.20 ms |
| TF-IDF | Dev (20) | 5/20 (25%) | 11/20 (55%) | 8/20 (40%) | 4.23 ms |
| Hybrid RRF | Dev (20) | 7/20 (35%) | 13/20 (65%) | 12/20 (60%) | 27.07 ms |
| Dense | Test (5) | 3/5 (60%) | 3/5 (60%) | 3/5 (60%) | 24.50 ms |
| TF-IDF | Test (5) | 2/5 (40%) | 4/5 (80%) | 4/5 (80%) | 3.57 ms |
| Hybrid RRF | Test (5) | 3/5 (60%) | 5/5 (100%) | 5/5 (100%) | 46.48 ms |

### Two improvement configurations evaluated on dev

Both configurations retrieve up to five candidates, keep the top-three cutoff for ranking metrics, and use a 3,200-character context budget.

| Configuration | Hypothesis | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median latency |
|---|---|---|---:|---:|---:|---:|
| Dense top-5 / context 3,200 | A larger candidate set and context may capture relevant evidence beyond the original top three. | Dev (20) | 8/20 (40%) | 12/20 (60%) | 11/20 (55%) | 29.96 ms |
| Hybrid top-5 / context 3,200 | RRF can combine complementary semantic and lexical rankings; larger candidate/context limits may improve evidence coverage. | Dev (20) | 7/20 (35%) | 13/20 (65%) | 13/20 (65%) | 41.65 ms |

**Selection made using dev only:** `hybrid_top5_context3200`. It had the highest development Evidence Hit@3 (13/20) and Context Evidence Rate (13/20). Dense ranked more gold evidence first (8/20 versus 7/20), but the selection criterion prioritized top-three evidence availability and final-context coverage.

### Held-out test after configuration selection

| Configuration | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median latency |
|---|---|---:|---:|---:|---:|
| Selected hybrid top-5 / context 3,200 | Test (5) | 3/5 (60%) | 5/5 (100%) | 5/5 (100%) | 28.41 ms |

The selected configuration retained 100% Evidence Hit@3 and 100% context evidence coverage on the five fresh answerable test questions. The test sample is small (n=5), so this result should not be interpreted as a precise estimate of general performance. Test results were recorded after the configuration had been selected on dev; they were not used for further tuning.

All reported retrieval experiments used local vector/lexical computation; reported service cost was USD 0. Median latency is measured by the pipeline around retrieval and context construction and is not an end-to-end agent/model latency.

## 3. Development error analysis

Two development failures were inspected from the selected hybrid top-5/context-3,200 traces:

1. **Q01 — retrieval miss.** None of the five returned chunks matched the gold evidence. Several results came from the correct Chapter 3 variables-and-mutability section, but they covered adjacent material (shadowing, immutability, and other variable concepts) rather than the precise passage identifying `mut` as the keyword. This is a near-topic ranking failure: section-level relevance did not guarantee passage-level evidence retrieval.
2. **Q02 — context-packing miss.** A retrieved result was marked as containing the gold evidence, but the final packed context did not include that chunk. The retrieval list contained the evidence, while the packer did not include it within the available context budget. This separates retrieval recall from context construction: evidence can be retrieved but still be omitted from the model input.

The improvement experiments tested candidate depth and context budget, not a new ranker or evidence-aware packing algorithm. These errors are therefore documented as limitations rather than claimed as fixed.

## 4. Agent implementation and evaluation

- **Agent architecture:** a controller/model loop calls a retrieval tool named `search_knowledge_base`; retrieved passages are returned to the model; the model either searches again or answers with exact chunk IDs in square brackets.
- **Retriever options:** dense (baseline), TF-IDF, and hybrid RRF. The live runner exposes `--method`, `--top-k`, `--budget`, and `--max-searches`, so baseline and selected retrieval can be run with the same controller.
- **Selected agent settings:** hybrid retrieval, top-5, 3,200-character context, up to three searches.
- **Baseline agent settings:** dense retrieval, top-3, 2,400-character context, up to three searches.
- **Citation safeguard:** the loop rejects citations that are not present in the accumulated context and abstains on invalid/empty citations. This validates ID membership only; each cited passage still requires manual semantic review.
- **Error handling:** provider/retriever failures are written to trace files with status `error`; no mock output is substituted.
- **Model configuration:** Gemini Developer API via `GEMINI_MODEL` and optional `GEMINI_FALLBACK_MODEL` in a local, git-ignored `.env`. Never commit API credentials.

### Live agent test (real model, test split)

The live agent was evaluated on all seven test cases using `gemini-3.5-flash-lite`, with real function calling and the Day 6 retriever. The dense baseline used top-3 / 2,400 characters; the selected hybrid used top-5 / 3,200 characters; both were limited to two searches per case. Results below are from saved local traces under `starter/results/agent-runs/{dense,hybrid}/`.

| Case | Type | Dense baseline | Selected hybrid |
|---|---|---|---|
| Q21 Cargo workspace | Answerable | Abstain; 2 searches; 4.26 s | Abstain; 2 searches; 4.68 s |
| Q22 RefCell | Answerable | Answer; 1 search; 2.69 s | Answer; 1 search; 2.90 s |
| Q23 Thread model | Answerable | Answer; 1 search; 2.79 s | Answer; 1 search; 7.45 s |
| Q24 Irrefutable patterns | Answerable | Answer; 1 search; 3.48 s | Provider read-timeout; 0 searches; 14.06 s |
| Q25 Metaprogramming | Answerable | Answer; 1 search; 3.14 s | Answer; 1 search; 4.98 s |
| U03 2026 language market share | Unanswerable | Correct abstention; 1 search; 2.33 s | Provider HTTP 429 rate-limit error; 0 searches; 0.57 s |
| U04 Vietnam Rust job vacancies | Unanswerable | Correct abstention; 1 search; 2.61 s | Correct abstention; 1 search; 3.17 s |

The answerable responses produced by both configurations for Q22, Q23 and Q25, and by dense for Q24, matched the benchmark's expected facts. Q21 was a **false abstention** for both configurations: the selected hybrid trace made two searches and returned chunks from the Cargo-workspaces section, while the independent retrieval benchmark had already found its gold evidence in context. This is an agent answer-generation/abstention failure despite good retrieval. The dense baseline also answered Q24 correctly; the hybrid Q24 run did not complete and is not scored as an answer or abstention.

**Observed aggregate (test cases):**
- Dense: 7/7 cases completed; retrieval tool used in 7/7; 8 total searches; 4/5 answerable questions received correct supported answers (grades 2,2,2,2,0; mean **1.60/2**); 2/2 unanswerable cases correctly abstained; false abstention 1/5.
- Hybrid: 5/7 cases completed; retrieval tool used in 5/5 completed cases; 6 total searches; 3/4 completed answerable cases received correct supported answers (grades 0,2,2,2; mean **1.50/2** over completed answerable cases); U04 correctly abstained (1/1 completed unanswerable case); one answerable case had a provider timeout and one unanswerable case had a provider rate-limit error. False abstention 1/4 completed answerable cases. Provider errors are reported separately, not silently counted as abstentions.

The agent extracts one citation ID for each generated answer and checks that the ID exists in the accumulated context. This is a syntactic safeguard, not a semantic citation verifier. The answer text was compared with the expected answer; final citation validity still requires a human to inspect the cited passage text in the saved traces. Therefore, citation support is not claimed as a fully audited percentage in this report.

**Provider limitations:** the original Gemini model repeatedly timed out. The faster `gemini-3.5-flash-lite` model enabled successful live calls, but the hybrid Q24 call hit a 12-second read timeout and hybrid U03 hit HTTP 429 because the free-tier per-minute request quota was exceeded. No further API retries were made. These are provider errors, not retrieval scores.

Trace paths:
- Dense: `starter/results/agent-runs/dense/Q21.json` through `Q25.json`, plus `U03.json` and `U04.json`.
- Hybrid: `starter/results/agent-runs/hybrid/Q21.json` through `Q25.json`, plus `U03.json` and `U04.json`.

Do not interpret mock unit tests as live agent evidence.

## 5. Reproduction commands

Run from the repository root in PowerShell, with the Day 6 virtual environment:

```powershell
$py = "$HOME\Documents\K5-L3B-DAY06-DangNguyenNhatMinh-2A202604118-DataIngestion\starter\.venv\Scripts\python.exe"

# Prepare is already complete; rerun only if local generated files are missing.
& $py -X utf8 -m unittest discover -s tests -v
& $py starter\day7_pipeline.py benchmark --split dev
& $py starter\run_improvement_experiments.py --split dev

# Held-out test and selected configuration evaluation
& $py starter\day7_pipeline.py benchmark --split test
& $py starter\run_improvement_experiments.py --split test --selected hybrid_top5_context3200

# Live agent: baseline dense on each test case (Q21–Q25, U03–U04)
& $py starter\run_agent.py --id Q21 --method dense --top-k 3 --budget 2400

# Live agent: selected hybrid configuration
& $py starter\run_agent.py --id Q21 --method hybrid --top-k 5 --budget 3200
```

Generated data and result traces live under `starter/data/` and `starter/results/`; these paths are ignored by Git by default. They can be recreated by running preparation and the documented benchmark commands. The Day 6 database is read-only during Day 7 export/search.

## 6. Limitations and conclusion

The retrieval benchmark, two dev improvement configurations, development error inspection, held-out selected-config test, 22 unit tests, and a real-model agent run on all seven test cases are complete. The selected hybrid retrieval result is 5/5 Evidence Hit@3 and 5/5 context evidence coverage on answerable test questions. The agent results are imperfect: both configurations falsely abstained on Q21; hybrid had a read-timeout on Q24 and a free-tier rate-limit error on U03. The dense baseline completed all cases and correctly abstained on both unanswerable questions, while the selected hybrid completed five of seven. Citation IDs are checked syntactically by the agent, but a human must still inspect each cited passage for semantic support. The small answerable test sample (n=5), passage-level retrieval misses, context-packing omissions, agent false abstention, API latency/rate limits, and lack of a completed semantic citation audit are limitations. The optional 100-page multimodal challenge was not attempted.
