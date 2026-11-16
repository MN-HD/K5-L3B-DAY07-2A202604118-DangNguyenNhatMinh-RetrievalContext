# Day 7 implementation notes

This implementation reuses the existing Day 6 PostgreSQL/pgvector index and local FastEmbed model. It does not write to Day 6.

## 1. Environment

Run commands from the Day 7 repository root using the existing Day 6 interpreter:

```powershell
$py = "$HOME\Documents\K5-L3B-DAY06-DangNguyenNhatMinh-2A202604118-DataIngestion\starter\.venv\Scripts\python.exe"
```

The Day 6 environment already contains psycopg, FastEmbed, python-dotenv and requests. The Day 7 Gemini SDK install is optional; the controller uses requests.

## 2. Prepare local Day 7 data

```powershell
& $py starter/day7_pipeline.py prepare
```

This exports strategy B (1,309 chunks) with SELECT-only database access. It maps Day 6 gold page IDs to `metadata.page_id`, and normalizes gold evidence whitespace to the Day 6 corpus convention. The original 20 questions become dev because they were already exposed in Day 6 evaluation; five newly written questions form the Day 7 answerable test set. Two unanswerable cases are defined for each split.

Generated local files are under `starter/data/` and are ignored by Git by default.

## 3. Compare retrieval on dev

```powershell
& $py starter/day7_pipeline.py benchmark --split dev
```

The three methods are: dense pgvector cosine search, TF-IDF lexical search, and hybrid Reciprocal Rank Fusion (RRF). Results and per-query traces are saved to `starter/results/`. Review at least two dev errors and select a configuration before any test run.

## 4. Run two improvement configurations on dev

```powershell
& $py starter/run_improvement_experiments.py --split dev
```

This compares dense top-5 with a 3,200-character context and hybrid RRF top-5 with the same context budget. It records two hypotheses and the same Hit@1/Evidence Hit@3/Context Evidence Rate metrics. Compare these results with the top-3/2,400-character baseline in `starter/results/dev-*`, then select a configuration using dev only.

## 5. Run test after selecting a configuration

```powershell
& $py starter/day7_pipeline.py benchmark --split test
& $py starter/run_improvement_experiments.py --split test --selected hybrid_top5_context3200
```

Replace `hybrid_top5_context3200` with `dense_top5_context3200` if that is the configuration selected on dev. The second command evaluates only the chosen configuration. The test split is five new answerable questions. The four unanswerable agent cases are in `starter/data/agent_cases.json`.

## 6. Real agent

The Gemini controller uses the Gemini Developer API function-calling interface and bounded HTTP timeouts. It tries the configured model and, once if it gets HTTP 503, a distinct fallback model. A 503 is a provider availability error, not an invalid key. Set `GEMINI_MODEL` and optionally `GEMINI_FALLBACK_MODEL` in the local Day 7 `.env`. Never commit the key.

```powershell
& $py starter/run_agent.py --id Q01
& $py starter/run_agent.py --id U03
```

Check the resulting trace under `starter/results/agent-runs/`. Citation IDs must be read and manually verified against their cited text; ID membership alone does not prove grounding.

## 7. Tests

```powershell
& $py -X utf8 -m unittest discover -s tests -v
```

Do not claim full completion until live retrieval, agent runs, manual citation/answer grading, required dev improvement experiments, report, and GitHub submission are complete.
