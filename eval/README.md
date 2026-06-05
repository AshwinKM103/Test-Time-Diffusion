# Evaluation Harness — TTD-DR

This directory contains evaluation scripts for the TTD-DR pipeline.
Reports are generated first and stored as JSON Lines files; judging is performed in a separate pass so you do not need to hold the pipeline and judge in memory simultaneously.

---

## Supported Benchmarks

| Benchmark | Script | Ground-truth type | Auto-scoreable? |
|---|---|---|---|
| **HLE** (Humanity's Last Exam) | `eval_hle.py` | Closed-ended short answer | ✅ LLM-judge (GRADE: C/I) |
| **GAIA** (validation set) | `eval_gaia.py` | Closed-ended short answer | ✅ Normalised string-match + LLM-judge |

> DeepConsult and LongForm Research were excluded from this harness — they require human pairwise evaluation or access to non-public datasets.

---

## Installation

```bash
# From the project root
pip install datasets tavily-python
```

For GAIA, some users need to authenticate with Hugging Face:
```bash
huggingface-cli login
# or set HF_TOKEN=hf_xxx in your .env file
```

---

## Environment Setup

Copy `.env.example` to `.env` and fill in the required keys:

```bash
cp .env.example .env
# Edit .env and set:
#   TAVILY_API_KEY=tvly-your_key_here
#   RETRIEVAL_BACKEND=tavily
#   MAX_SEARCH_ITERATIONS=20
```

---

## Step 1 — Generate Reports

### HLE

```bash
# Run on 20 HLE-Search questions, 20 iterations each
python -m eval.eval_hle \
    --output_dir eval/results/hle \
    --num_samples 20 \
    --max_iterations 20
```

This writes a `hle_reports_<timestamp>.jsonl` file to `eval/results/hle/`.  
Each line is a JSON record containing the question, ground-truth answer, generated report, and empty `grade` / `grade_rationale` fields.

### GAIA

```bash
# Run on 20 GAIA Level-1 questions (easiest; best for initial validation)
python -m eval.eval_gaia \
    --output_dir eval/results/gaia \
    --num_samples 20 \
    --level 1 \
    --max_iterations 20
```

For all levels: omit `--level`.

---

## Step 2 — Judge Reports (separate pass)

Use `eval/judge_prompts.py` to grade stored reports with an LLM judge.

### HLE — Correctness grading

```python
from eval.judge_prompts import CORRECTNESS_JUDGE_PROMPT, parse_judge_scores
from src.generator import get_llm_response
import json

with open("eval/results/hle/hle_reports_TIMESTAMP.jsonl") as f:
    for line in f:
        record = json.loads(line)
        if record.get("generated_report") is None:
            continue
        prompt = CORRECTNESS_JUDGE_PROMPT.format(
            question=record["question"],
            ground_truth=record["ground_truth_answer"],
            generated_answer=record["generated_report"],
        )
        response = get_llm_response(prompt, "You are a strict grading assistant.")
        scores = parse_judge_scores(response)
        print(f"ID: {record['id']} | GRADE: {scores.get('GRADE')}")
```

### GAIA — Combined quality scoring

```python
from eval.judge_prompts import COMBINED_JUDGE_PROMPT, parse_judge_scores

# Use the same pattern as above, substituting COMBINED_JUDGE_PROMPT
# and record["question"] as the query.
```

---

## Output Format

Each JSONL record contains:

```json
{
  "id": "hle-12345",
  "question": "...",
  "ground_truth_answer": "...",
  "generated_report": "Final Answer: ...\n\n## Section 1 ...",
  "elapsed_seconds": 142.3,
  "max_iterations": 20,
  "retrieval_backend": "tavily",
  "grade": null,
  "grade_rationale": null
}
```

The `grade` and `grade_rationale` fields are populated during the judge pass.

---

## Budget Notes (Tavily Free Tier)

- Free tier: **1,000 credits/month**; basic search = 1 credit per request.
- At 20 iterations per query: **50 queries** exhaust the monthly budget.
- For large evaluation runs (100+ queries), consider:
  - Switching to `RETRIEVAL_BACKEND=fineweb` if you have a FineWeb key.
  - Using `--max_iterations 5` for a faster preliminary sweep.

---

## Results Directory Structure

```
eval/
├── results/
│   ├── hle/
│   │   ├── hle_reports_20260605_123456.jsonl   # Generated reports
│   │   └── hle_meta_20260605_123456.json       # Run metadata
│   └── gaia/
│       ├── gaia_reports_level1_20260605_123456.jsonl
│       └── gaia_meta_level1_20260605_123456.json
├── __init__.py
├── eval_hle.py
├── eval_gaia.py
├── judge_prompts.py
└── README.md
```
