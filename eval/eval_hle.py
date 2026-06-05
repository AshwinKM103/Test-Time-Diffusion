"""
HLE (Humanity's Last Exam) evaluation script for TTD-DR.

This script:
  1. Loads a subset of the HLE dataset (cais/hle on Hugging Face).
  2. Filters to the HLE-Search subset — questions that require intensive
     web search and reasoning rather than pure recall.
  3. Runs the TTD-DR pipeline for each question.
  4. Stores the generated report alongside metadata as a JSON Lines file.

Reports are NOT judged here. Run the correctness judge separately using
judge_prompts.CORRECTNESS_JUDGE_PROMPT against the stored reports.

Usage:
    python -m eval.eval_hle \\
        --output_dir eval/results/hle \\
        --num_samples 20 \\
        --max_iterations 20

Requirements:
    pip install datasets

Environment variables (set in .env):
    TAVILY_API_KEY  (or FINEWEB_API_KEY if RETRIEVAL_BACKEND=fineweb)
    RETRIEVAL_BACKEND
    MAX_SEARCH_ITERATIONS  (overridden by --max_iterations if provided)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from src.utils.logger import CustomLogger
logger = CustomLogger.setup_task_logger(__name__, output_dir="outputs/logs")

load_dotenv()

# Ensure the project root is on sys.path so `src` is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_rag_static  # noqa: E402
from src.tracking.wandb_utils import init_wandb_run, log_evaluation_artifact  # noqa: E402


def load_hle_dataset(num_samples: int, search_only: bool = True):
    """
    Load HLE questions from Hugging Face.

    Fetches the Humanity's Last Exam (HLE) dataset test split.
    Optionally filters the questions to keep those more likely to require
    web search (by excluding very short exact-match answers).

    Args:
        num_samples (int): Maximum number of questions to evaluate.
        search_only (bool): If True, filter to questions that likely require web search. Default is True.

    Returns:
        List[dict]: A list of task dictionaries containing question, answer, and metadata.

    Raises:
        ImportError: If the datasets package is not installed.

    Example:
        >>> questions = load_hle_dataset(num_samples=5, search_only=True)
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from exc

    logger.info("Loading cais/hle dataset from Hugging Face...")
    ds = load_dataset("cais/hle", split="test")

    items = []
    for row in ds:
        items.append(
            {
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "answer_type": row.get("answer_type", ""),
                "category": row.get("category", ""),
            }
        )

    if search_only:
        # Heuristic: keep items whose answer is not a single short token
        # (pure formula / single-word) — these are more likely to benefit from search.
        items = [
            it for it in items
            if len(str(it.get("answer", ""))) > 3
        ]
        logger.info(f"After HLE-Search heuristic filter: {len(items)} questions")

    # Shuffle deterministically and take the first num_samples
    import random
    random.seed(42)
    random.shuffle(items)
    return items[:num_samples]


def run_hle_evaluation(
    output_dir: str,
    num_samples: int,
    max_iterations: int,
):
    """
    Run TTD-DR on HLE questions and store results as JSONL.

    Evaluates a set of HLE questions using the Test-Time-Diffusion pipeline,
    and logs the generated reports and metadata to the output directory and W&B.

    Args:
        output_dir (str): Directory to save the results and metadata files.
        num_samples (int): Maximum number of questions to evaluate.
        max_iterations (int): Maximum number of search iterations per query.

    Returns:
        None

    Example:
        >>> run_hle_evaluation("results", 5, 10)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = out_path / f"hle_reports_{timestamp}.jsonl"
    meta_file = out_path / f"hle_meta_{timestamp}.json"

    questions = load_hle_dataset(num_samples=num_samples)
    logger.info(f"Running evaluation on {len(questions)} HLE questions → {results_file}")

    # Override MAX_SEARCH_ITERATIONS for this run
    os.environ["MAX_SEARCH_ITERATIONS"] = str(max_iterations)

    run_meta = {
        "benchmark": "HLE",
        "timestamp": timestamp,
        "num_samples": len(questions),
        "max_iterations": max_iterations,
        "retrieval_backend": os.getenv("RETRIEVAL_BACKEND", "tavily"),
        "results_file": str(results_file),
    }

    run = init_wandb_run(
        project="Test-Time-Diffusion",
        config=run_meta,
        job_type="eval",
        group="HLE",
        tags=["hle-search"]
    )

    completed = 0
    errors = 0

    with open(results_file, "w") as fout:
        for idx, item in enumerate(questions):
            logger.info(
                f"[{idx + 1}/{len(questions)}] Evaluating HLE question id={item['id']!r}"
            )
            start_time = time.time()
            try:
                generated_report = run_rag_static(item["question"])
                elapsed = time.time() - start_time

                record = {
                    "id": item["id"],
                    "question": item["question"],
                    "ground_truth_answer": item["answer"],
                    "answer_type": item["answer_type"],
                    "category": item["category"],
                    "generated_report": generated_report,
                    "elapsed_seconds": round(elapsed, 2),
                    "max_iterations": max_iterations,
                    "retrieval_backend": os.getenv("RETRIEVAL_BACKEND", "tavily"),
                    # Judgement fields — to be filled in a separate judging pass
                    "grade": None,
                    "grade_rationale": None,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                completed += 1
                logger.success(f"  ✓ Done in {elapsed:.1f}s")

            except Exception as exc:
                elapsed = time.time() - start_time
                logger.error(f"  ✗ Failed after {elapsed:.1f}s: {exc}")
                error_record = {
                    "id": item["id"],
                    "question": item["question"],
                    "ground_truth_answer": item["answer"],
                    "answer_type": item["answer_type"],
                    "category": item["category"],
                    "generated_report": None,
                    "error": str(exc),
                    "elapsed_seconds": round(elapsed, 2),
                    "max_iterations": max_iterations,
                    "retrieval_backend": os.getenv("RETRIEVAL_BACKEND", "tavily"),
                    "grade": None,
                    "grade_rationale": None,
                }
                fout.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                fout.flush()
                errors += 1

    run_meta["completed"] = completed
    run_meta["errors"] = errors
    with open(meta_file, "w") as f:
        json.dump(run_meta, f, indent=2)

    run.log({
        "completed": completed,
        "errors": errors
    })
    
    log_evaluation_artifact(run, str(results_file), name="hle_results_jsonl")
    log_evaluation_artifact(run, str(meta_file), name="hle_meta_json")
    run.finish()

    logger.info(
        f"\n{'='*60}\n"
        f"HLE evaluation complete.\n"
        f"  Completed : {completed}\n"
        f"  Errors    : {errors}\n"
        f"  Reports   : {results_file}\n"
        f"  Meta      : {meta_file}\n"
        f"{'='*60}\n"
        f"Next step: run the correctness judge against {results_file}\n"
        f"using eval/judge_prompts.CORRECTNESS_JUDGE_PROMPT."
    )


import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for the HLE evaluation script.
    Triggered via Hydra configuration.
    """
    run_hle_evaluation(
        output_dir=cfg.eval.hle_output_dir,
        num_samples=cfg.eval.num_samples,
        max_iterations=cfg.eval.max_iterations,
    )

if __name__ == "__main__":
    main()
