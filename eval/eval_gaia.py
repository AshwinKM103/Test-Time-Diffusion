"""
GAIA validation-set evaluation script for TTD-DR.

This script:
  1. Loads the GAIA validation set from Hugging Face (gaia-benchmark/GAIA).
  2. Optionally filters by difficulty level (1, 2, or 3).
  3. Runs the TTD-DR pipeline for each question.
  4. Stores the generated report alongside the ground-truth answer as JSONL.

Reports are NOT scored here. GAIA answers can be scored with a normalised
string-match in a separate pass (no human preferences required).

Note on GAIA access:
  The GAIA test split requires a Hugging Face login and leaderboard submission.
  The validation split is publicly accessible and is used here.
  Run `huggingface-cli login` or set HF_TOKEN in your environment if needed.

Usage:
    python -m eval.eval_gaia \\
        --output_dir eval/results/gaia \\
        --num_samples 20 \\
        --level 1 \\
        --max_iterations 20

Requirements:
    pip install datasets

Environment variables (set in .env):
    TAVILY_API_KEY  (or FINEWEB_API_KEY if RETRIEVAL_BACKEND=fineweb)
    RETRIEVAL_BACKEND
    HF_TOKEN        (only needed if the GAIA repo requires authentication)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from src.utils.logger import CustomLogger
logger = CustomLogger.setup_task_logger(__name__, output_dir="outputs/logs")

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_rag_static  # noqa: E402
from src.tracking.wandb_utils import init_wandb_run, log_evaluation_artifact  # noqa: E402


def normalize_answer(text: str) -> str:
    """
    Normalise a GAIA answer string for string-match scoring.

    Strips punctuation, extra whitespace, and lowercases the text,
    following the GAIA official scoring convention.

    Args:
        text (str): The raw answer string.

    Returns:
        str: The normalized answer string.

    Example:
        >>> normalize_answer(" Hello, World! ")
        'hello world'
    """
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_short_answer(generated_report: str) -> str:
    """
    Heuristically extract the short answer from a generated report.

    The FINAL_REPORT_PROMPT instructs the model to start with a
    "Final Answer:" paragraph. This function extracts the first sentence
    of that paragraph as the candidate answer for string matching.
    If the prefix is not found, falls back to the first 200 characters.

    Args:
        generated_report (str): The full generated report text.

    Returns:
        str: The extracted short answer.

    Example:
        >>> report = "Final Answer: Paris is the capital.\\nDetails follow."
        >>> extract_short_answer(report)
        'Paris is the capital'
    """
    if "Final Answer:" in generated_report:
        after = generated_report.split("Final Answer:", 1)[1].strip()
        # Take the first sentence
        sentence = re.split(r"[.\n]", after)[0].strip()
        return sentence
    # Fallback: first 200 characters of the report
    return generated_report[:200].strip()


def load_gaia_dataset(num_samples: int, level: int | None = None):
    """
    Load GAIA validation questions from Hugging Face.

    Fetches the GAIA 2023_all validation split and optionally filters by
    difficulty level. Tasks requiring file attachments are automatically filtered out.

    Args:
        num_samples (int): Maximum number of questions to evaluate.
        level (int | None): Optional difficulty filter (1, 2, or 3). None means all levels. Default is None.

    Returns:
        List[dict]: A list of task dictionaries containing question and answer data.

    Raises:
        ImportError: If the datasets package is not installed.
        Exception: If loading the dataset from Hugging Face fails.

    Example:
        >>> questions = load_gaia_dataset(num_samples=5, level=1)
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from exc

    hf_token = os.getenv("HF_TOKEN")
    logger.info("Loading gaia-benchmark/GAIA validation split from Hugging Face...")

    try:
        ds = load_dataset(
            "gaia-benchmark/GAIA",
            "2023_all",
            split="validation",
            token=hf_token,
        )
    except Exception as exc:
        logger.error(
            f"Failed to load GAIA dataset: {exc}\n"
            "If the repo requires authentication, set HF_TOKEN in your .env file."
        )
        raise

    items = []
    for row in ds:
        item_level = row.get("Level", 0)
        if level is not None and item_level != level:
            continue
        items.append(
            {
                "task_id": row.get("task_id", ""),
                "question": row.get("Question", ""),
                "final_answer": row.get("Final answer", ""),
                "level": item_level,
                "file_name": row.get("file_name", ""),
                "annotator_metadata": row.get("Annotator Metadata", {}),
            }
        )

    # Filter out tasks that reference a file attachment (requires multi-modal handling)
    items_no_file = [it for it in items if not it.get("file_name")]
    if items_no_file:
        logger.info(
            f"Filtered {len(items) - len(items_no_file)} tasks with file attachments "
            f"(require multi-modal handling). Remaining: {len(items_no_file)}"
        )
        items = items_no_file

    import random
    random.seed(42)
    random.shuffle(items)
    return items[:num_samples]


def run_gaia_evaluation(
    output_dir: str,
    num_samples: int,
    level: int | None,
    max_iterations: int,
):
    """
    Run TTD-DR on GAIA validation questions and store results as JSONL.

    Evaluates a set of GAIA questions using the Test-Time-Diffusion pipeline,
    computes preliminary string-match accuracy, and logs the results and
    metadata to the output directory and Weights & Biases.

    Args:
        output_dir (str): Directory to save the results and metadata files.
        num_samples (int): Maximum number of questions to evaluate.
        level (int | None): Difficulty level to evaluate (1, 2, or 3), or None for all.
        max_iterations (int): Maximum number of search iterations per query.

    Returns:
        None

    Example:
        >>> run_gaia_evaluation("results", 5, 1, 10)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    level_tag = f"_level{level}" if level else "_alllevels"
    results_file = out_path / f"gaia_reports{level_tag}_{timestamp}.jsonl"
    meta_file = out_path / f"gaia_meta{level_tag}_{timestamp}.json"

    questions = load_gaia_dataset(num_samples=num_samples, level=level)
    logger.info(f"Running evaluation on {len(questions)} GAIA questions → {results_file}")

    os.environ["MAX_SEARCH_ITERATIONS"] = str(max_iterations)

    run_meta = {
        "benchmark": "GAIA",
        "timestamp": timestamp,
        "num_samples": len(questions),
        "level_filter": level,
        "max_iterations": max_iterations,
        "retrieval_backend": os.getenv("RETRIEVAL_BACKEND", "tavily"),
        "results_file": str(results_file),
    }

    run = init_wandb_run(
        project="Test-Time-Diffusion",
        config=run_meta,
        job_type="eval",
        group="GAIA",
        tags=[f"level{level}" if level else "alllevels"]
    )

    completed = 0
    errors = 0
    correct = 0  # Preliminary string-match score (not the full GAIA scorer)

    with open(results_file, "w") as fout:
        for idx, item in enumerate(questions):
            logger.info(
                f"[{idx + 1}/{len(questions)}] GAIA task_id={item['task_id']!r} "
                f"level={item['level']}"
            )
            start_time = time.time()
            try:
                generated_report = run_rag_static(item["question"])
                elapsed = time.time() - start_time

                # Preliminary string-match (normalised)
                predicted = normalize_answer(extract_short_answer(generated_report))
                reference = normalize_answer(item["final_answer"])
                is_correct = reference in predicted or predicted in reference

                if is_correct:
                    correct += 1

                record = {
                    "task_id": item["task_id"],
                    "question": item["question"],
                    "ground_truth_answer": item["final_answer"],
                    "level": item["level"],
                    "generated_report": generated_report,
                    "extracted_answer": extract_short_answer(generated_report),
                    "preliminary_correct": is_correct,
                    "elapsed_seconds": round(elapsed, 2),
                    "max_iterations": max_iterations,
                    "retrieval_backend": os.getenv("RETRIEVAL_BACKEND", "tavily"),
                    # Fields for a more thorough judging pass later
                    "judge_grade": None,
                    "judge_rationale": None,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                completed += 1
                logger.success(
                    f"  ✓ Done in {elapsed:.1f}s | string-match={'✓' if is_correct else '✗'}"
                )

            except Exception as exc:
                elapsed = time.time() - start_time
                logger.error(f"  ✗ Failed after {elapsed:.1f}s: {exc}")
                error_record = {
                    "task_id": item["task_id"],
                    "question": item["question"],
                    "ground_truth_answer": item["final_answer"],
                    "level": item["level"],
                    "generated_report": None,
                    "extracted_answer": None,
                    "preliminary_correct": None,
                    "error": str(exc),
                    "elapsed_seconds": round(elapsed, 2),
                    "max_iterations": max_iterations,
                    "retrieval_backend": os.getenv("RETRIEVAL_BACKEND", "tavily"),
                    "judge_grade": None,
                    "judge_rationale": None,
                }
                fout.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                fout.flush()
                errors += 1

    preliminary_accuracy = correct / completed if completed > 0 else 0.0
    run_meta["completed"] = completed
    run_meta["errors"] = errors
    run_meta["preliminary_string_match_accuracy"] = round(preliminary_accuracy, 4)

    with open(meta_file, "w") as f:
        json.dump(run_meta, f, indent=2)

    run.log({
        "accuracy": preliminary_accuracy,
        "completed": completed,
        "errors": errors
    })
    
    log_evaluation_artifact(run, str(results_file), name="gaia_results_jsonl")
    log_evaluation_artifact(run, str(meta_file), name="gaia_meta_json")
    run.finish()

    logger.info(
        f"\n{'='*60}\n"
        f"GAIA evaluation complete.\n"
        f"  Completed              : {completed}\n"
        f"  Errors                 : {errors}\n"
        f"  Preliminary accuracy   : {preliminary_accuracy:.1%} (string-match)\n"
        f"  Reports                : {results_file}\n"
        f"  Meta                   : {meta_file}\n"
        f"{'='*60}\n"
        f"Note: The preliminary accuracy is a normalised string-match heuristic.\n"
        f"For official GAIA scoring, submit to the leaderboard or apply the\n"
        f"GAIA official scoring script to the extracted_answer field.\n"
        f"A separate judge pass using CORRECTNESS_JUDGE_PROMPT can also be run."
    )


import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for the GAIA evaluation script.
    Triggered via Hydra configuration.
    """
    run_gaia_evaluation(
        output_dir=cfg.eval.gaia_output_dir,
        num_samples=cfg.eval.num_samples,
        level=cfg.eval.gaia_level,
        max_iterations=cfg.eval.max_iterations,
    )

if __name__ == "__main__":
    main()
