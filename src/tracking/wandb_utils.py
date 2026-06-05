"""
WandB Utilities Module

Provides helper functions for initializing and logging to Weights & Biases.

The implementation supports:
    - Initializing W&B runs with custom configurations
    - Logging specific files as evaluation artifacts

Key classes / functions:
    - init_wandb_run: Initializes a W&B run with a resolved config dict.
    - log_evaluation_artifact: Logs a single file as an evaluation artifact to W&B.

Version:
    - 05-Jun-2026 (Version 1.0): Initial implementation and documentation.
"""
import os
import wandb

def init_wandb_run(project, config, job_type, group=None, tags=None):
    """
    Initializes a W&B run with a resolved configuration dictionary.

    Args:
        project (str): The name of the W&B project.
        config (dict): The configuration dictionary for the run.
        job_type (str): The type of job (e.g., 'eval', 'train').
        group (Optional[str]): The group name to categorize the run. Default is None.
        tags (Optional[List[str]]): A list of tags for the run. Default is None.

    Returns:
        wandb.sdk.wandb_run.Run: The initialized W&B run object.

    Example:
        >>> run = init_wandb_run("my-project", {"lr": 0.01}, "train")
    """
    run = wandb.init(
        project=project,
        config=config,
        job_type=job_type,
        group=group,
        tags=tags
    )
    return run

def log_evaluation_artifact(run, file_path, name):
    """
    Logs a single file as an evaluation artifact to the active W&B run.

    Args:
        run (wandb.sdk.wandb_run.Run): The active W&B run object.
        file_path (str): The local path to the file being logged.
        name (str): The name to assign to the artifact in W&B.

    Returns:
        None

    Example:
        >>> log_evaluation_artifact(run, "results.json", "evaluation_results")
    """
    artifact = wandb.Artifact(name=name, type="evaluation")
    artifact.add_file(file_path)
    run.log_artifact(artifact)
