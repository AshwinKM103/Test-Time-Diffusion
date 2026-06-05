"""
Custom Logging System for ML Pipelines and Applications

This module provides a rich, flexible, and extensible logging system designed
for machine learning pipelines and general applications. It leverages the 'rich'
library for beautiful console output, provides multiple file-based handlers
(including JSONL), and integrates seamlessly with Weights & Biases.

The implementation supports:
    - Richly formatted console logging with styled, boxed log levels.
    - Task-specific loggers that write to a rich console, a human-readable
      .txt file, and a machine-readable .jsonl file.
    - Optional, one-line integration with Weights & Biases for experiment tracking.
    - A utility to display dataclass, dict, or OmegaConf configurations in a clean table.
    - A factory for creating custom, rich-styled progress bars.
    - Utilities for silencing noisy third-party loggers.

Key classes / functions:
    - CustomLogger: The main static class encapsulating all functionalities.
    - RichBoxedLevelFormatter: A custom formatter for visually distinct console log levels.
    - BoxedLevelFormatter: A custom formatter for clean, simple file logs.
    - JsonFormatter: A formatter for machine-readable JSONL output.

Author:
    - Ashwin K M (ashwinkm@iisc.ac.in)

Version:
    - 1.1 (12-Nov-2025): Merged ML-focused and App-focused loggers.
"""

# Standard Library Imports
import logging
import json
from pathlib import Path
from typing import Optional, Union, Any, Dict, List, Tuple
from dataclasses import is_dataclass, asdict

# Add SUCCESS level for loguru compatibility
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)
logging.Logger.success = success

# Third-Party Imports
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.theme import Theme

try:
    from omegaconf import DictConfig
except ImportError:
    DictConfig = None  # type: ignore

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore


class CustomLogger:
    """
    A centralized static class for handling rich logging, progress bars,
    and configuration display.
    """

    # Define custom theme, merging styles from both modules
    _custom_theme = Theme({
        # From File 1
        "logging.level.debug": "cyan",
        "logging.level.info": "green",
        "logging.level.warning": "yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        # From File 2
        "info": "cyan",
        "success_emoji": "bold green",
        "failure_emoji": "bold red",
        "skip_emoji": "bold yellow",
    })

    # Centralized console instance
    console = Console(theme=_custom_theme)

    # Style map for the RichBoxedLevelFormatter (from File 2)
    _RICH_LEVEL_STYLES = {
        logging.DEBUG: "[dim cyan]   DEBUG   [/dim cyan]",
        logging.INFO: "[bold white on blue]   INFO    [/bold white on blue]",
        SUCCESS_LEVEL_NUM: "[bold white on green] SUCCESS   [/bold white on green]",
        logging.WARNING: "[bold black on yellow] WARNING   [/bold black on yellow]",
        logging.ERROR: "[bold white on red]   ERROR   [/bold white on red]",
        logging.CRITICAL: "[bold white on magenta] CRITICAL  [/bold white on magenta]",
    }

    class RichBoxedLevelFormatter(logging.Formatter):
        """
        A custom log formatter that styles log levels with rich, boxed text
        for console output. (Based on File 2's formatter)
        """
        def format(self, record: logging.LogRecord) -> str:
            """Formats a log record with a styled, boxed level name."""
            record.levelname_styled = CustomLogger._RICH_LEVEL_STYLES.get(
                record.levelno, record.levelname
            )
            # Add marker for compatibility with BoxedLevelFormatter
            record._rich_handler = True
            return super().format(record)

    class BoxedLevelFormatter(logging.Formatter):
        """
        A custom log formatter that boxes the log level for plain file output.
        (Based on File 1's formatter)
        """
        LEVEL_STYLES = {
            "DEBUG": " D ",
            "INFO": " I ",
            "SUCCESS": " S ",
            "WARNING": " W ",
            "ERROR": " E ",
            "CRITICAL": " C ",
        }

        def format(self, record):
            # Only modify levelname if it's not for a Rich handler
            if not hasattr(record, '_rich_handler'):
                record.levelname = f"[{self.LEVEL_STYLES.get(record.levelname, record.levelname)}]"
            return super().format(record)

    class JsonFormatter(logging.Formatter):
        """
        A custom formatter to output log records as JSONL strings.
        (Based on File 1's formatter)
        """
        def format(self, record):
            log_record = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
            }
            
            # Add exception info if present
            if record.exc_info:
                log_record['exc_info'] = self.formatException(record.exc_info)
            
            # Add stack info if present
            if record.stack_info:
                log_record['stack_info'] = self.formatStack(record.stack_info)
                
            # Add any extra fields
            if hasattr(record, '__dict__'):
                extra = record.__dict__.copy()
                # Remove standard keys
                for key in ('args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                            'funcName', 'levelname', 'levelno', 'lineno', 'module',
                            'msecs', 'message', 'msg', 'name', 'pathname', 'process',
                            'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'):
                    extra.pop(key, None)
                # Clean up any internal-only keys
                extra.pop('_rich_handler', None)
                extra.pop('levelname_styled', None)
                log_record.update(extra)

            return json.dumps(log_record)

    @staticmethod
    def setup_root_logger(level: int = logging.INFO) -> None:
        """
        Sets up the root logger with a RichHandler and our custom formatter.

        Args:
            level (int): The logging level to set for the root logger.
        """
        formatter = CustomLogger.RichBoxedLevelFormatter(
            fmt="%(levelname_styled)s %(message)s"
        )
        handler = RichHandler(
            console=CustomLogger.console,
            show_time=True,
            log_time_format="[%X]",
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            show_level=False,  # We handle level display in our formatter
            keywords=[]  # Disable Rich's default keyword highlighting
        )
        handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
    
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
        root_logger.addHandler(handler)

    @staticmethod
    def setup_task_logger(
        name: str,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        file_log_level: int = logging.DEBUG,
        console_log_level: int = logging.INFO,
        clear_handlers: bool = True,
    ) -> logging.Logger:
        """
        Creates a named logger for a specific task with console and file handlers.

        This comprehensive logger will output to:
        1. The Console (using RichBoxedLevelFormatter)
        2. A human-readable 'logs.txt' file (using BoxedLevelFormatter)
        3. A machine-readable 'logs.jsonl' file (using JsonFormatter)

        Args:
            name (str): The name of the logger.
            output_dir (Optional[Union[str, Path]]): If provided, logs will be
                written to 'logs.txt' and 'logs.jsonl' in this directory.
            verbose (bool): If True, sets the console log level to DEBUG.
            file_log_level (int): The minimum level for messages to be
                written to the log files.
            console_log_level (int): The minimum level for messages to be
                displayed in the console.
            clear_handlers (bool): Whether to clear existing handlers on the logger.

        Returns:
            logging.Logger: The configured logger instance.
        """
        logger = logging.getLogger(name)
        # Set logger to the lowest level to capture all messages
        logger.setLevel(min(file_log_level, console_log_level))
        logger.propagate = False  

        if clear_handlers and logger.hasHandlers():
            logger.handlers.clear()

        # 1. Add Rich console handler
        if not any(isinstance(h, RichHandler) for h in logger.handlers):
            effective_console_level = logging.DEBUG if verbose else console_log_level
            console_handler = RichHandler(
                console=CustomLogger.console, 
                rich_tracebacks=True,
                show_path=False,
                markup=True,
                log_time_format="[%X]",
                omit_repeated_times=False,
                show_level=False,  # We handle level display
                keywords=[]
            )
            console_formatter = CustomLogger.RichBoxedLevelFormatter(
                fmt="%(levelname_styled)s %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(effective_console_level)
            logger.addHandler(console_handler)

        # 2. Add file handlers if output directory is specified
        if output_dir:
            log_dir = Path(output_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # 2a. Human-readable .txt log
            log_path_txt = log_dir / "logs.txt"
            file_handler_txt = logging.FileHandler(log_path_txt, mode='a', encoding='utf-8')
            file_handler_txt.setLevel(file_log_level)
            file_handler_txt.setFormatter(CustomLogger.BoxedLevelFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logger.addHandler(file_handler_txt)
            
            # 2b. Machine-readable .jsonl log
            log_path_jsonl = log_dir / "logs.jsonl"
            file_handler_jsonl = logging.FileHandler(log_path_jsonl, mode='a', encoding='utf-8')
            file_handler_jsonl.setLevel(file_log_level)
            file_handler_jsonl.setFormatter(CustomLogger.JsonFormatter())
            logger.addHandler(file_handler_jsonl)

        return logger

    @staticmethod
    def attach_wandb_logger(logger: logging.Logger, wandb_run) -> None:
        """
        Attaches a custom handler to forward logs to Weights & Biases.
        (From File 1)

        Args:
            logger (logging.Logger): The logger instance to attach to.
            wandb_run: The active wandb run object (from wandb.init()).
        """
        if wandb is None:
            logger.warning("wandb library not found. Skipping WandB logging.")
            return

        class WandBHandler(logging.Handler):
            def emit(self, record):
                log_entry = self.format(record)
                # Clean the log entry of Rich markup for WandB
                import re
                clean_log = re.sub(r'\[/?[^\]]*\]', '', log_entry)
                # Log to a custom key like "console_logs"
                wandb_run.log({"console_logs": clean_log}, commit=False)

        wandb_handler = WandBHandler()
        wandb_handler.setLevel(logging.INFO)
        # Use a simple formatter for wandb
        wandb_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(wandb_handler)
        logger.info("Attached Weights & Biases logger.")

    @staticmethod
    def log_config(config: Any, logger: Optional[logging.Logger] = None) -> None:
        """
        Logs a configuration object as a rich Table.
        (From File 1, supports dict, dataclass, and OmegaConf)

        Args:
            config (Any): A dataclass, dict, or OmegaConf DictConfig object.
            logger (Optional[logging.Logger]): If provided, used to log errors.
        """
        table = Table(title="Experiment Configuration", show_header=True, header_style="bold magenta", show_lines=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        config_dict: Dict[str, Any]
        if is_dataclass(config):
            config_dict = asdict(config)
        elif DictConfig and isinstance(config, DictConfig):
            config_dict = config # type: ignore
        elif isinstance(config, dict):
            config_dict = config
        else:
            msg = f"Configuration object of type {type(config)} not supported for table logging."
            if logger:
                logger.warning(msg)
            else:
                CustomLogger.console.print(f"[yellow]WARNING:[/] {msg}")
            return

        for key, value in config_dict.items():
            table.add_row(str(key), str(value))
        
        CustomLogger.console.print(table)

    @staticmethod
    def create_custom_progress_bar(
        color: str = "cyan",
        show_time: bool = True,
        show_spinner: bool = True,
        spinner_style: str = "dots",
        disable: bool = False,
        metrics: List[Tuple[str, str]] = [("Loss", ".4f")],
    ) -> Progress:
        """
        Creates a customizable Rich progress bar for tracking tasks.
        (From File 2, with MofNColumn added from File 1)

        Args:
            color (str): The color for the progress bar and spinner.
            show_time (bool): If True, includes the "Time Remaining" column.
            show_spinner (bool): If True, includes a spinner animation.
            spinner_style (str): The style of the spinner (e.g., "dots", "moon").
            disable (bool): If True, disables rendering of the progress bar.
            metrics (List[Tuple[str, str]]): Custom metrics to display,
                as (name, format_string) tuples.

        Returns:
            Progress: A configured `rich.progress.Progress` instance.
        """
        columns = []

        if show_spinner:
            columns.append(SpinnerColumn(spinner_name=spinner_style, style=color))

        columns.extend([
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, style=color, complete_style=f"bold {color}"),
            TaskProgressColumn(),
            MofNCompleteColumn(),  # Added from File 1
        ])

        for name, fmt in metrics:
            columns.append(TextColumn(f"[{name}: {{task.fields[{name}]:{fmt}}}]", justify="right"))

        if show_time:
            columns.append(TimeRemainingColumn())

        progress = Progress(*columns, console=CustomLogger.console, expand=True, disable=disable)
        return progress

    @staticmethod
    def disable_other_loggers(level: int = logging.WARNING) -> None:
        """
        Disable or set higher level for noisy third-party loggers.
        (From File 1)
        
        Args:
            level (int): The logging level to set for third-party loggers.
        """
        noisy_loggers = [
            'transformers',
            'datasets',
            'deepspeed',
            'torch.distributed',
            'accelerate',
            'wandb',
            'h5py',
            'matplotlib',
            'PIL'
        ]
        
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(level)

    @staticmethod
    def enable_file_logging(
        log_path: Union[str, Path],
        log_format: str = "%(asctime)s — %(levelname)s — %(message)s"
    ) -> None:
        """
        Adds a file handler to the root logger to capture all logs to a file.
        (From File 2)

        Args:
            log_path (Union[str, Path]): The file path where logs will be written.
            log_format (str): The format string for log messages in the file.
        """
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        f_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(log_format)
        f_handler.setFormatter(formatter)
        f_handler.setLevel(logging.DEBUG)

        logging.getLogger().addHandler(f_handler)

    @staticmethod
    def log_metrics_table(
        metrics: Dict[str, Any],
        title: str = "Evaluation Metrics",
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Logs evaluation metrics as a rich Table to console and optionally to logger.
        
        This is useful for displaying DPO training metrics in a clean, organized format.
        
        Args:
            metrics (Dict[str, Any]): Dictionary of metric names to values
            title (str): Title for the table
            logger (Optional[logging.Logger]): If provided, also logs metrics to this logger
        """
        table = Table(title=title, show_header=True, header_style="bold magenta", show_lines=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", justify="right")
        
        for metric_name, value in sorted(metrics.items()):
            # Format value based on type
            if isinstance(value, float):
                if abs(value) < 0.01 or abs(value) > 1000:
                    formatted_value = f"{value:.6e}"  # Scientific notation for very small/large
                else:
                    formatted_value = f"{value:.4f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"  # Add thousand separators
            else:
                formatted_value = str(value)
            
            table.add_row(metric_name, formatted_value)
        
        CustomLogger.console.print(table)
        
        # Also log to file if logger provided
        if logger:
            logger.info(f"{title}:")
            for metric_name, value in sorted(metrics.items()):
                logger.info(f"  {metric_name}: {value}")