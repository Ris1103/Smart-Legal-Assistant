"""
Centralised logging configuration for the Legal Advisor API.

Call setup_logging() once at application startup (main.py lifespan).
Every module then uses the standard pattern:

    import logging
    logger = logging.getLogger(__name__)

Log level and file output are controlled by settings:
    LOG_LEVEL        — DEBUG | INFO | WARNING | ERROR  (default: INFO)
    LOG_TO_FILE      — true | false                   (default: true)
    LOG_DIR          — directory for log files         (default: logs)
    LOG_MAX_BYTES    — rotate after N bytes            (default: 10 MB)
    LOG_BACKUP_COUNT — how many rotated files to keep  (default: 5)
"""

import logging
import logging.handlers
import os
import pathlib
import sys
from typing import Optional


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_setup_done = False


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure the root logger once.

    - Console handler: always active, streams to stdout
    - File handler (app.log): all records at *level* and above; rotated by size
    - Error file handler (error.log): ERROR and above only
    """
    global _setup_done
    if _setup_done:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any handlers that basicConfig or third-party libs may have added
    root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- Console ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # --- File handlers ---
    if log_to_file:
        log_path = pathlib.Path(log_dir)
        # log_dir is relative to the app/ working directory
        if not log_path.is_absolute():
            app_dir = pathlib.Path(__file__).resolve().parent.parent
            log_path = app_dir / log_dir
        log_path.mkdir(parents=True, exist_ok=True)

        # All logs
        app_handler = logging.handlers.RotatingFileHandler(
            log_path / "app.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        app_handler.setLevel(numeric_level)
        app_handler.setFormatter(formatter)
        root.addHandler(app_handler)

        # Errors only
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "error.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root.addHandler(error_handler)

    # Silence overly chatty third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _setup_done = True
    logging.getLogger(__name__).info(
        "Logging initialised | level=%s | file=%s | dir=%s",
        level.upper(),
        log_to_file,
        log_dir,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Convenience wrapper — same as logging.getLogger but documents intent."""
    return logging.getLogger(name)
