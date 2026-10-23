"""Console + file logging setup shared by the scripts."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> None:
    """Log to stdout and, if given, to ``log_file`` (appending, UTF-8)."""
    handlers: list = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers, force=True)
