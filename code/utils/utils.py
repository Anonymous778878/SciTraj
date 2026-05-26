"""
Shared utilities for the SciTraj-V2 pipeline.
- Config loader (reads config/main.yaml)
- Simple logger
- Path helpers
- Text utilities
"""
import json
import logging
import re
import sys
from pathlib import Path

import yaml


def load_config(config_path: str = "config/main.yaml") -> dict:
    """Load YAML config. Every script reads the same file.

    Honors environment variable SCITRAJ_SEED_OVERRIDE: if set to an integer,
    overrides cfg['project']['seed']. Used by the multi-seed runner.
    """
    import os
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    seed_env = os.environ.get("SCITRAJ_SEED_OVERRIDE")
    if seed_env:
        try:
            cfg.setdefault("project", {})["seed"] = int(seed_env)
        except ValueError:
            pass
    return cfg


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with clean formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path) -> Path:
    """Create directory (and parents) if missing. Return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path) -> None:
    """Write JSON to disk. Creates parent dir if needed."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    """Load JSON from disk."""
    with open(path) as f:
        return json.load(f)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace/newlines to single spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    """Fast word count."""
    return len((text or "").split())


def count_alpha(text: str) -> int:
    """Count alphabetic characters. Used for language/quality checks."""
    if not text:
        return 0
    return sum(1 for c in text if c.isalpha())


# Constants used across scripts
BOILERPLATE_MARKER = "Requests for name changes"
MIN_ABSTRACT_CHARS = 100
MIN_WORDS = 10
MIN_ALPHA = 50
