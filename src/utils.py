"""Utility helpers for timing, memory tracking, artifact persistence, and logging."""

import os
import random
import time
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator
import joblib
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across random and numpy."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(name: str = "enzyme_ml", level: int = logging.INFO) -> logging.Logger:
    """Configure a clean console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@contextmanager
def timer(description: str) -> Generator[None, None, None]:
    """Context manager to measure and print elapsed execution time."""
    start_time = time.perf_counter()
    print(f"[{description}] Started...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        mins, secs = divmod(elapsed, 60)
        if mins > 0:
            print(f"[{description}] Completed in {int(mins)}m {secs:.2f}s ({elapsed:.2f}s total)")
        else:
            print(f"[{description}] Completed in {elapsed:.2f}s")


def save_artifact(obj: Any, path: Path) -> Path:
    """Save an object to disk via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_artifact(path: Path) -> Any:
    """Load a serialized object from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)
