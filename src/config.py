"""Project configuration and path resolution module.

Centralizes paths, hyperparameters, constants, and deterministic settings
across all pipeline components.
"""

from pathlib import Path
from typing import List

# Dynamic Project Root Resolution
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw" / "swissprot-ec"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Raw Parquet Filenames
TRAIN_PARQUET: str = "train-00000-of-00001.parquet"
DEV_PARQUET: str = "dev-00000-of-00001.parquet"
TEST_PARQUET: str = "test-00000-of-00001.parquet"

# Processed File Paths
CLEANED_DATA_PATH: Path = PROCESSED_DATA_DIR / "cleaned_sequences.parquet"
DEDUP_DATA_PATH: Path = PROCESSED_DATA_DIR / "deduplicated_sequences.parquet"
TRAIN_SPLIT_PATH: Path = PROCESSED_DATA_DIR / "train_split.parquet"
VAL_SPLIT_PATH: Path = PROCESSED_DATA_DIR / "val_split.parquet"
TEST_SPLIT_PATH: Path = PROCESSED_DATA_DIR / "test_split.parquet"

# Output Directories
MODELS_DIR: Path = PROJECT_ROOT / "models" / "saved"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
PREDICTIONS_DIR: Path = RESULTS_DIR / "predictions"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
REPORT_FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Ensure runtime directories exist
for path in [
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    REPORT_FIGURES_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Determinism
RANDOM_SEED: int = 42

# Biological Constants
STANDARD_AA: List[str] = sorted(list("ACDEFGHIKLMNPQRSTVWY"))
VALID_EC_CLASSES: List[int] = [1, 2, 3, 4, 5, 6]

# Splitting Ratios
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# Feature Engineering Parameters
AAC_DIM: int = 20
DPC_DIM: int = 400
SVD_COMPONENTS: int = 128
SELECT_K_FEATURES: int = 500

# CV & Training Settings
CV_FOLDS: int = 5
N_JOBS: int = 2
