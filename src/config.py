from pathlib import Path

# Output paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Input data paths
ENRON_CSV = RAW_DIR / "Enron.csv"
LINGSPAM_CSV = RAW_DIR / "Ling.csv"

# Data columns
COLUMN_ALIASES = {
    "subject": ["subject"],
    "body": ["body"],
    "label": ["label"],
}

# Label mapping from data source
LABEL_MAP_MALICIOUS = {1, "1"}
LABEL_MAP_SAFE = {0, "0"}

# Reproducibility
RANDOM_STATE = 42
