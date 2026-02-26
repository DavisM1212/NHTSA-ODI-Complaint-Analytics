from pathlib import Path

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTRACTED_DATA_DIR = DATA_DIR / "extracted"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_project_directories():
    required_dirs = [
        DOCS_DIR,
        DOCS_DIR / "screenshots",
        RAW_DATA_DIR,
        EXTRACTED_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUTS_DIR,
        SRC_DIR,
        SCRIPTS_DIR
    ]
    for path in required_dirs:
        path.mkdir(parents=True, exist_ok=True)
