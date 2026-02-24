# -----------------------------------------------------------------------------
# Project constants
# -----------------------------------------------------------------------------
PROJECT_NAME = "NHTSA-ODI-COMPLAINT-ANALYTICS"
TARGET_PYTHON_VERSION = "3.13.12"
DEFAULT_RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# Raw file expectations
# -----------------------------------------------------------------------------
EXPECTED_COMPLAINT_ZIP_NAMES = [
    "COMPLAINTS_RECEIVED_2020-2024.zip",
    "COMPLAINTS_RECEIVED_2025-2026.zip",
]

ALTERNATE_COMPLAINT_ZIP_NAMES = ["complaints_2020_2024.zip", "complaints_2025_2026.zip"]

EXPECTED_RECALL_ZIP_NAMES = ["RCL_FROM_2020_2024.zip", "RCL_FROM_2025_2026.zip"]

ALTERNATE_RECALL_ZIP_NAMES = ["rcl_from_2020_2024.zip", "rcl_from_2025_2026.zip"]


# -----------------------------------------------------------------------------
# Data parsing hints
# -----------------------------------------------------------------------------
DATE_COLUMN_HINTS = ["date", "received", "incident"]

MODEL_YEAR_COLUMN_CANDIDATES = ["model_year", "modelyear", "veh_year", "year"]

COMMON_ODI_COMPLAINT_ID_COLUMNS = ["odi_number", "odino", "cmplid", "complaint_number"]

PREFERRED_OUTPUT_FORMAT = "parquet"
