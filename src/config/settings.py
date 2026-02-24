import os

from src.config.constants import DEFAULT_RANDOM_SEED, PREFERRED_OUTPUT_FORMAT

# -----------------------------------------------------------------------------
# Runtime settings
# -----------------------------------------------------------------------------
OUTPUT_FORMAT = os.getenv("ODI_OUTPUT_FORMAT", PREFERRED_OUTPUT_FORMAT).strip().lower()
OVERWRITE_EXTRACTED = os.getenv("ODI_OVERWRITE_EXTRACTED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
COMBINE_PROCESSED_OUTPUTS = os.getenv(
    "ODI_COMBINE_PROCESSED", "true"
).strip().lower() not in {"0", "false", "no", "n"}
RANDOM_SEED = int(os.getenv("ODI_RANDOM_SEED", str(DEFAULT_RANDOM_SEED)))


# -----------------------------------------------------------------------------
# Output naming
# -----------------------------------------------------------------------------
COMBINED_COMPLAINT_OUTPUT_STEM = "odi_complaints_combined"
INGEST_MANIFEST_NAME = "ingest_odi_manifest.csv"
RECALL_EXTRACT_MANIFEST_NAME = "ingest_recalls_extract_manifest.csv"
