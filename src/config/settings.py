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
    "y"
}
RANDOM_SEED = int(os.getenv("ODI_RANDOM_SEED", str(DEFAULT_RANDOM_SEED)))
