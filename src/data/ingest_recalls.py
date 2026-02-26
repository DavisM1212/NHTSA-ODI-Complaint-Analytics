import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import settings
from src.config.paths import (
    EXTRACTED_DATA_DIR,
    OUTPUTS_DIR,
    RAW_DATA_DIR,
    ensure_project_directories,
)
from src.data.io_utils import discover_zip_files, safe_extract_zip


# -----------------------------------------------------------------------------
# Recall extraction placeholder
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract NHTSA recall zip files (starter workflow placeholder)"
    )
    parser.add_argument(
        "--overwrite-extracted",
        action="store_true",
        default=settings.OVERWRITE_EXTRACTED,
        help="Re-extract files even if they already exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    recall_zips = discover_zip_files(RAW_DATA_DIR, include_terms=["rcl", "recall"])
    if not recall_zips:
        print("[info] No recall zip files found in data/raw")
        print("[hint] This project supports optional recall joins later")
        return 0

    manifest_rows = []
    for zip_path in recall_zips:
        extract_dir = EXTRACTED_DATA_DIR
        print(f"[extract] {zip_path.name} -> {extract_dir}")
        extracted_paths = safe_extract_zip(
            zip_path, extract_dir, overwrite=args.overwrite_extracted
        )
        for extracted_path in extracted_paths:
            manifest_rows.append(
                {
                    "source_zip": zip_path.name,
                    "extracted_file": str(
                        Path(extracted_path).relative_to(EXTRACTED_DATA_DIR.parent)
                    ),
                }
            )

    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_path = OUTPUTS_DIR / settings.RECALL_EXTRACT_MANIFEST_NAME
        manifest_df.to_csv(manifest_path, index=False)
        print(f"[write] {manifest_path}")

    print(
        "[note] Recall normalization/join logic is intentionally left as a starter placeholder"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
