import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import settings
from src.config.constants import EXPECTED_COMPLAINT_ZIP_NAMES
from src.config.paths import (
    EXTRACTED_DATA_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ensure_project_directories,
)
from src.data.io_utils import (
    discover_zip_files,
    minor_preprocess_complaints,
    read_tabular_file,
    safe_extract_zip,
    sanitize_name,
    write_dataframe,
)
from src.data.schema_checks import (
    collect_schema_report,
    get_schema_columns,
    print_schema_report,
)


# -----------------------------------------------------------------------------
# Discovery and processing
# -----------------------------------------------------------------------------
def find_complaint_zip_files():
    expected_paths = []
    for file_name in EXPECTED_COMPLAINT_ZIP_NAMES:
        path = RAW_DATA_DIR / file_name
        if path.exists():
            expected_paths.append(path)

    if expected_paths:
        return expected_paths

    return discover_zip_files(RAW_DATA_DIR, include_terms=["complaint"])


def tabular_candidates(extracted_paths):
    allowed_suffixes = {".txt", ".csv", ".tsv"}
    files = []
    for path in extracted_paths:
        if Path(path).suffix.lower() in allowed_suffixes:
            files.append(Path(path))
    return sorted(files)


def process_table_file(table_path, source_zip_name, output_format):
    complaint_columns = get_schema_columns("complaints")
    df = read_tabular_file(
        table_path,
        header=None,
        column_names=complaint_columns,
    )
    df = minor_preprocess_complaints(df)
    df["source_zip"] = source_zip_name
    df["source_file"] = table_path.name

    report = collect_schema_report(
        df,
        dataset_name=table_path.name,
        schema_name="complaints",
    )
    print_schema_report(report)

    base_name = sanitize_name(f"{table_path.stem}_processed")
    output_stem = PROCESSED_DATA_DIR / base_name
    output_path = write_dataframe(
        df, output_stem, prefer_parquet=output_format == "parquet"
    )

    manifest_row = {
        "source_zip": source_zip_name,
        "source_file": table_path.name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "processed_output": str(output_path.relative_to(PROCESSED_DATA_DIR.parent)),
    }
    return df, output_path, manifest_row


def process_zip_file(zip_path, overwrite_extracted=False, output_format="parquet"):
    zip_path = Path(zip_path)
    extract_dir = EXTRACTED_DATA_DIR
    print("")
    print(f"[extract] {zip_path.name} -> {extract_dir}")
    extracted_paths = safe_extract_zip(
        zip_path, extract_dir, overwrite=overwrite_extracted
    )
    candidate_files = tabular_candidates(extracted_paths)

    if not candidate_files:
        print(f"[warn] No tabular files found after extracting {zip_path.name}")
        return [], []

    frames = []
    manifest_rows = []
    for table_path in candidate_files:
        print(f"[process] {table_path.name}")
        df, output_path, manifest_row = process_table_file(
            table_path, zip_path.name, output_format
        )
        print(f"[write] {output_path}")
        frames.append(df)
        manifest_rows.append(manifest_row)

    return frames, manifest_rows


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract NHTSA ODI complaint zip files and create pandas-friendly outputs"
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv"],
        default=settings.OUTPUT_FORMAT
        if settings.OUTPUT_FORMAT in {"parquet", "csv"}
        else "parquet",
        help="Preferred processed output format",
    )
    parser.add_argument(
        "--overwrite-extracted",
        action="store_true",
        default=settings.OVERWRITE_EXTRACTED,
        help="Re-extract files even if they already exist under data/extracted",
    )
    parser.add_argument(
        "--no-combine",
        action="store_true",
        default=not settings.COMBINE_PROCESSED_OUTPUTS,
        help="Skip creating a combined complaint dataset",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    zip_files = find_complaint_zip_files()
    if not zip_files:
        print("[error] No complaint zip files found in data/raw")
        print(
            "[hint] Add complaint zip files such as COMPLAINTS_RECEIVED_2020-2024.zip and COMPLAINTS_RECEIVED_2025-2026.zip"
        )
        return 1

    print(f"[info] Found {len(zip_files)} complaint zip file(s)")
    print(f"[info] Preferred output format: {args.output_format}")
    if args.no_combine:
        print("[info] Combined dataset output disabled")

    all_frames = []
    all_manifest_rows = []

    for zip_path in zip_files:
        frames, manifest_rows = process_zip_file(
            zip_path,
            overwrite_extracted=args.overwrite_extracted,
            output_format=args.output_format,
        )
        all_manifest_rows.extend(manifest_rows)
        if not args.no_combine:
            all_frames.extend(frames)

    if all_manifest_rows:
        manifest_df = pd.DataFrame(all_manifest_rows)
        manifest_path = OUTPUTS_DIR / settings.INGEST_MANIFEST_NAME
        manifest_df.to_csv(manifest_path, index=False)
        print("")
        print(f"[write] {manifest_path}")

    if not args.no_combine and all_frames:
        print("")
        print("[combine] Building combined complaints dataset")
        combined_df = pd.concat(all_frames, ignore_index=True, sort=False)
        combined_output = write_dataframe(
            combined_df,
            PROCESSED_DATA_DIR / settings.COMBINED_COMPLAINT_OUTPUT_STEM,
            prefer_parquet=args.output_format == "parquet",
        )
        summary_path = OUTPUTS_DIR / "odi_complaints_combined_summary.csv"
        summary_df = pd.DataFrame(
            [
                {
                    "rows": int(len(combined_df)),
                    "columns": int(len(combined_df.columns)),
                    "processed_output": str(
                        combined_output.relative_to(PROCESSED_DATA_DIR.parent)
                    ),
                }
            ]
        )
        summary_df.to_csv(summary_path, index=False)
        print(f"[write] {combined_output}")
        print(f"[write] {summary_path}")

    print("")
    print("[done] ODI complaint ingestion finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
