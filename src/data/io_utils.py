import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from src.config.constants import DATE_COLUMN_HINTS, MODEL_YEAR_COLUMN_CANDIDATES


# -----------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------
def sanitize_name(value):
    text = str(value).strip()
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "dataset"


def discover_zip_files(raw_dir, include_terms):
    include_terms = [term.lower() for term in include_terms]
    zip_paths = []
    for path in sorted(Path(raw_dir).glob("*.zip")):
        name_lower = path.name.lower()
        if any(term in name_lower for term in include_terms):
            zip_paths.append(path)
    return zip_paths


def safe_extract_zip(zip_path, target_dir, overwrite=False):
    zip_path = Path(zip_path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    extracted_paths = []

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            relative_path = Path(member.filename)
            destination = (target_dir / relative_path).resolve()
            if target_root not in destination.parents and destination != target_root:
                raise ValueError(
                    f"Unsafe zip member path detected in {zip_path.name}: {member.filename}"
                )

            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not overwrite:
                extracted_paths.append(destination)
                continue

            with (
                archive.open(member, "r") as source_handle,
                destination.open("wb") as destination_handle
            ):
                shutil.copyfileobj(source_handle, destination_handle)

            extracted_paths.append(destination)

    return sorted(set(extracted_paths))


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------
def detect_delimiter(file_path):
    file_path = Path(file_path)
    try:
        sample = file_path.read_text(encoding="latin-1", errors="ignore")[:8000]
    except OSError:
        return "\t"

    lines = [line for line in sample.splitlines() if line.strip()]
    # The first line is usually the least chaotic place to guess a delimiter
    header = lines[0] if lines else sample
    header_counts = {
        "\t": header.count("\t"),
        "|": header.count("|"),
        ",": header.count(",")
    }
    header_delimiter = max(header_counts, key=header_counts.get)
    if header_counts[header_delimiter] > 0:
        return header_delimiter

    sample_counts = {
        "\t": sample.count("\t"),
        "|": sample.count("|"),
        ",": sample.count(",")
    }
    delimiter = max(sample_counts, key=sample_counts.get)
    if sample_counts[delimiter] == 0:
        return "\t"
    return delimiter


def read_tabular_file(file_path, header="infer", column_names=None):
    file_path = Path(file_path)
    delimiter = detect_delimiter(file_path)
    encodings = ["utf-8", "latin-1"]
    last_error = None

    for encoding in encodings:
        try:
            read_kwargs = {
                "sep": delimiter,
                "dtype": str,
                "low_memory": False,
                "encoding": encoding,
                "on_bad_lines": "skip"
            }

            if header != "infer":
                read_kwargs["header"] = header

            if column_names is not None:
                read_kwargs["names"] = list(column_names)

            return pd.read_csv(file_path, **read_kwargs)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to read {file_path} as a tabular file") from last_error


def minor_preprocess_complaints(df):
    working_df = df.copy()

    object_columns = working_df.select_dtypes(include=["object"]).columns.tolist()
    for column in object_columns:
        working_df[column] = working_df[column].astype("string").str.strip()
        working_df[column] = working_df[column].replace({"": pd.NA})

    for column in working_df.columns:
        if any(hint in column for hint in DATE_COLUMN_HINTS):
            parsed = pd.to_datetime(working_df[column], errors="coerce")
            if parsed.notna().sum() > 0:
                working_df[column] = parsed

    for column in MODEL_YEAR_COLUMN_CANDIDATES:
        if column in working_df.columns:
            working_df[column] = pd.to_numeric(
                working_df[column], errors="coerce"
            ).astype("Int64")

    return working_df


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def write_dataframe(df, output_stem, prefer_parquet=True):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    if prefer_parquet:
        try:
            import pyarrow as pa

            output_path = output_stem.with_suffix(".parquet")
            df.to_parquet(output_path, index=False)
            return output_path
        except Exception:
            pass

    output_path = output_stem.with_suffix(".csv")
    df.to_csv(output_path, index=False)
    return output_path
