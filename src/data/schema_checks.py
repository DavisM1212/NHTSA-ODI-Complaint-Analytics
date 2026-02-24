import re
from functools import lru_cache

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from src.config.constants import (
    COMMON_ODI_COMPLAINT_ID_COLUMNS,
    MODEL_YEAR_COLUMN_CANDIDATES,
)
from src.config.paths import DOCS_DIR

# -----------------------------------------------------------------------------
# Schema doc parsing config
# -----------------------------------------------------------------------------
FIELD_ROW_PATTERN = re.compile(r"^\s*(\d+)\s+([A-Z0-9_]+)\s+([A-Z]+)\((\d+)\)\s*(.*)$")
CODE_LINE_PATTERN = re.compile(r"^\s*([A-Z0-9]{1,10})\s*=")
BRACKET_CODES_PATTERN = re.compile(r"\[([A-Z0-9,\s/]+)\]")

SCHEMA_DOCS = {"complaints": DOCS_DIR / "CMPL.txt", "recalls": DOCS_DIR / "RCL.txt"}

SCHEMA_OPTIONAL_COLUMNS = {
    "complaints": set(),
    "recalls": {"do_not_drive", "park_outside"},
}

SCHEMA_DATE_COLUMNS_OVERRIDES = {
    "complaints": {"faildate", "datea", "ldate", "purch_dt", "manuf_dt"},
    "recalls": {"bgman", "endman", "odate", "rcdate", "datea"},
}

SCHEMA_ENUM_OVERRIDES = {"recalls": {"influenced_by": {"MFR", "OVSC", "ODI"}}}

SCHEMA_LENGTH_OVERRIDES = {
    "recalls": {
        # Field changed in 2025; allow slightly larger legacy values without failing
        "fmvss": 6
    }
}

ALLOWED_PIPELINE_EXTRA_COLUMNS = {"source_zip", "source_file"}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def _normalize_identifier(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "column"


def _preview_list(values, max_items=8):
    if not values:
        return "[]"
    if len(values) <= max_items:
        return "[" + ", ".join(values) + "]"
    shown = ", ".join(values[:max_items])
    return f"[{shown}, ... +{len(values) - max_items} more]"


def _clean_text_series(series):
    text = series.astype("string").str.strip()
    return text.replace({"": pd.NA})


def _extract_codes_from_line(line):
    codes = []

    code_match = CODE_LINE_PATTERN.match(line)
    if code_match:
        codes.append(code_match.group(1).upper())

    for match in BRACKET_CODES_PATTERN.finditer(line.upper()):
        raw_items = match.group(1).split(",")
        for item in raw_items:
            cleaned = item.strip().upper()
            if cleaned:
                codes.append(cleaned)

    return codes


def _is_probably_blank_line(line):
    stripped = line.strip()
    return stripped == "" or set(stripped) <= {"=", "-"}


# -----------------------------------------------------------------------------
# Schema doc parsing
# -----------------------------------------------------------------------------
def _parse_schema_doc(doc_path, schema_name):
    if not doc_path.exists():
        raise FileNotFoundError(f"Schema doc not found: {doc_path}")

    raw_text = doc_path.read_text(encoding="latin-1", errors="ignore")
    lines = raw_text.splitlines()

    fields = []
    current_field = None
    in_field_section = False

    for line in lines:
        if "FIELDS:" in line:
            in_field_section = True
            continue

        if not in_field_section:
            continue

        if _is_probably_blank_line(line):
            continue

        row_match = FIELD_ROW_PATTERN.match(line)
        if row_match:
            field_index = int(row_match.group(1))
            field_name_raw = row_match.group(2).strip()
            field_type = row_match.group(3).strip().upper()
            field_size = int(row_match.group(4))
            description_line = row_match.group(5).strip()

            current_field = {
                "index": field_index,
                "name_raw": field_name_raw,
                "name": _normalize_identifier(field_name_raw),
                "type": field_type,
                "size": field_size,
                "description_lines": [],
                "allowed_codes": [],
            }
            if description_line:
                current_field["description_lines"].append(description_line)
                current_field["allowed_codes"].extend(
                    _extract_codes_from_line(description_line)
                )
            fields.append(current_field)
            continue

        if current_field is None:
            continue

        continuation = line.strip()
        if continuation:
            current_field["description_lines"].append(continuation)
            current_field["allowed_codes"].extend(
                _extract_codes_from_line(continuation)
            )

    if not fields:
        raise ValueError(f"No schema fields parsed from {doc_path}")

    normalized_fields = []
    for field in sorted(fields, key=lambda item: item["index"]):
        description_text = " ".join(field["description_lines"]).strip()
        allowed_codes = sorted({code for code in field["allowed_codes"] if code})
        is_date = field["name"] in SCHEMA_DATE_COLUMNS_OVERRIDES.get(schema_name, set())
        if "YYYYMMDD" in description_text.upper():
            is_date = True

        is_yes_no = (
            "'Y' OR 'N'" in description_text.upper()
            or "Y/N" in description_text.upper()
        )
        normalized_fields.append(
            {
                "index": field["index"],
                "name_raw": field["name_raw"],
                "name": field["name"],
                "type": field["type"],
                "size": field["size"],
                "description": description_text,
                "is_date": is_date,
                "is_yes_no": is_yes_no,
                "allowed_codes": allowed_codes,
            }
        )

    expected_columns = [field["name"] for field in normalized_fields]
    optional_columns = SCHEMA_OPTIONAL_COLUMNS.get(schema_name, set())
    required_columns = [
        name for name in expected_columns if name not in optional_columns
    ]

    return {
        "schema_name": schema_name,
        "doc_path": str(doc_path),
        "fields": normalized_fields,
        "field_map": {field["name"]: field for field in normalized_fields},
        "expected_columns": expected_columns,
        "required_columns": required_columns,
        "optional_columns": sorted(optional_columns),
    }


@lru_cache(maxsize=1)
def _schema_catalog():
    catalog = {}
    errors = {}
    for schema_name, doc_path in SCHEMA_DOCS.items():
        try:
            catalog[schema_name] = _parse_schema_doc(doc_path, schema_name)
        except Exception as exc:
            errors[schema_name] = str(exc)

    return {"schemas": catalog, "errors": errors}


def get_schema_spec(schema_name):
    catalog_info = _schema_catalog()
    if schema_name in catalog_info["errors"]:
        raise RuntimeError(
            f"Schema '{schema_name}' could not be parsed: {catalog_info['errors'][schema_name]}"
        )
    if schema_name not in catalog_info["schemas"]:
        available = sorted(catalog_info["schemas"].keys())
        raise KeyError(
            f"Schema '{schema_name}' not found (available: {available})"
        )
    return catalog_info["schemas"][schema_name]


def get_schema_columns(schema_name):
    schema_spec = get_schema_spec(schema_name)
    return list(schema_spec["expected_columns"])


def _detect_schema_name(columns):
    column_set = set(columns)
    catalog = _schema_catalog()["schemas"]

    best_schema_name = None
    best_score = 0.0
    best_overlap = 0

    for schema_name, schema_spec in catalog.items():
        expected_set = set(schema_spec["expected_columns"])
        overlap = len(column_set & expected_set)
        score = overlap / max(len(expected_set), 1)
        if score > best_score:
            best_score = score
            best_overlap = overlap
            best_schema_name = schema_name

    if best_score < 0.35:
        return None, best_score, best_overlap

    return best_schema_name, best_score, best_overlap


# -----------------------------------------------------------------------------
# Field validation helpers
# -----------------------------------------------------------------------------
def _allowed_values_for_field(schema_name, field):
    override_values = (
        SCHEMA_ENUM_OVERRIDES.get(schema_name, {}).get(field["name"])
        if schema_name in SCHEMA_ENUM_OVERRIDES
        else None
    )
    if override_values:
        return sorted(set(override_values))

    if field.get("is_yes_no"):
        return ["Y", "N"]

    if field.get("allowed_codes"):
        return field["allowed_codes"]

    return []


def _validate_date_field(series):
    result = {
        "checked": True,
        "non_null_count": 0,
        "placeholder_zero_count": 0,
        "invalid_date_count": 0,
        "invalid_examples": [],
    }

    if is_datetime64_any_dtype(series):
        result["non_null_count"] = int(series.notna().sum())
        return result

    text = _clean_text_series(series)
    non_null_mask = text.notna()
    result["non_null_count"] = int(non_null_mask.sum())

    if result["non_null_count"] == 0:
        return result

    upper_text = text.str.upper()
    zero_mask = upper_text.isin(["0", "00000000", "0000-00-00"]).fillna(False)
    result["placeholder_zero_count"] = int(zero_mask.sum())

    check_mask = non_null_mask & ~zero_mask
    if int(check_mask.sum()) == 0:
        return result

    digits_mask = text.str.fullmatch(r"\d{8}").fillna(False)
    parseable = pd.to_datetime(text.where(check_mask), format="%Y%m%d", errors="coerce")
    invalid_mask = check_mask & (~digits_mask | parseable.isna())

    result["invalid_date_count"] = int(invalid_mask.sum())
    if result["invalid_date_count"] > 0:
        examples = sorted(text[invalid_mask].dropna().astype(str).unique().tolist())
        result["invalid_examples"] = examples[:5]

    return result


def _validate_numeric_field(series, max_digits):
    result = {
        "checked": True,
        "non_null_count": 0,
        "non_numeric_count": 0,
        "non_integer_count": 0,
        "digits_overflow_count": 0,
        "invalid_examples": [],
    }

    text = _clean_text_series(series)
    non_null_mask = text.notna()
    result["non_null_count"] = int(non_null_mask.sum())
    if result["non_null_count"] == 0:
        return result

    numeric_values = pd.to_numeric(text, errors="coerce")
    non_numeric_mask = non_null_mask & numeric_values.isna()
    result["non_numeric_count"] = int(non_numeric_mask.sum())

    parseable_mask = non_null_mask & ~numeric_values.isna()
    if int(parseable_mask.sum()) > 0:
        parseable_values = numeric_values[parseable_mask]
        integer_mask = (parseable_values % 1).abs() < 1e-9
        result["non_integer_count"] = int((~integer_mask).sum())

        digit_lengths = (
            text[parseable_mask].str.replace(r"[^\d]", "", regex=True).str.len()
        )
        overflow_mask = digit_lengths > int(max_digits)
        result["digits_overflow_count"] = int(overflow_mask.sum())

    total_invalid_mask = non_numeric_mask.copy()
    if int(parseable_mask.sum()) > 0:
        parseable_index = text[parseable_mask].index
        non_integer_series = pd.Series(False, index=parseable_index)
        non_integer_series.loc[parseable_index] = (
            numeric_values[parseable_mask] % 1
        ).abs() >= 1e-9
        total_invalid_mask = total_invalid_mask | non_integer_series.reindex(
            text.index, fill_value=False
        )

    if result["non_numeric_count"] > 0 or result["non_integer_count"] > 0:
        examples = sorted(
            text[total_invalid_mask].dropna().astype(str).unique().tolist()
        )
        result["invalid_examples"] = examples[:5]

    return result


def _validate_char_length(series, max_length, allow_datetime=False):
    result = {
        "checked": True,
        "non_null_count": 0,
        "too_long_count": 0,
        "max_observed_length": 0,
    }

    if allow_datetime and is_datetime64_any_dtype(series):
        result["checked"] = False
        return result

    text = _clean_text_series(series)
    non_null_mask = text.notna()
    result["non_null_count"] = int(non_null_mask.sum())
    if result["non_null_count"] == 0:
        return result

    lengths = text[non_null_mask].str.len()
    if len(lengths) == 0:
        return result

    result["max_observed_length"] = int(lengths.max())
    result["too_long_count"] = int((lengths > int(max_length)).sum())
    return result


def _validate_enum_field(series, allowed_values):
    result = {
        "checked": True,
        "non_null_count": 0,
        "invalid_value_count": 0,
        "invalid_examples": [],
    }

    if not allowed_values:
        result["checked"] = False
        return result

    text = _clean_text_series(series)
    normalized = text.str.upper()
    non_null_mask = normalized.notna()
    result["non_null_count"] = int(non_null_mask.sum())
    if result["non_null_count"] == 0:
        return result

    allowed_set = {value.upper() for value in allowed_values}
    invalid_mask = non_null_mask & ~normalized.isin(list(allowed_set)).fillna(False)
    result["invalid_value_count"] = int(invalid_mask.sum())

    if result["invalid_value_count"] > 0:
        examples = sorted(
            normalized[invalid_mask].dropna().astype(str).unique().tolist()
        )
        result["invalid_examples"] = examples[:8]

    return result


def _find_id_column(df):
    for column in COMMON_ODI_COMPLAINT_ID_COLUMNS:
        if column in df.columns:
            return column
    if "record_id" in df.columns:
        return "record_id"
    return None


def _find_model_year_column(df):
    if "yeartxt" in df.columns:
        return "yeartxt"
    for column in MODEL_YEAR_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    return None


# -----------------------------------------------------------------------------
# Report builder
# -----------------------------------------------------------------------------
def collect_schema_report(df, dataset_name, schema_name=None):
    report = {
        "dataset_name": dataset_name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()) if len(df) <= 500000 else None,
        "id_column_found": None,
        "id_null_count": None,
        "id_duplicate_count": None,
        "model_year_column_found": None,
        "model_year_out_of_range_count": None,
        "schema_name": None,
        "schema_doc_path": None,
        "schema_match_score": None,
        "expected_columns": None,
        "present_expected_columns": None,
        "missing_columns": [],
        "missing_optional_columns": [],
        "extra_columns": [],
        "unexpected_extra_columns": [],
        "column_order_matches_expected": None,
        "field_checks": {},
        "field_issue_counts": {
            "date_invalid_total": 0,
            "date_zero_placeholder_total": 0,
            "numeric_non_numeric_total": 0,
            "numeric_non_integer_total": 0,
            "numeric_digits_overflow_total": 0,
            "char_length_overflow_total": 0,
            "enum_invalid_total": 0,
        },
        "warnings": [],
        "errors": [],
    }

    catalog_info = _schema_catalog()
    catalog = catalog_info["schemas"]
    if catalog_info["errors"]:
        for failed_schema_name, message in catalog_info["errors"].items():
            report["warnings"].append(
                f"Could not parse schema doc for {failed_schema_name}: {message}"
            )

    if schema_name is None:
        detected_schema_name, match_score, overlap_count = _detect_schema_name(
            df.columns.tolist()
        )
    else:
        detected_schema_name = schema_name
        if schema_name in catalog:
            expected_set = set(catalog[schema_name]["expected_columns"])
            overlap_count = len(expected_set & set(df.columns))
            match_score = overlap_count / max(len(expected_set), 1)
        else:
            overlap_count = 0
            match_score = 0.0

    report["schema_name"] = detected_schema_name
    report["schema_match_score"] = match_score

    schema_spec = catalog.get(detected_schema_name) if detected_schema_name else None
    if schema_spec:
        report["schema_doc_path"] = schema_spec["doc_path"]

        expected_columns = schema_spec["expected_columns"]
        expected_set = set(expected_columns)
        optional_set = set(schema_spec["optional_columns"])  # noqa: F841

        present_expected = [
            column for column in expected_columns if column in df.columns
        ]
        missing_required = [
            column
            for column in schema_spec["required_columns"]
            if column not in df.columns
        ]
        missing_optional = [
            column
            for column in schema_spec["optional_columns"]
            if column not in df.columns
        ]
        extra_columns = [column for column in df.columns if column not in expected_set]
        unexpected_extra_columns = [
            column
            for column in extra_columns
            if column not in ALLOWED_PIPELINE_EXTRA_COLUMNS
        ]

        actual_expected_order = [
            column for column in df.columns if column in expected_set
        ]
        report["expected_columns"] = len(expected_columns)
        report["present_expected_columns"] = len(present_expected)
        report["missing_columns"] = missing_required
        report["missing_optional_columns"] = missing_optional
        report["extra_columns"] = extra_columns
        report["unexpected_extra_columns"] = unexpected_extra_columns
        report["column_order_matches_expected"] = (
            actual_expected_order == present_expected
        )

        if missing_required:
            report["errors"].append(
                f"Missing required columns for {detected_schema_name}: {_preview_list(missing_required)}"
            )

        if unexpected_extra_columns:
            report["warnings"].append(
                f"Unexpected extra columns: {_preview_list(unexpected_extra_columns)}"
            )

        if report["column_order_matches_expected"] is False:
            report["warnings"].append(
                "Expected columns are present but column order differs from schema doc"
            )

        field_map = schema_spec["field_map"]
        schema_length_overrides = SCHEMA_LENGTH_OVERRIDES.get(detected_schema_name, {})

        for column_name in present_expected:
            field = field_map[column_name]
            column_report = {"type": field["type"], "size": field["size"]}
            series = df[column_name]

            if field["type"] == "NUMBER":
                numeric_report = _validate_numeric_field(series, field["size"])
                column_report["numeric"] = numeric_report
                report["field_issue_counts"]["numeric_non_numeric_total"] += (
                    numeric_report["non_numeric_count"]
                )
                report["field_issue_counts"]["numeric_non_integer_total"] += (
                    numeric_report["non_integer_count"]
                )
                report["field_issue_counts"]["numeric_digits_overflow_total"] += (
                    numeric_report["digits_overflow_count"]
                )

                if numeric_report["non_numeric_count"] > 0:
                    report["errors"].append(
                        f"{column_name}: {numeric_report['non_numeric_count']:,} non-numeric values"
                    )
                if numeric_report["non_integer_count"] > 0:
                    report["warnings"].append(
                        f"{column_name}: {numeric_report['non_integer_count']:,} non-integer numeric values"
                    )
                if numeric_report["digits_overflow_count"] > 0:
                    report["warnings"].append(
                        f"{column_name}: {numeric_report['digits_overflow_count']:,} values exceed NUMBER({field['size']}) digit length"
                    )

            if field.get("is_date"):
                date_report = _validate_date_field(series)
                column_report["date"] = date_report
                report["field_issue_counts"]["date_invalid_total"] += date_report[
                    "invalid_date_count"
                ]
                report["field_issue_counts"]["date_zero_placeholder_total"] += (
                    date_report["placeholder_zero_count"]
                )

                if date_report["invalid_date_count"] > 0:
                    report["errors"].append(
                        f"{column_name}: {date_report['invalid_date_count']:,} invalid YYYYMMDD date values"
                    )
                if date_report["placeholder_zero_count"] > 0:
                    report["warnings"].append(
                        f"{column_name}: {date_report['placeholder_zero_count']:,} placeholder zero date values"
                    )

            if field["type"] == "CHAR":
                allowed_max_length = schema_length_overrides.get(
                    column_name, field["size"]
                )
                char_report = _validate_char_length(
                    series,
                    allowed_max_length,
                    allow_datetime=field.get("is_date", False),
                )
                column_report["char_length"] = char_report
                report["field_issue_counts"]["char_length_overflow_total"] += (
                    char_report["too_long_count"]
                )

                if char_report["too_long_count"] > 0:
                    if allowed_max_length != field["size"]:
                        report["warnings"].append(
                            f"{column_name}: {char_report['too_long_count']:,} values exceed doc CHAR({field['size']}) length (allowed legacy max {allowed_max_length})"
                        )
                    else:
                        report["errors"].append(
                            f"{column_name}: {char_report['too_long_count']:,} values exceed CHAR({allowed_max_length})"
                        )

                allowed_values = _allowed_values_for_field(detected_schema_name, field)
                enum_report = _validate_enum_field(series, allowed_values)
                column_report["enum"] = enum_report
                report["field_issue_counts"]["enum_invalid_total"] += enum_report[
                    "invalid_value_count"
                ]

                if enum_report["invalid_value_count"] > 0:
                    report["warnings"].append(
                        f"{column_name}: {enum_report['invalid_value_count']:,} values outside documented codes"
                    )

            report["field_checks"][column_name] = column_report

    else:
        if detected_schema_name and detected_schema_name not in catalog:
            report["warnings"].append(
                f"Schema '{detected_schema_name}' requested but not available"
            )
        elif detected_schema_name is None:
            report["warnings"].append(
                "Could not confidently match dataset columns to complaints or recalls schema"
            )

    id_column = _find_id_column(df)
    if id_column:
        report["id_column_found"] = id_column
        report["id_null_count"] = int(df[id_column].isna().sum())
        if len(df) <= 1000000:
            report["id_duplicate_count"] = int(df[id_column].duplicated().sum())

    model_year_column = _find_model_year_column(df)
    if model_year_column:
        report["model_year_column_found"] = model_year_column
        values = pd.to_numeric(df[model_year_column], errors="coerce")
        invalid = values.notna() & ~(values.eq(9999) | values.between(1900, 2100))
        report["model_year_out_of_range_count"] = int(invalid.sum())
        if report["model_year_out_of_range_count"] > 0:
            report["warnings"].append(
                f"{model_year_column}: {report['model_year_out_of_range_count']:,} values outside expected range (1900-2100 or 9999)"
            )

    return report


def print_schema_report(report):
    print("")
    print(f"[schema] {report['dataset_name']}")

    if report.get("schema_name"):
        schema_label = report["schema_name"]
        score = report.get("schema_match_score")
        if score is None:
            print(f"  schema: {schema_label}")
        else:
            print(f"  schema: {schema_label} (match={score:.2f})")
    else:
        print("  schema: unknown")

    print(f"  rows: {report['rows']:,}")
    print(f"  columns: {report['columns']:,}")

    expected_columns = report.get("expected_columns")
    if expected_columns is not None:
        present_expected = report.get("present_expected_columns", 0)
        print(f"  expected_columns: {present_expected:,}/{expected_columns:,}")

        missing_columns = report.get("missing_columns", [])
        missing_optional = report.get("missing_optional_columns", [])
        if missing_columns:
            print(
                f"  missing_required: {len(missing_columns):,} {_preview_list(missing_columns)}"
            )
        else:
            print("  missing_required: 0")

        if missing_optional:
            print(
                f"  missing_optional: {len(missing_optional):,} {_preview_list(missing_optional)}"
            )

        unexpected_extra_columns = report.get("unexpected_extra_columns", [])
        if unexpected_extra_columns:
            print(
                f"  unexpected_extra: {len(unexpected_extra_columns):,} {_preview_list(unexpected_extra_columns)}"
            )

        if report.get("column_order_matches_expected") is not None:
            print(
                f"  column_order_matches_expected: {report['column_order_matches_expected']}"
            )

    if report["duplicate_rows"] is None:
        print("  duplicate_rows: skipped (dataset too large for quick check)")
    else:
        print(f"  duplicate_rows: {report['duplicate_rows']:,}")

    if report["id_column_found"]:
        id_line = f"  id_column: {report['id_column_found']} (nulls={report['id_null_count']:,}"
        if report["id_duplicate_count"] is not None:
            id_line += f", duplicates={report['id_duplicate_count']:,}"
        id_line += ")"
        print(id_line)
    else:
        print("  id_column: not found")

    if report["model_year_out_of_range_count"] is None:
        print("  model_year_range_check: skipped (no model year column found)")
    else:
        column_name = report.get("model_year_column_found") or "model_year"
        print(
            f"  {column_name}_out_of_range_count: {report['model_year_out_of_range_count']:,}"
        )

    issue_counts = report.get("field_issue_counts", {})
    if issue_counts:
        print(
            "  field_issue_totals: "
            f"date_invalid={issue_counts.get('date_invalid_total', 0):,}, "
            f"date_zero={issue_counts.get('date_zero_placeholder_total', 0):,}, "
            f"non_numeric={issue_counts.get('numeric_non_numeric_total', 0):,}, "
            f"length_overflow={issue_counts.get('char_length_overflow_total', 0):,}, "
            f"enum_invalid={issue_counts.get('enum_invalid_total', 0):,}"
        )

    print(f"  warnings: {len(report.get('warnings', [])):,}")
    print(f"  errors: {len(report.get('errors', [])):,}")

    if report.get("errors"):
        print("  error_examples:")
        for message in report["errors"][:5]:
            print(f"    - {message}")
        if len(report["errors"]) > 5:
            print(f"    - ... +{len(report['errors']) - 5} more")

    if report.get("warnings"):
        print("  warning_examples:")
        for message in report["warnings"][:5]:
            print(f"    - {message}")
        if len(report["warnings"]) > 5:
            print(f"    - ... +{len(report['warnings']) - 5} more")
