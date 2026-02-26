import importlib
import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TARGET_PYTHON_VERSION = "3.13.12"
REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRS = [
    "docs",
    "docs/screenshots",
    "data",
    "data/raw",
    "data/extracted",
    "data/processed",
    "data/outputs",
    "src",
    "scripts",
]

REQUIRED_IMPORTS = ["pandas", "numpy", "sklearn", "matplotlib", "pyarrow"]

OPTIONAL_IMPORTS = ["seaborn"]

COMPLAINT_ZIP_NAME_CANDIDATES = [
    "COMPLAINTS_RECEIVED_2020-2024.zip",
    "COMPLAINTS_RECEIVED_2025-2026.zip",
]


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------
def print_status(level, message):
    print(f"[{level}] {message}")


def print_section(title):
    print("")
    print(f"=== {title} ===")


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------
def check_python_version(results):
    version = ".".join(str(part) for part in sys.version_info[:3])
    print_status("INFO", f"Python executable: {sys.executable}")
    print_status("INFO", f"Python version: {version}")

    if version == TARGET_PYTHON_VERSION:
        results["passes"].append(
            f"Python version matches target ({TARGET_PYTHON_VERSION})"
        )
        return

    if version.startswith("3.13."):
        results["warnings"].append(
            f"Python patch version is {version}; recommended version is {TARGET_PYTHON_VERSION}"
        )
        return

    results["warnings"].append(
        f"Python major/minor version is {version}; recommended version is {TARGET_PYTHON_VERSION}"
    )


def check_imports(results):
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            results["passes"].append(f"Import OK: {module_name}")
        except Exception as exc:
            results["failures"].append(
                f"Missing or broken import: {module_name} ({exc})"
            )

    for module_name in OPTIONAL_IMPORTS:
        try:
            importlib.import_module(module_name)
            results["passes"].append(f"Optional import OK: {module_name}")
        except Exception:
            results["warnings"].append(
                f"Optional package not installed: {module_name} (plots will still work without it)"
            )


def check_directories(results):
    for relative_path in REQUIRED_DIRS:
        path = REPO_ROOT / relative_path
        if path.exists() and path.is_dir():
            results["passes"].append(f"Directory exists: {relative_path}")
        else:
            results["failures"].append(f"Missing directory: {relative_path}")


def check_raw_data(results):
    raw_dir = REPO_ROOT / "data" / "raw"
    if not raw_dir.exists():
        results["failures"].append("data/raw does not exist")
        return

    zip_files = sorted(path.name for path in raw_dir.glob("*.zip"))
    if not zip_files:
        results["failures"].append("No zip files found in data/raw")
        results["actions"].append(
            "Place NHTSA ODI complaint zip files in data/raw (for example COMPLAINTS_RECEIVED_2020-2024.zip)"
        )
        return

    complaint_zip_matches = [name for name in zip_files if "complaint" in name.lower()]
    if complaint_zip_matches:
        results["passes"].append(
            f"Complaint zip(s) found: {', '.join(complaint_zip_matches)}"
        )
    else:
        results["warnings"].append(
            "Zip files exist in data/raw, but no complaint zip names were detected"
        )

    missing_named_candidates = [
        name for name in COMPLAINT_ZIP_NAME_CANDIDATES if name not in zip_files
    ]
    if len(missing_named_candidates) == len(COMPLAINT_ZIP_NAME_CANDIDATES):
        results["warnings"].append(
            "Complaint zip filenames differ from README examples (this is okay if names still contain 'complaint')"
        )


def check_output_write_access(results):
    output_dir = REPO_ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_file = output_dir / ".write_test.tmp"

    try:
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        results["passes"].append("Write access OK: data/outputs")
    except Exception as exc:
        results["failures"].append(f"Cannot write to data/outputs ({exc})")
        results["actions"].append(
            "Close files using data/outputs and verify folder permissions"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    os.chdir(REPO_ROOT)
    results = {"passes": [], "warnings": [], "failures": [], "actions": []}

    print("NHTSA ODI Complaint Analytics - Install Verification")
    print(f"Repo root: {REPO_ROOT}")

    print_section("Python")
    check_python_version(results)

    print_section("Imports")
    check_imports(results)

    print_section("Folders")
    check_directories(results)

    print_section("Data")
    check_raw_data(results)
    check_output_write_access(results)

    print_section("Summary")
    print_status("PASS", str(len(results["passes"])))
    print_status("WARN", str(len(results["warnings"])))
    print_status("FAIL", str(len(results["failures"])))

    if results["warnings"]:
        print("")
        print("Warnings")
        for message in results["warnings"]:
            print(f"  - {message}")

    if results["failures"]:
        print("")
        print("Failures")
        for message in results["failures"]:
            print(f"  - {message}")

    if results["actions"] or results["failures"]:
        print("")
        print("Actionable next steps")
        if results["actions"]:
            for message in results["actions"]:
                print(f"  - {message}")
        if results["failures"]:
            print("  - Re-run the setup script after fixing the issues")

    if results["failures"]:
        return 1

    print("")
    print("Install verification completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
