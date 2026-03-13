"""Small repo guardrails for raw data hashes and notebook hygiene"""

import fnmatch
import hashlib
import json
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
HASH_FILE = RAW_DIR / "SHA256SUMS.txt"
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
GITATTRIBUTES_FILE = REPO_ROOT / ".gitattributes"


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def print_status(level, message):
    print(f"[{level}] {message}")


def print_section(title):
    print("")
    print(f"=== {title} ===")


# -----------------------------------------------------------------------------
# Raw data checks
# -----------------------------------------------------------------------------
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def check_raw_hashes(results):
    if not HASH_FILE.exists():
        results["failures"].append("Missing raw hash manifest: data/raw/SHA256SUMS.txt")
        return

    manifest_paths = set()
    lines = HASH_FILE.read_text(encoding="utf-8").splitlines()

    # If a raw zip changes, we want it to be loud and boringly obvious
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(None, 1)
        if len(parts) != 2:
            results["failures"].append(
                f"Malformed hash manifest line {line_no} in data/raw/SHA256SUMS.txt"
            )
            continue

        expected_hash = parts[0].upper()
        rel_path = parts[1].strip().replace("\\", "/")
        file_path = REPO_ROOT / rel_path
        manifest_paths.add(rel_path)

        if not file_path.exists():
            results["failures"].append(f"Missing raw file listed in manifest: {rel_path}")
            continue

        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            results["failures"].append(f"Hash mismatch for {rel_path}")
            continue

        results["passes"].append(f"Raw hash OK: {rel_path}")

    raw_zip_paths = {
        path.relative_to(REPO_ROOT).as_posix() for path in RAW_DIR.glob("*.zip")
    }
    extra_paths = sorted(raw_zip_paths - manifest_paths)
    if extra_paths:
        results["failures"].append(
            "Raw zip files missing from manifest: " + ", ".join(extra_paths)
        )


# -----------------------------------------------------------------------------
# Notebook checks
# -----------------------------------------------------------------------------
def load_notebookstrip_patterns():
    if not GITATTRIBUTES_FILE.exists():
        return []

    patterns = []
    for raw_line in GITATTRIBUTES_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        if "filter=notebookstrip" not in parts[1:]:
            continue

        patterns.append(parts[0].replace("\\", "/"))

    return patterns


def notebook_should_be_cleared(rel_path, patterns):
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns)


def check_notebooks(results):
    patterns = load_notebookstrip_patterns()

    if not NOTEBOOK_DIR.exists():
        results["warnings"].append("Notebook folder not found, skipping notebook checks")
        return

    notebook_paths = sorted(NOTEBOOK_DIR.rglob("*.ipynb"))
    if not notebook_paths:
        results["passes"].append("No notebooks found under notebooks/")
        return

    if not patterns:
        results["warnings"].append(
            "No notebookstrip patterns found in .gitattributes, skipping notebook clear checks"
        )
        return

    # Only designated exploration notebooks need to travel light
    for path in notebook_paths:
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        if not notebook_should_be_cleared(rel_path, patterns):
            results["passes"].append(
                f"Notebook can keep outputs by default: {rel_path}"
            )
            continue

        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            results["failures"].append(f"Notebook JSON read failed for {rel_path} ({exc})")
            continue

        output_cells = 0
        executed_cells = 0

        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs"):
                output_cells += 1
            if cell.get("execution_count") is not None:
                executed_cells += 1

        if output_cells or executed_cells:
            results["failures"].append(
                f"Exploration notebook must be cleared: {rel_path} "
                f"(output_cells={output_cells}, executed_cells={executed_cells})"
            )
            continue

        results["passes"].append(f"Exploration notebook cleared: {rel_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    results = {"passes": [], "warnings": [], "failures": []}

    print("NHTSA ODI Complaint Analytics - Repo Integrity Check")
    print(f"Repo root: {REPO_ROOT}")

    print_section("Raw Data")
    check_raw_hashes(results)

    print_section("Notebooks")
    check_notebooks(results)

    print_section("Summary")
    print_status("PASS", str(len(results["passes"])))
    print_status("WARN", str(len(results["warnings"])))
    print_status("FAIL", str(len(results["failures"])))

    if results["warnings"]:
        print("")
        print("Warnings")
        for message in results["warnings"]:
            print_status("WARN", message)

    if results["failures"]:
        print("")
        print("Failures")
        for message in results["failures"]:
            print_status("FAIL", message)
        return 1

    print("")
    print("Repo integrity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
