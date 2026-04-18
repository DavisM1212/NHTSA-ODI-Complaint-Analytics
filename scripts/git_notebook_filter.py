import argparse
import json
import sys


# -----------------------------------------------------------------------------
# Filter actions
# -----------------------------------------------------------------------------
def clean_notebook(stdin_text):
    notebook = json.loads(stdin_text)

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell["execution_count"] = None
        cell["outputs"] = []

    return json.dumps(notebook, indent=2) + "\n"


def smudge_notebook(stdin_text):
    return stdin_text


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Strip outputs from designated exploration notebooks during git clean"
    )
    parser.add_argument(
        "--mode",
        choices=["clean", "smudge"],
        required=True,
        help="Filter mode requested by git"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stdin_text = sys.stdin.read()

    if args.mode == "clean":
        sys.stdout.write(clean_notebook(stdin_text))
        return 0

    sys.stdout.write(smudge_notebook(stdin_text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
