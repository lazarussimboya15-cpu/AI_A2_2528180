"""Single entry point: python run_all.py --student-id ug2528180@ines.ac.rw --seed 8180"""
import argparse
import sys

from src.pipeline import run_pipeline
from src.validation import ValidationError


def main() -> int:
    p = argparse.ArgumentParser(description="Recreate data, train models, write artifacts.")
    p.add_argument("--student-id", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--report", action="store_true", help="also build the PDF report")
    a = p.parse_args()
    try:
        run_pipeline(a.student_id, a.seed)
        if a.report:
            from src.report import build_report
            print("Report:", build_report())
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
