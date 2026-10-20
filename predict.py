"""CLI prediction: python predict.py --json samples/valid.json"""
import argparse
import json
import sys

from src.inference import InputValidationError, predict_record


def main() -> int:
    p = argparse.ArgumentParser(description="Predict for one farm record (JSON file).")
    p.add_argument("--json", required=True, help="path to a JSON file with one record")
    a = p.parse_args()
    try:
        with open(a.json, encoding="utf-8") as fh:
            record = json.load(fh)
        print(json.dumps(predict_record(record), indent=2))
        return 0
    except InputValidationError as exc:
        print(json.dumps({"error": "invalid input", "details": exc.errors}, indent=2),
              file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
