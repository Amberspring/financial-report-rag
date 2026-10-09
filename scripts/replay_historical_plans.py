"""Replay saved model plans through today's calculator, without model calls."""

import argparse
from collections import Counter
import json
from pathlib import Path
from finrag.calculator import execute, CalculationError


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = ROOT / "results/finqa-model-20261008.predictions.jsonl"
    labels = {row["id"]: row for row in json.loads((ROOT / "data/finqa-test-labels.json").read_text(encoding="utf-8"))}
    rows = []
    for line in predictions.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        old = item["prediction"]
        if old.get("reason") != "Mixed cell units":
            continue
        trace = old["model_trace"]
        plan = json.loads(trace["response"]["choices"][0]["message"]["content"])
        try:
            new = execute(trace["request"]["retrieved_tables"], plan)
            outcome = "now_executes"
        except (CalculationError, ValueError, KeyError, TypeError) as error:
            new = None
            outcome = "still_refused: " + str(error)
        rows.append({"id": item["id"], "historical_reason": old["reason"], "replay_outcome": outcome,
                     "new_value": new["value"] if new else None, "new_unit": new["unit"] if new else None,
                     "gold_exe_ans_posthoc_only": labels[item["id"]]["exe_ans"]})
    report = {"scope": "Saved-plan CPU replay; not a new model run or accuracy result",
              "counts": dict(Counter(row["replay_outcome"] for row in rows)), "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
