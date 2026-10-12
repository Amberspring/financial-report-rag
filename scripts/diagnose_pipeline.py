"""Post-hoc stage diagnosis of frozen FinQA predictions; never feeds labels to a predictor."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from finrag.scoring import execution_match


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(predictions, queries, labels):
    scope = {row["id"]: row["doc_id"] for row in queries}
    gold = {row["id"]: row for row in labels}
    if set(scope) != set(gold):
        raise ValueError("Query and label IDs differ")
    groups = defaultdict(dict)
    for row in predictions:
        case_id = row["id"]
        key = (row["mode"], row.get("rerank", False))
        if case_id not in scope or case_id in groups[key]:
            raise ValueError("Unknown or duplicate prediction ID")
        groups[key][case_id] = row["prediction"]
    results = []
    for (mode, rerank), cases in sorted(groups.items()):
        if set(cases) != set(scope):
            raise ValueError("Incomplete prediction mode")
        counts = defaultdict(int)
        for case_id, prediction in cases.items():
            target = scope[case_id]
            label = gold[case_id]
            hits = prediction.get("hits", [])
            target_hits = [hit for hit in hits[:5] if hit["chunk"]["doc_id"] == target]
            evidence = set().union(*(hit["chunk"]["evidence_keys"] for hit in target_hits))
            gold_keys = set(label["gold_inds"])
            selected = {"text_" + str(ref["text_index"]) if "text_index" in ref
                        else "table_" + str(ref["row"])
                        for ref in prediction.get("operands", []) if ref["doc_id"] == target}
            correct, _ = execution_match(prediction, label)
            answered = not prediction["refused"]
            counts["target_report_top1"] += bool(hits and hits[0]["chunk"]["doc_id"] == target)
            counts["target_report_top10"] += any(hit["chunk"]["doc_id"] == target for hit in hits[:10])
            counts["gold_evidence_all_top5"] += bool(gold_keys and gold_keys <= evidence)
            counts["gold_evidence_selected_proxy"] += bool(answered and gold_keys and gold_keys <= selected)
            counts["answered"] += answered
            counts["correct"] += correct
            counts["wrong_answer"] += bool(answered and not correct)
        n = len(cases)
        results.append({"mode": mode, "rerank": rerank, "n": n, "counts": dict(counts),
                        "report_top1_rate": counts["target_report_top1"] / n,
                        "report_top10_rate": counts["target_report_top10"] / n,
                        "gold_evidence_all_top5_rate": counts["gold_evidence_all_top5"] / n,
                        "selected_gold_evidence_proxy_rate": counts["gold_evidence_selected_proxy"] / n,
                        "answer_coverage": counts["answered"] / n,
                        "end_to_end_accuracy": counts["correct"] / n,
                        "accuracy_given_answer": counts["correct"] / counts["answered"] if counts["answered"] else None})
    return results


def main():
    parser = argparse.ArgumentParser()
    for name in ("predictions", "queries", "labels", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name)) for name in ("predictions", "queries", "labels")}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    raw = [json.loads(line) for line in paths["predictions"].open(encoding="utf-8")]
    queries = json.loads(paths["queries"].read_text(encoding="utf-8"))
    labels = json.loads(paths["labels"].read_text(encoding="utf-8"))
    report = {"scope": "post-hoc diagnosis of frozen global CPU rule predictions",
              "input_sha256": {name: sha(path) for name, path in paths.items()},
              "limitations": "Gold evidence keys proxy evidence selection, not operand values, unit correctness or independent PDF verification. No model quality gain is implied.",
              "modes": diagnose(raw, queries, labels)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["modes"]))


if __name__ == "__main__":
    main()
