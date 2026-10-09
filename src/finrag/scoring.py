"""Score numerical execution values in FinQA's raw value convention."""

from decimal import Decimal, InvalidOperation


def execution_match(prediction, label):
    if prediction.get("refused") or "value" not in prediction:
        return False, "refused"
    try:
        value = Decimal(str(prediction["value"]))
        gold = Decimal(str(label["exe_ans"]))
    except (InvalidOperation, TypeError, KeyError):
        return False, "non_numeric_reference_or_prediction"
    if not value.is_finite() or not gold.is_finite():
        return False, "non_finite"
    if prediction.get("unit") == "percent":
        value /= 100
        tolerance = max(Decimal("0.0005"), abs(gold) * Decimal("0.001"))
    else:
        tolerance = max(Decimal("0.01"), abs(gold) * Decimal("0.001"))
    return abs(value - gold) <= tolerance, "percent_normalized" if prediction.get("unit") == "percent" else "raw_units"
