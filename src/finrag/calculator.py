"""Decimal calculator with evidence-bound operands; no eval or gold-program access."""

from decimal import Decimal, InvalidOperation
import re


class CalculationError(ValueError):
    pass


def number(text):
    currencies = {symbol for symbol in ("$", "€", "£") if symbol in str(text)}
    if len(currencies) > 1:
        raise CalculationError("Ambiguous currency")
    s = (
        str(text)
        .strip()
        .replace(",", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace(" ", "")
    )
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    unit = (
        "percent"
        if s.endswith("%")
        else (
            {"$": "currency:$", "€": "currency:€", "£": "currency:£"}[next(iter(currencies))]
            if currencies
            else "reported_units"
        )
    )
    s = s.removesuffix("%")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        raise CalculationError("Cell is not a single number")
    try:
        value = Decimal(s)
    except InvalidOperation as e:
        raise CalculationError("Invalid decimal") from e
    return value, unit


def cell_unit(table, row, col):
    value, unit = number(table[row][col])
    if unit != "reported_units":
        return value, unit, "cell"
    # Financial tables often print a currency symbol only in the first year of a row.
    if col >= len(table[0]) or not re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", str(table[0][col])):
        return value, unit, "cell"
    row_units = set()
    for column, cell in enumerate(table[row][1:], 1):
        try:
            _, candidate = number(cell)
        except CalculationError:
            continue
        if candidate != "reported_units":
            if column >= len(table[0]) or not re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", str(table[0][column])):
                return value, unit, "cell"
            row_units.add(candidate)
    if len(row_units) == 1 and next(iter(row_units)).startswith("currency:"):
        return value, next(iter(row_units)), "same_row"
    return value, unit, "cell"


def text_number(paragraph, quote):
    """Accept one verbatim numeric token, including its currency or percent marker."""
    if not isinstance(quote, str):
        raise CalculationError("Text quote must be a string")
    matches = re.finditer(
        r"(?<![\w.($€£])(?:[$€£]\s*)?-?\d[\d,]*(?:\.\d+)?\s*%?(?![\w.)%])",
        paragraph,
    )
    tokens = [m.group().strip() for m in matches
              if not re.search(r"[($€£-]\s*$", paragraph[:m.start()])
              and not re.match(r"\s*\)", paragraph[m.end():])]
    selected = [token for token in tokens if token == quote.strip()]
    if len(selected) != 1:
        raise CalculationError("Text quote must match one unique numeric token")
    value, unit = number(selected[0])
    # shortcut: prose scale is rejected until its PDF context and unit conversion are verified.
    if re.search(r"\b(?:thousands?|millions?|billions?|trillions?)\b", paragraph, re.I):
        raise CalculationError("Prose scale is not yet supported")
    return value, unit


def execute(docs, plan):
    """Resolve table cells or verbatim text tokens; constants and unseen values are forbidden."""
    if not isinstance(plan, dict):
        raise CalculationError("Plan must be an object")
    lookup = {d["doc_id"]: d for d in docs}
    refs = plan.get("operands", [])
    if not isinstance(refs, list) or not 1 <= len(refs) <= 12:
        raise CalculationError("Require 1..12 evidence operands")
    values = []
    units = []
    trace = []
    for ref in refs:
        if not isinstance(ref, dict) or not isinstance(ref.get("doc_id"), str):
            raise CalculationError("Operand must have a string document ID")
        d = lookup.get(ref.get("doc_id"))
        if not d:
            raise CalculationError("Unknown document")
        if "text_index" in ref:
            if set(ref) != {"doc_id", "text_index", "quote"}:
                raise CalculationError("Unexpected text operand fields")
            index = ref["text_index"]
            paragraphs = d["pre_text"] + d["post_text"]
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(paragraphs):
                raise CalculationError("Text index out of range")
            paragraph = paragraphs[index]
            val, unit = text_number(paragraph, ref["quote"])
            source = {"text_index": index, "quote": paragraph, "selected_number": ref["quote"],
                      "unit_source": "text_token"}
        else:
            if ref.get("expected_label") is not None and not isinstance(ref["expected_label"], str):
                raise CalculationError("Expected label must be a string")
            if set(ref) - {"doc_id", "row", "column", "expected_label", "expected_year"}:
                raise CalculationError("Unknown operand fields")
            row, col = ref.get("row"), ref.get("column")
            if (not isinstance(row, int) or isinstance(row, bool) or not isinstance(col, int)
                    or isinstance(col, bool) or row <= 0 or col <= 0):
                raise CalculationError("Cell indices must be positive integers")
            table = d["table"]
            if row >= len(table) or col >= len(table[row]):
                raise CalculationError("Cell out of range")
            label = str(table[row][0])
            header = str(table[0][col]) if col < len(table[0]) else ""
            if ref.get("expected_label") and ref["expected_label"].casefold() not in label.casefold():
                raise CalculationError("Metric label mismatch")
            if ref.get("expected_year") and str(ref["expected_year"]) not in header + " " + label:
                raise CalculationError("Period mismatch")
            val, unit, unit_source = cell_unit(table, row, col)
            source = {"row": row, "column": col, "row_label": label, "column_header": header,
                      "quote": str(table[row][col]), "unit_source": unit_source}
        values.append(val)
        units.append(unit)
        trace.append(
            {
                "doc_id": d["doc_id"],
                "company": d["company"],
                "report_year": d["report_year"],
                "page": d["page"],
                **source,
                "value": str(val),
                "unit": unit,
            }
        )
    if len({t["company"] for t in trace}) > 1 and not plan.get(
        "allow_cross_company", False
    ):
        raise CalculationError("Cross-company arithmetic needs explicit authorization")
    if len({t["doc_id"] for t in trace}) > 1:
        raise CalculationError(
            "Cross-document scale and currency conversion are not supported"
        )
    if len(set(units)) > 1:
        raise CalculationError("Mixed cell units")
    op = plan.get("operation")
    if op in ("subtract", "divide", "growth", "percentage") and len(values) != 2:
        raise CalculationError("Binary operation needs two operands")
    if op == "sum":
        result = sum(values)
    elif op == "average":
        result = sum(values) / len(values)
    elif op == "subtract":
        result = values[0] - values[1]
    elif op in ("divide", "percentage", "growth"):
        if values[1] == 0:
            raise CalculationError("Zero denominator")
        result = (
            (values[0] - values[1]) / values[1] * 100
            if op == "growth"
            else values[0] / values[1] * (100 if op == "percentage" else 1)
        )
    else:
        raise CalculationError("Operation not allowed")
    unit = (
        "percent"
        if op in ("percentage", "growth")
        else "ratio"
        if op == "divide"
        else units[0]
    )
    return {
        "value": str(result),
        "unit": unit,
        "operation": op,
        "operands": trace,
        "source_scale": "As reported by source; no implicit scale conversion",
        "mode": "deterministic_calculation",
        "refused": False,
        "citations": [
            {
                "id": i + 1,
                "doc_id": t["doc_id"],
                "page": t["page"],
                "quote": t["quote"],
                **({"text_index": t["text_index"], "selected_number": t["selected_number"]}
                   if "text_index" in t else {"row": t["row"], "column": t["column"]}),
            }
            for i, t in enumerate(trace)
        ],
    }


def propose(question, doc):
    """Deliberately narrow baseline: two years + uniquely matched metric in a year-column table."""
    q = question.lower()
    years = re.findall(r"\b(?:19|20)\d{2}\b", q)
    if any(
        term in q
        for term in ("basis point", "percentage point", "compound", "annualized")
    ):
        raise CalculationError("Unsupported rate convention")
    # ponytail: narrow rule planner, not general numerical QA; use explicit cell plans for other questions.
    table = doc["table"]
    if len(years) != 2 or not table:
        raise CalculationError("Planner supports exactly two explicit years")
    cols = []
    for year in years:
        found = [i for i, h in enumerate(table[0]) if i > 0 and year in str(h)]
        if len(found) != 1:
            raise CalculationError("Year column is ambiguous or missing")
        cols.append(found[0])
    stop = {
        "what",
        "was",
        "the",
        "in",
        "of",
        "for",
        "from",
        "to",
        "and",
        "by",
        "is",
        "a",
        "an",
        "company",
        "percentage",
        "percent",
        "change",
        "growth",
        "increase",
        "decrease",
        "between",
        "rate",
        "net",
    }
    words = set(re.findall(r"[a-z]+", q)) - stop
    candidates = []
    for i, row in enumerate(table[1:], 1):
        tokens = set(re.findall(r"[a-z]+", str(row[0]).lower())) - stop
        if tokens and tokens <= words:
            candidates.append((len(tokens), i))
    if not candidates:
        raise CalculationError("No unambiguous metric row")
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise CalculationError("Ambiguous metric")
    row = candidates[0][1]
    if any(w in q for w in ("percentage", "percent", "growth", "rate")):
        op = "growth"
    elif any(w in q for w in ("change", "increase", "decrease", "difference")):
        op = "subtract"
    else:
        raise CalculationError("Unsupported question operation")
    # Most change questions request later minus earlier; preserve deterministic chronological order.
    order = sorted(zip(years, cols), reverse=True)
    return {
        "operation": op,
        "operands": [
            {
                "doc_id": doc["doc_id"],
                "row": row,
                "column": col,
                "expected_year": year,
                "expected_label": str(table[row][0]),
            }
            for year, col in order
        ],
    }
