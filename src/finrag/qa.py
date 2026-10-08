"""Global retrieval -> narrow evidence-bound calculation, without a supplied gold document."""
from .calculator import propose, execute, CalculationError
import re


def numerical_answer(retriever, docs, question, mode="hybrid", rerank=False):
    hits = retriever.search(question, 10, mode, rerank)
    by_id = {d["doc_id"]: d for d in docs}
    # ponytail: lexical gate and narrow planner; use a separately evaluated model planner for broader FinQA coverage.
    candidates = []
    reasons = []
    for doc_id in dict.fromkeys(h["chunk"]["doc_id"] for h in hits):
        doc = by_id.get(doc_id)
        if doc is None:
            continue
        company = doc["company"].casefold()
        if company not in re.findall(r"[a-z0-9]+", question.casefold()):
            reasons.append("An explicit matching company symbol is required")
            continue
        try:
            value = execute(docs, propose(question, doc))
            # Every selected cell must belong to a retrieved table, never fetch unseen operands.
            tables = [h for h in hits if h["chunk"]["doc_id"] == doc_id and h["chunk"]["kind"] == "table"]
            if not tables or not all(any("table_" + str(o["row"]) in h["chunk"]["evidence_keys"] for h in tables) for o in value["operands"]):
                raise CalculationError("Required table evidence was not retrieved")
            candidates.append(value)
        except CalculationError as e:
            reasons.append(str(e))
    if len(candidates) != 1:
        return {"answer": "Unsupported or ambiguous evidence: require one retrieved, company-matched table.",
                "refused": True, "citations": [], "hits": hits, "reasons": reasons}
    value = candidates[0]
    return {**value, "answer": value["value"] + " " + value["unit"], "hits": hits}
