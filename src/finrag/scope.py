"""Resolve a question's company and report year without using benchmark labels."""

import re


def report_ids(docs, question):
    companies = {d["company"] for d in docs if re.search(
        r"(?<![A-Za-z0-9])" + re.escape(d["company"]) + r"(?![A-Za-z0-9])", question, re.I)}
    if len(companies) != 1:
        return set()
    company = next(iter(companies))
    company_docs = [d for d in docs if d["company"] == company]
    years = set(re.findall(r"report\s+year\s*[:=]?\s*((?:19|20)\d{2})", question, re.I))
    if not years:
        mentioned = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", question))
        years = mentioned & {str(d["report_year"]) for d in company_docs}
    if len(years) != 1:
        return set()
    year = next(iter(years))
    return {d["doc_id"] for d in company_docs if str(d["report_year"]) == year}
