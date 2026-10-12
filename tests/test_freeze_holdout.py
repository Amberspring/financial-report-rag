import importlib.util
from pathlib import Path

import pytest


def test_frozen_selection_excludes_prior_ids_and_entire_reports():
    path = Path(__file__).resolve().parents[1] / "scripts/freeze_holdout.py"
    spec = importlib.util.spec_from_file_location("freeze_holdout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"id": f"{company}/{year}/page_1.pdf-{number}",
             "filename": f"{company}/{year}/page_1.pdf", "qa": {"question": "q"}}
            for company, year, number in (("A", "2001", 1), ("A", "2001", 2),
                                          ("B", "2002", 1), ("C", "2003", 1))]
    eligible, chosen = module.select_rows(rows, {rows[2]["id"]}, {("A", "2001")}, 1)
    assert len(eligible) == 1 and chosen[0][0] == rows[3]["id"]
    with pytest.raises(ValueError, match="Fewer than 2"):
        module.select_rows(rows, {rows[2]["id"]}, {("A", "2001")}, 2)
