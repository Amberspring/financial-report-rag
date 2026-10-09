from finrag.scoring import execution_match


def test_finqa_percent_execution_uses_fraction_units():
    assert execution_match({"refused": False, "value": "-21.0", "unit": "percent"}, {"exe_ans": -0.21029}) == (True, "percent_normalized")
    assert execution_match({"refused": False, "value": "-18.0", "unit": "percent"}, {"exe_ans": -0.21029})[0] is False
    assert execution_match({"refused": False, "value": "925031", "unit": "currency:$"}, {"exe_ans": 925031}) == (True, "raw_units")
    assert execution_match({"refused": False, "value": "1", "unit": "ratio"}, {"exe_ans": "yes"})[0] is False
