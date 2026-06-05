from src.generator import _parse_evolution_response

def test_parse_evolution_response_valid():
    resp = "HELPFULNESS_SCORE: 8\nCOMPREHENSIVENESS_SCORE: 7\nREVISED_TEXT: Improved text"
    revised, h, c = _parse_evolution_response(resp, "fallback")
    assert revised == "Improved text"
    assert h == 8
    assert c == 7

def test_parse_evolution_response_missing_fields():
    resp = "Some random output without markers."
    revised, h, c = _parse_evolution_response(resp, "fallback")
    assert revised == "fallback"
    assert h == 5
    assert c == 5
