import pytest

from ine_ocr.mrz import MRZError, assemble, icao_check_digit, parse


def _synthetic_mrz() -> str:
    cic = "123456789"
    birth = "900101"
    expiry = "300101"
    return (
        f"IDMEX{cic}{icao_check_digit(cic)}<<"
        f"1234567890123{birth}{icao_check_digit(birth)}H"
        f"{expiry}{icao_check_digit(expiry)}MEX<01<<12345<0"
        "PRUEBA<CONTROL<<PERSONA<SINTETICA<<"
    )


def test_parse_synthetic_mrz_checks_all_supported_digits():
    result = parse(_synthetic_mrz())
    assert result["validacionMRZAllOk"]
    assert result["cic"] == "123456789"
    assert result["fechaNacimiento"] == "01/01/1990"
    assert result["vigencia"] == "2030"


@pytest.mark.parametrize("observed_prefix", ["1DMEX", "10MEX", "188EX"])
def test_parse_canonicalizes_bounded_ine_prefix_confusions(observed_prefix):
    value = observed_prefix + _synthetic_mrz()[5:]
    result = parse(value)
    assert result["cic"] == "123456789"
    assert result["validacionMRZAllOk"]


def test_assemble_canonicalizes_prefix_before_joining_rows():
    value = _synthetic_mrz()
    lines = [
        {"text": "1DMEX" + value[5:35], "box": [[0, 0], [1, 0]]},
        {"text": value[35:65], "box": [[0, 1], [1, 1]]},
        {"text": value[65:], "box": [[0, 2], [1, 2]]},
    ]
    assert assemble(lines) == value


def test_parse_rejects_unbounded_prefix_repair():
    with pytest.raises(MRZError, match="mrz_prefix_invalid"):
        parse("12345" + _synthetic_mrz()[5:])


def test_mrz_separator_glyph_normalization_still_requires_numeric_checks():
    value = _synthetic_mrz().replace("<", "≤")
    assert parse(assemble([{"text": value}]))["validacionMRZAllOk"]
    corrupted = value[:5] + "9" + value[6:]
    assert not parse(assemble([{"text": corrupted}]))["validacionMRZAllOk"]
