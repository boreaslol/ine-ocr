import pytest

from ine_ocr.curp import compute_check_digit
from ine_ocr.document_fields import extract_known_fields, reading_rotation


def _synthetic_curp(gender: str = "H") -> str:
    prefix = f"AECD000101{gender}DFBCD0"
    return prefix + compute_check_digit(prefix)


def test_extract_known_fields_keeps_only_check_valid_curp():
    valid = _synthetic_curp("X")
    fields = extract_known_fields([{"text": f"CURP {valid} GENERO NB"}])
    assert fields["curp"] == valid
    assert fields["sexo"] == ""
    assert fields["sexoRaw"] == "NB"
    assert fields["genero"]["sexoRegistralCurp"] == "X"


def test_extract_known_fields_drops_invalid_check_digit():
    valid = _synthetic_curp()
    invalid = valid[:-1] + str((int(valid[-1]) + 1) % 10)
    assert "curp" not in extract_known_fields([{"text": invalid}])


def test_extract_known_front_fields_from_labelled_blocks():
    valid = _synthetic_curp()
    fields = extract_known_fields([
        {"text": "NOMBRE"},
        {"text": "PRUEBA"},
        {"text": "CONTROL"},
        {"text": "PERSONA SINTETICA"},
        {"text": "DOMICILIO"},
        {"text": "AV UNO 25"},
        {"text": "COL CENTRO 06000"},
        {"text": "CUAUHTEMOC CDMX"},
        {"text": "CLAVE DE ELECTOR ABCD900101HDFBCD01"},
        {"text": f"CURP {valid}"},
        {"text": "AÑO DE REGISTRO 2018 03"},
        {"text": "ESTADO 9 MUNICIPIO 14"},
        {"text": "SECCIÓN 123 LOCALIDAD 1"},
        {"text": "VIGENCIA 2020-2030"},
        {"text": "FECHA DE NACIMIENTO 01/01/2000"},
        {"text": "SEXO H"},
    ])
    assert fields["primerApellido"] == "PRUEBA"
    assert fields["segundoApellido"] == "CONTROL"
    assert fields["nombres"] == "PERSONA SINTETICA"
    assert fields["calle"] == "AV UNO 25"
    assert fields["colonia"] == "COL CENTRO 06000"
    assert fields["ciudad"] == "CUAUHTEMOC CDMX"
    assert fields["registro"] == "2018 03"
    assert fields["estado"] == "09"
    assert fields["municipio"] == "014"
    assert fields["seccion"] == "0123"
    assert fields["localidad"] == "0001"
    assert fields["emision"] == "2020"
    assert fields["vigencia"] == "2030"
    assert fields["fechaNacimiento"] == "01/01/2000"
    assert fields["_yearPairs"] == [["2020", "2030"]]


def test_registration_and_section_stay_in_their_columns():
    fields = extract_known_fields([
        _positioned("NOMBRE", 100, 40),
        _positioned("CURP", 100, 170, width=60),
        _positioned("AÑO DE REGISTRO", 400, 170, width=150),
        _positioned(_synthetic_curp(), 100, 184, width=220),
        _positioned("201103", 400, 185, width=70),
        _positioned("SECCION", 250, 200, width=70),
        _positioned("VIGENCIA", 400, 200, width=70),
        _positioned("2030", 400, 215, width=45),
        _positioned("0123", 250, 216, width=45),
    ])

    assert fields["registro"] == "2011 03"
    assert fields["seccion"] == "0123"
    assert fields["vigencia"] == "2030"


def test_extract_clave_when_label_is_truncated_but_value_is_complete():
    fields = extract_known_fields([
        {"text": "LECTOR ABCDBC00010100H001"},
    ])

    assert fields["claveElector"] == "ABCDBC00010100H001"


def test_name_candidates_continue_past_interleaved_gender_label():
    fields = extract_known_fields([
        {"text": "NOMBRE"},
        {"text": "SEXO M"},
        {"text": "CONTROL"},
        {"text": "PERSONA SINTETICA"},
        {"text": "DOMICILIO"},
    ])

    assert fields["_nameCandidates"] == ["CONTROL", "PERSONA SINTETICA"]
    assert "primerApellido" not in fields


def _positioned(text, left, top, width=140, height=12):
    return {
        "text": text,
        "score": 0.99,
        "box": [[left, top], [left + width, top],
                [left + width, top + height], [left, top + height]],
    }


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_name_layout_is_invariant_to_card_rotation(rotation):
    lines = [
        _positioned("INSTITUTO NACIONAL ELECTORAL", 100, 10),
        _positioned("NOMBRE", 100, 40, width=60),
        _positioned("SEXO H", 400, 45, width=60),
        _positioned("DE PRUEBA", 100, 55),
        _positioned("CONTROL", 100, 70),
        _positioned("PERSONA SINTETICA", 100, 85),
        _positioned("DOMICILIO", 100, 110),
        _positioned(f"CURP {_synthetic_curp()}", 100, 170, width=200),
    ]
    for line in lines:
        for point in line["box"]:
            for _ in range(rotation // 90):
                point[:] = [-point[1], point[0]]
    lines.sort(key=lambda line: min(point[1] for point in line["box"]))

    fields = extract_known_fields(lines)

    assert fields["primerApellido"] == "DE PRUEBA"
    assert fields["segundoApellido"] == "CONTROL"
    assert fields["nombres"] == "PERSONA SINTETICA"


@pytest.mark.parametrize("garbled_label", ["CPUF", "OMICILIO"])
def test_name_block_does_not_extend_into_unrecognized_address_label(garbled_label):
    fields = extract_known_fields([
        _positioned("NOMBRE", 100, 40, width=60),
        _positioned("PRUEBA", 100, 55),
        _positioned("CONTROL", 100, 70),
        _positioned("PERSONA SINTETICA", 100, 85),
        _positioned(garbled_label, 100, 120),
        _positioned("AND", 100, 135),
        _positioned("CURP", 100, 170),
    ])

    assert fields["nombres"] == "PERSONA SINTETICA"


def test_year_values_are_associated_with_their_label_column():
    fields = extract_known_fields([
        _positioned("NOMBRE", 100, 40),
        _positioned("CURP", 100, 170),
        _positioned("EMISION", 100, 200, width=70),
        _positioned("SECCION", 250, 200, width=70),
        _positioned("VIGENCIA", 400, 200, width=70),
        _positioned("0001", 250, 215, width=45),
        _positioned("2030", 400, 216, width=45),
        _positioned("2020", 100, 217, width=45),
    ])

    assert fields["emision"] == "2020"
    assert fields["vigencia"] == "2030"


def test_section_number_is_not_an_issue_year():
    fields = extract_known_fields([{"text": "EMISION 0001"}])

    assert "emision" not in fields


def test_printed_birth_date_can_be_corroborated_by_curp_without_label():
    fields = extract_known_fields([
        {"text": f"CURP {_synthetic_curp()}"},
        {"text": "01/01/2000"},
        {"text": "02/01/2000"},
    ])

    assert fields["fechaNacimiento"] == "01/01/2000"


def test_birth_date_is_not_guessed_from_ambiguous_century():
    fields = extract_known_fields([
        {"text": f"CURP {_synthetic_curp()}"},
        {"text": "01/01/2000"},
        {"text": "01/01/1900"},
    ])

    assert "fechaNacimiento" not in fields


def test_check_valid_curp_can_anchor_orientation_without_a_curp_label():
    lines = [
        _positioned("INSTITUTO NACIONAL ELECTORAL", 100, 10),
        _positioned("CREDENCIAL PARA VOTAR", 100, 25),
        _positioned(_synthetic_curp(), 100, 170),
    ]
    for line in lines:
        line["box"] = [[-point[1], point[0]] for point in line["box"]]

    assert reading_rotation(lines) == 270
