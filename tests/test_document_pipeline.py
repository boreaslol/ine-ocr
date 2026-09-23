import numpy as np
import pytest

from ine_ocr.curp import compute_check_digit
from ine_ocr.document_pipeline import (
    DocumentPipeline, _cross_channel_name_consensus, _mrz_spaced_name, _name_consensus, _prefer_printed_name,
)
from ine_ocr.mrz import icao_check_digit, parse as parse_mrz


def _synthetic_curp() -> str:
    prefix = "AECD000101HDFBCD0"
    return prefix + compute_check_digit(prefix)


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


class FakePreprocessor:
    def process(self, _data, side):
        value = 10 if side == "front" else 20
        image = np.full((630, 1000, 3), value, dtype=np.uint8)
        quality = {"ok": True, "issues": [], "cardDetected": True}
        return image, quality, image, image


@pytest.mark.parametrize(
    ("maternal", "raw_given", "expected"),
    [
        ("CONTROL", "PERSONA<SINTE", "PERSONA SINTETICA"),
        ("DISTINTO", "PERSONA<SINTE", "PERSONASINTETICA"),
        ("CONTROL", "OTRA<SINTE", "PERSONASINTETICA"),
    ],
)
def test_missing_mrz_name_separator_can_restore_only_corroborated_spaces(
    monkeypatch, maternal, raw_given, expected
):
    pipeline = DocumentPipeline(
        line_reader=FakeLineReader(), preprocessor=FakePreprocessor(),
        qr_decoder=FakeQRDecoder(),
    )
    front = {
        "curp": _synthetic_curp(), "primerApellido": "PRUEBA",
        "segundoApellido": maternal, "nombres": "PERSONASINTETICA",
    }
    raw_mrz = _synthetic_mrz().replace(
        "PRUEBA<CONTROL<<PERSONA<SINTETICA<<", "PRUEBA<CONTROL<" + raw_given
    )
    monkeypatch.setattr(pipeline, "_front_fields", lambda *_: (front, {}))
    monkeypatch.setattr(pipeline, "_mrz", lambda *_: (raw_mrz, parse_mrz(raw_mrz), {}))
    result = pipeline.extract(b"front", b"back")
    assert result["nombres"] == expected
    assert result["nombres"].replace(" ", "") == front["nombres"]
    assert result["fieldConfidence"]["nombres"] == 0.8
    if " " in expected:
        assert result["fieldSources"]["nombres"] == "front_ocr_mrz_word_boundaries"


class FakeLineReader:
    def read_lines(self, image):
        if image.mean() < 15:
            return [{"text": f"CURP {_synthetic_curp()} SEXO H", "box": [[0, 0], [1, 0]]}]
        return [{"text": _synthetic_mrz(), "box": [[0, 0], [1, 0]]}]


class FakeQRDecoder:
    def decode(self, image):
        return ["https://qr.ine.mx/synthetic-test"] if image.mean() >= 15 else []


class StructuredQRDecoder:
    def decode(self, image):
        if image.mean() >= 15:
            return ["https://qr.ine.mx/123456789012345987654321/20240101/P/000001"]
        return []


class QROnlyBackLineReader:
    def read_lines(self, image):
        if image.mean() < 15:
            return [{"text": f"CURP {_synthetic_curp()} SEXO H", "box": [[0, 0], [1, 0]]}]
        return [{"text": "NO MRZ", "box": [[0, 0], [1, 0]]}]


class DetailRescueLineReader:
    def read_lines(self, image):
        if image.shape[1] > 1000:
            return [{"text": "VIGENCIA 2020-2030", "box": [[0, 0], [1, 0]]}]
        return [{"text": f"CURP {_synthetic_curp()}", "box": [[0, 0], [1, 0]]}]


class DetailPreprocessor:
    def process(self, _data, side):
        standardized = np.full((630, 1000, 3), 10 if side == "front" else 20, dtype=np.uint8)
        detail = np.full((1008, 1600, 3), 10 if side == "front" else 20, dtype=np.uint8)
        quality = {"ok": True, "issues": [], "cardDetected": True}
        return standardized, quality, detail, standardized


class CorroboratedNameLineReader:
    def read_lines(self, image):
        if image.mean() < 15:
            return [
                {"text": "NOMBRE", "box": [[0, 0], [1, 0]]},
                {"text": "MUÑOZ", "box": [[0, 1], [1, 1]]},
                {"text": "CONTROL", "box": [[0, 2], [1, 2]]},
                {"text": "JOSE", "box": [[0, 3], [1, 3]]},
                {"text": f"CURP {_synthetic_curp()}", "box": [[0, 4], [1, 4]]},
            ]
        value = _synthetic_mrz().replace(
            "PRUEBA<CONTROL<<PERSONA<SINTETICA<<",
            "MUNOZ<CONTROL<<JOSE<<",
        )
        return [{"text": value, "box": [[0, 0], [1, 0]]}]


class WideBandMRZLineReader:
    def read_lines(self, image):
        if image.shape[0] >= 650:
            return [{"text": _synthetic_mrz(), "box": [[0, 0], [1, 0]]}]
        return [{"text": "PRUEBA<CONTROL<<PERSONA", "box": [[0, 0], [1, 0]]}]


class PartialNumericMRZReader:
    def read_lines(self, image):
        return [{"text": "9001011H3001011MEX<01<<12345<0"}]


class RegionMRZRefiner:
    def read_regions(self, image, lines, *, minimum_text=3):
        assert minimum_text == 0
        return [{"text": _synthetic_mrz()}]


def test_regional_mrz_rescue_requires_and_returns_full_check_valid_mrz():
    pipeline = DocumentPipeline(line_reader=PartialNumericMRZReader(), front_refiner=RegionMRZRefiner())
    image = np.zeros((1008, 1600, 3), dtype=np.uint8)
    value, fields, processing = pipeline._mrz(image, image)
    assert value == _synthetic_mrz()
    assert fields["validacionMRZAllOk"]
    assert processing["regionRescue"]
    assert processing["attempts"] == 1


def test_front_refinement_evidence_stays_request_local_and_out_of_json():
    import json

    class EvidenceRefiner:
        def refine(self, data, result, *, evidence):
            assert data == b"front"
            assert evidence["image"].shape == (630, 1000, 3)
            assert evidence["lines"]
            assert "_refinementInput" not in result

    pipeline = DocumentPipeline(
        line_reader=FakeLineReader(), preprocessor=FakePreprocessor(),
        qr_decoder=FakeQRDecoder(), front_refiner=EvidenceRefiner(),
    )
    result = pipeline.extract(b"front", b"back")
    assert json.loads(json.dumps(result))["curp"] == _synthetic_curp()


def test_name_adapter_is_an_explicit_name_only_overlay():
    class StubNameAdapter:
        def predict_from_bytes(self, data):
            raise AssertionError("name_model_should_reuse_front_evidence")

        def predict_from_evidence(self, image, lines):
            assert image.shape == (630, 1000, 3)
            assert lines
            return {"nombres": {"text": "PERSONA NUEVA", "confidence": 0.99}}, {"rotation": 0}

        def apply_to_result(self, result, predictions):
            assert predictions["nombres"]["text"] == "PERSONA NUEVA"
            result["nombres"] = predictions["nombres"]["text"]
            result.setdefault("processing", {})["nameModel"] = {
                "enabled": True,
                "changedFields": ["nombres"],
                "scope": "name_fields_only",
            }
            return ["nombres"]

    pipeline = DocumentPipeline(
        line_reader=FakeLineReader(), preprocessor=FakePreprocessor(),
        qr_decoder=FakeQRDecoder(), name_adapter=StubNameAdapter(),
    )
    result = pipeline.extract(b"front")
    assert result["nombres"] == "PERSONA NUEVA"
    assert result["processing"]["nameModel"]["scope"] == "name_fields_only"


def test_document_pipeline_ready_initializes_runtime_components():
    class ReadyComponent:
        def __init__(self, value):
            self.value = value
            self.ready_calls = 0

        def ready(self):
            self.ready_calls += 1

    preprocessor = ReadyComponent("preprocessor")
    line_reader = ReadyComponent("line_reader")
    qr_decoder = ReadyComponent("qr_decoder")
    pipeline = DocumentPipeline(
        line_reader=line_reader,
        preprocessor=preprocessor,
        qr_decoder=qr_decoder,
    )
    pipeline.ready()
    assert [component.ready_calls for component in (preprocessor, line_reader, qr_decoder)] == [1, 1, 1]


class FallbackClaveLineReader:
    def read_lines(self, image):
        if image.shape[1] > 1000:
            return [{"text": f"CURP {_synthetic_curp()}", "box": [[0, 0], [1, 0]]}]
        return [{"text": "LECTOR ABCDBC00010100H001", "box": [[0, 0], [1, 0]]}]


class PartialPrintedNameLineReader:
    def read_lines(self, image):
        if image.mean() < 15:
            return [
                {"text": "NOMBRE", "box": [[0, 0], [1, 0]]},
                {"text": "SEXO H", "box": [[0, 1], [1, 1]]},
                {"text": "CONTROL", "box": [[0, 2], [1, 2]]},
                {"text": "PERSONA SINTETICA", "box": [[0, 3], [1, 3]]},
                {"text": f"CURP {_synthetic_curp()}", "box": [[0, 4], [1, 4]]},
            ]
        value = _synthetic_mrz().replace(
            "PRUEBA<CONTROL<<PERSONA<SINTETICA<<",
            "PRUEBA<CONTROL<<PERSONA<SINTET<<",
        )
        return [{"text": value, "box": [[0, 0], [1, 0]]}]


def test_full_document_pipeline_combines_validated_channels():
    pipeline = DocumentPipeline(
        line_reader=FakeLineReader(),
        preprocessor=FakePreprocessor(),
        qr_decoder=FakeQRDecoder(),
    )
    result = pipeline.extract(b"front", b"back")
    assert result["channels"] == {
        "frontCurp": True,
        "frontCurpCheckOk": True,
        "mrz": True,
        "mrzCheckOk": True,
        "qr": True,
        "anyStructured": True,
    }
    assert result["validacionMRZ"]["cic"] == "OK"
    assert "ocrLines" not in result


def test_front_detail_rescue_fills_missing_years():
    pipeline = DocumentPipeline(
        line_reader=DetailRescueLineReader(),
        preprocessor=DetailPreprocessor(),
        qr_decoder=FakeQRDecoder(),
    )
    result = pipeline.extract(b"front")
    assert result["emision"] == "2020"
    assert result["vigencia"] == "2030"
    assert result["processing"]["front"]["detailRescue"] is True


def test_front_roi_sidecar_can_replace_complete_detail_rescue():
    class SidecarReader:
        def __init__(self):
            self.detail_reads = 0
            self.sidecar_reads = 0

        def read_lines(self, image):
            if image.shape[1] > 1000:
                self.detail_reads += 1
                raise AssertionError("complete_roi_should_skip_full_detail_read")
            return [{"text": f"CURP {_synthetic_curp()}"}]

        def read_lines_from_regions_sidecar(self, image, lines):
            self.sidecar_reads += 1
            return [{"text": text} for text in [
                "NOMBRE", "PRUEBA", "CONTROL", "PERSONA SINTETICA",
                "DOMICILIO", "CALLE 1", "COLONIA 2", "CIUDAD 3",
                "CLAVE DE ELECTOR ABCDBC00010100H001",
                f"CURP {_synthetic_curp()}",
                "ANO DE REGISTRO 2020 01", "FECHA DE NACIMIENTO 01/01/2000",
                "VIGENCIA 2020-2030", "SEXO H",
            ]]

    reader = SidecarReader()
    pipeline = DocumentPipeline(
        line_reader=reader, preprocessor=DetailPreprocessor(), qr_decoder=FakeQRDecoder(),
    )

    result = pipeline.extract(b"front")

    assert reader.sidecar_reads == 1
    assert reader.detail_reads == 0
    assert result["processing"]["front"]["detailRescueSource"] == "front_roi_sidecar"
    assert result["nombres"] == "PERSONA SINTETICA"


def test_front_roi_sidecar_rejects_conflicting_or_incomplete_fields():
    from ine_ocr.document_pipeline import _accepted_roi_fields

    primary = {"curp": _synthetic_curp(), "nombres": "PERSONA SINTETICA"}
    assert _accepted_roi_fields(primary, {"curp": _synthetic_curp()}) is None
    assert _accepted_roi_fields(
        primary,
        {
            "curp": _synthetic_curp(),
            "claveElector": "ABCDBC00010100H001",
            "emision": "2020", "vigencia": "2030", "fechaNacimiento": "01/01/2000",
            "primerApellido": "PRUEBA", "segundoApellido": "CONTROL",
            "nombres": "PERSONA DISTINTA",
        },
    ) is None


def test_official_qr_fills_cic_when_mrz_is_unavailable():
    pipeline = DocumentPipeline(
        line_reader=QROnlyBackLineReader(),
        preprocessor=FakePreprocessor(),
        qr_decoder=StructuredQRDecoder(),
    )
    result = pipeline.extract(b"front", b"back")
    assert result["cic"] == "987654321"
    assert result["fieldSources"]["cic"] == "qr_official_token"
    assert result["qrValidation"]["cic"] == "STRUCTURE_OK"


def test_printed_name_is_kept_only_when_mrz_corroborates_it():
    pipeline = DocumentPipeline(
        line_reader=CorroboratedNameLineReader(),
        preprocessor=FakePreprocessor(),
        qr_decoder=FakeQRDecoder(),
    )
    result = pipeline.extract(b"front", b"back")
    assert result["primerApellido"] == "MUÑOZ"
    assert result["fieldSources"]["primerApellido"] == "front_ocr_mrz_corroborated"


def test_printed_name_can_extend_a_truncated_given_name_or_surname():
    assert _prefer_printed_name("nombres", "SINTETICA", "SINTET") is True
    assert _prefer_printed_name("primerApellido", "CONTROL", "CONTRO") is True


def test_printed_name_can_restore_a_token_missing_from_truncated_mrz():
    assert _prefer_printed_name("nombres", "PERSONA SINTETICA", "PERSONA") is True


def test_printed_name_cannot_replace_a_different_or_too_short_mrz_name():
    assert _prefer_printed_name("nombres", "PERSONA SINTETICA", "CONTROL") is False
    assert _prefer_printed_name("primerApellido", "DE PRUEBA", "DE") is False
    assert _prefer_printed_name("segundoApellido", "CONTROLEXTRA", "CONTROL") is False


def test_mrz_word_boundaries_can_extend_to_the_full_printed_name():
    assert _mrz_spaced_name(
        "PERSONASINTETICA", "PRUEBA<CONTROL<<PERSONA<SINTET",
    ) == "PERSONA SINTETICA"


def test_mrz_spacing_preserves_printed_spanish_diacritics():
    assert _mrz_spaced_name("MUÑOZDEPRUEBA", "MUNOZ<DE<PRUEBA<<PERSONA") == "MUÑOZ DE PRUEBA"


def test_mrz_scan_keeps_first_line_when_card_is_small_in_photo():
    pipeline = DocumentPipeline(line_reader=WideBandMRZLineReader())
    detail = np.zeros((1008, 1600, 3), dtype=np.uint8)
    natural = np.zeros((630, 1000, 3), dtype=np.uint8)

    value, fields, processing = pipeline._mrz(detail, natural)

    assert value == _synthetic_mrz()
    assert fields["validacionMRZAllOk"] is True
    assert processing == {
        "source": "rectified_detail",
        "orientation": "0",
        "bandStart": 0.35,
        "attempts": 2,
    }


def test_front_fields_keep_values_found_before_valid_curp_source():
    pipeline = DocumentPipeline(
        line_reader=FallbackClaveLineReader(),
        preprocessor=DetailPreprocessor(),
        qr_decoder=FakeQRDecoder(),
    )

    result = pipeline.extract(b"front")

    assert result["curp"] == _synthetic_curp()
    assert result["claveElector"] == "ABCDBC00010100H001"


def test_partial_printed_name_extends_validated_mrz_name():
    pipeline = DocumentPipeline(
        line_reader=PartialPrintedNameLineReader(),
        preprocessor=FakePreprocessor(),
        qr_decoder=FakeQRDecoder(),
    )

    result = pipeline.extract(b"front", b"back")

    assert result["primerApellido"] == "PRUEBA"
    assert result["segundoApellido"] == "CONTROL"
    assert result["nombres"] == "PERSONA SINTETICA"
    assert result["fieldSources"]["nombres"] == "front_ocr_mrz_corroborated"


def test_valid_curp_orientation_does_not_inherit_wrong_rotation_names():
    class RotatedReader:
        calls = 0

        def read_lines(self, image):
            self.calls += 1
            names = ["PRUEBA", "CONTROL", "PERSONA SINTETICA"]
            if self.calls == 1:
                names = ["INCORRECTO", "OTRO", "ERROR"]
            lines = [{"text": name} for name in ["NOMBRE", *names, "DOMICILIO"]]
            if self.calls > 1:
                lines.append({"text": f"CURP {_synthetic_curp()}"})
            return lines

    reader = RotatedReader()
    pipeline = DocumentPipeline(line_reader=reader, preprocessor=FakePreprocessor())

    result = pipeline.extract(b"front")

    assert result["processing"]["front"]["orientation"] == "180"
    assert result["primerApellido"] == "PRUEBA"
    assert result["segundoApellido"] == "CONTROL"
    assert result["nombres"] == "PERSONA SINTETICA"


def test_compound_surname_slots_follow_front_rows_corroborated_by_mrz():
    class CompoundSurnameReader:
        def read_lines(self, image):
            if image.mean() < 15:
                return [{"text": name} for name in [
                    "NOMBRE", "DEPRUEBA", "CONTROL", "PERSONA SINTETICA",
                    "DOMICILIO", f"CURP {_synthetic_curp()}",
                ]]
            return [{"text": _synthetic_mrz().replace("PRUEBA<CONTROL", "DE<PRUEBA<CONTROL")}]

    pipeline = DocumentPipeline(
        line_reader=CompoundSurnameReader(),
        preprocessor=FakePreprocessor(),
        qr_decoder=FakeQRDecoder(),
    )

    result = pipeline.extract(b"front", b"back")

    assert result["primerApellido"] == "DE PRUEBA"
    assert result["segundoApellido"] == "CONTROL"
    assert result["nombres"] == "PERSONA SINTETICA"


def _positioned_names():
    lines = []
    for index, text in enumerate([
        "NOMBRE", "PRUEBA", "CONTROL", "PERSONA SINTETICA",
        "DOMICILIO", f"CURP {_synthetic_curp()}",
    ]):
        top = index * 20
        lines.append({
            "text": text,
            "score": 0.99,
            "box": [[100, top], [300, top], [300, top + 12], [100, top + 12]],
        })
    return lines


def test_high_resolution_rescue_uses_upright_natural_image_for_vertical_text():
    class PortraitPreprocessor:
        def process(self, data, side):
            return (
                np.zeros((630, 1000, 3), dtype=np.uint8),
                {"ok": True},
                np.zeros((1008, 1600, 3), dtype=np.uint8),
                np.zeros((1600, 900, 3), dtype=np.uint8),
            )

    class PortraitReader:
        shapes = []

        def read_lines(self, image):
            self.shapes.append(image.shape)
            lines = _positioned_names()
            if len(self.shapes) == 1:
                for line in lines:
                    line["box"] = [[-point[1], point[0]] for point in line["box"]]
            return lines

    reader = PortraitReader()
    pipeline = DocumentPipeline(line_reader=reader, preprocessor=PortraitPreprocessor())

    result = pipeline.extract(b"front")

    assert reader.shapes == [(630, 1000, 3), (900, 1600, 3)]
    assert result["processing"]["front"]["detailRescueSource"] == "upright_natural"
    assert result["processing"]["front"]["detailRescueOrientation"] == "270"
    assert result["nombres"] == "PERSONA SINTETICA"


def test_two_high_confidence_front_reads_can_correct_unchecked_mrz_name():
    class ConsensusReader:
        def read_lines(self, image):
            if image.mean() < 15:
                return _positioned_names()
            return [{"text": _synthetic_mrz().replace("PRUEBA", "PRUEVA")}]

    pipeline = DocumentPipeline(
        line_reader=ConsensusReader(), preprocessor=DetailPreprocessor(), qr_decoder=FakeQRDecoder(),
    )

    result = pipeline.extract(b"front", b"back")

    assert result["primerApellido"] == "PRUEBA"
    assert result["fieldSources"]["primerApellido"] == "front_ocr_consensus"
    assert result["validacionMRZ"]["cic"] == "OK"


def test_consensus_requires_two_high_confidence_matching_reads():
    first = {"nombres": "PERSONASINTETICA", "_nameScores": {"nombres": 0.99}}
    second = {"nombres": "PERSONA SINTETICA", "_nameScores": {"nombres": 0.97}}
    assert _name_consensus(first, second) == {"nombres": "PERSONA SINTETICA"}
    assert _name_consensus(first, {**second, "_nameScores": {"nombres": 0.7}}) == {}
    assert _name_consensus(first, {**second, "nombres": "OTRA PERSONA"}) == {}


def test_numeric_mrz_checks_do_not_validate_the_name_characters():
    pipeline = DocumentPipeline(
        line_reader=FakeLineReader(), preprocessor=FakePreprocessor(), qr_decoder=FakeQRDecoder(),
    )

    result = pipeline.extract(b"front", b"back")

    assert result["fieldSources"]["cic"] == "mrz_validated"
    assert result["fieldSources"]["nombres"] == "mrz_unchecked_name"
    assert result["fieldConfidence"]["nombres"] < result["fieldConfidence"]["cic"]


def test_malformed_clave_does_not_block_complete_detail_read():
    class ClaveReader:
        def read_lines(self, image):
            clave = "ABCDBC00010100H001" if image.shape[1] > 1000 else "ABCDDC00010100H01"
            return [{"text": f"CURP {_synthetic_curp()} CLAVE DE ELECTOR {clave}"}]

    pipeline = DocumentPipeline(line_reader=ClaveReader(), preprocessor=DetailPreprocessor())

    result = pipeline.extract(b"front")

    assert result["claveElector"] == "ABCDBC00010100H001"


def test_clave_with_conflicting_birth_digits_is_not_accepted():
    class ClaveReader:
        def read_lines(self, image):
            return [{"text": f"CURP {_synthetic_curp()} CLAVE DE ELECTOR ABCDBC99010100H001"}]

    pipeline = DocumentPipeline(line_reader=ClaveReader(), preprocessor=DetailPreprocessor())

    result = pipeline.extract(b"front")

    assert not result.get("claveElector")


def test_front_consensus_still_restores_explicit_mrz_word_boundaries():
    class SpacingReader:
        def read_lines(self, image):
            if image.mean() >= 15:
                return [{"text": _synthetic_mrz()}]
            lines = _positioned_names()
            lines[3]["text"] = "PERSONASINTETICA"
            return lines

    pipeline = DocumentPipeline(
        line_reader=SpacingReader(), preprocessor=DetailPreprocessor(), qr_decoder=FakeQRDecoder(),
    )

    result = pipeline.extract(b"front", b"back")

    assert result["nombres"] == "PERSONA SINTETICA"


def _cross_channel_inputs():
    front = {
        "curp": _synthetic_curp(), "fechaNacimiento": "01/01/2000",
        "primerApellido": "PRUEBA", "segundoApellido": "CONTROL", "nombres": "PERSONA SINTETICA",
        "_nameScores": {"primerApellido": 0.99, "segundoApellido": 0.99, "nombres": 0.99},
        "_nameConsensus": {"primerApellido": "PRUEBA"},
    }
    birth = "000101"
    raw = _synthetic_mrz().replace(f"900101{icao_check_digit('900101')}", f"{birth}{icao_check_digit(birth)}")
    raw = raw.replace("PRUEBA", "PRAXXA")
    return front, raw, parse_mrz(raw)


def test_cross_channel_consensus_requires_corresponding_identity_and_name_anchor():
    front, _, mrz = _cross_channel_inputs()
    assert _cross_channel_name_consensus("primerApellido", front, mrz)


@pytest.mark.parametrize("change", ["no_consensus", "invalid_curp", "missing_birth", "different_birth",
                                    "curp_birth_conflict", "invalid_mrz", "no_anchor", "weak_anchor", "short_anchor"])
def test_cross_channel_consensus_rejects_uncorroborated_inputs(change):
    front, _, mrz = _cross_channel_inputs()
    if change == "no_consensus":
        front.pop("_nameConsensus")
    elif change == "invalid_curp":
        front["curp"] = "invalid"
    elif change == "missing_birth":
        front.pop("fechaNacimiento")
    elif change == "different_birth":
        mrz["fechaNacimiento"] = "01/01/1990"
    elif change == "curp_birth_conflict":
        front["fechaNacimiento"] = mrz["fechaNacimiento"] = "01/01/1990"
    elif change == "invalid_mrz":
        mrz["validacionMRZAllOk"] = False
    elif change == "no_anchor":
        mrz["segundoApellido"], mrz["nombres"] = "DIFERENTE", "DISTINTO"
    elif change == "weak_anchor":
        front["_nameScores"] = {"primerApellido": 0.99, "segundoApellido": 0.5, "nombres": 0.5}
    else:
        front["segundoApellido"] = mrz["segundoApellido"] = "DE"
        mrz["nombres"] = "DISTINTO"
    assert not _cross_channel_name_consensus("primerApellido", front, mrz)


@pytest.mark.parametrize("enabled,expected", [(False, "PRAXXA"), (True, "PRUEBA")])
def test_cross_channel_consensus_is_opt_in_and_changes_only_the_name(monkeypatch, enabled, expected):
    front, raw, mrz = _cross_channel_inputs()
    pipeline = DocumentPipeline(line_reader=FakeLineReader(), preprocessor=FakePreprocessor(),
                                qr_decoder=FakeQRDecoder(), cross_channel_name_consensus=enabled)
    monkeypatch.setattr(pipeline, "_front_fields", lambda *_: (front, {}))
    monkeypatch.setattr(pipeline, "_mrz", lambda *_: (raw, mrz, {}))
    result = pipeline.extract(b"front", b"back")
    assert result["primerApellido"] == expected
    assert result["segundoApellido"] == "CONTROL"
    assert result["nombres"] == "PERSONA SINTETICA"
    assert result["curp"] == front["curp"]
    assert result["cic"] == mrz["cic"]
    assert result["fechaNacimiento"] == mrz["fechaNacimiento"]
    assert result["vigencia"] == mrz["vigencia"]
    assert result["fieldSources"]["primerApellido"] == (
        "front_ocr_cross_channel_consensus" if enabled else "mrz_unchecked_name"
    )
