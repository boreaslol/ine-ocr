from types import SimpleNamespace

import json
import numpy as np
import pytest

from ine_ocr.name_onnx import NameLineAdapter


def _line(text, top):
    return {"text": text, "score": 0.99, "box": [[20, top], [180, top], [180, top + 12], [20, top + 12]]}


def test_name_adapter_preserves_numeric_and_label_lines():
    lines = [_line("NOMBRE", 5), _line("PRUEBA", 25), _line("DOS", 45),
             _line("PERSONA", 65), _line("DOMICILIO", 100), _line("CURP", 120),
             _line("ABCD900101HDFXYZ09", 140)]
    reader = SimpleNamespace(cpu_threads=2, read_lines=lambda image: lines)
    recognizer = SimpleNamespace(predict=lambda crops: [{"text": text, "confidence": 0.99} for text in ["PRUEBA", "DOS", "PERSONA SINTETICA"]])
    result = NameLineAdapter(reader, recognizer).read_lines(np.full((180, 220, 3), 255, dtype=np.uint8))
    assert result[3]["text"] == "PERSONA SINTETICA"
    assert result[-1] == lines[-1]
    assert result[-2] == lines[-2]
    assert result[0] == lines[0]


def test_low_confidence_or_machine_tokens_do_not_replace_names():
    lines = [_line("NOMBRE", 5), _line("PRUEBA", 25), _line("DOS", 45), _line("PERSONA", 65), _line("DOMICILIO", 100)]
    reader = SimpleNamespace(cpu_threads=2, read_lines=lambda image: lines)
    recognizer = SimpleNamespace(predict=lambda crops: [{"text": "<UNKNOWN", "confidence": 1.0}, {"text": "OTHER", "confidence": 0.2}, {"text": "", "confidence": 1.0}])
    assert NameLineAdapter(reader, recognizer).read_lines(np.full((140, 220, 3), 255, dtype=np.uint8)) == lines


def test_name_model_cannot_silently_enter_machine_line_decoder(tmp_path):
    from ine_ocr.alphabet import LATIN_NAME_ALPHABET
    from ine_ocr.runtime import OnnxMachineLineRecognizer

    model = tmp_path / "name.onnx"
    model.with_suffix(".onnx.json").write_text(json.dumps({"alphabet": LATIN_NAME_ALPHABET}))
    with pytest.raises(ValueError, match="use_name_recognizer"):
        OnnxMachineLineRecognizer(str(model))


def test_digit_predictions_cannot_replace_name_lines():
    lines = [_line("NOMBRE", 5), _line("PRUEBA", 25), _line("DOS", 45), _line("PERSONA", 65), _line("DOMICILIO", 100)]
    reader = SimpleNamespace(cpu_threads=2, read_lines=lambda image: lines)
    recognizer = SimpleNamespace(predict=lambda crops: [{"text": "2023 2033", "confidence": 1.0}] * len(crops))
    assert NameLineAdapter(reader, recognizer).read_lines(np.full((140, 220, 3), 255, dtype=np.uint8)) == lines


def test_name_overlay_changes_only_name_fields():
    from ine_ocr.name_onnx import NameLineAdapter

    result = {"nombres": "OLD", "calle": "KEEP", "fieldSources": {}, "fieldConfidence": {}}
    adapter = NameLineAdapter.__new__(NameLineAdapter)
    changed = adapter.apply_to_result(result, {"nombres": {"text": "NEW", "confidence": 0.99}})
    assert changed == ["nombres"]
    assert result["nombres"] == "NEW"
    assert result["calle"] == "KEEP"
    assert result["processing"]["nameModel"]["scope"] == "name_fields_only"


@pytest.mark.parametrize("source", [
    "front_ocr_consensus", "front_ocr_mrz_corroborated", "front_ocr_cross_channel_consensus",
])
def test_name_overlay_does_not_override_high_confidence_front_consensus(source):
    result = {
        "nombres": "CONSENSUS",
        "fieldSources": {"nombres": source},
        "fieldConfidence": {"nombres": 0.95},
    }
    adapter = NameLineAdapter.__new__(NameLineAdapter)

    changed = adapter.apply_to_result(
        result, {"nombres": {"text": "MODEL", "confidence": 0.99}}
    )

    assert changed == []
    assert result["nombres"] == "CONSENSUS"
    assert result["processing"]["nameModel"]["skippedProtectedFields"] == ["nombres"]


def test_name_overlay_can_replace_unchecked_mrz_name():
    result = {
        "nombres": "MRZ NAME",
        "fieldSources": {"nombres": "mrz_unchecked_name"},
        "fieldConfidence": {"nombres": 0.8},
    }
    adapter = NameLineAdapter.__new__(NameLineAdapter)

    changed = adapter.apply_to_result(
        result, {"nombres": {"text": "PRINTED NAME", "confidence": 0.99}}
    )

    assert changed == ["nombres"]
    assert result["nombres"] == "PRINTED NAME"
    assert result["processing"]["nameModel"]["changedFieldConfidences"] == {"nombres": 0.99}
