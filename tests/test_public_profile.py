import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat

from PIL import Image
import pytest

from ine_ocr.document_onnx import model_directory
from ine_ocr.document_preprocess import CardPreprocessor, ImageDecodeError, bytes_to_image
from ine_ocr.qr import extract_official_fields


ROOT = Path(__file__).resolve().parents[1]


def test_public_preprocessor_needs_no_custom_weight(monkeypatch):
    monkeypatch.delenv("INE_OCR_YOLO_MODEL_PATH", raising=False)
    processor = CardPreprocessor()
    processor.ready()
    assert processor.yolo_model_path is None
    buffer = io.BytesIO()
    Image.new("RGB", (1000, 630), "white").save(buffer, format="PNG")
    normalized, quality, _, _ = processor.process(buffer.getvalue(), "front")
    assert normalized.shape == (630, 1000, 3)
    assert quality["fullFrameAssumed"]


def test_explicit_missing_custom_detector_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="card_detector_not_ready"):
        CardPreprocessor(str(tmp_path / "missing.onnx")).ready()


def test_pixel_limit_precedes_large_image_decode(monkeypatch):
    monkeypatch.setattr("ine_ocr.document_preprocess.MAX_DECODED_PIXELS", 10)
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buffer, format="PNG")
    with pytest.raises(ImageDecodeError, match="decoded_image_too_large"):
        bytes_to_image(buffer.getvalue())


def test_byom_requires_operator_manifest_and_matching_hashes(monkeypatch, tmp_path):
    monkeypatch.setenv("INE_OCR_RECOGNITION_ONNX_DIR", str(tmp_path))
    directory = tmp_path / "PP-OCRv6_small_rec"
    directory.mkdir()
    hashes = {}
    for filename in ("inference.onnx", "inference.yml"):
        data = ("synthetic-fixture-" + filename).encode()
        (directory / filename).write_bytes(data)
        hashes[filename] = hashlib.sha256(data).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(hashes))
    assert model_directory(directory.name) == str(directory)
    (directory / "inference.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="onnx_recognition_asset_unverified"):
        model_directory(directory.name)


def test_synthetic_qr_fields_require_official_host_and_no_conflict():
    path = "/000000000000000123456789/20000101/P/000001"
    assert extract_official_fields(["https://qr.ine.mx" + path]) == {"cic": "123456789"}
    assert extract_official_fields(["https://qr.ine.mx.example.org" + path]) == {}
    assert extract_official_fields(["https://qr.ine.mx" + path, "https://qr.ine.mx" + path.replace("123456789", "987654321")]) == {}


def test_credentials_are_unique_private_and_not_overwritten(tmp_path, capsys):
    specification = importlib.util.spec_from_file_location("setup_env", ROOT / "scripts/setup_env.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    target = tmp_path / ".env"
    module.create_environment(target)
    content = target.read_text()
    assert len(content.split("=", 1)[1].strip()) == 64
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert content.split("=", 1)[1].strip() not in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        module.create_environment(target)
    assert target.read_text() == content


def test_default_deployment_has_no_custom_weights_and_is_loopback_only():
    import yaml

    config = yaml.safe_load((ROOT / "compose.yaml").read_text())["services"]["ocr"]
    assert config["ports"] == ["127.0.0.1:8100:8100"]
    assert config["read_only"] and config["cap_drop"] == ["ALL"]
    assert config["mem_limit"] == "6g"
    assert config["cpus"] == 2
    assert not any("MODEL_PATH" in key for key in config["environment"])
    allowlist = (ROOT / ".dockerignore").read_text().splitlines()
    assert allowlist[0] == "**"
    assert not any(line in {"!.git", "!.env", "!artifacts/**", "!models/**"} for line in allowlist)
