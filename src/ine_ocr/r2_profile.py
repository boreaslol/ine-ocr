"""Frozen R2 model identity and inference configuration, without deployment secrets."""

from importlib.resources import files
import json
import os
from pathlib import Path

from .artifacts import sha256_file


def contract():
    return json.loads(files("ine_ocr").joinpath("r2-profile.json").read_text())


def model_root():
    return Path(os.getenv("INE_OCR_R2_MODEL_DIR", "/opt/models/r2")).expanduser().resolve()


def runtime_environment(root):
    profile = contract()
    root = Path(root).expanduser().resolve()
    return {
        **profile["configuration"],
        **{key: str(root / value) for key, value in profile["paths"].items()},
        "INE_OCR_PROFILE": "r2-v1",
        "INE_OCR_R2_MODEL_DIR": str(root),
        "INE_OCR_EXPECTED_NAME_MODEL_SHA256": profile["files"]["name/ine-names.onnx"]["sha256"],
        "INE_OCR_EXPECTED_NAME_SIDECAR_SHA256": profile["files"]["name/ine-names.onnx.json"]["sha256"],
    }


def verify_models(root=None):
    root = Path(root).resolve() if root is not None else model_root()
    verified = {}
    for name, entry in contract()["files"].items():
        target = root / name
        if target.is_symlink() or not target.is_file() or target.stat().st_size != entry["bytes"] or sha256_file(str(target)) != entry["sha256"]:
            raise RuntimeError("r2_asset_identity_mismatch:" + name)
        verified[name] = entry["sha256"]
    return verified


def verify_runtime():
    for key, value in runtime_environment(model_root()).items():
        if os.getenv(key) != value:
            raise RuntimeError("r2_configuration_mismatch:" + key)
    for key in ("INE_OCR_FIELD_SELECTOR", "INE_OCR_MODEL_PATH", "INE_OCR_TEXT_DET_LIMIT_SIDE_LEN", "INE_OCR_TEXT_DET_LIMIT_TYPE"):
        if os.getenv(key):
            raise RuntimeError("r2_unsupported_override:" + key)
    return verify_models()


def serve():
    for key, value in runtime_environment(model_root()).items():
        if key in os.environ and os.environ[key] != value:
            raise RuntimeError("r2_configuration_mismatch:" + key)
        os.environ[key] = value
    verify_runtime()
    from .api import main
    main()
