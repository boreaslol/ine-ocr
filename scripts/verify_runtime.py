"""Verify public assets and write a non-secret, build-specific runtime manifest."""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform

from fetch_wechat_qr_models import EXPECTED


ROOT = Path(__file__).resolve().parents[1]


def verify_assets():
    expected = json.loads((ROOT / "deploy/public-models.json").read_text())["sha256"]
    paddle_root = Path(os.getenv("PADDLE_PDX_CACHE_HOME", "~/.paddlex")).expanduser() / "official_models"
    qr_root = Path(os.getenv("INE_OCR_WECHAT_QR_MODEL_DIR", "~/.cache/wechat_qrcode")).expanduser()
    verified = {}
    for family, directory, hashes in (("paddle", paddle_root, expected), ("wechat", qr_root, EXPECTED)):
        for name, wanted in hashes.items():
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if actual != wanted:
                raise RuntimeError("public_model_checksum_mismatch")
            verified[family + "/" + name] = actual
    return verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = {
        "profile": "public-base-v1", "release_commit": os.getenv("RELEASE_COMMIT", "uncommitted"),
        "release_tag": os.getenv("RELEASE_TAG", "local"), "build_time": os.getenv("RELEASE_BUILD_TIME", "unknown"),
        "python": platform.python_version(), "entrypoint": "ine_ocr.api:app", "port": 8100,
        "card_detector": "contour", "private_weights_included": False,
        "dependencies": {package: metadata.version(package) for package in (
            "numpy", "Pillow", "paddlepaddle", "paddleocr", "paddlex", "opencv-contrib-python", "onnxruntime", "fastapi", "uvicorn"
        )}, "model_sha256": verify_assets(),
    }
    if args.output:
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"profile": manifest["profile"], "verified_assets": len(manifest["model_sha256"])}))


if __name__ == "__main__":
    main()
