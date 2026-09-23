#!/usr/bin/env python3
"""Fetch and verify the fixed WeChatQRCode runtime model set."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"
RAW_BASE_URL = f"https://raw.githubusercontent.com/opencv/opencv_zoo/{REVISION}/models/qrcode_wechatqrcode"
MEDIA_BASE_URL = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/models/qrcode_wechatqrcode"
EXPECTED = {
    "detect_2021nov.prototxt": "e8acfc395caf443a47f15686a9b9207b36cb8f7e6ceb8fbaf6466665e68a9466",
    "detect_2021nov.caffemodel": "cc49b8c9babaf45f3037610fe499df38c8819ebda29e90ca9f2e33270f6ef809",
    "sr_2021nov.prototxt": "8ae41acba97e8b4a8e741ee350481e49b8e01d787193f470a4c95ee1c02d5b61",
    "sr_2021nov.caffemodel": "e5d36889d8e6ef2f1c1f515f807cec03979320ac81792cd8fb927c31fd658ae3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_url(name: str) -> str:
    base_url = RAW_BASE_URL if name.endswith(".prototxt") else MEDIA_BASE_URL
    return f"{base_url}/{name}"


def main() -> int:
    output = Path(
        os.getenv("INE_OCR_WECHAT_QR_MODEL_DIR", "~/.cache/wechat_qrcode")
    ).expanduser()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    for name, expected in EXPECTED.items():
        destination = output / name
        if destination.is_file() and sha256(destination) == expected:
            continue
        descriptor, temporary_name = tempfile.mkstemp(prefix=f"{name}.", dir=output)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with urllib.request.urlopen(model_url(name), timeout=30) as source, temporary.open("wb") as target:
                total = 0
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > 32 * 1024 * 1024:
                        raise RuntimeError("model_download_size_limit")
                    target.write(chunk)
            if sha256(temporary) != expected:
                raise RuntimeError(f"wechat_qr_checksum_mismatch:{name}")
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    print({"modelDir": str(output), "verified": len(EXPECTED)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
