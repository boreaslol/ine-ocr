"""Local QR decoding with WeChatQRCode and OpenCV fallback."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def extract_official_fields(values: list[str]) -> dict[str, str]:
    cics = set()
    for value in values:
        try:
            parsed = urlparse(value)
        except ValueError:
            continue
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or not (
            hostname == "qr.ine.mx" or hostname.endswith(".qr.ine.mx")
        ):
            continue
        parts = [part for part in parsed.path.split("/") if part]
        if parts and len(parts[0]) == 24 and parts[0].isdigit():
            cics.add(parts[0][-9:])
    return {"cic": next(iter(cics))} if len(cics) == 1 else {}


class QRDecoder:
    def __init__(self, model_dir: str | None = None) -> None:
        self.model_dir = Path(
            model_dir or os.getenv("INE_OCR_WECHAT_QR_MODEL_DIR", "~/.cache/wechat_qrcode")
        ).expanduser()
        self._wechat = None
        self._wechat_checked = False

    def _get_wechat(self):
        if self._wechat_checked:
            return self._wechat
        self._wechat_checked = True
        try:
            import cv2
            paths = {
                "detect_proto": self.model_dir / "detect_2021nov.prototxt",
                "detect_model": self.model_dir / "detect_2021nov.caffemodel",
                "sr_proto": self.model_dir / "sr_2021nov.prototxt",
                "sr_model": self.model_dir / "sr_2021nov.caffemodel",
            }
            if hasattr(cv2, "wechat_qrcode_WeChatQRCode") and all(path.is_file() for path in paths.values()):
                self._wechat = cv2.wechat_qrcode_WeChatQRCode(
                    str(paths["detect_proto"]),
                    str(paths["detect_model"]),
                    str(paths["sr_proto"]),
                    str(paths["sr_model"]),
                )
        except Exception:
            self._wechat = None
        return self._wechat

    def ready(self) -> None:
        if self._get_wechat() is None:
            raise RuntimeError("qr_decoder_not_ready")

    def decode(self, image) -> list[str]:
        try:
            import cv2
        except ImportError:
            return []
        decoded: list[str] = []
        wechat = self._get_wechat()
        if wechat is not None:
            try:
                texts, _ = wechat.detectAndDecode(image)
                decoded.extend(str(text) for text in texts if text)
            except Exception:
                pass
        if not decoded:
            try:
                ok, values, _, _ = cv2.QRCodeDetector().detectAndDecodeMulti(image)
                if ok:
                    decoded.extend(str(value) for value in values if value)
            except Exception:
                pass
        return list(dict.fromkeys(decoded))
