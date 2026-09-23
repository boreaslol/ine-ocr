"""Exercise public models with a synthetic image; never print credentials or fields."""

import base64
import io
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont


def synthetic_image():
    image = Image.new("RGB", (1000, 630), "white")
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((8, 8, 990, 620), outline="black", width=3)
    font = ImageFont.load_default(size=34)
    for offset, text in enumerate(("SYNTHETIC TEST", "NOMBRE", "PRUEBA", "CONTROL", "PERSONA SINTETICA", "DOMICILIO")):
        drawing.text((260, 50 + offset * 75), text, fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def local_token():
    token = os.getenv("INE_OCR_API_BEARER_TOKEN", "")
    if not token:
        content = (Path(__file__).resolve().parents[1] / ".env").read_text()
        tokens = [line.partition("=")[2] for line in content.splitlines() if line.startswith("INE_OCR_API_BEARER_TOKEN=")]
        if len(tokens) != 1:
            raise RuntimeError("local_credential_missing")
        token = tokens[0]
    if not token:
        raise RuntimeError("local_credential_missing")
    return token


def main():
    token = local_token()
    endpoint = "http://127.0.0.1:8100"

    def call(path, *, body=None, supplied=token):
        headers = {"Authorization": "Bearer " + supplied}
        data = None if body is None else json.dumps(body).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(endpoint + path, data=data, headers=headers)
        try:
            with urlopen(request, timeout=180) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    checks = (
        (call("/healthz")[0] == 200, "health"),
        (call("/readyz")[0] == 200, "readiness"),
        (call("/readyz", supplied="wrong-token")[0] == 401, "readiness_auth"),
        (call("/v1/ine/extract", supplied="wrong-token", body={"id": "invalid"})[0] == 401, "extraction_auth"),
        (call("/v1/ine/extract", body={"id": "invalid"})[0] == 400, "invalid_image"),
    )
    for passed, name in checks:
        if not passed:
            raise RuntimeError("smoke_failed:" + name)
    started = time.monotonic()
    status, result = call("/v1/ine/extract", body={"id": base64.b64encode(synthetic_image()).decode()})
    if status != 200 or result.get("status") != "OK":
        raise RuntimeError("synthetic_extraction_failed")
    if any("ENGINE_UNAVAILABLE" in warning for warning in result.get("warnings", [])):
        raise RuntimeError("synthetic_engine_unavailable")
    if not any(result.get(field) for field in ("nombres", "primerApellido", "segundoApellido")):
        raise RuntimeError("synthetic_text_not_extracted")
    print(json.dumps({"smoke": "passed", "input": "synthetic_in_memory", "elapsed_seconds": round(time.monotonic() - started, 2)}))


if __name__ == "__main__":
    main()
