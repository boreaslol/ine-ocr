"""CPU-only API for full-document INE extraction and an experimental line model."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import logging
import os
import re
import threading
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .artifacts import sha256_file
from .runtime import OnnxMachineLineRecognizer
from .transport import RequestGate

MAX_LINE_IMAGE_BYTES = int(os.getenv("INE_OCR_MAX_LINE_IMAGE_BYTES", str(5 * 1024 * 1024)))
MAX_DOCUMENT_IMAGE_BYTES = int(
    os.getenv("INE_OCR_MAX_DOCUMENT_IMAGE_BYTES", str(15 * 1024 * 1024))
)
MODEL_PATH = os.getenv("INE_OCR_MODEL_PATH", "")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
REQUEST_LOGGER = logging.getLogger("ine_ocr.request")
if not REQUEST_LOGGER.handlers:
    request_handler = logging.StreamHandler()
    request_handler.setFormatter(logging.Formatter("%(message)s"))
    REQUEST_LOGGER.addHandler(request_handler)
REQUEST_LOGGER.setLevel(logging.INFO)
REQUEST_LOGGER.propagate = False


def _valid_api_token(value: str) -> bool:
    return (
        32 <= len(value.encode("utf-8")) <= 512
        and value == value.strip()
        and all(character.isprintable() and not character.isspace() for character in value)
    )


class RecognitionRequest(BaseModel):
    image: str = Field(min_length=4)
    lineType: str = Field(default="unknown", pattern=r"^(unknown|curp|mrz)$")


class DocumentRecognitionRequest(BaseModel):
    id: str = Field(min_length=4)
    idReverso: str | None = Field(default=None, min_length=4)


def _decode_image(value: str, *, max_bytes: int) -> bytes:
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="invalid_base64_image") from error
    if not payload or len(payload) > max_bytes:
        raise HTTPException(status_code=413, detail="image_size_out_of_range")
    return payload


class _LazyDocumentPipeline:
    def __init__(self) -> None:
        consensus = os.getenv("INE_OCR_CROSS_CHANNEL_NAME_CONSENSUS", "0")
        if consensus not in {"0", "1"}:
            raise ValueError("name_consensus_flag_invalid")
        self.cross_channel_name_consensus = consensus == "1"
        self.name_model_path = os.getenv("INE_OCR_NAME_MODEL_PATH", "")
        self.expected_name_model_sha256 = os.getenv("INE_OCR_EXPECTED_NAME_MODEL_SHA256", "")
        self.expected_name_sidecar_sha256 = os.getenv("INE_OCR_EXPECTED_NAME_SIDECAR_SHA256", "")
        for digest in (self.expected_name_model_sha256, self.expected_name_sidecar_sha256):
            if digest and re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("expected_name_model_sha256_invalid")
        if bool(self.expected_name_model_sha256) != bool(self.expected_name_sidecar_sha256):
            raise ValueError("name_release_requires_weight_and_sidecar_pins")
        self._pipeline = None
        self._lock = threading.Lock()
        self._ready = False
        self._ready_lock = threading.Lock()
        self._inference_slot = threading.BoundedSemaphore(1)

    def _get(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    from .document_pipeline import DocumentPipeline
                    self._pipeline = (
                        DocumentPipeline(cross_channel_name_consensus=True)
                        if self.cross_channel_name_consensus else DocumentPipeline()
                    )
        return self._pipeline

    def extract(self, front: bytes, back: bytes | None = None) -> dict:
        if not self._inference_slot.acquire(blocking=False):
            raise RuntimeError("document_pipeline_busy")
        try:
            self.ready()
            return self._get().extract(front, back)
        finally:
            self._inference_slot.release()

    def ready(self) -> None:
        if self._ready:
            return
        with self._ready_lock:
            if self._ready:
                return
            try:
                if os.getenv("INE_OCR_PROFILE") == "r2-v1":
                    from .r2_profile import verify_runtime
                    verify_runtime()
                if self.expected_name_model_sha256 and (
                    not self.name_model_path
                    or sha256_file(self.name_model_path) != self.expected_name_model_sha256
                    or sha256_file(self.name_model_path + ".json") != self.expected_name_sidecar_sha256
                ):
                    raise RuntimeError("name_model_release_hash_mismatch")
                self._get().ready()
            except Exception as error:
                raise RuntimeError("document_pipeline_not_ready") from error
            self._ready = True


def _request_id(value: str | None) -> str:
    return value if value and REQUEST_ID_PATTERN.fullmatch(value) else uuid4().hex


def _request_log(payload: dict) -> None:
    REQUEST_LOGGER.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def create_app(
    model_path: str | None = None,
    *,
    document_pipeline=None,
    api_bearer_token: str | None = None,
) -> FastAPI:
    selected_model = model_path or MODEL_PATH
    selected_token = (
        os.getenv("INE_OCR_API_BEARER_TOKEN", "")
        if api_bearer_token is None
        else api_bearer_token
    )
    bind_host = os.getenv("INE_OCR_HOST", "127.0.0.1")
    if bind_host not in {"127.0.0.1", "localhost", "::1"} and not selected_token:
        raise ValueError("api_bearer_token_required_for_external_bind")
    if selected_token and not _valid_api_token(selected_token):
        raise ValueError("api_bearer_token_invalid")
    recognizer = OnnxMachineLineRecognizer(selected_model) if selected_model else None
    selected_document_pipeline = document_pipeline or _LazyDocumentPipeline()
    app = FastAPI(title="INE OCR", version="0.2.0")
    app.add_middleware(RequestGate, bearer_token=selected_token)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        return JSONResponse({"detail": "invalid_request"}, status_code=422)

    @app.middleware("http")
    async def audit_request(http_request: Request, call_next):
        started = time.perf_counter()
        request_id = _request_id(http_request.headers.get("X-Request-ID"))
        http_request.state.request_id = request_id
        status_code = 500
        try:
            response = await call_next(http_request)
            status_code = response.status_code
        except Exception:
            response = None
            raise
        finally:
            audit = getattr(http_request.state, "audit", {})
            _request_log({
                "event": "request",
                "requestId": request_id,
                "method": http_request.method,
                "path": http_request.url.path if http_request.url.path in {
                    "/healthz", "/readyz", "/v1/ine/extract", "/v1/ine/recognize-machine-line"
                } else "unmatched",
                "status": status_code,
                "elapsedMs": round((time.perf_counter() - started) * 1000, 2),
                **audit,
            })
        response.headers["X-Request-ID"] = request_id
        return response

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if not selected_token:
            return
        scheme, _, supplied = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, selected_token):
            raise HTTPException(
                status_code=401,
                detail="unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/healthz")
    def health():
        return {
            "ok": True,
            "profile": os.getenv("INE_OCR_PROFILE", "custom"),
            "documentPipeline": "configured",
            "lineRecognizer": "configured" if recognizer is not None else "not_configured",
            "nameModel": "configured" if os.getenv("INE_OCR_NAME_MODEL_PATH") else "disabled",
            "crossChannelNameConsensus": getattr(
                selected_document_pipeline, "cross_channel_name_consensus", False
            ),
            "expectedNameModelSha256": getattr(
                selected_document_pipeline, "expected_name_model_sha256", ""
            ) or None,
            "expectedNameSidecarSha256": getattr(
                selected_document_pipeline, "expected_name_sidecar_sha256", ""
            ) or None,
            "runtime": "cpu",
            "release": {
                "commit": os.getenv("RELEASE_COMMIT", "uncommitted"),
                "tag": os.getenv("RELEASE_TAG", "local"),
                "artifactSha256": os.getenv("RELEASE_ARTIFACT_SHA256", "unknown"),
                "buildTime": os.getenv("RELEASE_BUILD_TIME", "unknown"),
            },
        }

    @app.get("/readyz")
    def ready():
        try:
            ready_method = getattr(selected_document_pipeline, "ready", None)
            if callable(ready_method):
                ready_method()
        except Exception as error:
            REQUEST_LOGGER.error("document_pipeline_not_ready")
            raise HTTPException(status_code=503, detail="document_pipeline_not_ready") from error
        return {
            "ok": True,
            "documentPipeline": "ready",
            "profile": os.getenv("INE_OCR_PROFILE", "custom"),
            "lineRecognizer": "configured" if recognizer is not None else "not_configured",
            "nameModel": "configured" if os.getenv("INE_OCR_NAME_MODEL_PATH") else "disabled",
            "crossChannelNameConsensus": getattr(
                selected_document_pipeline, "cross_channel_name_consensus", False
            ),
            "expectedNameModelSha256": getattr(
                selected_document_pipeline, "expected_name_model_sha256", ""
            ) or None,
            "expectedNameSidecarSha256": getattr(
                selected_document_pipeline, "expected_name_sidecar_sha256", ""
            ) or None,
            "runtime": "cpu",
            "release": {
                "commit": os.getenv("RELEASE_COMMIT", "uncommitted"),
                "tag": os.getenv("RELEASE_TAG", "local"),
                "artifactSha256": os.getenv("RELEASE_ARTIFACT_SHA256", "unknown"),
                "buildTime": os.getenv("RELEASE_BUILD_TIME", "unknown"),
            },
        }

    @app.post("/v1/ine/extract", dependencies=[Depends(authorize)])
    async def extract_document(request: DocumentRecognitionRequest, http_request: Request):
        started = time.perf_counter()
        front = _decode_image(request.id, max_bytes=MAX_DOCUMENT_IMAGE_BYTES)
        back = (
            _decode_image(request.idReverso, max_bytes=MAX_DOCUMENT_IMAGE_BYTES)
            if request.idReverso
            else None
        )
        http_request.state.audit = {
            "imageCount": 2 if back is not None else 1,
            "frontBytes": len(front),
            "backBytes": len(back) if back is not None else 0,
        }
        try:
            result = await run_in_threadpool(selected_document_pipeline.extract, front, back)
        except ValueError as error:
            raise HTTPException(status_code=400, detail="invalid_document_image") from error
        except RuntimeError as error:
            headers = {"Retry-After": "1"} if str(error) == "document_pipeline_busy" else None
            detail = "document_pipeline_busy" if headers else "document_pipeline_unavailable"
            raise HTTPException(status_code=503, detail=detail, headers=headers) from error
        except Exception as error:
            REQUEST_LOGGER.error("document_pipeline_failed")
            raise HTTPException(status_code=503, detail="document_pipeline_unavailable") from error
        result["elapsedMs"] = round((time.perf_counter() - started) * 1000, 2)
        http_request.state.audit.update({
            "pipelineStatus": result.get("status"),
            "warningCount": len(result.get("warnings", [])),
            "structuredChannels": sum(
                bool(value) for value in (result.get("channels") or {}).values()
            ),
        })
        return result

    @app.post("/v1/ine/recognize-machine-line", dependencies=[Depends(authorize)])
    async def recognize(request: RecognitionRequest, http_request: Request):
        if recognizer is None:
            raise HTTPException(status_code=503, detail="model_not_configured")
        started = time.perf_counter()
        image = _decode_image(request.image, max_bytes=MAX_LINE_IMAGE_BYTES)
        http_request.state.audit = {
            "imageCount": 1,
            "frontBytes": len(image),
            "lineType": request.lineType,
        }
        try:
            result = await run_in_threadpool(recognizer.recognize, image, request.lineType)
        except ValueError as error:
            raise HTTPException(status_code=400, detail="invalid_line_image") from error
        except Exception as error:
            raise HTTPException(status_code=503, detail="line_model_unavailable") from error
        result["elapsedMs"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    return app


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run("ine_ocr.api:app", host=os.getenv("INE_OCR_HOST", "127.0.0.1"),
                port=int(os.getenv("INE_OCR_PORT", "8100")), workers=1, access_log=False)
