import base64
import sys
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from ine_ocr import api
from ine_ocr.api import create_app


class StubDocumentPipeline:
    def extract(self, front, back=None):
        return {"status": "OK", "frontBytes": len(front), "hasBack": back is not None}


class ReadyDocumentPipeline(StubDocumentPipeline):
    def __init__(self):
        self.ready_calls = 0

    def ready(self):
        self.ready_calls += 1


def test_document_endpoint_accepts_two_images_without_logging_content():
    client = TestClient(create_app(document_pipeline=StubDocumentPipeline()))
    encoded = base64.b64encode(b"synthetic-image").decode()
    response = client.post("/v1/ine/extract", json={"id": encoded, "idReverso": encoded})
    assert response.status_code == 200
    assert response.json()["hasBack"] is True
    assert response.headers["X-Request-ID"]


def test_ready_endpoint_loads_document_pipeline():
    pipeline = ReadyDocumentPipeline()
    client = TestClient(create_app(document_pipeline=pipeline))
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["documentPipeline"] == "ready"
    assert pipeline.ready_calls == 1


def test_lazy_ready_warms_underlying_document_pipeline(monkeypatch):
    calls = []

    class FakePipeline:
        def ready(self):
            calls.append("ready")

    module = ModuleType("ine_ocr.document_pipeline")
    module.DocumentPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "ine_ocr.document_pipeline", module)

    lazy = api._LazyDocumentPipeline()
    lazy.ready()
    lazy.ready()

    assert calls == ["ready"]


def test_document_endpoint_rejects_invalid_base64():
    client = TestClient(create_app(document_pipeline=StubDocumentPipeline()))
    response = client.post("/v1/ine/extract", json={"id": "not-base64"})
    assert response.status_code == 400


def test_document_endpoint_uses_document_specific_size_limit(monkeypatch):
    monkeypatch.setattr(api, "MAX_LINE_IMAGE_BYTES", 4)
    monkeypatch.setattr(api, "MAX_DOCUMENT_IMAGE_BYTES", 8)
    client = TestClient(create_app(document_pipeline=StubDocumentPipeline()))
    encoded = base64.b64encode(b"123456").decode()
    response = client.post("/v1/ine/extract", json={"id": encoded})
    assert response.status_code == 200
    assert response.json()["frontBytes"] == 6


def test_document_endpoint_rejects_document_over_its_limit(monkeypatch):
    monkeypatch.setattr(api, "MAX_DOCUMENT_IMAGE_BYTES", 4)
    client = TestClient(create_app(document_pipeline=StubDocumentPipeline()))
    encoded = base64.b64encode(b"123456").decode()
    response = client.post("/v1/ine/extract", json={"id": encoded})
    assert response.status_code == 413


def test_document_endpoint_requires_configured_bearer_token():
    token = "a" * 32
    client = TestClient(
        create_app(document_pipeline=StubDocumentPipeline(), api_bearer_token=token)
    )
    encoded = base64.b64encode(b"synthetic-image").decode()
    missing = client.post("/v1/ine/extract", json={"id": encoded})
    accepted = client.post(
        "/v1/ine/extract",
        headers={"Authorization": f"Bearer {token}"},
        json={"id": encoded},
    )
    assert missing.status_code == 401
    assert accepted.status_code == 200


def test_external_bind_requires_token(monkeypatch):
    monkeypatch.setenv("INE_OCR_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="api_bearer_token_required_for_external_bind"):
        create_app(document_pipeline=StubDocumentPipeline(), api_bearer_token="")


def test_validation_and_pipeline_errors_do_not_echo_input():
    class FailingPipeline:
        def extract(self, front, back=None):
            raise ValueError("synthetic-private-data")

    client = TestClient(create_app(document_pipeline=FailingPipeline()))
    response = client.post("/v1/ine/extract", json={"id": {"secret": "synthetic-private-data"}})
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_request"}
    response = client.post("/v1/ine/extract", json={"id": base64.b64encode(b"synthetic-private-data").decode()})
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid_document_image"}


def test_ready_is_authenticated_when_token_is_set():
    pipeline = ReadyDocumentPipeline()
    token = "synthetic-test-token-" * 3
    client = TestClient(create_app(document_pipeline=pipeline, api_bearer_token=token))
    assert client.get("/readyz").status_code == 401
    assert pipeline.ready_calls == 0
    assert client.get("/readyz", headers={"Authorization": "Bearer " + token}).status_code == 200
    assert pipeline.ready_calls == 1
