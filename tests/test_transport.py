import asyncio
import json

import pytest
from starlette.responses import JSONResponse

from ine_ocr.transport import RequestGate


TOKEN = "synthetic-test-token-" * 3


async def invoke(gate, *, headers=None, chunks=None, receive=None, path="/v1/ine/extract"):
    messages = []
    pending = list(chunks if chunks is not None else [b"{}"])

    async def default_receive():
        chunk = pending.pop(0)
        return {"type": "http.request", "body": chunk, "more_body": bool(pending)}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "method": "POST", "path": path, "headers": headers or []}
    await gate(scope, receive or default_receive, send)
    status = next((message["status"] for message in messages if message["type"] == "http.response.start"), None)
    return status, messages


async def echo_size(scope, receive, send):
    message = await receive()
    await JSONResponse({"bytes": len(message["body"])})(scope, receive, send)


def test_unauthorized_upload_is_rejected_without_reading_body():
    async def forbidden_receive():
        raise AssertionError("unauthorized_body_read")

    status, _ = asyncio.run(invoke(RequestGate(echo_size, bearer_token=TOKEN), receive=forbidden_receive))
    assert status == 401


@pytest.mark.parametrize("declared, expected", [(b"-1", 400), (b"invalid", 400), (b"11", 413)])
def test_declared_size_rejected_before_reading(declared, expected):
    async def forbidden_receive():
        raise AssertionError("oversize_body_read")

    status, _ = asyncio.run(invoke(RequestGate(echo_size, max_bytes=10), headers=[(b"content-length", declared)], receive=forbidden_receive))
    assert status == expected


def test_chunked_limit_and_slot_recovery():
    async def scenario():
        gate = RequestGate(echo_size, max_bytes=5, max_inflight=1)
        assert (await invoke(gate, chunks=[b"123", b"456"]))[0] == 413
        status, messages = await invoke(gate, chunks=[b"12", b"345"])
        assert status == 200
        assert json.loads(messages[-1]["body"]) == {"bytes": 5}

    asyncio.run(scenario())


def test_timeout_disconnect_and_saturation_release_capacity():
    async def scenario():
        gate = RequestGate(echo_size, max_inflight=1, upload_timeout=0.01)

        async def slow_receive():
            await asyncio.sleep(1)

        async def disconnected_receive():
            return {"type": "http.disconnect"}

        assert (await invoke(gate, receive=slow_receive))[0] == 408
        assert (await invoke(gate, receive=disconnected_receive))[0] is None
        async with gate.slots:
            assert (await invoke(gate))[0] == 503
        assert (await invoke(gate))[0] == 200

    asyncio.run(scenario())


def test_authenticated_request_passes_and_wrong_case_does_not():
    gate = RequestGate(echo_size, bearer_token=TOKEN)
    assert asyncio.run(invoke(gate, headers=[(b"authorization", ("Bearer " + TOKEN).encode())]))[0] == 200
    assert asyncio.run(invoke(gate, headers=[(b"authorization", ("bearer " + TOKEN).encode())]))[0] == 401
