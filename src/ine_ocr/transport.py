"""Authenticate and bound document uploads before JSON parsing."""

import asyncio
import hmac

from starlette.responses import JSONResponse


class RequestGate:
    def __init__(self, app, *, bearer_token="", max_bytes=44 * 1024 * 1024,
                 max_inflight=2, upload_timeout=30):
        self.app = app
        self.token = bearer_token
        self.max_bytes = max_bytes
        self.slots = asyncio.Semaphore(max_inflight)
        self.upload_timeout = upload_timeout

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        protected = scope["path"].startswith("/v1/") or scope["path"] == "/readyz"
        if not protected:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))

        async def reject(status, detail, response_headers=None):
            response = JSONResponse({"detail": detail}, status_code=status, headers=response_headers)
            await response(scope, receive, send)

        if self.token:
            supplied = headers.get(b"authorization", b"")
            expected = ("Bearer " + self.token).encode()
            if not hmac.compare_digest(supplied, expected):
                return await reject(401, "unauthorized", {"WWW-Authenticate": "Bearer"})
        if scope["method"] != "POST":
            return await self.app(scope, receive, send)
        try:
            declared = int(headers.get(b"content-length", b"0"))
            if declared < 0:
                raise ValueError
        except ValueError:
            return await reject(400, "invalid_content_length")
        if declared > self.max_bytes:
            return await reject(413, "request_body_too_large")
        if self.slots.locked():
            return await reject(503, "document_pipeline_busy", {"Retry-After": "1"})
        async with self.slots:
            body = bytearray()
            try:
                async with asyncio.timeout(self.upload_timeout):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        chunk = message.get("body", b"")
                        if len(body) + len(chunk) > self.max_bytes:
                            return await reject(413, "request_body_too_large")
                        body.extend(chunk)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                return await reject(408, "upload_timeout")
            delivered = False

            async def buffered_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, buffered_receive, send)
