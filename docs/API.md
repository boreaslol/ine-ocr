# API contract

## Endpoints

| Method / path | Authentication | Meaning |
| --- | --- | --- |
| `GET /healthz` | None | Process liveness and release metadata; not model readiness |
| `GET /readyz` | Bearer | Loads/warms the configured OCR, preprocessor and QR decoder; 503 if unavailable |
| `POST /v1/ine/extract` | Bearer | Full document field extraction |
| `POST /v1/ine/recognize-machine-line` | Bearer | Optional BYOM experiment; returns 503 in the frozen R2 profile |

Extraction body is JSON: `id` is the required front image in base64; `idReverso` is an optional back image in base64. A base64 data URI is also accepted. Remote URLs and PDF files are not an input contract; use decoded raster images such as JPEG or PNG.

Limits: 15 MiB decoded bytes per document image, 50 million decoded pixels per image, 44 MiB total request body, 30 seconds for upload, at most two admitted uploads/requests and one document inference per process. The default container starts one worker. Set a caller response timeout suitable for CPU inference (the example uses 120 seconds); this is not a latency SLA. Front-and-back orientation retries can take longer than a front-only synthetic check.

Authentication and total body limits run before JSON parsing. Invalid JSON/schema responses do not echo the submitted input. Body limits also cover chunked uploads. The configured token must be 32–512 bytes with no whitespace. The supported header format is exactly `Authorization: Bearer <token>`.

## Result

`status: "OK"` means the extraction pipeline returned, **not that required fields were found or are correct**. A valid image can return 200 with missing fields and warnings. Consumers must inspect fields and channels, request corrections or a new image as needed, and never treat missing fields as successful verification.

Depending on evidence, fields may include `nombres`, `primerApellido`, `segundoApellido`, `curp`, `cic`, `fechaNacimiento`, `claveElector`, `vigencia`, `calle`, `colonia` and `ciudad`. Unavailable fields are omitted rather than invented. Dates extracted into `fechaNacimiento` use `DD/MM/YYYY`.

- `channels`: availability/check results for front CURP, MRZ and QR.
- `fieldSources`: which extraction channel supplied each populated field.
- `fieldConfidence`: rule/model-derived ranking hints, **not calibrated probabilities**.
- `imageQuality`: blur, brightness and glare indicators for each side.
- `processing`: orientation/source/attempt diagnostics.
- `warnings`: optional list of quality, channel failure or disagreement warnings.
- `elapsedMs`: server extraction timing, excluding client upload/network time.

Names that agree after normalization can still differ in accents, spacing or spelling; neither such agreement nor a valid check digit proves a real person's identity. Do not turn these diagnostics into a user acceptance rate without a representative, consented evaluation.

## Errors and retries

| Status | Meaning / action |
| --- | --- |
| 400 | Invalid base64/image/content-length; fix the request |
| 401 | Missing or incorrect token |
| 408 | Upload timed out; retry a smaller/faster upload |
| 413 | Image or body exceeds byte limits |
| 422 | Malformed JSON/schema; generic `invalid_request`, no input echo |
| 503 + `Retry-After: 1` | Inference/admission busy; bounded exponential backoff with jitter |
| Other 503 | Models/pipeline unavailable; check authenticated readiness and operators' logs |

Use a small finite retry budget for 408/503; never retry invalid credentials or schema indefinitely. `X-Request-ID` accepts 1–64 ASCII letters/digits/`.`/`_`/`-`; otherwise a random ID is generated. These IDs are logged, so never put names, identity numbers or secrets in them. Responses contain personal data; disable body capture in tracing/proxies.
