# INE OCR

Self-hosted, CPU-only text extraction for Mexican INE documents. A small FastAPI service combines printed-text OCR, MRZ check digits, local QR decoding, and field-level provenance. No hosted OCR account is required.

**This is the public base-model edition, not a release of any private fine-tuned model.** It contains no identity images, production configuration, credentials, private weights, or training datasets. There is no measured accuracy guarantee for this edition. Treat results as editable form prefills, never as identity verification or proof that a document is genuine.

## Deploy

Supported container target: **Linux x86-64**, Docker Engine with Compose v2, Git, Python 3.11+ and Pillow on the host. Plan for 2 CPU cores, 6 GiB container memory, and several GiB of image/build storage. These are resource limits, not throughput guarantees. A GPU is not required. The first build downloads Python packages and public models; subsequent OCR requests use local models.

```bash
git clone https://github.com/boreaslol/ine-ocr.git
cd ine-ocr
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'Pillow>=10.4,<13'
python scripts/setup_env.py
bash scripts/deploy.sh
```

`setup_env.py` creates a unique token in an owner-only `.env` and never prints it. Keep that file private. `deploy.sh` accepts only a clean commit contained in `origin/main`, builds on the deployment host, checks authenticated readiness, performs a real synthetic-image inference, and records commit, image ID, dependencies and public-model hashes in ignored `artifacts/`. It does **not** upload documents or deployment receipts anywhere.

- Local endpoint: `http://127.0.0.1:8100/v1/ine/extract`.
- Authorization: `Authorization: Bearer <your locally generated token>`.
- OpenAPI: `http://127.0.0.1:8100/docs`.
- No real credentials are supplied by this repository. Each installation generates its own.

The container runs as a non-root user with a read-only filesystem, bounded memory/CPU/processes, bounded logs and an ephemeral `/tmp`. It publishes **localhost only**. For another backend to call it, add your own authenticated HTTPS reverse proxy or private-network tunnel, restrict callers, set request size/time/rate limits, and test the actual caller path. Do not change the binding to a public unauthenticated HTTP listener. See [operations](docs/OPERATIONS.md).

## Call the API

Set `INE_OCR_API_BEARER_TOKEN` in your caller's secret manager or environment. Pass local images, not URLs:

```python
import base64
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

payload = {"id": base64.b64encode(Path("front.jpg").read_bytes()).decode()}
request = Request(
    "http://127.0.0.1:8100/v1/ine/extract",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json",
             "Authorization": "Bearer " + os.environ["INE_OCR_API_BEARER_TOKEN"]},
)
with urlopen(request, timeout=120) as response:
    result = json.load(response)
```

Add `idReverso` with the back image when available. Avoid logging `payload` or `result`: both can contain personal information. See [API contract](docs/API.md) for limits, errors, missing fields and confidence semantics.

## Included

- Contour-based card extraction, orientation retries, image normalization and quality warnings.
- Public PaddleOCR `PP-OCRv6_small_det` and `PP-OCRv6_small_rec` models.
- CURP format/check-digit checks, MRZ parsing/checks, and local OpenCV WeChat QR decoding. Decoded links are **not fetched**.
- Optional bring-your-own ONNX hooks; no custom trained weights are included or required by the default deployment.
- Synthetic tests, authenticated upload limits, readiness checks, version receipts, and a CI container smoke test.

The contour fallback can be weaker on cluttered, skewed or poorly cropped photos than a trained card detector. Name spelling, accents, truncation and multi-part surnames require user confirmation. Channel agreement and `fieldConfidence` are heuristics, not calibrated correctness probabilities. There is no supplier-comparison score or user-acceptance claim for this edition.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]' opencv-contrib-python==4.10.0.84
python -m pytest -q
```

Unit tests use synthetic data and do not download models. To exercise the real CPU service, use the container deployment above. Container CI builds from the public commit and runs real public-model inference; it is not a benchmark on identity documents. Direct dependencies are pinned in `deploy/requirements.cpu.txt`; transitive packages and the OS base may evolve between builds. Preserve the verified image ID for rollback rather than rebuilding an old source revision and assuming identical bytes.

## Privacy and licensing

Never commit scans, OCR outputs, credentials or real-person fixtures, including in issues and pull requests. By default this service has no database and no image retention: OCR intermediate PNGs are removed after use and live on container tmpfs. It logs request IDs, timing, sizes and statuses, not image content or recognized fields. Use non-personal request IDs; upstream proxies and application logs need their own privacy controls.

Code: [Apache-2.0](LICENSE). Public models and libraries retain their own licenses; see [third-party notices](THIRD_PARTY.md). This project is independent and is not an official INE service.

## 中文说明

这是可独立部署的公开基础模型版，不包含内部训练权重、真实证件、生产地址或 token，也不继承内部版本的精度结论。按上述命令在自己的 Linux x86-64 服务器部署并生成自己的 token；默认仅监听本机。跨机器调用须自行配置 HTTPS/内网及访问限制。识别结果适合预填，必须允许用户核对修改，不可当作实名认证或证件真伪判断。
