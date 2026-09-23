# INE OCR

Self-hosted, CPU-only text extraction for Mexican INE documents. The **R2 release includes the trained name and card-detection models**, printed-text OCR, MRZ check digits, local QR decoding, and field-level provenance. No hosted OCR account or access to the authors' servers is required.

**This is the complete R2 inference edition, licensed AGPL-3.0-only.** It includes the previously retained R2 weights and frozen inference rules, but no identity images, private deployment configuration, credentials or training datasets. The earlier Apache-2.0 base-only edition remains available at [its original commit](https://github.com/boreaslol/ine-ocr/tree/972240cf27349e6db4b2734bc77447b25fed84bc). Treat results as editable form prefills, never as identity verification or proof that a document is genuine.

## Deploy

Supported container target: **Linux x86-64**, Docker Engine with Compose v2, Git, Python 3.11+ and Pillow on the host. Plan for 2 CPU cores, 6 GiB container memory, and several GiB of image/build storage. These are resource limits, not throughput guarantees. A GPU is not required. The first build downloads pinned Python packages, the roughly 61 MB R2 model bundle and public base models; subsequent OCR requests use local models.

```bash
git clone https://github.com/boreaslol/ine-ocr.git
cd ine-ocr
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'Pillow>=10.4,<13'
python scripts/setup_env.py
bash scripts/deploy.sh
```

`setup_env.py` creates a unique token in an owner-only `.env` and never prints it. Keep that file private. `deploy.sh` accepts only a clean commit contained in `origin/main`, builds on the deployment host, checks authenticated R2 readiness, performs a real synthetic-image inference, and records commit, image ID, dependencies and all model hashes in ignored `artifacts/`. It does **not** upload documents or deployment receipts anywhere. Missing or changed R2 weights fail verification; there is no silent base-model fallback.

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

- Fine-tuned YOLOv8n card detection, contour/natural-image fallbacks, orientation retries and quality warnings.
- Public PaddleOCR small text detector and pinned ONNX text recognition.
- R2 fine-tuned name CRNN, original vocabulary/preprocessing, 0.9 admission threshold and protected-source/cross-channel fusion.
- CURP format/check-digit checks, MRZ parsing/checks, and local OpenCV WeChat QR decoding. Decoded links are **not fetched**.
- Editable tensor-only source weights, architecture descriptions and model export tooling; no executable pickle checkpoints.
- Synthetic tests, authenticated upload limits, readiness checks, version receipts, and a CI container smoke test.

R2 reached **92.522% joint normalized-name agreement** with third-party reference outputs on a frozen 1,003-document development set. This is not human-adjudicated accuracy or a guarantee on your inputs. See the [model card](docs/MODEL_CARD.md) for denominators, limitations and the distinction between inference reproduction and retraining. Name spelling, accents, truncation and multi-part surnames require user confirmation. `fieldConfidence` is not a calibrated correctness probability.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]' opencv-contrib-python==4.10.0.84
python -m pytest -q
```

Unit tests use synthetic data and do not download models. To exercise the real CPU service, use the container deployment above. Container CI builds from the public commit and runs real R2 inference on synthetic content; it is not a benchmark on identity documents. The evaluated runtime packages are pinned in `deploy/requirements.r2.txt` and the Python base is digest-pinned. Preserve the verified image ID for rollback rather than rebuilding a source revision and assuming identical bytes. See [editable model sources](docs/MODEL_SOURCES.md) for safe restoration and export.

## Privacy and licensing

Never commit scans, OCR outputs, credentials or real-person fixtures, including in issues and pull requests. By default this service has no database and no image retention: OCR intermediate PNGs are removed after use and live on container tmpfs. It logs request IDs, timing, sizes and statuses, not image content or recognized fields. Use non-personal request IDs; upstream proxies and application logs need their own privacy controls.

Full R2 distribution: [AGPL-3.0-only](LICENSE). Upstream models and libraries retain their own licenses; see [third-party notices](THIRD_PARTY.md). This project is independent and is not an official INE service. Read the AGPL requirements before modifying/serving it or embedding it into another application.

## 中文说明

这是包含 **R2 已训练权重及完整推理规则**的可部署版本，不再是只有公开基础模型的简化版。完整发行版采用 AGPL-3.0；原 Apache 基础版保留在历史提交中。模型包不含真实证件、训练标签、生产地址或 token。按上述命令在自己的 Linux x86-64 服务器部署并生成自己的 token；默认仅监听本机。历史 92.522% 是特定样本的供应商姓名一致率，并非保证真实精度。跨机器调用须自行配置 HTTPS/内网及访问限制，结果必须允许用户核对修改。
