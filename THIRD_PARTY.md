# Third-party components

This repository's license does not relicense its dependencies or externally downloaded models. Preserve upstream notices when redistributing containers or models. Installed Python distributions retain their package license files; this document lists the primary runtime components, not an exhaustive transitive dependency inventory.

| Component | Source | Upstream license |
| --- | --- | --- |
| PaddleOCR | https://github.com/PaddlePaddle/PaddleOCR | Apache-2.0; copy in `licenses/PaddleOCR-LICENSE.txt` |
| PaddlePaddle / PaddleX | https://github.com/PaddlePaddle/Paddle / https://github.com/PaddlePaddle/PaddleX | Apache-2.0 |
| PP-OCRv6 small detector | https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det | Model card declares Apache-2.0 |
| PP-OCRv6 small recognizer | https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec | Model card declares Apache-2.0 |
| OpenCV / WeChatQRCode | https://github.com/opencv/opencv / https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/qrcode_wechatqrcode | Apache-2.0; QR license copy in `licenses/WeChatQRCode-LICENSE.txt` |
| ONNX Runtime | https://github.com/microsoft/onnxruntime | MIT |
| FastAPI / Uvicorn | https://github.com/fastapi/fastapi / https://github.com/Kludex/uvicorn | MIT / BSD-3-Clause |
| NumPy / Pillow / PyYAML | https://numpy.org / https://python-pillow.org / https://pyyaml.org | BSD-3-Clause / HPND / MIT |

The repository does not redistribute weight binaries. Container builds download the two public Paddle models and four WeChat QR files and verify SHA-256 values from `deploy/public-models.json` and `scripts/fetch_wechat_qr_models.py`. An upstream change fails the verification rather than silently accepting replacement weights. Additional models selected by an operator require their own license and provenance review.
