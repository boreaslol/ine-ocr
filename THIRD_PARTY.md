# Third-party components

This repository's license does not relicense its dependencies or externally downloaded models. Preserve upstream notices when redistributing containers or models. Installed Python distributions retain their package license files; this document lists the primary runtime components, not an exhaustive transitive dependency inventory.

| Component | Source | Upstream license |
| --- | --- | --- |
| YOLOv8n card model / authoring implementation | https://github.com/ultralytics/ultralytics/tree/v8.4.137 | AGPL-3.0; the full R2 distribution follows AGPL-3.0-only |
| R2 name CRNN / released source weights | `src/ine_ocr/model.py` and the R2 model release | AGPL-3.0-only |
| PaddleOCR | https://github.com/PaddlePaddle/PaddleOCR | Apache-2.0; copy in `licenses/PaddleOCR-LICENSE.txt` |
| PaddlePaddle / PaddleX | https://github.com/PaddlePaddle/Paddle / https://github.com/PaddlePaddle/PaddleX | Apache-2.0 |
| PP-OCRv6 small detector | https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det | Model card declares Apache-2.0 |
| PP-OCRv6 small recognizer | https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec | Model card declares Apache-2.0 |
| OpenCV / WeChatQRCode | https://github.com/opencv/opencv / https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/qrcode_wechatqrcode | Apache-2.0; QR license copy in `licenses/WeChatQRCode-LICENSE.txt` |
| ONNX Runtime | https://github.com/microsoft/onnxruntime | MIT |
| FastAPI / Uvicorn | https://github.com/fastapi/fastapi / https://github.com/Kludex/uvicorn | MIT / BSD-3-Clause |
| NumPy / Pillow / PyYAML | https://numpy.org / https://python-pillow.org / https://pyyaml.org | BSD-3-Clause / HPND / MIT |

Git history does not embed weight binaries. GitHub Releases distributes the R2 model bundle, including its ONNX inference weights and editable SafeTensors/architecture files. Container builds verify that bundle against `src/ine_ocr/r2-profile.json`, and the public Paddle/QR assets against their separate manifests. Changed bytes fail verification. Additional models selected by an operator require their own license and provenance review.
