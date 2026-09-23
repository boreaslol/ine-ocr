"""Isolated text-recognition worker for fixed document regions."""

from __future__ import annotations

import os


_RECOGNIZER = None
_RECOGNITION_MODEL = None


def initialize(model_name: str, cpu_threads: int) -> None:
    global _RECOGNIZER, _RECOGNITION_MODEL
    if model_name not in {"PP-OCRv6_small_rec", "en_PP-OCRv5_mobile_rec"}:
        raise ValueError("unsupported_roi_sidecar_model")
    if not 1 <= cpu_threads <= 16:
        raise ValueError("roi_sidecar_cpu_threads_out_of_range")
    os.environ.setdefault("PADDLE_PDX_CPU_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from paddleocr import TextRecognition

    from .document_onnx import recognition_options

    _RECOGNIZER = TextRecognition(
        model_name=model_name,
        cpu_threads=cpu_threads,
        **recognition_options(model_name, cpu_threads),
    )
    pipeline = getattr(_RECOGNIZER, "paddlex_pipeline", None)
    runtime_pipeline = getattr(pipeline, "_pipeline", None)
    _RECOGNITION_MODEL = getattr(runtime_pipeline, "text_rec_model", None)


def recognize(crops: list) -> list[dict]:
    if _RECOGNIZER is None:
        raise RuntimeError("roi_sidecar_not_initialized")
    if _RECOGNITION_MODEL is not None:
        predictions = _RECOGNITION_MODEL.predict(crops, batch_size=16)
    else:
        predictions = _RECOGNIZER.predict(crops, batch_size=16)
    return [
        {
            "text": str(prediction["rec_text"]),
            "score": round(float(prediction["rec_score"]), 4),
        }
        for prediction in predictions
    ]
