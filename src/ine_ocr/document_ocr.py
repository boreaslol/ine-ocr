"""Lazy PaddleOCR text-line adapter for the full-document pipeline."""

from __future__ import annotations

import os
import multiprocessing
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor

from .document_onnx import document_options


class OCREngineUnavailable(RuntimeError):
    pass


class PaddleLineReader:
    def __init__(self, tier: str | None = None) -> None:
        self.tier = tier or os.getenv("INE_OCR_PADDLE_TIER", "small")
        if self.tier not in {"tiny", "small", "medium"}:
            raise ValueError("unsupported_paddle_tier")
        self.cpu_threads = int(os.getenv("INE_OCR_PADDLE_CPU_THREADS", "2"))
        if not 1 <= self.cpu_threads <= 16:
            raise ValueError("paddle_cpu_threads_out_of_range")
        self.enable_mkldnn = os.getenv("INE_OCR_PADDLE_ENABLE_MKLDNN", "0").lower() in {
            "1", "true", "yes", "on"
        }
        detection_limit = os.getenv("INE_OCR_TEXT_DET_LIMIT_SIDE_LEN")
        self.text_detection_limit = int(detection_limit) if detection_limit else None
        if self.text_detection_limit is not None and not 320 <= self.text_detection_limit <= 4096:
            raise ValueError("text_detection_limit_out_of_range")
        self.text_detection_limit_type = (
            os.getenv("INE_OCR_TEXT_DET_LIMIT_TYPE", "max")
            if self.text_detection_limit is not None
            else None
        )
        if self.text_detection_limit_type is not None and self.text_detection_limit_type not in {
            "max", "min", "resize_long"
        }:
            raise ValueError("text_detection_limit_type_invalid")
        self.roi_sidecar_enabled = os.getenv("INE_OCR_FRONT_ROI_SIDECAR", "0").lower() in {
            "1", "true", "yes", "on"
        }
        self.roi_sidecar_model = os.getenv(
            "INE_OCR_FRONT_ROI_SIDECAR_MODEL", "en_PP-OCRv5_mobile_rec"
        )
        if self.roi_sidecar_enabled and self.roi_sidecar_model not in {
            "PP-OCRv6_small_rec", "en_PP-OCRv5_mobile_rec"
        }:
            raise ValueError("unsupported_roi_sidecar_model")
        sidecar_threads = int(os.getenv("INE_OCR_FRONT_ROI_SIDECAR_CPU_THREADS", "1"))
        if not 1 <= sidecar_threads <= 16:
            raise ValueError("roi_sidecar_cpu_threads_out_of_range")
        self.roi_sidecar_cpu_threads = sidecar_threads
        self.roi_sidecar_timeout_seconds = float(
            os.getenv("INE_OCR_FRONT_ROI_SIDECAR_TIMEOUT_SECONDS", "20")
        )
        if not 1 <= self.roi_sidecar_timeout_seconds <= 120:
            raise ValueError("roi_sidecar_timeout_out_of_range")
        self._engine = None
        self._lock = threading.Lock()
        self._roi_sidecar_executor = None
        self._roi_sidecar_lock = threading.Lock()
        self.last_roi_sidecar_stats = {}

    def _get_engine(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as error:
                raise OCREngineUnavailable("paddleocr_not_installed") from error
            detection_options = {}
            if self.text_detection_limit is not None:
                detection_options = {
                    "text_det_limit_side_len": self.text_detection_limit,
                    "text_det_limit_type": self.text_detection_limit_type,
                }
            self._engine = PaddleOCR(
                text_detection_model_name=f"PP-OCRv6_{self.tier}_det",
                text_recognition_model_name=f"PP-OCRv6_{self.tier}_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                cpu_threads=self.cpu_threads,
                enable_mkldnn=self.enable_mkldnn,
                **detection_options,
                **document_options(
                    self.tier, self.cpu_threads, enable_mkldnn=self.enable_mkldnn
                ),
            )
        return self._engine

    def ready(self) -> None:
        self._get_engine()
        import numpy as np

        self.read_lines(np.full((96, 384, 3), 255, dtype=np.uint8))
        if self.roi_sidecar_enabled:
            self._run_roi_sidecar([np.full((48, 320, 3), 255, dtype=np.uint8)])

    def _get_roi_sidecar_executor(self):
        if not self.roi_sidecar_enabled:
            raise OCREngineUnavailable("roi_sidecar_disabled")
        if self._roi_sidecar_executor is None:
            from .roi_sidecar import initialize

            self._roi_sidecar_executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize,
                initargs=(self.roi_sidecar_model, self.roi_sidecar_cpu_threads),
            )
        return self._roi_sidecar_executor

    def _run_roi_sidecar(self, crops: list) -> list[dict]:
        from .roi_sidecar import recognize

        started = time.perf_counter()
        with self._roi_sidecar_lock:
            executor = self._get_roi_sidecar_executor()
            future = executor.submit(recognize, crops)
            try:
                predictions = future.result(timeout=self.roi_sidecar_timeout_seconds)
                self.last_roi_sidecar_stats = {
                    "regions": len(crops),
                    "predictions": len(predictions),
                    "nonemptyPredictions": sum(bool(str(item.get("text") or "").strip()) for item in predictions),
                    "meanScore": round(
                        sum(float(item.get("score") or 0) for item in predictions)
                        / max(1, len(predictions)),
                        4,
                    ),
                    "elapsedMs": round((time.perf_counter() - started) * 1000, 3),
                }
                return predictions
            except Exception as error:
                self.last_roi_sidecar_stats = {
                    "regions": len(crops),
                    "elapsedMs": round((time.perf_counter() - started) * 1000, 3),
                    "error": type(error).__name__,
                }
                executor.shutdown(wait=False, cancel_futures=True)
                self._roi_sidecar_executor = None
                raise OCREngineUnavailable("roi_sidecar_failed") from error

    def read_lines_from_regions_sidecar(self, image, lines: list[dict]) -> list[dict]:
        from .front_refinement import crop_lines

        crops, selected = crop_lines(image, lines, minimum_text=1)
        if not crops:
            raise OCREngineUnavailable("roi_sidecar_no_regions")
        predictions = self._run_roi_sidecar(crops)
        if len(predictions) != len(selected):
            raise OCREngineUnavailable("roi_sidecar_prediction_count_mismatch")
        recognized = [dict(line) for line in lines]
        for index, prediction in zip(selected, predictions, strict=True):
            recognized[index] = {
                **recognized[index],
                "text": prediction["text"],
                "score": prediction["score"],
            }
        return recognized

    def read_lines(self, image) -> list[dict]:
        try:
            import cv2
        except ImportError as error:
            raise OCREngineUnavailable("opencv_not_installed") from error
        descriptor, path = tempfile.mkstemp(prefix="ine_ocr_", suffix=".png")
        os.close(descriptor)
        try:
            if not cv2.imwrite(path, image):
                raise OCREngineUnavailable("temporary_image_encode_failed")
            with self._lock:
                result = self._get_engine().predict(path)
            page = result[0]
            texts = page["rec_texts"]
            scores = page["rec_scores"]
            boxes = page.get("rec_polys")
            if boxes is None:
                boxes = page.get("dt_polys")
            lines = []
            for index, (text, score) in enumerate(zip(texts, scores)):
                box = boxes[index].tolist() if boxes is not None and len(boxes) > index else None
                lines.append({"text": text, "score": round(float(score), 4), "box": box})
            lines.sort(key=lambda line: (
                min((point[1] for point in line["box"]), default=0),
                min((point[0] for point in line["box"]), default=0),
            ))
            return lines
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
