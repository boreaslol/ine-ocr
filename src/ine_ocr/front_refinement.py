"""Optional high-resolution front reading and learned candidate selection."""

from __future__ import annotations

import time
import re
import threading

import numpy as np

from .curp import is_valid_curp
from .document_fields import extract_known_fields, reading_rotation
from .document_onnx import recognition_options
from .document_preprocess import bytes_to_image, prescale, standardize
from .field_selection import (
    FIELDS,
    FieldSelector,
    preserve_printed_diacritics,
    signature,
)


def corroborated_name_changes(baseline: dict, views: dict) -> dict:
    names = FIELDS[:3]
    sources = baseline.get("fieldSources", {})
    if not any(sources.get(field) == "mrz_unchecked_name" for field in names):
        return {}
    curp = baseline.get("curp")
    if not curp or not is_valid_curp(curp):
        return {}
    readers = [views.get(source, {}) for source in ("small_upright", "english_regions")]
    if any(reader.get("curp") != curp for reader in readers):
        return {}
    for field in names:
        if any(
            not isinstance(reader.get(field), str)
            or not reader[field].strip()
            or not 0.95 <= reader.get("_nameScores", {}).get(field, 0) <= 1.0
            for reader in readers
        ):
            return {}
        if signature(readers[0][field]) != signature(readers[1][field]):
            return {}
        original = str(baseline.get(field) or "")
        if sources.get(field) != "mrz_unchecked_name" and signature(
            original, compact=True
        ) != signature(readers[0][field], compact=True):
            return {}
    changes = {}
    for field in names:
        if sources.get(field) not in {"mrz_unchecked_name", "front_ocr_mrz_corroborated"}:
            continue
        value = preserve_printed_diacritics(
            readers[0][field], str(baseline.get(field) or "")
        )
        if value != baseline.get(field):
            changes[field] = {
                "value": value,
                "source": "cross_model_name_block",
                "confidence": min(reader["_nameScores"][field] for reader in readers),
            }
    return changes


def refinement_indices(lines: list[dict]) -> set[int]:
    fields = extract_known_fields(lines)
    values = [str(fields.get(field) or "") for field in FIELDS]
    values.extend(fields.get("_nameCandidates", []))
    signatures = {signature(value, compact=True) for value in values if value}
    indices = set()
    for index, line in enumerate(lines):
        text = str(line.get("text") or "")
        compact = signature(text, compact=True)
        if len(compact) >= 3 and any(compact in value for value in signatures):
            indices.add(index)
        if not fields.get("curp") and (
            "CURP" in text.upper() or re.fullmatch(r"[A-Z0-9]{18}", compact)
        ):
            indices.add(index)
    return indices


def region_view(lines: list[dict], recognized: list[dict]) -> list[dict]:
    if len(lines) != len(recognized):
        raise ValueError("region_line_count_mismatch")
    indices = refinement_indices(lines)
    return [
        recognized[index] if index in indices else line
        for index, line in enumerate(lines)
    ]


def crop_lines(
    image, lines: list[dict], *, minimum_text: int = 3
) -> tuple[list, list[int]]:
    import cv2

    crops, selected = [], []
    for index, line in enumerate(lines):
        if not line.get("box") or len(str(line.get("text", ""))) < minimum_text:
            continue
        points = np.asarray(line["box"], dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            continue
        width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        if min(width, height) < 2:
            continue
        target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        matrix = cv2.getPerspectiveTransform(points, target)
        crop = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        if crop.shape[0] > crop.shape[1] * 1.5:
            crop = np.ascontiguousarray(np.rot90(crop))
        crops.append(crop)
        selected.append(index)
    return crops, selected


class FrontRefiner:
    def __init__(self, line_reader, model_path: str) -> None:
        self.line_reader = line_reader
        self.selector = FieldSelector(model_path)
        self.recognizer = None
        self._recognition_lock = threading.RLock()

    def ready(self) -> None:
        from paddleocr import TextRecognition

        with self._recognition_lock:
            if self.recognizer is None:
                self.recognizer = TextRecognition(
                    model_name="en_PP-OCRv5_mobile_rec",
                    cpu_threads=self.line_reader.cpu_threads,
                    **recognition_options(
                        "en_PP-OCRv5_mobile_rec", self.line_reader.cpu_threads
                    ),
                )

    def observe(
        self, data: bytes, *, rotation_hint: int | None = None
    ) -> tuple[dict, dict]:
        started = time.perf_counter()
        natural = prescale(bytes_to_image(data), max_side=2200)
        _, thumbnail, _ = standardize(natural)
        rotation = reading_rotation(self.line_reader.read_lines(thumbnail))
        if rotation is None:
            rotation = rotation_hint
        upright = np.ascontiguousarray(np.rot90(natural, -int(rotation or 0) // 90))
        lines = self.line_reader.read_lines(upright)
        english_lines = self.read_regions(upright, lines, field_only=True)
        return {
            "small_upright": extract_known_fields(lines),
            "english_regions": extract_known_fields(english_lines),
        }, {
            "rotation": rotation,
            "elapsedMs": round((time.perf_counter() - started) * 1000, 3),
        }

    def read_regions(
        self,
        image,
        lines: list[dict],
        *,
        minimum_text: int = 3,
        field_only: bool = False,
    ) -> list[dict]:
        from paddleocr import TextRecognition

        crops, selected = crop_lines(image, lines, minimum_text=minimum_text)
        if field_only:
            indices = refinement_indices(lines)
            retained = [
                (crop, index)
                for crop, index in zip(crops, selected)
                if index in indices
            ]
            crops = [crop for crop, _ in retained]
            selected = [index for _, index in retained]
        predictions = []
        if crops:
            with self._recognition_lock:
                if self.recognizer is None:
                    self.ready()
                predictions = list(self.recognizer.predict(crops, batch_size=16))
        english_lines = [dict(line) for line in lines]
        for index, prediction in zip(selected, predictions, strict=True):
            english_lines[index] = {
                **lines[index],
                "text": str(prediction["rec_text"]),
                "score": float(prediction["rec_score"]),
            }
        return english_lines

    def apply(self, result: dict, views: dict, metadata: dict) -> None:
        processing = {**metadata, "changedFields": [], "enabled": True}
        result.setdefault("processing", {})["refinement"] = processing
        if result.get("curp") and any(
            view.get("curp") and view["curp"] != result["curp"]
            for view in views.values()
        ):
            processing["rejected"] = "identity_conflict"
            return
        changes = self.selector.select(result, views)
        changes.update(corroborated_name_changes(result, views))
        for field, choice in changes.items():
            result[field] = choice["value"]
            source = choice["source"]
            result.setdefault("fieldSources", {})[field] = (
                "front_ocr_" if source == "cross_model_name_block" else "front_ocr_ranked_"
            ) + source
            result.setdefault("fieldConfidence", {})[field] = min(
                0.9, choice["confidence"]
            )
            processing["changedFields"].append(field)
        curps = {
            view["curp"]
            for view in views.values()
            if view.get("curp") and is_valid_curp(view["curp"])
        }
        if not result.get("curp") and len(curps) == 1:
            curp = next(iter(curps))
            birth = str(result.get("fechaNacimiento") or "").split("/")
            if len(birth) == 3 and curp[4:10] == birth[2][-2:] + birth[1] + birth[0]:
                result["curp"] = curp
                result.setdefault("fieldSources", {})["curp"] = (
                    "front_ocr_refined_check_valid"
                )
                result.setdefault("fieldConfidence", {})["curp"] = 0.9
                result.setdefault("genero", {})["sexoRegistralCurp"] = curp[10]
                processing["changedFields"].append("curp")

    def refine(
        self, data: bytes, result: dict, *, evidence: dict | None = None
    ) -> None:
        if evidence is not None:
            started = time.perf_counter()
            lines = evidence["lines"]
            english_lines = self.read_regions(evidence["image"], lines, field_only=True)
            views = {
                "small_upright": extract_known_fields(lines),
                "english_regions": extract_known_fields(english_lines),
            }
            self.apply(
                result,
                views,
                {
                    "source": "existing_pipeline_regions",
                    "additionalDetectionCalls": 0,
                    "elapsedMs": round((time.perf_counter() - started) * 1000, 3),
                },
            )
            return
        front = result.get("processing", {}).get("front", {})
        hint = None
        if front.get("detailRescueSource") == "upright_natural":
            hint = int(front["detailRescueOrientation"])
        elif front.get("source") == "standardized_natural":
            hint = int(front.get("orientation") or 0)
        views, metadata = self.observe(data, rotation_hint=hint)
        self.apply(result, views, metadata)
