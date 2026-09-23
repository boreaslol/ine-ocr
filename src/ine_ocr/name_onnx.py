"""Opt-in Latin-name ONNX inference and local document-evaluation adapter."""

from __future__ import annotations

import json
from pathlib import Path
import threading

import numpy as np

from .alphabet import LATIN_NAME_ALPHABET, greedy_decode, normalize_text
from .artifacts import sha256_file
from .image_preprocessing import prepare_image_array
from .name_crops import compose_name_crop, name_regions
from .document_fields import reading_rotation
from .document_preprocess import bytes_to_image, prescale, standardize


class NameOnnxRecognizer:
    def __init__(self, model: str | Path, *, cpu_threads: int = 1) -> None:
        import onnxruntime as ort

        path = Path(model).resolve()
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        if metadata.get("alphabet") != LATIN_NAME_ALPHABET:
            raise ValueError("name_model_alphabet_mismatch")
        if sha256_file(str(path)) != metadata.get("onnx_sha256"):
            raise ValueError("name_model_hash_mismatch")
        if not metadata.get("verification", {}).get("passed"):
            raise ValueError("name_model_export_parity_not_verified")
        shape = metadata["input"]["shape"]
        if shape[1] != 1 or shape[2] != 48 or not 1 <= cpu_threads <= 4:
            raise ValueError("name_model_input_or_threads_invalid")
        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.height, self.width = int(shape[2]), int(shape[3])
        self.preprocessing = metadata["preprocessing"]
        self.model_sha256 = metadata["onnx_sha256"]
        self.lock = threading.Lock()

    def ready(self) -> None:
        from PIL import Image

        self.predict([Image.new("RGB", (self.width * 2, self.height * 2), "white")])

    def predict(self, crops: list) -> list[dict]:
        if not crops:
            return []
        images = np.stack([
            prepare_image_array(crop, self.width, self.height,
                                resize_mode=self.preprocessing["resize_mode"],
                                isolate_foreground=self.preprocessing["isolate_foreground"])
            for crop in crops
        ])
        with self.lock:
            logits = self.session.run(["logits"], {"image": images})[0]
        if logits.shape[-1] != len(LATIN_NAME_ALPHABET) + 1:
            raise ValueError("name_model_output_classes_mismatch")
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        results = []
        for row, scores in zip(logits.argmax(axis=-1), probabilities, strict=True):
            confidence = []
            previous = -1
            for index, value in enumerate(row):
                if value and value != previous:
                    confidence.append(float(scores[index, value]))
                previous = value
            text = normalize_text(greedy_decode(row.tolist(), alphabet=LATIN_NAME_ALPHABET), alphabet=LATIN_NAME_ALPHABET)
            results.append({"text": text, "confidence": sum(confidence) / len(confidence) if confidence else 0.0})
        return results


class NameLineAdapter:
    protected_result_sources = {
        "front_ocr_consensus", "front_ocr_mrz_corroborated",
        "front_ocr_cross_channel_consensus",
    }

    def __init__(self, line_reader, recognizer: NameOnnxRecognizer, *, threshold: float = 0.9) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("name_adapter_threshold_out_of_range")
        self.line_reader = line_reader
        self.cpu_threads = line_reader.cpu_threads
        self.recognizer = recognizer
        self.threshold = threshold

    def ready(self) -> None:
        ready_method = getattr(self.recognizer, "ready", None)
        if callable(ready_method):
            ready_method()

    def predict_fields(self, image, lines: list[dict] | None = None) -> dict[str, dict]:
        lines = lines if lines is not None else self.line_reader.read_lines(image)
        regions = name_regions(lines)
        fields, crops = [], []
        for field, pieces in regions.items():
            try:
                crop = compose_name_crop(image, pieces)
            except ValueError:
                continue
            fields.append(field)
            crops.append(crop)
        predictions = {}
        for field, prediction in zip(fields, self.recognizer.predict(crops), strict=True):
            text = prediction["text"]
            if not text or "<" in text or any(character.isdigit() for character in text) or not any(character.isalpha() for character in text) or prediction["confidence"] < self.threshold:
                continue
            predictions[field] = {
                "text": text,
                "confidence": prediction["confidence"],
            }
        return predictions

    def predict_from_bytes(self, data: bytes) -> tuple[dict[str, dict], dict]:
        natural = prescale(bytes_to_image(data), max_side=2200)
        _, thumbnail, _ = standardize(natural)
        rotation = reading_rotation(self.line_reader.read_lines(thumbnail))
        upright = np.ascontiguousarray(np.rot90(natural, -int(rotation or 0) // 90))
        lines = self.line_reader.read_lines(upright)
        predictions, metadata = self.predict_from_evidence(upright, lines)
        metadata["source"] = "front_name_region"
        metadata["rotation"] = rotation
        return predictions, metadata

    def predict_from_evidence(self, image, lines: list[dict]) -> tuple[dict[str, dict], dict]:
        return self.predict_fields(image, lines), {
            "source": "front_name_region",
            "fields": sorted(name_regions(lines)),
        }

    def apply_to_result(self, result: dict, predictions: dict[str, dict]) -> list[str]:
        changed = []
        skipped = []
        confidences = {}
        for field, prediction in predictions.items():
            value = prediction["text"]
            if value == result.get(field):
                continue
            if (
                result.get(field)
                and (result.get("fieldSources") or {}).get(field) in self.protected_result_sources
            ):
                skipped.append(field)
                continue
            result[field] = value
            result.setdefault("fieldSources", {})[field] = "front_name_onnx"
            result.setdefault("fieldConfidence", {})[field] = min(0.9, prediction["confidence"])
            changed.append(field)
            confidences[field] = round(float(prediction["confidence"]), 4)
        result.setdefault("processing", {})["nameModel"] = {
            "enabled": True,
            "changedFields": changed,
            "changedFieldConfidences": confidences,
            "skippedProtectedFields": sorted(skipped),
            "scope": "name_fields_only",
        }
        return changed

    def read_lines(self, image) -> list[dict]:
        lines = self.line_reader.read_lines(image)
        predictions = self.predict_fields(image, lines)
        replacements = {}
        removed = set()
        regions = name_regions(lines)
        for field, prediction in predictions.items():
            pieces = regions[field]
            selected = []
            for index, line in enumerate(lines):
                box = line.get("box")
                if not box:
                    continue
                center_x = sum(point[0] for point in box) / len(box)
                center_y = sum(point[1] for point in box) / len(box)
                if any(min(point[0] for point in piece["box"]) <= center_x <= max(point[0] for point in piece["box"])
                       and min(point[1] for point in piece["box"]) <= center_y <= max(point[1] for point in piece["box"])
                       for piece in pieces):
                    selected.append(index)
            if not selected or set(selected) & (removed | set(replacements)):
                continue
            coordinates = [point for piece in pieces for point in piece["box"]]
            left, right = min(point[0] for point in coordinates), max(point[0] for point in coordinates)
            top, bottom = min(point[1] for point in coordinates), max(point[1] for point in coordinates)
            replacements[selected[0]] = {**lines[selected[0]], "text": prediction["text"], "score": prediction["confidence"],
                                         "box": [[left, top], [right, top], [right, bottom], [left, bottom]]}
            removed.update(selected[1:])
        return [replacements.get(index, line) for index, line in enumerate(lines) if index not in removed]
