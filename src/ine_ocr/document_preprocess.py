"""INE card detection, rectification, normalization, and quality scoring."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CARD_WIDTH = 1000
CARD_HEIGHT = 630
DETAIL_WIDTH = 1600
DETAIL_HEIGHT = 1008
MAX_SIDE = 1600
MAX_DECODED_PIXELS = int(os.getenv("INE_OCR_MAX_DECODED_PIXELS", "50000000"))
ASPECT_MIN = 1.35
ASPECT_MAX = 1.85
MIN_AREA_RATIO = 0.04


class ImageDecodeError(ValueError):
    pass


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv_not_installed") from error
    return cv2


def bytes_to_image(data: bytes):
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width * height > MAX_DECODED_PIXELS:
                raise ImageDecodeError("decoded_image_too_large")
            image = ImageOps.exif_transpose(image).convert("RGB")
            array = np.asarray(image)
    except ImageDecodeError:
        raise
    except Exception as error:
        raise ImageDecodeError("image_decode_failed") from error
    return _cv2().cvtColor(array, _cv2().COLOR_RGB2BGR)


def image_to_bytes(image, *, extension: str = ".jpg", quality: int = 85) -> bytes:
    cv2 = _cv2()
    parameters = [cv2.IMWRITE_JPEG_QUALITY, quality] if extension in {".jpg", ".jpeg"} else []
    ok, buffer = cv2.imencode(extension, image, parameters)
    if not ok:
        raise ImageDecodeError("image_encode_failed")
    return buffer.tobytes()


def prescale(image, max_side: int = MAX_SIDE):
    height, width = image.shape[:2]
    current_max = max(height, width)
    if current_max <= max_side:
        return image
    scale = max_side / current_max
    return _cv2().resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=_cv2().INTER_AREA,
    )


def standardize(image, *, width: int = CARD_WIDTH, height: int = CARD_HEIGHT):
    cv2 = _cv2()
    source_height, source_width = image.shape[:2]
    canvas = np.zeros((height, width, 3), np.uint8)
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    top = (height - resized_height) // 2
    left = (width - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    encoded = image_to_bytes(canvas)
    return encoded, canvas, {
        "stdSize": f"{width}x{height}",
        "stdFormat": ".jpg",
        "stdQuality": 85,
        "stdBytes": len(encoded),
    }


def quality_score(image) -> dict:
    cv2 = _cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    glare = float((gray > 245).mean())
    issues = []
    if blur < 60:
        issues.append("blurry")
    if brightness < 50:
        issues.append("too_dark")
    if brightness > 235:
        issues.append("too_bright")
    if glare > 0.35:
        issues.append("glare")
    return {
        "blurScore": round(blur, 1),
        "brightness": round(brightness, 1),
        "glareRatio": round(glare, 3),
        "ok": not issues,
        "issues": issues,
    }


def _order_points(points: np.ndarray) -> np.ndarray:
    rectangle = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    rectangle[0], rectangle[2] = points[np.argmin(sums)], points[np.argmax(sums)]
    differences = np.diff(points, axis=1)
    rectangle[1], rectangle[3] = points[np.argmin(differences)], points[np.argmax(differences)]
    return rectangle


def _contour_candidates(edge_source, area_minimum: float):
    cv2 = _cv2()
    edges = cv2.dilate(edge_source, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if (
            len(approximation) == 4
            and cv2.isContourConvex(approximation)
            and cv2.contourArea(approximation) >= area_minimum
        ):
            yield approximation


class CardPreprocessor:
    def __init__(self, yolo_model_path: str | None = None) -> None:
        selected_model = yolo_model_path or os.getenv("INE_OCR_YOLO_MODEL_PATH", "")
        self.yolo_model_path = Path(selected_model).expanduser() if selected_model else None
        self._yolo = None
        self._yolo_checked = False
        self.last_geometry = None

    def _get_yolo(self):
        if self._yolo_checked:
            return self._yolo
        self._yolo_checked = True
        if self.yolo_model_path is None or not self.yolo_model_path.is_file():
            return None
        try:
            import onnxruntime as ort
            self._yolo = ort.InferenceSession(
                str(self.yolo_model_path), providers=["CPUExecutionProvider"]
            )
        except Exception:
            self._yolo = None
        return self._yolo

    def ready(self) -> None:
        _cv2()
        if self.yolo_model_path is not None and (
            not self.yolo_model_path.is_file() or self._get_yolo() is None
        ):
            raise RuntimeError("card_detector_not_ready")

    def _yolo_detect(self, image):
        session = self._get_yolo()
        if session is None:
            return None, 0.0
        try:
            cv2 = _cv2()
            source_height, source_width = image.shape[:2]
            scale = min(640 / source_width, 640 / source_height)
            resized_width = max(1, round(source_width * scale))
            resized_height = max(1, round(source_height * scale))
            resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
            left = (640 - resized_width) // 2
            top = (640 - resized_height) // 2
            canvas[top:top + resized_height, left:left + resized_width] = resized
            tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            tensor = np.expand_dims(np.ascontiguousarray(tensor), 0)
            output_name = session.get_outputs()[0].name
            input_name = session.get_inputs()[0].name
            predictions = session.run([output_name], {input_name: tensor})[0]
            predictions = predictions[0]
            if predictions.shape[0] < predictions.shape[1]:
                predictions = predictions.transpose(1, 0)
            if predictions.shape[1] < 5:
                return None, 0.0
            confidence_index = int(np.argmax(predictions[:, 4]))
            confidence = float(predictions[confidence_index, 4])
            if confidence < 0.4:
                return None, 0.0
            center_x, center_y, width, height = predictions[confidence_index, :4]
            box_left = (float(center_x) - float(width) / 2 - left) / scale
            box_top = (float(center_y) - float(height) / 2 - top) / scale
            box_right = (float(center_x) + float(width) / 2 - left) / scale
            box_bottom = (float(center_y) + float(height) / 2 - top) / scale
            quadrilateral = np.array(
                [
                    [box_left, box_top],
                    [box_right, box_top],
                    [box_right, box_bottom],
                    [box_left, box_bottom],
                ],
                dtype=np.float32,
            )
            return quadrilateral, confidence
        except Exception:
            return None, 0.0

    def detect(self, image, *, allow_classical: bool = True):
        cv2 = _cv2()
        quadrilateral, confidence = self._yolo_detect(image)
        if quadrilateral is not None:
            distance = lambda first, second: np.linalg.norm(  # noqa: E731
                quadrilateral[first] - quadrilateral[second]
            )
            estimated_width = (distance(0, 1) + distance(2, 3)) / 2
            estimated_height = (distance(1, 2) + distance(0, 3)) / 2
            if estimated_height > 0:
                aspect = max(estimated_width, estimated_height) / min(
                    estimated_width, estimated_height
                )
                if 1.15 < aspect < 2.1:
                    center_x, center_y = quadrilateral[:, 0].mean(), quadrilateral[:, 1].mean()
                    quadrilateral = (
                        quadrilateral - [center_x, center_y]
                    ) * 1.04 + [center_x, center_y]
                    quadrilateral[:, 0] = np.clip(
                        quadrilateral[:, 0], 0, image.shape[1] - 1
                    )
                    quadrilateral[:, 1] = np.clip(
                        quadrilateral[:, 1], 0, image.shape[0] - 1
                    )
                    return quadrilateral, True, estimated_height > estimated_width, confidence
        if not allow_classical:
            return None, False, False, 0.0

        height, width = image.shape[:2]
        scale = 800 / max(height, width)
        small = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
        ) if scale < 1 else image.copy()
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        raw_edges = cv2.Canny(gray, 40, 120)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,
            10,
        )
        adaptive = cv2.morphologyEx(
            adaptive,
            cv2.MORPH_CLOSE,
            np.ones((15, 15), np.uint8),
            iterations=2,
        )
        area_minimum = small.shape[0] * small.shape[1] * MIN_AREA_RATIO
        candidates = [
            *_contour_candidates(raw_edges, area_minimum),
            *_contour_candidates(adaptive, area_minimum),
        ]
        for approximation in candidates:
            quadrilateral = approximation.reshape(4, 2).astype(np.float32)
            straight_sides = 0
            for side_index in range(4):
                start = quadrilateral[side_index]
                end = quadrilateral[(side_index + 1) % 4]
                side_length = float(np.linalg.norm(end - start))
                if side_length < 30:
                    break
                band = np.zeros(raw_edges.shape, np.uint8)
                cv2.line(
                    band,
                    tuple(start.astype(int)),
                    tuple(end.astype(int)),
                    255,
                    thickness=max(4, small.shape[0] // 120),
                )
                region = cv2.bitwise_and(raw_edges, raw_edges, mask=band)
                lines = cv2.HoughLinesP(
                    region,
                    1,
                    np.pi / 180,
                    threshold=30,
                    minLineLength=side_length * 0.6,
                    maxLineGap=side_length * 0.08,
                )
                if lines is None:
                    continue
                side_angle = np.arctan2(end[1] - start[1], end[0] - start[0])
                for line in np.asarray(lines).reshape(-1, 4):
                    line_angle = np.arctan2(line[3] - line[1], line[2] - line[0])
                    difference = abs((line_angle - side_angle + np.pi / 2) % np.pi - np.pi / 2)
                    if difference < np.deg2rad(12):
                        straight_sides += 1
                        break
            if straight_sides < 3:
                continue
            points = approximation.reshape(4, 2).astype(np.float32) / (scale if scale < 1 else 1)
            rectangle = _order_points(points)
            distance = lambda first, second: np.linalg.norm(  # noqa: E731
                rectangle[first] - rectangle[second]
            )
            estimated_width = max(distance(0, 1), distance(2, 3))
            estimated_height = max(distance(1, 2), distance(0, 3))
            if estimated_height and ASPECT_MIN < estimated_width / estimated_height < ASPECT_MAX:
                return rectangle, True, False, 0.0
        return None, False, False, 0.0

    @staticmethod
    def rectify(image, rectangle, width: int, height: int):
        cv2 = _cv2()
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(rectangle, destination)
        return cv2.warpPerspective(image, matrix, (width, height))

    def process(self, data: bytes, side: str) -> tuple[object, dict, object, object]:
        cv2 = _cv2()
        natural = prescale(bytes_to_image(data))
        gray = cv2.cvtColor(natural, cv2.COLOR_BGR2GRAY)
        edge_density = float(
            (cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120) > 0).mean()
        )
        quadrilateral, found, portrait, confidence = self.detect(
            natural, allow_classical=False
        )
        if not found and edge_density <= 0.12:
            quadrilateral, found, portrait, confidence = self.detect(
                natural, allow_classical=True
            )

        full_frame_assumed = False
        if found:
            normalized = self.rectify(natural, quadrilateral, CARD_WIDTH, CARD_HEIGHT)
            detail = self.rectify(natural, quadrilateral, DETAIL_WIDTH, DETAIL_HEIGHT)
            if portrait:
                normalized = cv2.rotate(normalized, cv2.ROTATE_90_CLOCKWISE)
                detail = cv2.rotate(detail, cv2.ROTATE_90_CLOCKWISE)
        else:
            height, width = natural.shape[:2]
            aspect = max(width, height) / max(1, min(width, height))
            if ASPECT_MIN < aspect < ASPECT_MAX and width >= 600:
                normalized = natural if width >= height else cv2.rotate(
                    natural, cv2.ROTATE_90_CLOCKWISE
                )
                detail = normalized
                full_frame_assumed = True
            else:
                normalized = natural
                detail = natural

        self.last_geometry = {
            "quadrilateral": quadrilateral.tolist() if found else None,
            "portrait": bool(portrait),
            "naturalShape": list(natural.shape),
            "detailShape": list(detail.shape),
            "cardDetected": bool(found),
            "fullFrameAssumed": bool(full_frame_assumed),
        }

        quality = quality_score(normalized)
        quality.update({
            "side": side,
            "cardDetected": found,
            "fullFrameAssumed": full_frame_assumed,
            "edgeDensity": round(edge_density, 3),
            "detectorConfidence": round(confidence, 4) if confidence else None,
        })
        if not found and not full_frame_assumed:
            quality["issues"] = list(dict.fromkeys([*quality["issues"], "card_not_found"]))
            quality["ok"] = False
        encoded, standardized, metadata = standardize(normalized)
        quality.update(metadata)
        return standardized, quality, detail, natural
