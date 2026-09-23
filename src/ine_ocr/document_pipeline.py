"""Full INE document pipeline: front OCR, MRZ, QR, and validation."""

from __future__ import annotations

import re
import os
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np

from .curp import is_valid_curp
from .document_fields import CLAVE_ELECTOR_VALUE, _bounds, _plain, extract_known_fields, reading_rotation
from .document_ocr import OCREngineUnavailable, PaddleLineReader
from .document_preprocess import CARD_HEIGHT, CARD_WIDTH, CardPreprocessor, standardize
from .mrz import MRZError, assemble as assemble_mrz, parse as parse_mrz
from .qr import QRDecoder, extract_official_fields


@dataclass(frozen=True)
class ChannelCandidate:
    source: str
    orientation: str
    fields: dict


def _rotations(image):
    return (
        ("0", image),
        ("180", np.ascontiguousarray(np.rot90(image, 2))),
        ("90", np.ascontiguousarray(np.rot90(image, 3))),
        ("270", np.ascontiguousarray(np.rot90(image, 1))),
    )


def _official_qr_values(values: list[str]) -> list[str]:
    accepted = []
    for value in values:
        try:
            parsed = urlparse(value)
        except ValueError:
            continue
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme in {"http", "https"} and (
            hostname == "qr.ine.mx" or hostname.endswith(".qr.ine.mx")
        ):
            accepted.append(value)
    return list(dict.fromkeys(accepted))


def _name_signature(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.upper())
    plain = "".join(character for character in normalized if not unicodedata.combining(character))
    return " ".join(re.sub(r"[^A-Z ]", " ", plain).split())


def _prefer_printed_name(field: str, printed: str | None, mrz: str | None) -> bool:
    if not printed or not mrz:
        return False
    printed_signature = _name_signature(printed)
    mrz_signature = _name_signature(mrz)
    if printed_signature == mrz_signature:
        return True
    printed_compact = printed_signature.replace(" ", "")
    mrz_compact = mrz_signature.replace(" ", "")
    return bool(
        len(mrz_compact) >= 4
        and printed_compact.startswith(mrz_compact)
        and (
            field == "nombres"
            or len(printed_signature.split()[0]) - len(mrz_compact) <= 1
        )
    )


def _printed_surnames_match_mrz(front: dict, mrz: dict) -> bool:
    names = [front.get(field) for field in ("primerApellido", "segundoApellido")]
    if not all(names):
        return False
    joined = "".join(_name_signature(value).replace(" ", "") for value in names)
    raw = _name_signature(str(mrz.get("nameText") or "").replace("<", " ")).replace(" ", "")
    return len(joined) >= 6 and raw.startswith(joined)


def _mrz_spaced_name(printed: str, raw: str) -> str:
    tokens = [token for token in re.split(r"<+", raw) if token]
    signature = _name_signature(printed).replace(" ", "")
    letters = "".join(re.findall(r"[A-ZÀ-ÖØ-Þ]", unicodedata.normalize("NFC", printed.upper())))
    if len(letters) != len(signature):
        return printed
    choices = []
    for start in range(len(tokens)):
        for end in range(start + 2, len(tokens) + 1):
            observed = "".join(tokens[start:end])
            if observed == signature or (
                end == len(tokens) and len(observed) >= 4 and signature.startswith(observed)
            ):
                boundaries = {0, len(signature)}
                offset = 0
                for token in tokens[start:end - 1]:
                    offset += len(token)
                    boundaries.add(offset)
                offset = 0
                for word in printed.split()[:-1]:
                    offset += len(_name_signature(word).replace(" ", ""))
                    boundaries.add(offset)
                ordered = sorted(boundaries)
                value = " ".join(letters[left:right] for left, right in zip(ordered, ordered[1:]))
                choices.append((len(observed), value))
    if choices:
        return max(choices, key=lambda choice: choice[0])[1]
    return printed


def _name_consensus(first: dict, second: dict) -> dict[str, str]:
    consensus = {}
    for field in ("primerApellido", "segundoApellido", "nombres"):
        values = [observation.get(field) for observation in (first, second)]
        if not all(values) or any(
            observation.get("_nameScores", {}).get(field, 0) < 0.95
            for observation in (first, second)
        ):
            continue
        signatures = [_name_signature(value).replace(" ", "") for value in values]
        if signatures[0] == signatures[1]:
            consensus[field] = max(values, key=lambda value: len(value.split()))
    return consensus


def _single_name_error(first: str, second: str | None) -> bool:
    if not second:
        return False
    first = _name_signature(first).replace(" ", "")
    second = _name_signature(second).replace(" ", "")
    if min(len(first), len(second)) < 4:
        return False
    if len(first) == len(second):
        return sum(left != right for left, right in zip(first, second)) == 1
    shorter, longer = sorted((first, second), key=len)
    return len(longer) == len(shorter) + 1 and any(
        longer[:index] + longer[index + 1:] == shorter for index in range(len(longer))
    )


def _cross_channel_name_consensus(field: str, front: dict, mrz: dict) -> bool:
    consensus = front.get("_nameConsensus", {}).get(field)
    curp = str(front.get("curp") or "")
    birth = str(front.get("fechaNacimiento") or "")
    if (
        not consensus
        or not mrz.get(field)
        or not is_valid_curp(curp)
        or not mrz.get("validacionMRZAllOk")
        or re.fullmatch(r"\d{2}/\d{2}/\d{4}", birth) is None
        or birth != mrz.get("fechaNacimiento")
        or birth[-2:] + birth[3:5] + birth[:2] != curp[4:10]
    ):
        return False
    return any(
        other != field
        and front.get("_nameScores", {}).get(other, 0) >= 0.95
        and len(_name_signature(str(mrz.get(other) or "")).replace(" ", "")) >= 4
        and _prefer_printed_name(other, front.get(other), mrz.get(other))
        for other in ("nombres", "primerApellido", "segundoApellido")
    )


def _transform_line_boxes(lines: list[dict], transform, source_shape) -> list[dict]:
    transformed = []
    for line in lines:
        box = line.get("box")
        if box and len(box) == 4:
            box = [transform(float(point[0]), float(point[1]), source_shape) for point in box]
        transformed.append({**line, "box": box})
    return transformed


def _rotate_point(x: float, y: float, orientation: int, source_shape) -> list[float]:
    height, width = source_shape[:2]
    if orientation == 90:
        return [height - 1 - y, x]
    if orientation == 180:
        return [width - 1 - x, height - 1 - y]
    if orientation == 270:
        return [y, width - 1 - x]
    return [x, y]


def _oriented_detail_rois(
    lines: list[dict], natural, detail, orientation: str, target_orientation: str,
    geometry: dict | None,
) -> tuple[object, list[dict]]:
    import cv2

    angle = int(orientation)
    target_angle = int(target_orientation)
    candidate_shape = (CARD_HEIGHT, CARD_WIDTH)
    base_lines = _transform_line_boxes(
        lines,
        lambda x, y, shape: _rotate_point(x, y, (360 - angle) % 360, shape),
        (candidate_shape[1], candidate_shape[0]) if angle in {90, 270} else candidate_shape,
    )
    natural_height, natural_width = natural.shape[:2]
    scale = min(CARD_WIDTH / natural_width, CARD_HEIGHT / natural_height)
    resized_width = max(1, int(natural_width * scale))
    resized_height = max(1, int(natural_height * scale))
    left = (CARD_WIDTH - resized_width) // 2
    top = (CARD_HEIGHT - resized_height) // 2

    def to_natural(x: float, y: float, _shape) -> list[float]:
        return [(x - left) / scale, (y - top) / scale]

    natural_lines = _transform_line_boxes(base_lines, to_natural, candidate_shape)
    quadrilateral = (geometry or {}).get("quadrilateral")
    if not quadrilateral:
        oriented = next(image for name, image in _rotations(natural) if name == target_orientation)
        oriented_lines = _transform_line_boxes(
            natural_lines,
            lambda x, y, shape: _rotate_point(x, y, target_angle, shape),
            natural.shape,
        )
        return oriented, oriented_lines

    rectangle = np.asarray(quadrilateral, dtype=np.float32)
    destination = np.float32([
        [0, 0], [1600 - 1, 0], [1600 - 1, 1008 - 1], [0, 1008 - 1],
    ])
    matrix = cv2.getPerspectiveTransform(rectangle, destination)

    def to_detail(x: float, y: float, _shape) -> list[float]:
        point = cv2.perspectiveTransform(
            np.asarray([[[x, y]]], dtype=np.float32), matrix
        )[0, 0]
        return [float(point[0]), float(point[1])]

    detail_lines = _transform_line_boxes(natural_lines, to_detail, natural.shape)
    if (geometry or {}).get("portrait"):
        detail_lines = _transform_line_boxes(
            detail_lines,
            lambda x, y, shape: _rotate_point(x, y, 90, shape),
            (1008, 1600),
        )
    oriented = next(image for name, image in _rotations(detail) if name == target_orientation)
    oriented_lines = _transform_line_boxes(
        detail_lines,
        lambda x, y, shape: _rotate_point(x, y, target_angle, shape),
        detail.shape,
    )
    return oriented, oriented_lines


def _name_roi_indices(lines: list[dict]) -> set[int]:
    for anchor_index, anchor in enumerate(lines):
        if _plain(str(anchor.get("text") or "")) != "NOMBRE":
            continue
        anchor_bounds = _bounds(anchor)
        if not anchor_bounds:
            continue
        left, top, right, bottom = anchor_bounds
        height = bottom - top
        address_top = min(
            (
                bounds[1]
                for line in lines
                if _plain(str(line.get("text") or "")) == "DOMICILIO"
                and (bounds := _bounds(line))
                and bounds[1] > bottom
                and abs(bounds[0] - left) < height * 3
            ),
            default=None,
        )
        selected = []
        for index, line in enumerate(lines):
            if index == anchor_index:
                continue
            bounds = _bounds(line)
            if not bounds:
                continue
            candidate_left, candidate_top, _, _ = bounds
            if candidate_top < top or candidate_top - bottom > height * 10:
                continue
            if abs(candidate_left - left) > max(height * 2, (right - left) * 0.55):
                continue
            if address_top is not None and candidate_top >= address_top:
                continue
            plain = _plain(str(line.get("text") or ""))
            if any(plain == stop or plain.startswith(stop + " ") for stop in (
                "DOMICILIO", "CLAVE DE ELECTOR", "CURP", "ANO DE REGISTRO",
                "FECHA DE NACIMIENTO", "ESTADO", "MUNICIPIO", "LOCALIDAD",
                "SECCION", "EMISION", "VIGENCIA", "SEXO", "GENERO",
            )):
                continue
            selected.append(index)
        if len(selected) >= 3:
            return set(selected)
    return set()


def _pad_roi_boxes(lines: list[dict], indices: set[int]) -> list[dict]:
    padded = []
    for index, line in enumerate(lines):
        box = np.asarray(line.get("box") or [], dtype=np.float32)
        if index not in indices or box.shape != (4, 2):
            padded.append(dict(line))
            continue
        left, top = box.min(axis=0)
        right, bottom = box.max(axis=0)
        width, height = right - left, bottom - top
        horizontal = width * 0.06
        vertical = height * 0.25
        padded.append({
            **line,
            "box": [
                [float(left - horizontal), float(top - vertical)],
                [float(right + horizontal), float(top - vertical)],
                [float(right + horizontal), float(bottom + vertical)],
                [float(left - horizontal), float(bottom + vertical)],
            ],
        })
    return padded


_ROI_REQUIRED_FIELDS = (
    "curp", "claveElector", "emision", "vigencia", "fechaNacimiento",
    "primerApellido", "segundoApellido", "nombres",
)
_ROI_COMPARISON_FIELDS = (
    "curp", "claveElector", "registro", "estado", "municipio", "localidad",
    "seccion", "emision", "vigencia", "fechaNacimiento", "primerApellido",
    "segundoApellido", "nombres", "calle", "colonia", "ciudad", "sexo",
)


def _same_roi_value(field: str, first, second) -> bool:
    if field in {"primerApellido", "segundoApellido", "nombres"}:
        return _name_signature(str(first)) == _name_signature(str(second))
    return re.sub(r"\W", "", str(first).upper()) == re.sub(r"\W", "", str(second).upper())


def _accepted_roi_fields(primary: dict, candidate: dict) -> dict | None:
    if not candidate.get("curp") or candidate["curp"] != primary.get("curp"):
        return None
    if not is_valid_curp(candidate["curp"]):
        return None
    clave = str(candidate.get("claveElector") or "")
    if clave and (
        not CLAVE_ELECTOR_VALUE.fullmatch(clave)
        or clave[6:12] != candidate["curp"][4:10]
    ):
        candidate = {key: value for key, value in candidate.items() if key != "claveElector"}
    if any(
        primary.get(field) not in (None, "", {}, [])
        and candidate.get(field) not in (None, "", {}, [])
        and not _same_roi_value(field, primary[field], candidate[field])
        for field in _ROI_COMPARISON_FIELDS
    ):
        return None
    if any(
        primary.get(field) not in (None, "", {}, [])
        and candidate.get(field) in (None, "", {}, [])
        for field in _ROI_COMPARISON_FIELDS
    ):
        return None
    if any(candidate.get(field) in (None, "", {}, []) for field in _ROI_REQUIRED_FIELDS):
        return None
    return candidate


def _roi_rejection_code(primary: dict, candidate: dict) -> str | None:
    if not candidate.get("curp") or candidate["curp"] != primary.get("curp"):
        return "curp_missing_or_mismatch"
    if not is_valid_curp(candidate["curp"]):
        return "curp_invalid"
    clave = str(candidate.get("claveElector") or "")
    if clave and (
        not CLAVE_ELECTOR_VALUE.fullmatch(clave)
        or clave[6:12] != candidate["curp"][4:10]
    ):
        candidate = {key: value for key, value in candidate.items() if key != "claveElector"}
    for field in _ROI_COMPARISON_FIELDS:
        if (
            primary.get(field) not in (None, "", {}, [])
            and candidate.get(field) not in (None, "", {}, [])
            and not _same_roi_value(field, primary[field], candidate[field])
        ):
            return f"{field}_conflict"
    for field in _ROI_COMPARISON_FIELDS:
        if (
            primary.get(field) not in (None, "", {}, [])
            and candidate.get(field) in (None, "", {}, [])
        ):
            return f"{field}_missing"
    for field in _ROI_REQUIRED_FIELDS:
        if candidate.get(field) in (None, "", {}, []):
            return f"{field}_required_missing"
    return None


class DocumentPipeline:
    def __init__(
        self,
        *,
        line_reader=None,
        preprocessor=None,
        qr_decoder=None,
        front_refiner=None,
        name_adapter=None,
        cross_channel_name_consensus: bool = False,
    ) -> None:
        self.line_reader = line_reader or PaddleLineReader()
        self.preprocessor = preprocessor or CardPreprocessor()
        self.qr_decoder = qr_decoder or QRDecoder()
        self.front_refiner = front_refiner
        self.cross_channel_name_consensus = cross_channel_name_consensus
        selector_path = os.getenv("INE_OCR_FIELD_SELECTOR")
        if self.front_refiner is None and selector_path:
            from .front_refinement import FrontRefiner
            self.front_refiner = FrontRefiner(self.line_reader, selector_path)
        self.name_adapter = name_adapter
        name_model_path = os.getenv("INE_OCR_NAME_MODEL_PATH")
        if self.name_adapter is None and name_model_path:
            from .name_onnx import NameLineAdapter, NameOnnxRecognizer

            threshold = float(os.getenv("INE_OCR_NAME_MODEL_THRESHOLD", "0.9"))
            cpu_threads = int(os.getenv("INE_OCR_NAME_MODEL_CPU_THREADS", "1"))
            self.name_adapter = NameLineAdapter(
                self.line_reader,
                NameOnnxRecognizer(name_model_path, cpu_threads=cpu_threads),
                threshold=threshold,
            )

    def ready(self) -> None:
        seen = set()
        for component in (
            self.preprocessor,
            self.line_reader,
            self.qr_decoder,
            self.front_refiner,
            self.name_adapter,
        ):
            if component is None or id(component) in seen:
                continue
            seen.add(id(component))
            ready_method = getattr(component, "ready", None)
            if callable(ready_method):
                ready_method()

    def _front_fields(self, standardized, detail, natural) -> tuple[dict, dict]:
        attempts = 0
        fallback = ChannelCandidate("standardized_natural", "0", {})
        if natural.shape[:2] == (CARD_HEIGHT, CARD_WIDTH):
            standardized_natural = natural
        else:
            _, standardized_natural, _ = standardize(natural)
        sources = (
            ("standardized_natural", standardized_natural),
            ("rectified_detail", detail),
            ("standardized_card", standardized),
        )
        for source, image in sources:
            for orientation, candidate_image in _rotations(image):
                attempts += 1
                lines = self.line_reader.read_lines(candidate_image)
                fields = extract_known_fields(lines)
                fallback_fields = dict(fallback.fields)
                for field, value in fields.items():
                    if value not in (None, "", {}, []) and fallback_fields.get(field) in (
                        None, "", {}, []
                    ):
                        fallback_fields[field] = value
                if fallback_fields != fallback.fields:
                    fallback = ChannelCandidate(source, orientation, fallback_fields)
                if fields.get("curp") and is_valid_curp(fields["curp"]):
                    current_clave = str(fields.get("claveElector") or "")
                    if current_clave and (
                        not CLAVE_ELECTOR_VALUE.fullmatch(current_clave)
                        or current_clave[6:12] != fields["curp"][4:10]
                    ):
                        fields.pop("claveElector", None)
                    fallback_clave = str(fallback.fields.get("claveElector") or "")
                    if (
                        not fields.get("claveElector")
                        and CLAVE_ELECTOR_VALUE.fullmatch(fallback_clave)
                        and fallback_clave[6:12] == fields["curp"][4:10]
                    ):
                        fields["claveElector"] = fallback_clave
                    detail_rescue_fields = []
                    refinement_input = {"image": candidate_image, "lines": lines}
                    detail_rescue_source = None
                    detail_rescue_orientation = None
                    roi_sidecar_processing = None
                    if source != "rectified_detail":
                        detail_rescue_source = "rectified_detail"
                        detail_rescue_orientation = orientation
                        rescue_image = detail
                        layout_rotation = reading_rotation(lines)
                        if source == "standardized_natural" and layout_rotation:
                            rescue_image = natural
                            detail_rescue_source = "upright_natural"
                            detail_rescue_orientation = str((int(orientation) + layout_rotation) % 360)
                        oriented_detail = next(
                            image for name, image in _rotations(rescue_image)
                            if name == detail_rescue_orientation
                        )
                        detail_lines = None
                        detail_fields = {}
                        detail_evidence_image = oriented_detail
                        sidecar_reader = getattr(
                            self.line_reader, "read_lines_from_regions_sidecar", None
                        )
                        if (
                            source == "standardized_natural"
                            and callable(sidecar_reader)
                            and getattr(self.line_reader, "roi_sidecar_enabled", True)
                        ):
                            roi_sidecar_processing = {"attempted": True}
                            try:
                                sidecar_image, sidecar_input_lines = _oriented_detail_rois(
                                    lines,
                                    natural,
                                    rescue_image,
                                    orientation,
                                    detail_rescue_orientation,
                                    (
                                        getattr(self.preprocessor, "last_geometry", None)
                                        if rescue_image is detail else None
                                    ),
                                )
                                name_indices = _name_roi_indices(sidecar_input_lines)
                                sidecar_input_lines = _pad_roi_boxes(
                                    sidecar_input_lines, name_indices
                                )
                                sidecar_lines = sidecar_reader(
                                    sidecar_image,
                                    sidecar_input_lines,
                                )
                                sidecar_fields = extract_known_fields(sidecar_lines)
                                present_sidecar_fields = {
                                    field for field in _ROI_COMPARISON_FIELDS
                                    if sidecar_fields.get(field) not in (None, "", {}, [])
                                }
                                present_primary_fields = {
                                    field for field in _ROI_COMPARISON_FIELDS
                                    if fields.get(field) not in (None, "", {}, [])
                                }
                                roi_sidecar_processing.update({
                                    "presentFields": sorted(present_sidecar_fields),
                                    "missingRequiredFields": sorted(
                                        set(_ROI_REQUIRED_FIELDS) - present_sidecar_fields
                                    ),
                                    "missingPrimaryFields": sorted(
                                        present_primary_fields - present_sidecar_fields
                                    ),
                                    "acceptedFieldCount": len(present_sidecar_fields),
                                    "requiredFieldCount": sum(
                                        field in present_sidecar_fields for field in _ROI_REQUIRED_FIELDS
                                    ),
                                    "accepted": False,
                                    **getattr(self.line_reader, "last_roi_sidecar_stats", {}),
                                })
                                accepted = _accepted_roi_fields(fields, sidecar_fields)
                                if accepted is not None:
                                    detail_lines = sidecar_lines
                                    detail_fields = accepted
                                    detail_evidence_image = sidecar_image
                                    detail_rescue_source = "front_roi_sidecar"
                                    roi_sidecar_processing["accepted"] = True
                                else:
                                    roi_sidecar_processing["rejectedReason"] = (
                                        _roi_rejection_code(fields, sidecar_fields)
                                    )
                            except OCREngineUnavailable:
                                roi_sidecar_processing.update(
                                    getattr(self.line_reader, "last_roi_sidecar_stats", {})
                                )
                                pass
                        if detail_lines is None:
                            detail_lines = self.line_reader.read_lines(oriented_detail)
                            detail_fields = extract_known_fields(detail_lines)
                        if detail_fields.get("curp") not in (None, fields["curp"]):
                            detail_fields = {}
                        if detail_fields.get("curp") == fields["curp"] or (
                            detail_fields.get("nombres") and reading_rotation(detail_lines) == 0
                        ):
                            refinement_input = {
                                "image": detail_evidence_image, "lines": detail_lines
                            }
                        detail_clave = str(detail_fields.get("claveElector") or "")
                        if detail_clave and (
                            not CLAVE_ELECTOR_VALUE.fullmatch(detail_clave)
                            or detail_clave[6:12] != fields["curp"][4:10]
                        ):
                            detail_fields.pop("claveElector", None)
                        consensus = (
                            _name_consensus(fields, detail_fields)
                            if candidate_image.shape != oriented_detail.shape else {}
                        )
                        printed_candidates = list(dict.fromkeys([
                            *fields.get("_nameCandidates", []),
                            *detail_fields.get("_nameCandidates", []),
                        ]))
                        for field, value in detail_fields.items():
                            if value not in (None, "", {}, []) and fields.get(field) in (
                                None, "", {}, []
                            ):
                                fields[field] = value
                                detail_rescue_fields.append(field)
                        fields["_nameConsensus"] = consensus
                        fields["_nameCandidates"] = printed_candidates
                    if self.front_refiner is not None or self.name_adapter is not None:
                        fields["_refinementInput"] = refinement_input
                    processing = {
                        "source": source,
                        "orientation": orientation,
                        "attempts": attempts,
                        "detailRescue": bool(detail_rescue_fields),
                        "detailRescueFields": sorted(detail_rescue_fields),
                        "detailRescueSource": detail_rescue_source,
                        "detailRescueOrientation": detail_rescue_orientation,
                    }
                    if roi_sidecar_processing is not None:
                        processing["roiSidecar"] = roi_sidecar_processing
                    return fields, processing
        return fallback.fields, {
            "source": fallback.source,
            "orientation": fallback.orientation,
            "attempts": attempts,
            "detailRescue": False,
            "detailRescueFields": [],
        }

    def _mrz(self, detail, natural) -> tuple[str | None, dict | None, dict]:
        attempts = 0
        for source, image in (("rectified_detail", detail), ("natural", natural)):
            for orientation, candidate_image in _rotations(image):
                height = candidate_image.shape[0]
                for band_start in (0.45, 0.35):
                    attempts += 1
                    band = candidate_image[int(height * band_start):, :]
                    lines = self.line_reader.read_lines(band)
                    value = assemble_mrz(lines)
                    if not value:
                        break
                    try:
                        parsed = parse_mrz(value)
                    except (MRZError, IndexError, ValueError):
                        parsed = None
                    region_rescue = False
                    if (
                        self.front_refiner is not None
                        and not (parsed and parsed["validacionMRZAllOk"])
                        and band_start == 0.45
                        and any(re.search(r"\d{4}.{0,25}MEX", str(line.get("text") or "")) for line in lines)
                    ):
                        refined = assemble_mrz(self.front_refiner.read_regions(band, lines, minimum_text=0))
                        try:
                            rescued = parse_mrz(refined or "")
                        except (MRZError, IndexError, ValueError):
                            rescued = None
                        if rescued and rescued["validacionMRZAllOk"]:
                            value, parsed, region_rescue = refined, rescued, True
                    if not parsed:
                        continue
                    if parsed["validacionMRZAllOk"]:
                        return value, parsed, {
                            "source": source,
                            "orientation": orientation,
                            "bandStart": band_start,
                            "attempts": attempts,
                            **({"regionRescue": True} if region_rescue else {}),
                        }
        return None, None, {"source": None, "orientation": None, "attempts": attempts}

    def _qr(self, detail, natural) -> tuple[list[str], dict]:
        attempts = 0
        for source, image in (("rectified_detail", detail), ("natural", natural)):
            for orientation, candidate_image in _rotations(image):
                attempts += 1
                values = _official_qr_values(self.qr_decoder.decode(candidate_image))
                if values:
                    return values, {
                        "source": source,
                        "orientation": orientation,
                        "attempts": attempts,
                    }
        return [], {"source": None, "orientation": None, "attempts": attempts}

    def extract(self, front: bytes, back: bytes | None = None) -> dict:
        front_standardized, front_quality, front_detail, front_natural = self.preprocessor.process(
            front, "front"
        )
        warnings: list[str] = []
        try:
            front_fields, front_processing = self._front_fields(
                front_standardized, front_detail, front_natural
            )
        except OCREngineUnavailable:
            front_fields = {}
            front_processing = {"source": None, "orientation": None, "attempts": 0}
            warnings.append("OCR_ENGINE_UNAVAILABLE")

        mrz_value = None
        mrz_fields = None
        qr_values: list[str] = []
        back_quality = {"ok": None, "issues": ["not_provided"]}
        mrz_processing = {"source": None, "orientation": None, "attempts": 0}
        qr_processing = {"source": None, "orientation": None, "attempts": 0}
        if back is not None:
            back_standardized, back_quality, back_detail, back_natural = self.preprocessor.process(
                back, "back"
            )
            del back_standardized
            try:
                mrz_value, mrz_fields, mrz_processing = self._mrz(back_detail, back_natural)
            except OCREngineUnavailable:
                warnings.append("MRZ_OCR_ENGINE_UNAVAILABLE")
            qr_values, qr_processing = self._qr(back_detail, back_natural)

        result: dict = {
            "status": "OK",
            "tipo": "INE",
            "subTipo": None,
            "imageQuality": {"front": front_quality, "back": back_quality},
            "processing": {
                "front": front_processing,
                "mrz": mrz_processing,
                "qr": qr_processing,
            },
        }
        for field in (
            "curp", "claveElector", "registro", "estado", "municipio", "localidad",
            "seccion", "emision", "vigencia", "fechaNacimiento", "primerApellido",
            "segundoApellido", "nombres", "calle", "colonia", "ciudad", "sexo",
            "sexoRaw", "genero",
        ):
            if field in front_fields:
                result[field] = front_fields[field]

        field_sources: dict[str, str] = {}
        field_confidence: dict[str, float] = {}
        if result.get("curp"):
            field_sources["curp"] = "front_ocr_check_digit"
            field_confidence["curp"] = 0.99
        for field in (
            "claveElector", "registro", "estado", "municipio", "localidad", "seccion",
            "emision", "vigencia", "fechaNacimiento", "primerApellido", "segundoApellido",
            "nombres", "calle", "colonia", "ciudad", "sexo",
        ):
            if result.get(field) is not None:
                field_sources[field] = "front_ocr"
                field_confidence[field] = 0.80

        if mrz_fields:
            result["mrz"] = mrz_value
            result["validacionMRZ"] = {
                field: "OK" if valid else "DIFF"
                for field, valid in mrz_fields["validacionMRZ"].items()
                if valid is not None
            }
            for field in (
                "cic", "ocr", "identificadorCiudadano", "fechaNacimiento", "pais",
                "versionMRZ",
            ):
                if mrz_fields.get(field):
                    result[field] = mrz_fields[field]
                    field_sources[field] = "mrz_validated"
                    field_confidence[field] = 0.99
            if mrz_fields.get("vigencia"):
                result["vigencia"] = mrz_fields["vigencia"]
                field_sources["vigencia"] = "mrz_validated"
                field_confidence["vigencia"] = 0.99
            name_fields = ("nombres", "primerApellido", "segundoApellido")
            front_names = {field: result.get(field) for field in name_fields}
            mrz_names = {field: mrz_fields.get(field) for field in name_fields}
            printed_candidates = front_fields.get("_nameCandidates", [])
            surnames_corroborated = _printed_surnames_match_mrz(front_names, mrz_fields)
            for field in name_fields:
                consensus_name = front_fields.get("_nameConsensus", {}).get(field)
                printed_name = consensus_name or front_names[field]
                if not printed_name and mrz_names[field]:
                    matches = [
                        candidate
                        for candidate in printed_candidates
                        if _prefer_printed_name(field, candidate, mrz_names[field])
                    ]
                    if matches and len({_name_signature(value).replace(" ", "") for value in matches}) == 1:
                        printed_name = max(matches, key=lambda value: len(value.split()))
                        result[field] = printed_name
                baseline_consensus = consensus_name and (
                    _single_name_error(consensus_name, mrz_names[field])
                    or _prefer_printed_name("nombres", consensus_name, mrz_names[field])
                )
                cross_channel_consensus = (
                    self.cross_channel_name_consensus
                    and _cross_channel_name_consensus(field, front_fields, mrz_fields)
                )
                if baseline_consensus or cross_channel_consensus:
                    result[field] = _mrz_spaced_name(consensus_name, mrz_fields.get("nameText", ""))
                    field_sources[field] = (
                        "front_ocr_consensus" if baseline_consensus
                        else "front_ocr_cross_channel_consensus"
                    )
                    field_confidence[field] = 0.95
                elif _prefer_printed_name(field, printed_name, mrz_names[field]) or (
                    field in ("primerApellido", "segundoApellido") and surnames_corroborated
                ):
                    result[field] = _mrz_spaced_name(printed_name, mrz_fields.get("nameText", ""))
                    field_sources[field] = "front_ocr_mrz_corroborated"
                    field_confidence[field] = 0.99
                elif mrz_names[field]:
                    result[field] = mrz_names[field]
                    field_sources[field] = "mrz_unchecked_name"
                    field_confidence[field] = 0.80
                elif field == "nombres" and printed_name and surnames_corroborated:
                    spaced_name = _mrz_spaced_name(printed_name, mrz_fields.get("nameText", ""))
                    if spaced_name != printed_name:
                        result[field] = spaced_name
                        field_sources[field] = "front_ocr_mrz_word_boundaries"
            if not result.get("emision") and str(mrz_fields.get("vigencia") or "").isdigit():
                expiry_year = str(mrz_fields["vigencia"])
                issue_year = str(int(expiry_year) - 10)
                if any(
                    issue_year in pair and expiry_year in pair
                    for pair in front_fields.get("_yearPairs", [])
                ):
                    result["emision"] = issue_year
                    field_sources["emision"] = "front_ocr_mrz_corroborated"
                    field_confidence["emision"] = 0.99
            if result.get("sexo") is None:
                result["sexo"] = mrz_fields.get("sexo")
                result["sexoRaw"] = mrz_fields.get("sexoRaw")
                field_sources["sexo"] = "mrz_validated"
                field_confidence["sexo"] = 0.99

        if qr_values:
            qr_fields = extract_official_fields(qr_values)
            result["qrExtract"] = {"urls": qr_values}
            result["qrValidation"] = {
                "cic": "STRUCTURE_OK" if qr_fields.get("cic") else "UNAVAILABLE"
            }
            qr_cic = qr_fields.get("cic")
            if qr_cic and result.get("cic") and result["cic"] != qr_cic:
                result["qrValidation"]["cic"] = "DIFF"
                warnings.append("QR_CIC_MISMATCH")
            elif qr_cic and not result.get("cic"):
                result["cic"] = qr_cic
                field_sources["cic"] = "qr_official_token"
                field_confidence["cic"] = 0.99

        result["fieldSources"] = field_sources
        result["fieldConfidence"] = field_confidence
        if self.front_refiner is not None:
            self.front_refiner.refine(front, result, evidence=front_fields.get("_refinementInput"))
        if self.name_adapter is not None:
            evidence = front_fields.get("_refinementInput")
            predict_from_evidence = getattr(self.name_adapter, "predict_from_evidence", None)
            if evidence is not None and callable(predict_from_evidence):
                predictions, metadata = predict_from_evidence(
                    evidence["image"], evidence["lines"]
                )
            else:
                predictions, metadata = self.name_adapter.predict_from_bytes(front)
            self.name_adapter.apply_to_result(result, predictions)
            result["processing"]["nameModel"].update(metadata)

        channels = {
            "frontCurp": bool(result.get("curp")),
            "frontCurpCheckOk": bool(result.get("curp") and is_valid_curp(result["curp"])),
            "mrz": bool(mrz_fields),
            "mrzCheckOk": bool(mrz_fields and mrz_fields["validacionMRZAllOk"]),
            "qr": bool(qr_values),
        }
        channels["anyStructured"] = bool(
            channels["frontCurp"] or channels["mrz"] or channels["qr"]
        )
        result["channels"] = channels
        result["fieldSources"] = field_sources
        result["fieldConfidence"] = field_confidence
        if not front_quality.get("ok"):
            warnings.extend(f"FRONT_{issue.upper()}" for issue in front_quality.get("issues", []))
        if back is not None and not back_quality.get("ok"):
            warnings.extend(f"BACK_{issue.upper()}" for issue in back_quality.get("issues", []))
        if warnings:
            result["warnings"] = list(dict.fromkeys(warnings))
        return result
