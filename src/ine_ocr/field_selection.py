"""Feature-only ranking of observed front-field candidates on CPU."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np

FIELDS = ("primerApellido", "segundoApellido", "nombres", "calle", "colonia", "ciudad")
SOURCES = ("baseline", "small_upright", "english_regions")
FEATURE_VERSION = 1


def signature(value: str, *, compact: bool = False) -> str:
    plain = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.upper())
        if not unicodedata.combining(character)
    )
    if compact:
        return re.sub(r"[^A-Z0-9]", "", plain)
    return " ".join(plain.split())


def observations(baseline: dict, views: dict[str, dict], field: str) -> list[dict]:
    candidates = []
    for source, fields in (("baseline", baseline), *views.items()):
        value = fields.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        confidence = (
            fields.get("_nameScores", {}).get(field)
            if field in FIELDS[:3]
            else fields.get("_addressScore")
        )
        if confidence is None:
            confidence = fields.get("fieldConfidence", {}).get(field, 0.8)
        candidates.append(
            {
                "source": source,
                "value": value,
                "confidence": float(confidence),
                "complete_block": all(
                    fields.get(key)
                    for key in (FIELDS[:3] if field in FIELDS[:3] else FIELDS[3:])
                ),
            }
        )
    return candidates


def features(candidates: list[dict], field: str, baseline: str = "") -> np.ndarray:
    values = []
    for candidate in candidates:
        value = candidate["value"]
        normalized = signature(value)
        compact = signature(value, compact=True)
        agreements = sum(
            signature(other["value"]) == normalized for other in candidates
        )
        compact_agreements = sum(
            signature(other["value"], compact=True) == compact for other in candidates
        )
        values.append(
            [
                *(float(candidate["source"] == name) for name in SOURCES),
                *(float(field == name) for name in FIELDS),
                candidate["confidence"],
                float(candidate["complete_block"]),
                agreements / len(SOURCES),
                compact_agreements / len(SOURCES),
                min(len(value), 100) / 100,
                min(len(value.split()), 15) / 15,
                sum(character.isdigit() for character in value) / max(1, len(value)),
                float(bool(re.search(r"\d{5}\b", value))),
                float(", " in value),
                float(value.endswith(".")),
                float(bool(re.search(r"[A-Z]\d|\d[A-Z]", normalized))),
                float(normalized == signature(baseline)),
                float(compact == signature(baseline, compact=True)),
            ]
        )
    return np.asarray(values, dtype=np.float32)


def eligible(
    candidate: dict, candidates: list[dict], baseline: str, field: str
) -> bool:
    if candidate["source"] == "baseline":
        return True
    proposed = signature(candidate["value"], compact=True)
    original = signature(baseline, compact=True)
    if (
        field in FIELDS[:3]
        and proposed.startswith(original)
        and not preserves_word_boundaries(candidate["value"], baseline)
    ):
        return False
    if proposed == original:
        return True
    corroborations = sum(
        other["source"] != "baseline"
        and other["confidence"] >= 0.9
        and signature(other["value"], compact=True) == proposed
        for other in candidates
    )
    if corroborations < 2:
        return False
    if field in FIELDS[3:]:
        return True
    return not original or len(original) >= 4 and proposed.startswith(original)


def preserves_word_boundaries(proposed: str, baseline: str) -> bool:
    def boundaries(value):
        offsets = set()
        offset = 0
        for word in value.split()[:-1]:
            offset += len(signature(word, compact=True))
            offsets.add(offset)
        return offsets

    return boundaries(baseline) <= boundaries(proposed)


def preserve_printed_diacritics(proposed: str, baseline: str) -> str:
    if signature(proposed, compact=True) != signature(baseline, compact=True):
        return proposed
    original_letters = [character for character in baseline if character.isalnum()]
    proposed_letters = [character for character in proposed if character.isalnum()]
    if len(original_letters) != len(proposed_letters):
        return proposed
    offset = 0
    result = []
    for character in proposed:
        if character.isalnum():
            original = original_letters[offset]
            if ord(original) > 127:
                character = original
            offset += 1
        result.append(character)
    return "".join(result)


def consensus_choice(candidates: list[dict], baseline: str) -> dict | None:
    supported = [
        candidate
        for candidate in candidates
        if candidate["source"] != "baseline" and candidate["confidence"] >= 0.95
    ]
    if (
        len(supported) < 2
        or len({signature(candidate["value"]) for candidate in supported}) != 1
    ):
        return None
    proposed = signature(supported[0]["value"], compact=True)
    original = signature(baseline, compact=True)
    if proposed != original and not (
        len(original) >= 4 and proposed.startswith(original)
    ):
        return None
    if not preserves_word_boundaries(supported[0]["value"], baseline):
        return None
    return {**supported[0], "source": "consensus"}


class FieldSelector:
    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort

        path = Path(model_path).expanduser().resolve()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("feature_version") != str(FEATURE_VERSION):
            raise ValueError("field_selector_feature_version_mismatch")

    def select(self, baseline: dict, views: dict[str, dict]) -> dict[str, dict]:
        selected = {}
        for field in FIELDS:
            candidates = observations(baseline, views, field)
            if not candidates:
                continue
            matrix = features(candidates, field, str(baseline.get(field) or ""))
            logits = self.session.run(None, {"features": matrix})[0].reshape(-1)
            choices = [
                (float(score), candidate)
                for score, candidate in zip(logits, candidates)
                if eligible(
                    candidate, candidates, str(baseline.get(field) or ""), field
                )
            ]
            if choices:
                score, choice = max(choices, key=lambda item: item[0])
                if field in FIELDS[:3]:
                    choice = (
                        consensus_choice(candidates, str(baseline.get(field) or ""))
                        or choice
                    )
                choice = {
                    **choice,
                    "value": preserve_printed_diacritics(
                        choice["value"], str(baseline.get(field) or "")
                    ),
                }
                if choice["value"] != baseline.get(field):
                    selected[field] = {**choice, "rank_score": score}
        return selected
