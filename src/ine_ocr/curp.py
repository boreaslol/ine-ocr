"""CURP validation and format-constrained CTC decoding."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

from .alphabet import BLANK_INDEX, CHAR_TO_INDEX, INDEX_TO_CHAR, greedy_decode

CURP_LENGTH = 18
LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VOWELS_OR_X = frozenset("AEIOUX")
CONSONANTS_OR_X = frozenset("BCDFGHJKLMNPQRSTVWXYZ")
DIGITS = frozenset("0123456789")
ALPHANUMERIC = LETTERS | DIGITS
STATE_CODES = frozenset({
    "AS", "BC", "BS", "CC", "CH", "CL", "CM", "CS", "DF", "DG", "GR",
    "GT", "HG", "JC", "MC", "MN", "MS", "NE", "NL", "NT", "OC", "PL",
    "QR", "QT", "SL", "SP", "SR", "TC", "TL", "TS", "VZ", "YN", "ZS",
})
CURP_PATTERN = re.compile(r"^[A-Z][AEIOUX][A-Z]{2}\d{6}[HMX][A-Z]{2}[BCDFGHJKLMNPQRSTVWXYZ]{3}[A-Z0-9]\d$")
CHECKSUM_ALPHABET = "0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
CHECKSUM_VALUES = {character: index for index, character in enumerate(CHECKSUM_ALPHABET)}
NEGATIVE_INFINITY = float("-inf")


def compute_check_digit(prefix: str) -> str:
    if len(prefix) != CURP_LENGTH - 1:
        raise ValueError("curp_checksum_prefix_length_invalid")
    try:
        total = sum(CHECKSUM_VALUES[character] * (CURP_LENGTH - index)
                    for index, character in enumerate(prefix))
    except KeyError as error:
        raise ValueError("curp_checksum_prefix_character_invalid") from error
    return str((10 - total % 10) % 10)


def _date_is_valid(value: str) -> bool:
    if len(value) != 6 or not value.isdigit():
        return False
    month = int(value[2:4])
    day = int(value[4:6])
    if not 1 <= month <= 12:
        return False
    days_in_month = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    return 1 <= day <= days_in_month[month - 1]


def is_valid_curp(value: str) -> bool:
    if not CURP_PATTERN.fullmatch(value):
        return False
    return (
        _date_is_valid(value[4:10])
        and value[11:13] in STATE_CODES
        and value[-1] == compute_check_digit(value[:-1])
    )


def _allowed_extensions(prefix: str) -> frozenset[str]:
    position = len(prefix)
    if position >= CURP_LENGTH:
        return frozenset()
    if position == 0 or position in (2, 3):
        return LETTERS
    if position == 1:
        return VOWELS_OR_X
    if position in (4, 5):
        return DIGITS
    if position == 6:
        return frozenset("01")
    if position == 7:
        return frozenset("123456789") if prefix[6] == "0" else frozenset("012")
    if position == 8:
        return frozenset("0123")
    if position == 9:
        return frozenset(
            digit for digit in DIGITS if _date_is_valid(prefix[4:9] + digit)
        )
    if position == 10:
        return frozenset("HMX")
    if position == 11:
        return frozenset(code[0] for code in STATE_CODES)
    if position == 12:
        return frozenset(code[1] for code in STATE_CODES if code[0] == prefix[11])
    if position in (13, 14, 15):
        return CONSONANTS_OR_X
    if position == 16:
        return ALPHANUMERIC
    return frozenset(compute_check_digit(prefix))


def _log_add(left: float, right: float) -> float:
    if left == NEGATIVE_INFINITY:
        return right
    if right == NEGATIVE_INFINITY:
        return left
    maximum = max(left, right)
    return maximum + math.log1p(math.exp(-abs(left - right)))


def _update_beam(beams: dict[str, tuple[float, float]], prefix: str, *,
                 blank: float | None = None, non_blank: float | None = None) -> None:
    old_blank, old_non_blank = beams.get(
        prefix, (NEGATIVE_INFINITY, NEGATIVE_INFINITY)
    )
    beams[prefix] = (
        _log_add(old_blank, blank) if blank is not None else old_blank,
        _log_add(old_non_blank, non_blank) if non_blank is not None else old_non_blank,
    )


def constrained_ctc_decode(logits: np.ndarray | Sequence[Sequence[float]], *,
                           beam_width: int = 32) -> str:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("curp_logits_must_be_time_by_class")
    if beam_width < 1:
        raise ValueError("curp_beam_width_must_be_positive")
    maximum = values.max(axis=1, keepdims=True)
    normalized = values - maximum
    log_probabilities = normalized - np.log(np.exp(normalized).sum(axis=1, keepdims=True))
    beams: dict[str, tuple[float, float]] = {
        "": (0.0, NEGATIVE_INFINITY),
    }
    for frame in log_probabilities:
        next_beams: dict[str, tuple[float, float]] = {}
        for prefix, (blank_probability, non_blank_probability) in beams.items():
            total_probability = _log_add(blank_probability, non_blank_probability)
            _update_beam(
                next_beams,
                prefix,
                blank=total_probability + float(frame[BLANK_INDEX]),
            )
            if prefix:
                repeated_index = CHAR_TO_INDEX[prefix[-1]]
                _update_beam(
                    next_beams,
                    prefix,
                    non_blank=non_blank_probability + float(frame[repeated_index]),
                )
            for character in _allowed_extensions(prefix):
                character_index = CHAR_TO_INDEX[character]
                if prefix and character == prefix[-1]:
                    extension_probability = blank_probability + float(frame[character_index])
                else:
                    extension_probability = total_probability + float(frame[character_index])
                _update_beam(
                    next_beams,
                    prefix + character,
                    non_blank=extension_probability,
                )
        ranked = sorted(
            next_beams.items(),
            key=lambda item: _log_add(*item[1]),
            reverse=True,
        )
        beams = dict(ranked[:beam_width])
    complete = [
        (prefix, _log_add(*probabilities))
        for prefix, probabilities in beams.items()
        if len(prefix) == CURP_LENGTH and is_valid_curp(prefix)
    ]
    if complete:
        return max(complete, key=lambda item: item[1])[0]
    indices = values.argmax(axis=-1).tolist()
    return greedy_decode(indices)
