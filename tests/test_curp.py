import numpy as np

from ine_ocr.alphabet import BLANK_INDEX, CHAR_TO_INDEX, NUM_CLASSES
from ine_ocr.curp import compute_check_digit, constrained_ctc_decode, is_valid_curp


def _valid_example() -> str:
    prefix = "AECD000101HDFBCD0"
    return prefix + compute_check_digit(prefix)


def test_curp_validation_checks_structure_and_digit():
    value = _valid_example()
    assert is_valid_curp(value)
    assert not is_valid_curp(value[:-1] + str((int(value[-1]) + 1) % 10))


def test_constrained_decoder_resolves_letter_digit_confusion():
    value = _valid_example()
    logits = np.full((len(value) * 2, NUM_CLASSES), -8.0, dtype=np.float32)
    for position, character in enumerate(value):
        character_frame = position * 2
        logits[character_frame, CHAR_TO_INDEX[character]] = 8.0
        logits[character_frame + 1, BLANK_INDEX] = 8.0
    logits[0, CHAR_TO_INDEX["0"]] = 9.0
    assert constrained_ctc_decode(logits, beam_width=8) == value
