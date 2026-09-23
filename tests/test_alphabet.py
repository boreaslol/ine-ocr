import pytest

from ine_ocr.alphabet import encode_text, greedy_decode


def test_ctc_greedy_decode_removes_blanks_and_repeats():
    assert greedy_decode([0, 1, 1, 0, 2, 2, 0]) == "AB"


def test_encode_rejects_unsupported_characters():
    with pytest.raises(ValueError, match="unsupported_characters"):
        encode_text("ABC-")
