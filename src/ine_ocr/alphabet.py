"""Character vocabulary and CTC encoding helpers."""

from __future__ import annotations

import unicodedata

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
LATIN_NAME_ALPHABET = ALPHABET + " ÁÀÂÃÄÅÆÇÉÈÊËÍÌÎÏÑÓÒÔÕÖØÚÙÛÜÝŸŽŠŁĐŒ-.'"
BLANK_INDEX = 0
CHAR_TO_INDEX = {character: index + 1 for index, character in enumerate(ALPHABET)}
INDEX_TO_CHAR = {index: character for character, index in CHAR_TO_INDEX.items()}
NUM_CLASSES = len(ALPHABET) + 1


def normalize_text(value: str, *, alphabet: str = ALPHABET) -> str:
    normalized = unicodedata.normalize("NFC", (value or "").upper())
    return (" " if " " in alphabet else "").join(normalized.split())


def validate_text(value: str, *, alphabet: str = ALPHABET) -> str:
    text = normalize_text(value, alphabet=alphabet)
    unexpected = sorted(set(text) - set(alphabet))
    if unexpected:
        raise ValueError(f"unsupported_characters:{''.join(unexpected)}")
    if not text:
        raise ValueError("empty_label")
    return text


def encode_text(value: str, *, alphabet: str = ALPHABET):
    text = validate_text(value, alphabet=alphabet)
    import torch

    mapping = {character: index + 1 for index, character in enumerate(alphabet)}
    return torch.tensor([mapping[character] for character in text], dtype=torch.long)


def greedy_decode(indices: list[int], *, alphabet: str = ALPHABET) -> str:
    characters: list[str] = []
    previous = -1
    for index in indices:
        if index != previous and index != BLANK_INDEX:
            character = alphabet[int(index) - 1] if 1 <= int(index) <= len(alphabet) else None
            if character is not None:
                characters.append(character)
        previous = int(index)
    return "".join(characters)
