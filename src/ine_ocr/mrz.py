"""INE machine-readable zone assembly, parsing, and ICAO checks."""

from __future__ import annotations

import datetime as dt
import re

MRZ_CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
MRZ_LINE = re.compile(r"^[A-Z0-9<]{20,}$")
INE_PREFIX = re.compile(r"[I1L][D0O8][M8]EX")
GENDER_NORMALIZATION = {"H": "H", "M": "M", "X": "", "NB": ""}


class MRZError(ValueError):
    pass


def icao_check_digit(value: str) -> int:
    total = 0
    for index, character in enumerate(value):
        character_value = MRZ_CHARACTERS.index(character) if character in MRZ_CHARACTERS else 0
        total += character_value * (7, 3, 1)[index % 3]
    return total % 10


def _check(value: str, digit: str) -> bool:
    return digit.isdigit() and icao_check_digit(value) == int(digit)


def _birth_date(value: str) -> str:
    if len(value) != 6 or not value.isdigit():
        raise MRZError("mrz_birth_date_invalid")
    threshold = (dt.date.today().year - 18) % 100
    year_value = int(value[:2])
    year = 1900 + year_value if year_value > threshold else 2000 + year_value
    return f"{value[4:6]}/{value[2:4]}/{year}"


def _canonical_prefix(value: str) -> tuple[int, str] | None:
    match = INE_PREFIX.search(value)
    if match is None:
        return None
    return match.start(), "IDMEX" + value[match.end():]


def assemble(lines: list[dict]) -> str | None:
    rows: list[tuple[float, str]] = []
    for line in lines:
        text = re.sub(r"[^A-Z0-9<]", "", str(line.get("text") or "").replace("≤", "<").replace("＜", "<").replace(" ", "").upper())
        if len(text) < 20 or not (MRZ_LINE.fullmatch(text) or "IDMEX" in text):
            continue
        box = line.get("box") or []
        y_position = min((point[1] for point in box), default=0)
        rows.append((float(y_position), text))
    if not rows:
        return None
    rows.sort()
    for index, (_, text) in enumerate(rows):
        normalized = _canonical_prefix(text)
        if normalized is not None:
            _, canonical = normalized
            return canonical + "".join(row[1] for row in rows[index + 1:index + 3])
    return "".join(row[1] for row in rows[:3])


def parse(value: str) -> dict:
    compact = (value or "").strip().replace(" ", "")
    normalized = _canonical_prefix(compact[:5])
    if normalized is None or normalized[0] != 0:
        raise MRZError("mrz_prefix_invalid")
    compact = normalized[1] + compact[5:]
    remainder = compact[5:]
    if len(remainder) < 40:
        raise MRZError("mrz_too_short")
    cic, cic_digit = remainder[:9], remainder[9]
    if remainder[10:12] != "<<":
        raise MRZError("mrz_cic_separator_invalid")
    segment = remainder[12:]
    ocr_identifier = segment[:13]
    birth, birth_digit = segment[13:19], segment[19]
    gender = segment[20]
    expiry, expiry_digit = segment[21:27], segment[27]
    country = segment[28:31]
    tail = segment[31:]
    version = tail[1:3] if tail.startswith("<") else ""
    composite = ""
    composite_digit = ""
    if "<<" in tail:
        after_separator = tail.split("<<", 1)[1]
        composite = after_separator[:5]
        composite_digit = after_separator[6:7] if len(after_separator) > 6 else ""

    name_part = ""
    if composite:
        marker = composite + "<" + composite_digit
        marker_index = compact.find(marker)
        if marker_index >= 0:
            name_part = compact[marker_index + len(marker):].strip("<")
    given_names = paternal_name = maternal_name = ""
    if name_part:
        sections = [section for section in name_part.split("<<") if section]
        if sections:
            surnames = sections[0].split("<")
            paternal_name = surnames[0] if surnames else ""
            maternal_name = surnames[1] if len(surnames) > 1 else ""
        if len(sections) > 1:
            given_names = sections[1].replace("<", " ").strip()

    validation = {
        "cic": _check(cic, cic_digit),
        "fechaNacimiento": _check(birth, birth_digit),
        "vigencia": _check(expiry, expiry_digit),
        "composite": None,
    }
    return {
        "cic": cic,
        "ocr": ocr_identifier,
        "identificadorCiudadano": ocr_identifier[-9:] if len(ocr_identifier) == 13 else "",
        "fechaNacimiento": _birth_date(birth),
        "sexo": GENDER_NORMALIZATION.get(gender, gender),
        "sexoRaw": gender,
        "vigencia": f"20{expiry[:2]}",
        "pais": country,
        "versionMRZ": version,
        "composite": composite,
        "primerApellido": paternal_name,
        "segundoApellido": maternal_name,
        "nombres": given_names,
        "nameText": name_part,
        "validacionMRZ": validation,
        "validacionMRZAllOk": all(result for result in validation.values() if result is not None),
    }
