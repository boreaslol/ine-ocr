"""Deterministic field extraction for INE front-side OCR lines."""

from __future__ import annotations

import re
import statistics
import unicodedata
from datetime import date

from .curp import is_valid_curp

CURP_CANDIDATE = re.compile(r"\b([A-Z][AEIOUX][A-Z]{2}\d{6}[HMX][A-Z]{5}[0-9A-Z]\d)\b")
CLAVE_ELECTOR = re.compile(r"CLAVE\s*DE\s*ELECTOR[:\s]*([A-Z0-9]{17,18})")
CLAVE_ELECTOR_VALUE = re.compile(r"\b([A-Z]{6}\d{8}[HMX]\d{3})\b")
VIGENCIA = re.compile(r"VIGENCIA[:\s]*(\d{4})(?:\s*[-–]\s*(\d{4}))?")
EMISION = re.compile(r"EMISION[:\s]*(\d{4})")
REGISTRO = re.compile(r"ANO\s*DE\s*REGISTRO[:\s]*(\d{4})\s*(\d{2})")
BIRTH_DATE = re.compile(r"FECHA\s*DE\s*NACIMIENTO[:\s]*(\d{2})[/.-](\d{2})[/.-](\d{4})")
GENDER_LINE = re.compile(r"\b(SEXO|GENERO)[:\s]*(H|M|NB|X)\b")
NUMBER_FIELDS = {
    "estado": (re.compile(r"\bESTADO[:\s]*(\d{1,2})\b"), 2),
    "municipio": (re.compile(r"\bMUNICIPIO[:\s]*(\d{1,3})\b"), 3),
    "localidad": (re.compile(r"\bLOCALIDAD[:\s]*(\d{1,4})\b"), 4),
    "seccion": (re.compile(r"\bSECCION[:\s]*(\d{1,4})\b"), 4),
}
GENDER_NORMALIZATION = {"H": "H", "M": "M", "NB": "", "X": ""}
YEAR_VALUE = r"(?:19|20)\d{2}"

STOP_LABELS = (
    "DOMICILIO", "CLAVE DE ELECTOR", "CURP", "ANO DE REGISTRO", "FECHA DE NACIMIENTO",
    "ESTADO", "MUNICIPIO", "LOCALIDAD", "SECCION", "EMISION", "VIGENCIA", "SEXO",
    "GENERO",
)


def _plain(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.upper())
    return " ".join("".join(character for character in normalized if not unicodedata.combining(character)).split())


def _clean_value(value: str) -> str:
    return " ".join(value.upper().strip(" :;,-").split())


def _bounds(line: dict) -> tuple[float, float, float, float] | None:
    box = line.get("box") or []
    if len(box) < 4:
        return None
    left, right = min(point[0] for point in box), max(point[0] for point in box)
    top, bottom = min(point[1] for point in box), max(point[1] for point in box)
    return (left, top, right, bottom) if right > left and bottom > top else None


def reading_rotation(lines: list[dict]) -> int | None:
    positioned = [line for line in lines if _bounds(line)]
    if len(positioned) < 3:
        return None
    vertical = sum(
        bounds[3] - bounds[1] > (bounds[2] - bounds[0]) * 1.5
        for line in positioned if (bounds := _bounds(line))
    ) > len(positioned) / 2
    anchors = []
    for line in positioned:
        compact = re.sub(r"[^A-Z]", "", _plain(str(line.get("text") or "")))
        rank = next((rank for marker, rank in (
            ("INSTITUTONACIONAL", 0), ("CREDENCIALPARA", 0),
            ("NOMBRE", 1), ("DOMICILIO", 2), ("CLAVEDELECTOR", 3),
            ("CLAVEDEELECTOR", 3), ("CURP", 3),
        ) if compact.startswith(marker)), None)
        if rank is None and any(
            is_valid_curp(match.group(1))
            for match in CURP_CANDIDATE.finditer(_plain(str(line.get("text") or "")))
        ):
            rank = 3
        if rank is not None:
            bounds = _bounds(line)
            anchors.append((rank, (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2))
    direction = sum(
        (1 if later[coordinate] > earlier[coordinate] else -1)
        for earlier in anchors for later in anchors
        if earlier[0] < later[0]
        for coordinate in (1 if vertical else 2,)
    )
    if not direction:
        return None
    return (90 if direction > 0 else 270) if vertical else (0 if direction > 0 else 180)


def _layout_lines(lines: list[dict]) -> list[dict]:
    orientation = reading_rotation(lines)
    if orientation is None:
        return lines
    transformed = []
    for line in lines:
        box = line.get("box") or []
        if orientation == 90:
            box = [[-point[1], point[0]] for point in box]
        elif orientation == 270:
            box = [[point[1], -point[0]] for point in box]
        elif orientation == 180:
            box = [[-point[0], -point[1]] for point in box]
        transformed.append({**line, "box": box})
    return sorted(transformed, key=lambda line: (
        min((point[1] for point in line["box"]), default=0),
        min((point[0] for point in line["box"]), default=0),
    ))


def _spatial_name_block(lines: list[dict]) -> list[dict]:
    for index, label in enumerate(lines):
        if _plain(str(label.get("text") or "")) != "NOMBRE" or not _bounds(label):
            continue
        left, top, right, bottom = _bounds(label)
        line_height = bottom - top
        selected = []
        address_top = min((
            _bounds(candidate)[1] for candidate in lines
            if _bounds(candidate)
            and _plain(str(candidate.get("text") or "")) == "DOMICILIO"
            and _bounds(candidate)[1] > bottom
            and abs(_bounds(candidate)[0] - left) < line_height * 3
        ), default=None)
        for candidate in lines[index + 1:]:
            bounds = _bounds(candidate)
            if not bounds:
                continue
            candidate_left, candidate_top, _, _ = bounds
            if abs(candidate_left - left) > max(line_height * 2, (right - left) * 0.55):
                continue
            if candidate_top < top or candidate_top - bottom > line_height * 10:
                continue
            if address_top is not None and candidate_top >= address_top:
                break
            value = _clean_value(str(candidate.get("text") or ""))
            plain = _plain(value)
            if any(plain == stop or plain.startswith(stop + " ") for stop in STOP_LABELS):
                break
            if not re.fullmatch(r"[A-ZÀ-ÖØ-Þ .'-]{2,80}", value) or float(candidate.get("score", 1)) < 0.8:
                continue
            if "INSTITUTO" in plain or "CREDENCIAL" in plain:
                continue
            selected.append({**candidate, "text": value, "score": float(candidate.get("score", 1))})
            if len(selected) >= (6 if address_top is not None else 3):
                break
        return _merge_rows(selected)
    return []


def _merge_rows(lines: list[dict]) -> list[dict]:
    rows = []
    for line in lines:
        bounds = _bounds(line)
        if not bounds:
            rows.append(dict(line))
            continue
        left, top, right, bottom = bounds
        previous = rows[-1] if rows else None
        previous_bounds = _bounds(previous) if previous else None
        if previous_bounds:
            prior_left, prior_top, prior_right, prior_bottom = previous_bounds
            height = min(bottom - top, prior_bottom - prior_top)
            overlap = min(bottom, prior_bottom) - max(top, prior_top)
            if overlap > height * 0.55 and left >= prior_right - height * 0.2:
                previous["text"] += " " + line["text"]
                previous["score"] = min(previous.get("score", 1), line.get("score", 1))
                previous["box"] = [
                    [prior_left, min(top, prior_top)], [right, min(top, prior_top)],
                    [right, max(bottom, prior_bottom)], [prior_left, max(bottom, prior_bottom)],
                ]
                continue
        rows.append(dict(line))
    return rows


def _name_slots(rows: list[dict]) -> list[dict]:
    slots = []
    particles = {"DE", "DEL", "LA", "LAS", "LOS", "Y", "DE LA", "DE LAS", "DE LOS"}
    for row in rows:
        if slots and _plain(slots[-1]["text"]) in particles:
            slots[-1] = {
                "text": slots[-1]["text"] + " " + row["text"],
                "score": min(slots[-1].get("score", 1), row.get("score", 1)),
            }
        else:
            slots.append(dict(row))
    return slots


def _spatial_address_block(lines: list[dict]) -> list[dict]:
    def left_midpoint(line):
        points = sorted(line.get("box") or [], key=lambda point: point[0])[:2]
        return statistics.mean(point[1] for point in points) if len(points) == 2 else 0

    for anchor in lines:
        if _plain(str(anchor.get("text") or "")) != "DOMICILIO" or not _bounds(anchor):
            continue
        left, top, right, bottom = _bounds(anchor)
        height = bottom - top
        selected = []
        for candidate in sorted(lines, key=left_midpoint):
            bounds = _bounds(candidate)
            candidate_y = left_midpoint(candidate)
            if not bounds or candidate_y < left_midpoint(anchor) + height * 0.45:
                continue
            if bounds[1] > bottom + height * 8 or bounds[0] < left - height * 2:
                continue
            plain = _plain(str(candidate.get("text") or ""))
            compact = re.sub(r"[^A-Z]", "", plain)
            if any(compact.startswith(label.replace(" ", "")) for label in STOP_LABELS):
                if abs(bounds[0] - left) < height * 3:
                    break
                continue
            if float(candidate.get("score", 1)) < 0.8:
                continue
            selected.append({**candidate, "text": _clean_value(str(candidate.get("text") or ""))})
        rows = _merge_rows(selected)
        postal_rows = [index for index, row in enumerate(rows) if re.search(r"\d{5}\b", row["text"])]
        if len(postal_rows) == 1:
            postal_index = postal_rows[0]
            if 1 <= postal_index <= 4 and postal_index + 2 == len(rows):
                colony_start = next((index for index, row in enumerate(rows[:postal_index + 1])
                                     if re.match(r"^(?:COL|FRACC|LOC|U HAB|BARR|AMPL|RANCHO|EJIDO|RCHO)\b", row["text"])), postal_index)
                if colony_start == 0:
                    continue
                street_rows = rows[:colony_start]
                colony_rows = rows[colony_start:postal_index + 1]
                return [
                    {"text": " ".join(row["text"] for row in street_rows),
                     "score": min(row.get("score", 1) for row in street_rows)},
                    {"text": " ".join(row["text"] for row in colony_rows),
                     "score": min(row.get("score", 1) for row in colony_rows)}, rows[-1],
                ]
    return []


def _spatial_label_value(lines: list[dict], label: str, pattern: str) -> re.Match | None:
    for anchor in lines:
        plain = _plain(str(anchor.get("text") or ""))
        if re.sub(r"\s", "", plain) != label or not _bounds(anchor):
            continue
        left, top, right, bottom = _bounds(anchor)
        height = bottom - top
        choices = []
        for candidate in lines:
            bounds = _bounds(candidate)
            if not bounds or candidate is anchor:
                continue
            candidate_left, candidate_top, _, candidate_bottom = bounds
            if candidate_top < top or candidate_bottom - bottom > 3.5 * height:
                continue
            if abs(candidate_left - left) > max(height * 2, (right - left) * 0.55):
                continue
            value = _plain(str(candidate.get("text") or ""))
            match = re.fullmatch(pattern, value)
            if match is not None and float(candidate.get("score", 1)) >= 0.8:
                distance = abs(candidate_top - bottom) + abs(candidate_left - left)
                choices.append((distance, match))
        if choices:
            return min(choices, key=lambda choice: choice[0])[1]
    return None


def _block(
    lines: list[str],
    label: str,
    *,
    maximum: int,
    ignored_stops: tuple[str, ...] = (),
) -> list[str]:
    for index, line in enumerate(lines):
        plain = _plain(line)
        if plain == label or plain.startswith(label + " "):
            values = []
            inline = _clean_value(line[len(line.split()[0]):]) if plain.startswith(label + " ") else ""
            if inline:
                values.append(inline)
            for candidate in lines[index + 1:]:
                candidate_plain = _plain(candidate)
                if any(
                    candidate_plain == stop or candidate_plain.startswith(stop + " ")
                    for stop in ignored_stops
                ):
                    continue
                if any(candidate_plain == stop or candidate_plain.startswith(stop + " ") for stop in STOP_LABELS):
                    break
                cleaned = _clean_value(candidate)
                if cleaned and not any(
                    token in candidate_plain
                    for token in ("INSTITUTO NACIONAL", "CREDENCIAL PARA VOTAR")
                ):
                    values.append(cleaned)
                if len(values) >= maximum:
                    break
            return values
    return []


def extract_known_fields(lines: list[dict]) -> dict:
    lines = _layout_lines(lines)
    line_texts = [_clean_value(str(line.get("text") or "")) for line in lines]
    line_texts = [line for line in line_texts if line]
    full_text = "\n".join(line_texts)
    plain_text = _plain(full_text)
    output: dict = {}
    output["_yearPairs"] = []
    for line, plain_line in zip(line_texts, (_plain(value) for value in line_texts)):
        if "REGISTRO" in plain_line or "NACIMIENTO" in plain_line:
            continue
        years = list(dict.fromkeys(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", plain_line)))
        if len(years) >= 2:
            output["_yearPairs"].append(years)
    for match in CURP_CANDIDATE.finditer(plain_text):
        if is_valid_curp(match.group(1)):
            output["curp"] = match.group(1)
            break
    clave = CLAVE_ELECTOR.search(plain_text)
    if clave is None:
        clave = CLAVE_ELECTOR_VALUE.search(plain_text)
    if clave:
        value = re.sub(r"[^A-Z0-9]", "", clave.group(1))
        if len(value) in {17, 18}:
            output["claveElector"] = value
    vigencia = VIGENCIA.search(plain_text)
    if vigencia:
        if vigencia.group(2):
            output["emision"] = vigencia.group(1)
            output["vigencia"] = vigencia.group(2)
        else:
            output["vigencia"] = vigencia.group(1)
    emission = EMISION.search(plain_text)
    if emission:
        output["emision"] = emission.group(1)
    registration = REGISTRO.search(plain_text)
    if registration:
        output["registro"] = f"{registration.group(1)} {registration.group(2)}"
    birth_date = BIRTH_DATE.search(plain_text)
    if birth_date:
        output["fechaNacimiento"] = "/".join(birth_date.groups())
    if output.get("curp") and not output.get("fechaNacimiento"):
        dates = set()
        for match in re.finditer(r"\b(\d{2})[/.-](\d{2})[/.-]((?:19|20)\d{2})\b", plain_text):
            day, month, year = match.groups()
            if year[-2:] + month + day != output["curp"][4:10]:
                continue
            try:
                date(int(year), int(month), int(day))
            except ValueError:
                continue
            dates.add(f"{day}/{month}/{year}")
        if len(dates) == 1:
            output["fechaNacimiento"] = dates.pop()
    for field, (pattern, width) in NUMBER_FIELDS.items():
        match = pattern.search(plain_text)
        if match:
            output[field] = match.group(1).zfill(width)
        positioned_number = _spatial_label_value(lines, field.upper(), rf"(\d{{1,{width}}})")
        if positioned_number:
            output[field] = positioned_number.group(1).zfill(width)

    positioned_registration = _spatial_label_value(lines, "ANODEREGISTRO", rf"({YEAR_VALUE})\s*(\d{{2}})")
    if positioned_registration:
        output["registro"] = " ".join(positioned_registration.groups())

    validity = _spatial_label_value(lines, "VIGENCIA", f"({YEAR_VALUE})(?:\\s*[-– ]\\s*({YEAR_VALUE}))?")
    if validity:
        if validity.group(2):
            output["emision"], output["vigencia"] = validity.group(1), validity.group(2)
        else:
            output["vigencia"] = validity.group(1)
    issue = _spatial_label_value(lines, "EMISION", f"({YEAR_VALUE})")
    if issue:
        output["emision"] = issue.group(1)
    for field in ("emision", "vigencia"):
        if field in output and re.fullmatch(YEAR_VALUE, output[field]) is None:
            del output[field]

    names = _block(
        line_texts,
        "NOMBRE",
        maximum=3,
        ignored_stops=("SEXO", "GENERO"),
    )
    spatial_names = _name_slots(_spatial_name_block(lines))
    if spatial_names:
        names = [entry["text"] for entry in spatial_names]
    output["_nameCandidates"] = list(dict.fromkeys([
        *names,
        *[
            _clean_value(str(line.get("text") or "")) for line in lines
            if float(line.get("score", 1)) >= 0.95
            and re.fullmatch(r"[A-ZÀ-ÖØ-Þ .'-]{4,60}", _clean_value(str(line.get("text") or "")))
            and not any(label in _plain(str(line.get("text") or "")) for label in (*STOP_LABELS, "NOMBRE", "INSTITUTO", "CREDENCIAL"))
        ],
    ]))
    if len(names) >= 3:
        output["primerApellido"] = names[0]
        output["segundoApellido"] = names[1]
        output["nombres"] = " ".join(names[2:])
        if spatial_names:
            output["_nameScores"] = {
                "primerApellido": spatial_names[0]["score"],
                "segundoApellido": spatial_names[1]["score"],
                "nombres": statistics.mean(entry["score"] for entry in spatial_names[2:]),
            }
    address = _block(line_texts, "DOMICILIO", maximum=3)
    spatial_address = _spatial_address_block(lines)
    if spatial_address:
        address = [row["text"] for row in spatial_address]
        output["_addressScore"] = min(row.get("score", 1) for row in spatial_address)
    if len(address) >= 3:
        output["calle"], output["colonia"], output["ciudad"] = address[:3]
    elif len(address) == 2 and re.search(r"\b\d{5}\b", address[0]):
        output["colonia"], output["ciudad"] = address

    gender = GENDER_LINE.search(plain_text)
    visible = gender.group(2) if gender else None
    output["sexo"] = GENDER_NORMALIZATION.get(visible, visible) if visible else None
    output["sexoRaw"] = visible
    output["genero"] = {
        "tituloEnTarjeta": gender.group(1) if gender else None,
        "generoVisible": visible,
        "sexoRegistralCurp": output["curp"][10] if output.get("curp") else None,
        "sexoRegistralClave": (
            output["claveElector"][14]
            if len(output.get("claveElector", "")) >= 15
            else None
        ),
        "autoidentificacion": None,
    }
    return output
