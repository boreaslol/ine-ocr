"""Label-independent name-region composition for optional user-provided models."""

from PIL import Image

from .document_fields import _plain, _spatial_name_block
from .front_refinement import crop_lines


def name_regions(lines: list[dict]) -> dict[str, list[dict]]:
    rows = _spatial_name_block(lines)
    particles = {"DE", "DEL", "LA", "LAS", "LOS", "Y", "DE LA", "DE LAS", "DE LOS"}
    groups = []
    for row in rows:
        previous = " ".join(part["text"] for part in groups[-1]) if groups else ""
        if groups and _plain(previous) in particles:
            groups[-1].append(row)
        else:
            groups.append([row])
    if len(groups) < 3:
        return {}
    return {
        "primerApellido": groups[0],
        "segundoApellido": groups[1],
        "nombres": [line for group in groups[2:] for line in group],
    }


def compose_name_crop(image, lines: list[dict], height: int = 48) -> Image.Image:
    import cv2

    crops, indices = crop_lines(image, lines, minimum_text=1)
    if len(indices) != len(lines) or not crops:
        raise ValueError("name_region_geometry_incomplete")
    resized = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        width = max(1, round(gray.shape[1] * height / gray.shape[0]))
        if width > 4096:
            raise ValueError("name_region_aspect_ratio_invalid")
        resized.append(Image.fromarray(gray).resize((width, height), Image.Resampling.BICUBIC))
    gap = max(4, height // 6)
    width = sum(crop.width for crop in resized) + gap * (len(resized) - 1) + gap * 2
    canvas = Image.new("L", (width, height), 255)
    offset = gap
    for crop in resized:
        canvas.paste(crop, (offset, 0))
        offset += crop.width + gap
    return canvas
