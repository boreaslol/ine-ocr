"""NumPy/Pillow OCR-line preprocessing shared by training and CPU inference."""

from __future__ import annotations

import hashlib
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def seed_for(path: str, profile: str) -> int:
    digest = hashlib.sha256(f"{path}\0{profile}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _apply_stress(image: Image.Image, profile: str, generator: random.Random) -> Image.Image:
    if profile == "clean":
        return image
    if profile == "dark":
        return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(0.48)).enhance(0.82)
    if profile == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.25))
    if profile == "glare":
        overlay = Image.new("L", image.size, color=0)
        draw = ImageDraw.Draw(overlay)
        width, height = image.size
        band = max(10, width // 7)
        start = generator.randint(max(0, width // 8), max(width // 8, width - band))
        draw.polygon(
            [
                (start, 0),
                (min(width, start + band), 0),
                (min(width, start + band // 2), height),
                (max(0, start - band // 2), height),
            ],
            fill=150,
        )
        white = Image.new("L", image.size, color=255)
        return Image.composite(white, image, overlay)
    raise ValueError(f"unknown_stress_profile:{profile}")


def _augment(image: Image.Image, generator: random.Random) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(generator.uniform(0.55, 1.25))
    image = ImageEnhance.Contrast(image).enhance(generator.uniform(0.7, 1.35))
    if generator.random() < 0.30:
        image = image.filter(ImageFilter.GaussianBlur(radius=generator.uniform(0.25, 1.2)))
    if generator.random() < 0.20:
        image = _apply_stress(image, "glare", generator)
    if generator.random() < 0.25:
        image = image.rotate(
            generator.uniform(-2.0, 2.0),
            resample=Image.Resampling.BICUBIC,
            fillcolor=255,
        )
    return image


def isolate_foreground_line(image: Image.Image, label_type: str) -> Image.Image:
    grayscale = image.convert("L")
    array = np.asarray(grayscale)
    row_ink = (array < 205).sum(axis=1)
    active = row_ink >= max(2, int(array.shape[1] * 0.02))
    segments: list[tuple[int, int]] = []
    start = None
    gap = 0
    for row, enabled in enumerate(active):
        if enabled:
            if start is None:
                start = row
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 2:
                end = row - gap + 1
                if end - start >= 2:
                    segments.append((start, end))
                start = None
                gap = 0
    if start is not None:
        segments.append((start, len(active)))
    if not segments:
        return grayscale
    if label_type == "curp" and len(segments) > 1:
        start, end = max(
            segments,
            key=lambda segment: int(row_ink[segment[0]:segment[1]].sum()),
        )
    else:
        start, end = segments[0][0], segments[-1][1]
    margin = max(1, round((end - start) * 0.15))
    return grayscale.crop(
        (0, max(0, start - margin), grayscale.width, min(grayscale.height, end + margin))
    )


def prepare_image_array(
    image: Image.Image,
    width: int,
    height: int,
    *,
    training: bool = False,
    profile: str = "clean",
    seed: int | None = None,
    resize_mode: str = "stretch",
    isolate_foreground: bool = False,
    label_type: str = "unknown",
) -> np.ndarray:
    generator = random.Random(seed)
    image = image.convert("L")
    if isolate_foreground:
        image = isolate_foreground_line(image, label_type)
    if training:
        image = _augment(image, generator)
    else:
        image = _apply_stress(image, profile, generator)
    if resize_mode == "stretch":
        canvas = image.resize((width, height), Image.Resampling.BILINEAR)
    elif resize_mode == "fit":
        source_width, source_height = image.size
        scale = min(width / max(1, source_width), height / max(1, source_height))
        resized_width = max(1, min(width, round(source_width * scale)))
        resized_height = max(1, min(height, round(source_height * scale)))
        image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (width, height), color=255)
        canvas.paste(image, (0, (height - resized_height) // 2))
    else:
        raise ValueError(f"unknown_resize_mode:{resize_mode}")
    array = np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0
    return np.expand_dims(array, 0)
