"""Opt-in CPU recognition acceleration with the original Paddle detector."""

from __future__ import annotations

import os
import json
from pathlib import Path

from .artifacts import sha256_file

MODEL_NAMES = {"PP-OCRv6_small_rec", "en_PP-OCRv5_mobile_rec"}
MODEL_FILES = ("inference.onnx", "inference.yml")


def model_directory(model_name: str) -> str | None:
    configured = os.getenv("INE_OCR_RECOGNITION_ONNX_DIR")
    if not configured:
        return None
    if model_name not in MODEL_NAMES:
        raise ValueError("unsupported_onnx_recognition_model")
    directory = Path(configured).expanduser().resolve() / model_name
    manifest = json.loads((directory / "manifest.json").read_text())
    for filename in MODEL_FILES:
        expected = manifest.get(filename)
        path = directory / filename
        if not isinstance(expected, str) or len(expected) != 64 or not path.is_file() or sha256_file(str(path)) != expected:
            raise ValueError("onnx_recognition_asset_unverified")
    return str(directory)


def engine_config(cpu_threads: int) -> dict:
    if not 1 <= cpu_threads <= 16:
        raise ValueError("onnx_recognition_cpu_threads_out_of_range")
    return {
        "providers": ["CPUExecutionProvider"],
        "intra_op_num_threads": cpu_threads,
        "inter_op_num_threads": 1,
    }


def recognition_options(model_name: str, cpu_threads: int) -> dict:
    directory = model_directory(model_name)
    if directory is None:
        return {}
    return {
        "model_dir": directory,
        "engine": "onnxruntime",
        "engine_config": engine_config(cpu_threads),
        "device": "cpu",
    }


def document_options(tier: str, cpu_threads: int, *, enable_mkldnn: bool = False) -> dict:
    if not os.getenv("INE_OCR_RECOGNITION_ONNX_DIR"):
        return {}
    if tier != "small":
        raise ValueError("onnx_recognition_requires_small_tier")
    directory = model_directory("PP-OCRv6_small_rec")
    onnx_config = engine_config(cpu_threads)
    from paddleocr._common_args import parse_common_args, prepare_common_init_args
    from paddlex.inference import load_pipeline_config

    common = parse_common_args(
        {
            "device": "cpu",
            "cpu_threads": cpu_threads,
            "enable_mkldnn": enable_mkldnn,
        },
        default_enable_hpi=None,
    )
    native_config = prepare_common_init_args(None, common)["engine_config"][
        "paddle_static"
    ]
    config = load_pipeline_config("OCR")
    config["SubModules"]["TextDetection"].update(
        {
            "model_name": "PP-OCRv6_small_det",
            "engine": "paddle_static",
            "engine_config": native_config,
        }
    )
    config["SubModules"]["TextRecognition"].update(
        {
            "model_name": "PP-OCRv6_small_rec",
            "model_dir": directory,
            "engine": "onnxruntime",
            "engine_config": onnx_config,
        }
    )
    return {"paddlex_config": config, "engine": "onnxruntime", "device": "cpu"}


def verified_models() -> dict:
    hashes = {}
    for model_name in sorted(MODEL_NAMES):
        directory = model_directory(model_name)
        if directory is None:
            return {}
        for filename in MODEL_FILES:
            hashes[f"{model_name}/{filename}"] = sha256_file(
                str(Path(directory) / filename)
            )
    return hashes
