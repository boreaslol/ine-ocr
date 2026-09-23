"""ONNX Runtime CPU recognizer."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .alphabet import ALPHABET, greedy_decode
from .curp import constrained_ctc_decode, is_valid_curp
from .image_preprocessing import prepare_image_array


class OnnxMachineLineRecognizer:
    def __init__(self, model_path: str, *, image_width: int = 320, image_height: int = 48,
                 intra_op_threads: int = 0, curp_beam_width: int = 32) -> None:
        metadata_path = Path(model_path + ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        if metadata.get("alphabet", ALPHABET) != ALPHABET:
            raise ValueError("machine_recognizer_alphabet_mismatch_use_name_recognizer")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("onnxruntime_not_installed") from error
        options = ort.SessionOptions()
        if intra_op_threads > 0:
            options.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.image_width = image_width
        self.image_height = image_height
        self.curp_beam_width = curp_beam_width
        preprocessing = metadata.get("preprocessing") or {}
        self.resize_mode = preprocessing.get("resize_mode", "fit")
        self.isolate_foreground = bool(preprocessing.get("isolate_foreground", False))

    def recognize(self, image_bytes: bytes, label_type: str = "unknown") -> dict:
        with Image.open(io.BytesIO(image_bytes)) as image:
            tensor = prepare_image_array(
                image,
                self.image_width,
                self.image_height,
                resize_mode=self.resize_mode,
                isolate_foreground=self.isolate_foreground,
                label_type=label_type,
            )
        logits = self.session.run(["logits"], {"image": np.expand_dims(tensor, 0)})[0]
        if label_type == "curp":
            text = constrained_ctc_decode(logits[0], beam_width=self.curp_beam_width)
            return {"text": text, "decoder": "curp-constrained", "valid": is_valid_curp(text)}
        indices = np.argmax(logits[0], axis=-1).tolist()
        return {"text": greedy_decode(indices), "decoder": "ctc-greedy"}
