"""Restore editable model sources from tensors only; no pickle loading."""

import argparse
import json
from pathlib import Path
import tempfile


def load_name(root):
    from safetensors.torch import load_file
    from ine_ocr.alphabet import LATIN_NAME_ALPHABET
    from ine_ocr.model import MachineLineRecognizer

    root = Path(root)
    descriptor = json.loads((root / "source/name-architecture.json").read_text())
    if descriptor["alphabet"] != LATIN_NAME_ALPHABET:
        raise ValueError("name_alphabet_mismatch")
    model = MachineLineRecognizer(**descriptor["architecture"])
    model.load_state_dict(load_file(str(root / "source/name.safetensors")), strict=True)
    return model.eval(), descriptor


def load_card(root):
    from safetensors.torch import load_file
    from ultralytics import YOLO
    import yaml

    root = Path(root)
    descriptor = json.loads((root / "source/card-architecture.json").read_text())
    with tempfile.TemporaryDirectory() as temporary:
        configuration = Path(temporary) / "card.yaml"
        configuration.write_text(yaml.safe_dump(descriptor["architecture"]))
        wrapper = YOLO(str(configuration), task="detect")
    wrapper.model.load_state_dict(load_file(str(root / "source/card.safetensors")), strict=True)
    wrapper.model.names = {0: "ine_card"}
    wrapper.ckpt = {"model": wrapper.model}
    wrapper.model.eval()
    return wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("name", "card"))
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from ine_ocr.r2_profile import verify_models
    verify_models(args.models)
    import torch

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refuse_to_overwrite_export")
    if args.kind == "name":
        model, descriptor = load_name(args.models)
        sample = torch.zeros((1, 1, descriptor["height"], descriptor["width"]))
        options = {"input_names": ["image"], "output_names": ["logits"], "dynamic_axes": {"image": {0: "batch"}, "logits": {0: "batch"}}}
    else:
        model = load_card(args.models).model.float().fuse(verbose=False)
        head = model.model[-1]
        head.export = True
        head.format = "onnx"
        head.dynamic = False
        sample = torch.zeros((1, 3, 640, 640))
        options = {"input_names": ["images"], "output_names": ["output0"]}
    with torch.inference_mode():
        model(sample)
        torch.onnx.export(model, sample, str(args.output), opset_version=17, dynamo=False, **options)
    print("Export complete; verify numerical parity before deploying a rebuilt model.")


if __name__ == "__main__":
    main()
