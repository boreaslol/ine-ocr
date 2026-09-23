# Editable model sources

Install the optional authoring environment separately from the CPU serving image:

```bash
python -m pip install -e '.[model-tools]'
INE_OCR_R2_MODEL_DIR="$PWD/models/r2" python scripts/fetch_r2_models.py
python scripts/model_source.py name --models models/r2 --output artifacts/rebuilt-name.onnx
python scripts/model_source.py card --models models/r2 --output artifacts/rebuilt-card.onnx
```

Exports are authoring outputs, not automatically approved replacement models. Always compare tensors/logits, preprocessing and full-document behavior before replacing a deployed artifact. Exporter/toolchain differences may change file bytes. Serving uses the frozen release weights, not a freshly rebuilt export.

## Name recognizer

`source/name.safetensors` contains only the R2 state dictionary. `source/name-architecture.json` supplies dimensions and the alphabet. `scripts/model_source.py:load_name` reconstructs the exact CRNN without any `torch.load` or optimizer state. The model source is `src/ine_ocr/model.py` and preprocessing is `src/ine_ocr/image_preprocessing.py`.

A training template on your own authorized name-line crops is:

1. Split by source identity/document **before** making crops; keep training, model-selection and final test sources disjoint.
2. Restore the model with `load_name`, call `model.train()`, and prepare inputs with `prepare_image_array(..., width=384, height=48, resize_mode="fit", isolate_foreground=False)`.
3. Encode your labels with `LATIN_NAME_ALPHABET`; index 0 is the CTC blank and characters start at 1. The model returns `[batch, time, classes]`; use log-softmax, transpose to `[time, batch, classes]`, and `torch.nn.CTCLoss(blank=0)` with valid target/input lengths.
4. Choose an optimizer/schedule on your training/selection split, save only tensor state with `safetensors.torch.save_file`, and export/check CPU parity. Do not assume these choices reproduce the original run.
5. Assess the complete document pipeline against an unchanged R2 control on unseen sources, not just crop accuracy. Preserve missing/failed cases in the denominator.

No original labels, source-group identifiers, personal images, optimizer state or complete training history are included.

## Card detector

`source/card.safetensors` and `source/card-architecture.json` reconstruct the inherited one-class YOLOv8n via `load_card`. The upstream implementation is pinned to [Ultralytics 8.4.137](https://github.com/ultralytics/ultralytics/tree/v8.4.137). License and attribution are retained.

For a **new** training run on your own labelled images, import `load_card` from the script, obtain the returned YOLO wrapper and call its documented `train(data="your-dataset.yaml", epochs=..., imgsz=640)` API. Those are operator-selected settings, not a reconstruction of unknown historical settings. Class 0 is `ine_card`; do not train only on perfectly pre-cropped cards. Maintain independent localization and end-to-end evaluations.

Original `.pt` files are deliberately excluded because they contain unnecessary training metadata and Python objects. Only tensor dictionaries and a JSON architecture are distributed. The runtime ONNX already contains the deployable graph and parameters; it does not import Ultralytics or PyTorch while serving.
