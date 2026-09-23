# R2 model card

## What is released

The default deployment is the complete **R2-v1 inference composition**, not just a generic OCR engine or an experimental later checkpoint:

| Component | Frozen configuration |
| --- | --- |
| Card detector | Fine-tuned YOLOv8n, fixed `1×3×640×640` ONNX; natural-image fallbacks retained |
| Printed-text detection | Public PP-OCRv6 small Paddle detector |
| Printed-text recognition | Pinned PP-OCRv6 small ONNX export with its original vocabulary/YAML |
| Names | Fine-tuned CRNN: input `N×1×48×384`, hidden size 192, two bidirectional GRU layers, 76 CTC classes |
| Name admission | Confidence threshold 0.9, one name-model CPU thread, protected-source rules unchanged |
| Fusion | Cross-channel name consensus enabled; no experimental field selector |
| QR / MRZ | Local QR decoder, MRZ parsing/check digits and the existing field-selection rules |
| Auxiliary English recognizer | Included for provenance; ROI sidecar remains **disabled** in R2 |

The exact allowlist, hashes, byte sizes, configuration and download URL live in `src/ine_ocr/r2-profile.json`. The approximately 61 MB compressed release contains both runtime ONNX files and editable tensor-only source weights/architecture descriptions. Weights are not embedded in Git history. The build downloads one checksum-pinned GitHub release asset; a missing or changed asset fails the build. Startup verifies all required files and flags and cannot silently fall back to the public base profile.

R2 name ONNX identity: `a8e2afbbc28119ae4bab43abbe2cea526e359386030051d244f005f99e019eb5`. The name model, sidecar and recognition exports retain their evaluated bytes. Only the card ONNX's descriptive metadata containing an internal path was replaced; its computation graph and tensors are unchanged, with identical outputs in the sanitization probes. Its public file hash therefore differs from the original export. No server addresses, credentials, private source checkout paths, real document images, labels, optimizer state or training logs are distributed.

## Observed reference agreement

On a frozen development comparison of 1,003 documents from 1,000 source groups, the retained R2 composition produced:

| Metric | Matches / full denominator | Rate |
| --- | --- | --- |
| Joint name strict agreement | 905 / 1,003 | 90.229% |
| Joint name normalized agreement | 928 / 1,003 | 92.522% |
| Joint critical-field strict agreement | 801 / 1,003 | 79.860% |
| Joint critical-field normalized agreement | 822 / 1,003 | 81.954% |

These are **agreement with third-party reference outputs, not independently adjudicated human accuracy**. Joint matching uses available reference fields while retaining the full source denominator; 997 cases have all three name references. Normalized name comparison removes accents/nonletters and normalizes spacing; it is not an LLM semantic similarity score. The data are not published because they contain personal information. Results are not independently reproducible from this repository, are not a performance promise on another population, and are not user completion/correction measurements.

Later evaluated candidates did not demonstrate a full-document improvement under the frozen replacement criteria; R2 remains the release choice. This does not establish that R2 is universally superior or that small differences are statistically meaningful. Container smoke tests use synthetic inputs and verify execution/model identity only, not these agreement rates.

## Intended use and limitations

Use for **editable document-form prefilling**. Always show the extracted fields for confirmation/correction. Missing fields, false text, spelling/accent errors, damaged images and conflicting channels are possible. Check digits and QR/MRZ agreement do not prove identity or document authenticity. Do not use model confidence as a calibrated probability or as the sole basis of an eligibility decision.

The card detector is inherited from an earlier handoff; its pseudo-label training overrepresented full-frame cards. Its complete original training recipe is not available. Name training used private weakly labelled crops; private examples and labels are deliberately excluded. Static metadata checks are not a proof that trained weights have zero memorization risk.

## Inference reproduction versus retraining

Frozen inference is reproduced using released weights, preprocessing, vocabulary, decoder and flags. Relevant CPU dependencies are pinned to the evaluated runtime in `deploy/requirements.r2.txt`; the Python 3.12.14 base is digest-pinned. Server OS/kernel/CPU may still affect floating-point and timing behavior. Confirm representative results in your own authorized environment.

Editable source weights use SafeTensors, not executable pickle checkpoints. `src/ine_ocr/model.py` supplies the name architecture; the card architecture and upstream Ultralytics 8.4.137 implementation restore the detector. See [model source and retraining templates](MODEL_SOURCES.md). Private training data and exact original training histories are not supplied; reproducing the original training run or its evaluation score is **not** claimed.

## License

This full R2 distribution is **AGPL-3.0-only**. Ultralytics retains its attribution and upstream terms; Paddle/OpenCV components retain their respective licenses. The earlier Apache-2.0 base-only release remains available at commit `972240cf27349e6db4b2734bc77447b25fed84bc`. Consult the license and upstream guidance before embedding R2 in a closed-source application or network service. No affiliation with INE is claimed.
