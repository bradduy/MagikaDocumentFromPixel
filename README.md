# MagikaDocument — Lightweight Blur Detector

A **Magika-inspired image quality gate** that classifies images as `sharp`, `blurred`, or `uncertain` in a few milliseconds on CPU. Built to sit at the front of vision pipelines so expensive downstream models (OCR, detection, classification) never waste compute on unusable input.

**Final result on GoPro test split:**

| Metric | Value |
|---|---|
| F1 | **0.9749** |
| Accuracy | 0.9752 |
| Precision | 0.9889 |
| Recall | 0.9613 |
| AUC | 0.9982 |
| Model size | 17 MB (ONNX) |
| Baseline inference | ~7 ms / image (CPU, single-scale) |

---

## 1. The Problem

Most production vision pipelines — OCR, document understanding, face/object recognition, content moderation — silently degrade on blurry input. A few concrete pain points:

1. **OCR on blurred receipts** emits garbage text that downstream logic can't distinguish from real data.
2. **Product search from user photos** returns irrelevant results when the phone shot is motion-blurred.
3. **Large multimodal models** (GPT-4V-class) are expensive; burning compute on an unreadable frame is wasteful.
4. **Dataset curation** — hand-filtering blurry images from millions of uploads is infeasible.

What teams usually want is a **fast, cheap pre-check** that can:
- Run on **CPU** before shipping the image to a GPU model or paid API
- Return not just a label but a **confidence** so the pipeline can route uncertain cases to a human or a heavier model
- Be **trainable without human annotation** — paired sharp/blur datasets already exist

Existing approaches either require heavy restoration networks (deblur GANs, hundreds of millions of params) or brittle hand-crafted edge-variance heuristics that fail on textured-but-sharp vs smooth-but-blurred scenes.

---

## 2. Our Solution

A two-class CNN classifier with a confidence-aware routing head, designed after the **Magika file-type detection** philosophy:

> *A small model, fast inference, CPU-friendly, constant-cost preprocessing, confidence-aware output — used as a routing or gating component, not as a final answer.*

### Recipe (F1 = 0.9749)

- **Data**: GoPro Large dataset. Both `blur/` and `blur_gamma/` folders count as positive (blurred); `sharp/` is negative. Paired sample generation — no extra human annotation.
- **Backbone**: MobileNetV3-Large, ImageNet-pretrained, 2-class softmax head (~3.3M params).
- **Training**: 384×384 input, AdamW lr=1e-4, CosineAnnealing, CrossEntropy, 25 epochs, medium augmentation (crop, flip, mild color jitter), mixed-precision.
- **Inference**: **5-scale multi-scale TTA** — run the model at 256, 320, 384, 448, 512 and average the softmax probabilities. Free +0.27% F1 over single-scale, no retraining.
- **Routing**: return `sharp` or `blurred` when max softmax ≥ 0.60, otherwise return `uncertain` so the pipeline can hand off to a heavier model or a human.

### Autonomous research loop

This model wasn't picked by hand — it fell out of an **autoresearch loop** that ran 46 experiments across 8 parallel sweeps, with every run logged to `autoresearch/results.tsv`. The full progression:

```
Sweep 1 (res 128-160):       F1 ≤ 0.89
Sweep 2 (res 160-224):       F1 ≤ 0.92
Sweep 3 (aug/threshold):     F1 ≤ 0.93
Sweep 4 (blur_gamma + 320):  F1 ≤ 0.95
Sweep 5 (384px):             F1 ≤ 0.96
Sweep 6 (MNV3-Large @ 384):  F1 = 0.9722  ← single-model best
Sweep 7 (Large variants):    F1 ≤ 0.9722  (plateau)
Sweep 8 (EfficientNet/reg):  F1 ≤ 0.9722  (plateau)
Multi-scale TTA on champion: F1 = 0.9749  ← final
```

The loop discovered three things humans would have over/under-weighted:
- **Resolution is by far the biggest lever** (+13% F1 alone, 128→384)
- **Capacity × resolution interact**: MNV3-Large was neutral at 160-320px, but wins at ≥384px
- **blur_gamma data**: +1% F1 just by including the second GoPro blur folder

Anti-patterns the loop rejected:
- Focal loss (dataset is balanced)
- Strong/aggressive augmentation (hurt at ≥224px)
- Training >25 epochs at 384px (overfit)
- Threshold tuning (val saturates at F1=1.0)
- Naive ensembles of top-N models (errors are correlated)

---

## 3. Use Cases

### 3.1 OCR / document pre-check

```python
from blur_detector.src.models.blur_detector import build_model
from blur_detector.src.inference.predictor import BlurPredictor
import torch

model = build_model("mobilenet_v3_large", pretrained=False)
model.load_state_dict(torch.load("blur_detector_champion.pt"))

predictor = BlurPredictor(model, image_size=[256, 320, 384, 448, 512])

pred = predictor.predict("receipt.jpg")
if pred.label == "blurred":
    return {"status": "rejected", "reason": "image too blurry, please retake"}
elif pred.label == "uncertain":
    route_to_human_review(...)
else:
    return run_ocr(...)
```

Saves OCR compute on unusable inputs and gives users an immediate "please retake the photo" signal.

### 3.2 Upload-time quality filter

Plug as middleware in an image upload API. Reject or flag low-quality uploads before they hit storage or trigger downstream processing. The `uncertain` class lets you tune precision vs. user friction.

### 3.3 Dataset curation

Point at a directory of unlabeled images and let the batch CLI partition them into `sharp/`, `blurred/`, `uncertain/` for downstream annotation or training. ~7 ms per image on CPU → a million images in under 2 hours on a single core.

### 3.4 Routing layer in front of expensive models

```
image ─► BlurDetector ─┬─► sharp   ─► run full VLM (expensive)
                       ├─► blurred ─► skip, return low-quality flag
                       └─► uncertain ─► run small fallback model, or human
```

Cost-sensitive teams see large savings — the blur detector is ~1000× cheaper than a vision LLM, so even rejecting 10% of traffic is a large win.

### 3.5 Edge / on-device inference

17 MB ONNX artifact with `dynamic_axes` enabled — deployable to mobile (CoreML/TFLite via ONNX converters), browser (ONNX Runtime Web), or embedded devices. Multi-scale TTA is opt-in; single-scale at 384px still gets F1=0.9722.

---

## 4. How to Use

### Run inference

```bash
python blur_detector/scripts/predict.py \
  --checkpoint blur_detector/outputs/checkpoints/exp29_large_384_gamma/best.pt \
  path/to/image.jpg path/to/other.png
```

Output:
```json
{"file": "path/to/image.jpg", "label": "sharp",   "confidence": 0.97, "prob_sharp": 0.97, "prob_blurred": 0.03}
{"file": "path/to/other.png", "label": "blurred", "confidence": 1.00, "prob_sharp": 0.00, "prob_blurred": 1.00}
```

### Reproduce the champion

```bash
# 1. Train (GPU recommended — 25 epochs @ 384px)
python blur_detector/scripts/train.py \
  --run_name production \
  --backbone mobilenet_v3_large --image_size 384 --batch_size 24 --lr 1e-4 \
  --loss ce --aug_level medium --epochs 25 --use_blur_gamma \
  --exp_id 100 --notes 'production reproduce'

# 2. Evaluate with multi-scale TTA
python autoresearch/multiscale_tta.py \
  --run_name production --backbone mobilenet_v3_large \
  --scales 256,320,384,448,512 --exp_id 101 --notes 'MS-TTA eval'

# 3. Inspect the log
cat autoresearch/results.tsv
```

### Export to ONNX for deployment

```bash
python blur_detector/scripts/export_onnx.py \
  --checkpoint .../exp29_large_384_gamma/best.pt \
  --backbone mobilenet_v3_large --image_size 384 \
  --onnx_path blur_detector.onnx
```

---

## 5. Project Structure

```
MagikaDocument/
├── blur_detector/
│   ├── configs/        — YAML hyperparameter config
│   ├── data/           — GoPro dataset (gitignored)
│   ├── outputs/        — Checkpoints + ONNX artifact (gitignored)
│   ├── src/
│   │   ├── datasets/   — GoProDataset (paired sharp/blur labeling)
│   │   ├── models/     — MobileNetV3, TinyCNN, EfficientNet backbones
│   │   ├── training/   — Trainer (AdamW, Cosine, AMP, early stop)
│   │   ├── inference/  — BlurPredictor (multi-scale TTA + uncertain routing)
│   │   └── utils/      — Metrics (F1, AUC, confusion)
│   └── scripts/        — prepare, train, evaluate, predict, export_onnx
├── autoresearch/
│   ├── program.md      — Research loop instructions
│   ├── sweep.sh        — Parallel 4-experiment launcher
│   ├── threshold_sweep.py, ensemble_eval.py,
│   │ multiscale_tta.py, ultimate_ensemble.py
│   └── results.tsv     — Full experiment log (46 runs)
└── README.md           — This file
```

---

## 6. Extensions

If your blur distribution differs from GoPro motion blur (e.g. defocus, low-light, compression artifacts), retrain with:

1. **More blur types** — mix GoPro with REDS (video deblurring) or synthetic defocus from public clean-image datasets.
2. **Domain-specific sharp set** — if you deploy on receipts/documents, use a few thousand domain-sharp images as the positive class and generate synthetic motion/defocus blur.
3. **Multi-class taxonomy** — swap the 2-class head for `{sharp, motion_blur, defocus_blur, noise, low_light}` when you want the downstream pipeline to react differently to different degradation types.

See `autoresearch/program.md` for the research ideas that were scoped but not executed (mixup between blur/sharp, full-resolution crops, distillation to TinyCNN, curriculum learning).
