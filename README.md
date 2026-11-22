---
title: Aetheris PCam Pathology
emoji: 🔬
colorFrom: yellow
colorTo: gray
sdk: static
pinned: false
---

# Aetheris — PCam Computational Pathology Workstation

> **Live Demo:** **[Launch Aetheris Web App](https://sukuna-sama-code-pcam-cancer-detector.static.hf.space/)** · **[Hugging Face Space](https://huggingface.co/spaces/Sukuna-sama-code/Pcam-Cancer-Detector)**

**Aetheris** is an editorial computational pathology workstation for inspecting `96×96 px` H&E lymph-node histopathology patches (PatchCamelyon / PCam) using a **Center-Aware ResNet-34** deep neural network, `4×` rotational Test-Time Augmentation, analytical Layer-4 Grad-CAM explainability, and an interactive 3D nuclear morphometry Random Forest explorer.

It runs **100% client-side in your browser** for **$0 deployment** on **Hugging Face Static Spaces** or **GitHub Pages** via **ONNX Runtime Web (WebAssembly)**, with an optional local **FastAPI + PyTorch** backend mode.

---

## Architecture & Browser AI Pipeline

- **In-Browser CNN Inference (`static/onnx-engine.js`):** Runs `CenterAwareResNet34` exported from `pcam_deployment_bundle.pt` via `export_static_assets.py` into `static/models/pcam_resnet34_q8.onnx` (`20.76 MB`, per-channel INT8 weight compression). Outputs both malignancy logit and exact analytical Layer-4 Grad-CAM activation maps (`12×12`) in one pass through `onnxruntime-web@1.21.0` (single-threaded WASM + SIMD).
- **Center-Aware Dual Pooling:** Replaces the standard ResNet `7×7` stride-2 stem with a `3×3` stride-1 convolutional stem so `96×96` nuclei retain spatial fidelity, concatenating global pooling with central `32×32` ROI pooling to match PCam's central-region metastatic definition.
- **Display-Only Optics:** Client-side `6×` HD upscaling (`576×576`), luminance contrast stretching, and separable Unsharp Masking (`USM`) clarify visual inspection without altering the untouched `96×96` model tensors. Supports `.png`, `.jpg`, `.jpeg`, and `.tif`/`.tiff` uploads (`UTIF.js`).
- **Biomarker Random Forest (`static/models/biomarker_rf.json`):** Exports all `180` scikit-learn trees trained on the Wisconsin Diagnostic Breast Cancer dataset (`569` cases) for zero-latency client-side evaluation (`0.0000%` preset drift).

---

## Workspaces & UI Features

- **01 · Optics:** Reference `8`-patch H&E filmstrip + drag-and-drop upload, animated `0° / 90° / 180° / 270°` TTA sequence, synced Dual Viewports (`H&E Brightfield` + `Grad-CAM`), Interactive Draggable Optical Curtain slider, live Youden's J decision cutoff (`0.126`), Spatial Attribution breakdown, and JSON telemetry export.
- **02 · Batch:** Multi-slide cohort upload or 1-click screening across all `8` built-in reference patches, optional `train_labels.csv` (`id, label`) accuracy audit, verdict filter chips, and downloadable CSV audit report.
- **03 · 3D Biomarker:** Interactive Plotly WebGL 3D nuclear cluster scatter (`Mean Radius × Mean Concavity × Worst Radius`), `10` morphometry sliders, clinical presets (`Benign Typical`, `Borderline`, `Malignant High-Risk`), live SVG risk dial gauge, and feature importance chart.
- **04 · Insights:** Verified `500`-patch blind confusion matrix (`TN: 290, FP: 6, FN: 0, TP: 204`) and dual-pooling ResNet-34 architecture breakdown.
- **BGThemes, Help & Feedback:** `BGThemes` dropdown with `5` tailored visual themes (`Liquid Gold & Espresso`, `Espresso & Cream Marble`, `Mocha & Copper Swirl`, `Warm Cedar Wood Slats`, `Chestnut & Molten Amber`), a plain-English **`? Help`** clinical terminology guide, and a scroll-tracking theme-reactive **Heart Feedback** modal.

---

## Deploy for Free (Hugging Face Static Spaces / GitHub Pages)

This repository includes `sdk: static` in its YAML header and root `index.html` redirecting cleanly into `./static/index.html`:

- **Hugging Face Static Spaces:** Create a **Static → Blank** Space (`huggingface.co/new-space`), add your Space remote (`git remote add hf https://huggingface.co/spaces/<USER>/<SPACE>`), and push with Git LFS (`git lfs install && git push hf main`).
- **GitHub Pages:** In **Repository Settings → Pages**, set **Source** to **Deploy from a branch** (`main` / `/(root)`).

---

## Run Locally (Optional PyTorch FastAPI Backend Mode)

```bash
git lfs pull
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

- Open `http://127.0.0.1:7860/` to run the default **in-browser ONNX WebAssembly** engine.
- Open `http://127.0.0.1:7860/?backend=api` to route inference through native PyTorch in `server.py`.

---

## Verified 500-Patch Benchmark & Previews

On the `500`-patch blind evaluation (with checkpoint validation AUC `0.9667`, test ROC-AUC `0.9534`, calibrated cutoff `0.1261`), 4× TTA achieved **`0.9998` ROC-AUC**, **`98.80%` Accuracy**, **`100.00%` Sensitivity** (`TP=204, FN=0`), and **`97.97%` Specificity** (`TN=290, FP=6`). *For research & educational exploration only — not cleared for primary clinical diagnosis.*

<img width="977" height="508" alt="image" src="https://github.com/user-attachments/assets/6ab3ab9e-eee1-42ba-9823-28a251e445b0" />

