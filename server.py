import base64
import io
import json
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torchvision import models, transforms


# ==============================================================================
# 1. CENTER-AWARE RESNET-34 ARCHITECTURE (EXACT CHECKPOINT MATCH)
# ==============================================================================
class CenterAwareResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet34(weights=None)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.head = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        feat = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        global_pool = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        h, w = feat.shape[2], feat.shape[3]
        center_feat = feat[:, :, h // 3 : 2 * (h // 3), w // 3 : 2 * (w // 3)]
        center_pool = F.adaptive_avg_pool2d(center_feat, (1, 1)).flatten(1)
        return self.head(torch.cat([global_pool, center_pool], dim=1))


# ==============================================================================
# 2. MODEL LOADING & PREPROCESSING
# ==============================================================================
CKPT_PATH = os.environ.get("PCAM_CHECKPOINT", "pcam_deployment_bundle.pt")
SAMPLE_DIR = os.environ.get("PCAM_SAMPLE_DIR", "sample_biopsy_patches")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pcam_model: Optional[CenterAwareResNet34] = None
model_loaded = False
default_threshold = 0.1261
val_auc = 0.9667
test_auc = 0.9534
checkpoint_metrics: Dict[str, float] = {
    "best_val_auc": 0.9667,
    "test_roc_auc": 0.9534,
    "calibrated_test_acc": 0.8985,
    "calibrated_test_f1": 0.8944,
    "raw_test_acc_050": 0.8732,
    "optimal_threshold": 0.1261,
}

if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 1_000_000:
    try:
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            metrics = ckpt.get("metrics", {}) if isinstance(ckpt.get("metrics"), dict) else {}
            default_threshold = float(
                ckpt.get("optimal_threshold", metrics.get("optimal_threshold", ckpt.get("threshold", 0.1261)))
            )
            val_auc = float(metrics.get("best_val_auc", ckpt.get("val_auc", ckpt.get("best_val_auc", 0.9667))))
            test_auc = float(metrics.get("test_roc_auc", ckpt.get("test_auc_tta", ckpt.get("test_auc", 0.9534))))
            checkpoint_metrics = {
                "best_val_auc": round(val_auc, 4),
                "test_roc_auc": round(test_auc, 4),
                "calibrated_test_acc": round(float(metrics.get("calibrated_test_acc", 0.8985)), 4),
                "calibrated_test_f1": round(float(metrics.get("calibrated_test_f1", 0.8944)), 4),
                "raw_test_acc_050": round(float(metrics.get("raw_test_acc_050", 0.8732)), 4),
                "optimal_threshold": round(default_threshold, 4),
            }
        else:
            sd = ckpt.state_dict()

        clean_sd = {}
        for k, v in sd.items():
            nk = k
            for prefix in ("backbone.", "model.", "module."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]
            clean_sd[nk] = v

        pcam_model = CenterAwareResNet34().to(device)
        pcam_model.load_state_dict(clean_sd, strict=True)
        pcam_model.eval()
        model_loaded = True
    except Exception as exc:
        print(f"[WARN] Could not load {CKPT_PATH}: {exc}")

eval_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_4x_tta(img_pil: Image.Image):
    angles = [0, 90, 180, 270]
    rot_probs = {}
    with torch.no_grad():
        for angle in angles:
            rot_img = img_pil.rotate(angle) if angle > 0 else img_pil
            tensor = eval_transform(rot_img).unsqueeze(0).to(device)
            prob = torch.sigmoid(pcam_model(tensor)).item()
            rot_probs[f"{angle}°"] = float(prob)
    probs_arr = list(rot_probs.values())
    return float(np.mean(probs_arr)), float(np.std(probs_arr)), rot_probs


def generate_gradcam(img_pil: Image.Image) -> np.ndarray:
    activations, gradients = [], []

    def f_hook(_, __, out):
        activations.append(out)

    def b_hook(_, __, grad_out):
        gradients.append(grad_out[0])

    target = pcam_model.layer4[-1]
    hf = target.register_forward_hook(f_hook)
    hb = target.register_full_backward_hook(b_hook)
    try:
        tensor = eval_transform(img_pil).unsqueeze(0).to(device)
        tensor.requires_grad_(True)
        pcam_model.zero_grad()
        out = pcam_model(tensor)
        out.backward(torch.ones_like(out))

        grads = gradients[0].detach().cpu().numpy()[0]
        acts = activations[0].detach().cpu().numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max() + 1e-8
        return cv2.resize(cam, (96, 96), interpolation=cv2.INTER_CUBIC)
    finally:
        hf.remove()
        hb.remove()


def enhance_patch(rgb_96: np.ndarray, sharpen_strength: float = 1.25, apply_clahe: bool = True) -> np.ndarray:
    upscaled = cv2.resize(rgb_96, (576, 576), interpolation=cv2.INTER_LANCZOS4)
    if apply_clahe:
        lab = cv2.cvtColor(upscaled, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        l_blend = cv2.addWeighted(clahe.apply(l_ch), 0.55, l_ch, 0.45, 0)
        upscaled = cv2.cvtColor(cv2.merge((l_blend, a_ch, b_ch)), cv2.COLOR_LAB2RGB)
    if sharpen_strength > 0:
        blur = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.4)
        upscaled = cv2.addWeighted(upscaled, 1.0 + sharpen_strength, blur, -sharpen_strength, 0)
    return np.clip(upscaled, 0, 255).astype(np.uint8)


def _draw_roi_reticle(disp: np.ndarray, roi_color=(26, 168, 108)) -> None:
    """Draw the exact 32x32 center ROI boundary (192..384 on 576x576) plus subtle optical ticks."""
    x0, y0, x1, y1 = 192, 192, 384, 384
    cv2.rectangle(disp, (x0, y0), (x1, y1), roi_color, 2, lineType=cv2.LINE_AA)
    tick = 16
    accent = (255, 255, 255) if roi_color == (255, 255, 255) else (124, 232, 166)
    corners = [
        ((x0, y0), (x0 + tick, y0), (x0, y0 + tick)),
        ((x1, y0), (x1 - tick, y0), (x1, y0 + tick)),
        ((x0, y1), (x0 + tick, y1), (x0, y1 - tick)),
        ((x1, y1), (x1 - tick, y1), (x1, y1 - tick)),
    ]
    for p, px, py in corners:
        cv2.line(disp, p, px, accent, 3, lineType=cv2.LINE_AA)
        cv2.line(disp, p, py, accent, 3, lineType=cv2.LINE_AA)


def render_overlay_b64(
    img_pil: Image.Image,
    cam: Optional[np.ndarray],
    roi_color=(26, 168, 108),
    enhance: bool = True,
    sharpen: float = 1.25,
) -> str:
    base_rgb = np.array(img_pil.resize((96, 96), Image.Resampling.LANCZOS).convert("RGB"))
    if enhance:
        disp = enhance_patch(base_rgb, sharpen_strength=sharpen, apply_clahe=True)
    else:
        disp = cv2.resize(base_rgb, (576, 576), interpolation=cv2.INTER_NEAREST)

    if cam is not None:
        cam_hi = np.clip(cv2.resize(cam, (576, 576), interpolation=cv2.INTER_CUBIC), 0, 1)
        heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam_hi), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        disp = cv2.addWeighted(disp, 0.60, heat, 0.40, 0)

    _draw_roi_reticle(disp, roi_color=roi_color)
    buf = io.BytesIO()
    Image.fromarray(disp).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_thumb_b64(img_pil: Image.Image, size: int = 64) -> str:
    thumb = img_pil.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# ==============================================================================
# 3. BIOMARKER RANDOM FOREST PIPELINE
#    Plug-in point for external clinical models:
#      - For scikit-learn .pkl: import joblib; bio_clf = joblib.load("your_model.pkl")
#      - For Keras/TF .h5: from tensorflow import keras; bio_clf = keras.models.load_model("your_model.h5")
# ==============================================================================
BIO_FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "worst radius",
    "worst concavity",
]
_bc = load_breast_cancer(as_frame=True)
_bio_df = _bc.frame.copy()
_bio_df["malignant"] = (1 - _bio_df["target"]).astype(int)
_X_tr, _X_te, _y_tr, _y_te = train_test_split(
    _bio_df[BIO_FEATURES],
    _bio_df["malignant"],
    test_size=0.2,
    random_state=42,
    stratify=_bio_df["malignant"],
)
bio_clf = RandomForestClassifier(n_estimators=180, max_depth=8, random_state=42).fit(_X_tr, _y_tr)
bio_accuracy = float(bio_clf.score(_X_te, _y_te))

_benign_means = _bio_df[_bio_df["malignant"] == 0][BIO_FEATURES].median().to_dict()
_malig_means = _bio_df[_bio_df["malignant"] == 1][BIO_FEATURES].median().to_dict()


# ==============================================================================
# 4. FASTAPI APPLICATION & ROUTES
# ==============================================================================
app = FastAPI(title="Aetheris PCam Pathology API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def list_reference_samples():
    items = []
    if os.path.isdir(SAMPLE_DIR):
        for fname in sorted(os.listdir(SAMPLE_DIR)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                kind = "cancer" if "cancer" in fname.lower() else ("normal" if "normal" in fname.lower() else "other")
                prefix = "Metastatic reference" if kind == "cancer" else ("Benign reference" if kind == "normal" else "Biopsy patch")
                short_name = os.path.splitext(fname)[0].replace("_", " ")
                items.append({
                    "filename": fname,
                    "label": f"{prefix} — {fname}",
                    "short_name": short_name,
                    "kind": kind,
                    "thumb_url": f"/api/sample-thumb/{fname}",
                })
    return items


@app.get("/api/status")
def get_status():
    return {
        "model_loaded": model_loaded,
        "device": str(device).upper(),
        "default_threshold": round(default_threshold, 4),
        "val_auc": round(val_auc, 4),
        "test_auc": round(test_auc, 4),
        "checkpoint_metrics": checkpoint_metrics,
        "samples": list_reference_samples(),
    }


@app.get("/api/sample-thumb/{filename}")
def get_sample_thumb(filename: str):
    safe_name = os.path.basename(filename)
    fpath = os.path.join(SAMPLE_DIR, safe_name)
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail="Sample thumbnail not found.")
    if safe_name.lower().endswith((".tif", ".tiff")):
        with Image.open(fpath) as img:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
    return FileResponse(fpath)


@app.post("/api/analyze")
async def analyze_slide(
    sample_name: Optional[str] = Form(None),
    threshold: float = Form(0.1261),
    enhance: bool = Form(True),
    sharpen: float = Form(1.25),
    file: Optional[UploadFile] = File(None),
):
    if not model_loaded or pcam_model is None:
        raise HTTPException(status_code=503, detail="PyTorch checkpoint pcam_deployment_bundle.pt not loaded on server.")

    t0 = time.perf_counter()
    if file is not None:
        raw = await file.read()
        try:
            img_pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")
        specimen = file.filename or "uploaded_patch.png"
        source = "External Upload"
    elif sample_name:
        safe_name = os.path.basename(sample_name)
        fpath = os.path.join(SAMPLE_DIR, safe_name)
        if not os.path.isfile(fpath):
            raise HTTPException(status_code=404, detail="Reference biopsy patch not found.")
        img_pil = Image.open(fpath).convert("RGB")
        specimen = safe_name
        source = "Reference Biopsy Bank"
    else:
        raise HTTPException(status_code=400, detail="Provide either a reference sample or an uploaded biopsy file.")

    mean_intensity = float(np.mean(np.array(img_pil)))
    mean_prob, tta_std, rot_probs = predict_4x_tta(img_pil)
    cam_map = generate_gradcam(img_pil)

    center_cam = float(np.mean(cam_map[32:64, 32:64]))
    mask_outer = np.ones_like(cam_map, dtype=bool)
    mask_outer[32:64, 32:64] = False
    outer_cam = float(np.mean(cam_map[mask_outer]))
    peak_cam = float(np.max(cam_map[32:64, 32:64]))
    tta_stability = max(0.0, 1.0 - tta_std)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    return {
        "specimen": specimen,
        "source": source,
        "native_size": f"{img_pil.size[0]}×{img_pil.size[1]} px",
        "mean_intensity": round(mean_intensity, 1),
        "mean_prob": round(mean_prob * 100.0, 2),
        "raw_prob": round(mean_prob, 6),
        "threshold": round(threshold * 100.0, 2),
        "is_cancer": bool(mean_prob >= threshold),
        "rot_probs": {k: round(v * 100.0, 2) for k, v in rot_probs.items()},
        "tta_std": round(tta_std * 100.0, 2),
        "elapsed_ms": elapsed_ms,
        "attribution": {
            "center_peak": round(peak_cam * 100.0, 1),
            "center_mean": round(center_cam * 100.0, 1),
            "outer_mean": round(outer_cam * 100.0, 1),
            "tta_stability": round(tta_stability * 100.0, 1),
        },
        "roi_image": render_overlay_b64(img_pil, None, (26, 168, 108), enhance, sharpen),
        "cam_image": render_overlay_b64(img_pil, cam_map, (255, 255, 255), enhance, sharpen),
    }


@app.post("/api/batch")
async def batch_screen(
    threshold: float = Form(0.1261),
    use_builtin: bool = Form(True),
    files: Optional[List[UploadFile]] = File(None),
    gt_csv: Optional[UploadFile] = File(None),
):
    if not model_loaded or pcam_model is None:
        raise HTTPException(status_code=503, detail="PyTorch checkpoint not loaded.")

    t0 = time.perf_counter()
    gt_lookup: Dict[str, int] = {}
    if gt_csv is not None:
        csv_bytes = await gt_csv.read()
        try:
            gt_df = pd.read_csv(io.BytesIO(csv_bytes))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}")
        if "id" in gt_df.columns and "label" in gt_df.columns:
            gt_lookup = dict(zip(gt_df["id"].astype(str), gt_df["label"].astype(int)))

    batch_items = []
    if files:
        for f in files:
            content = await f.read()
            if not content:
                continue
            try:
                batch_items.append((f.filename or "patch.png", Image.open(io.BytesIO(content)).convert("RGB")))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Could not decode image '{f.filename}': {exc}")
    elif use_builtin:
        for item in list_reference_samples():
            fpath = os.path.join(SAMPLE_DIR, item["filename"])
            batch_items.append((item["filename"], Image.open(fpath).convert("RGB")))

    if not batch_items:
        raise HTTPException(status_code=400, detail="No slides provided for batch screening.")

    rows = []
    for fname, pil_img in batch_items:
        mp, _std, rp = predict_4x_tta(pil_img)
        pred = 1 if mp >= threshold else 0
        stem = os.path.splitext(fname)[0]
        true_lbl = gt_lookup.get(stem, 1 if "cancer" in fname.lower() else (0 if "normal" in fname.lower() else None))
        rows.append({
            "specimen": fname,
            "thumb_b64": encode_thumb_b64(pil_img, size=56),
            "verdict": "METASTATIC" if pred == 1 else "BENIGN",
            "prob_pct": round(mp * 100.0, 2),
            "risk_tier": "High Risk" if mp >= 0.75 else ("Moderate Risk" if mp >= threshold else "Low Risk"),
            "ground_truth": "Metastatic (1)" if true_lbl == 1 else ("Benign (0)" if true_lbl == 0 else "Unlabeled"),
            "audit": ("Correct" if pred == true_lbl else "Mismatch") if true_lbl is not None else "—",
            "rot_0": round(rp["0°"] * 100.0, 1),
            "rot_90": round(rp["90°"] * 100.0, 1),
            "rot_180": round(rp["180°"] * 100.0, 1),
            "rot_270": round(rp["270°"] * 100.0, 1),
            "_pred": pred,
            "_ok": int(pred == true_lbl) if true_lbl is not None else None,
        })

    rows.sort(key=lambda r: r["prob_pct"], reverse=True)
    pos_n = sum(r["_pred"] for r in rows)
    labeled = [r["_ok"] for r in rows if r["_ok"] is not None]
    verified_acc = round(float(np.mean(labeled) * 100.0), 2) if labeled else None

    ui_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    csv_rows = [{k: v for k, v in r.items() if k != "thumb_b64"} for r in ui_rows]
    csv_out = pd.DataFrame(csv_rows).to_csv(index=False)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    return {
        "total": len(ui_rows),
        "positive": pos_n,
        "negative": len(ui_rows) - pos_n,
        "verified_accuracy": verified_acc,
        "mean_risk": round(float(np.mean([r["prob_pct"] for r in ui_rows])), 2),
        "elapsed_ms": elapsed_ms,
        "rows": ui_rows,
        "csv": csv_out,
    }


@app.get("/api/biomarker/init")
def biomarker_init():
    benign = _bio_df[_bio_df["malignant"] == 0]
    malig = _bio_df[_bio_df["malignant"] == 1]
    defaults = {
        "mean radius": 14.8,
        "mean texture": 19.8,
        "mean perimeter": 96.5,
        "mean area": 680.0,
        "mean smoothness": 0.098,
        "mean compactness": 0.115,
        "mean concavity": 0.105,
        "mean concave points": 0.055,
        "worst radius": 17.5,
        "worst concavity": 0.32,
    }
    ranges = {
        "mean radius": [6.0, 29.0, 0.1],
        "mean texture": [9.0, 40.0, 0.2],
        "mean perimeter": [43.0, 190.0, 0.5],
        "mean area": [140.0, 2500.0, 5.0],
        "mean smoothness": [0.05, 0.16, 0.002],
        "mean compactness": [0.02, 0.35, 0.005],
        "mean concavity": [0.0, 0.43, 0.005],
        "mean concave points": [0.0, 0.20, 0.002],
        "worst radius": [7.0, 36.0, 0.1],
        "worst concavity": [0.0, 1.25, 0.01],
    }
    presets = {
        "benign": {f: round(float(_benign_means[f]), 3) for f in BIO_FEATURES},
        "borderline": defaults,
        "malignant": {f: round(float(_malig_means[f]), 3) for f in BIO_FEATURES},
    }
    return {
        "accuracy": round(bio_accuracy * 100.0, 1),
        "n_samples": int(len(_bio_df)),
        "features": BIO_FEATURES,
        "defaults": defaults,
        "ranges": ranges,
        "presets": presets,
        "benign": {
            "x": benign["mean radius"].round(3).tolist(),
            "y": benign["mean concavity"].round(4).tolist(),
            "z": benign["worst radius"].round(3).tolist(),
        },
        "malignant": {
            "x": malig["mean radius"].round(3).tolist(),
            "y": malig["mean concavity"].round(4).tolist(),
            "z": malig["worst radius"].round(3).tolist(),
        },
    }


class BiomarkerRequest(BaseModel):
    values: Dict[str, float]


@app.post("/api/biomarker/predict")
def biomarker_predict(req: BiomarkerRequest):
    missing = [f for f in BIO_FEATURES if f not in req.values]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing biomarker features: {', '.join(missing)}")

    row = [[float(req.values[f]) for f in BIO_FEATURES]]
    df_in = pd.DataFrame(row, columns=BIO_FEATURES)
    prob_mal = float(bio_clf.predict_proba(df_in)[0][1])
    imps = sorted(
        [{"feature": f.title(), "value": round(float(v * 100.0), 2)} for f, v in zip(BIO_FEATURES, bio_clf.feature_importances_)],
        key=lambda x: x["value"],
    )
    risk_tier = "High Risk" if prob_mal >= 0.70 else ("Borderline" if prob_mal >= 0.35 else "Low Risk")
    return {
        "malignant_prob": round(prob_mal * 100.0, 2),
        "benign_prob": round((1.0 - prob_mal) * 100.0, 2),
        "risk_tier": risk_tier,
        "importances": imps,
    }


class FeedbackRequest(BaseModel):
    rating: int
    category: str = "UI & Themes"
    message: str = ""
    theme: str = "gold-espresso"


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    if not (1 <= req.rating <= 5):
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5.")
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rating": int(req.rating),
        "category": (req.category or "General")[:60],
        "theme": (req.theme or "gold-espresso")[:40],
        "message": (req.message or "").strip()[:1000],
    }
    try:
        with open("feedback_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass
    return {"ok": True, "message": "Thank you for your feedback!"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
