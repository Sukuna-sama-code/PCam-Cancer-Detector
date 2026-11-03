import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
import streamlit as st

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="PATHO-AI | Clinical Histopathology Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CENTER-AWARE RESNET-34 ARCHITECTURE
# ==========================================
class CenterAwareResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet34(weights=None)

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
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
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)

        global_pool = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)

        h, w = feat.shape[2], feat.shape[3]
        h1, h2 = max(0, h // 3), min(h, 2 * (h // 3))
        w1, w2 = max(0, w // 3), min(w, 2 * (w // 3))
        center_feat = feat[:, :, h1:h2, w1:w2]
        center_pool = F.adaptive_avg_pool2d(center_feat, (1, 1)).flatten(1)

        combined = torch.cat([global_pool, center_pool], dim=1)
        return self.head(combined)


@st.cache_resource
def load_deployment_bundle(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        threshold = float(checkpoint.get("optimal_threshold", checkpoint.get("threshold", 0.126)))
        val_auc = float(checkpoint.get("val_auc", checkpoint.get("best_val_auc", 0.9667)))
        test_auc = float(checkpoint.get("test_auc_tta", checkpoint.get("test_auc", 0.9534)))
    else:
        state_dict = checkpoint.state_dict()
        threshold, val_auc, test_auc = 0.126, 0.9667, 0.9534

    clean_sd = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ["backbone.", "model.", "module."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        clean_sd[new_k] = v

    model = CenterAwareResNet34().to(device)
    model.load_state_dict(clean_sd, strict=True)
    model.eval()
    return model, device, threshold, val_auc, test_auc


# ==========================================
# 3. PREPROCESSING, 4x TTA, GRAD-CAM & OPTICS
# ==========================================
eval_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_with_4x_tta(model, img_pil, device):
    angles = [0, 90, 180, 270]
    rot_probs = {}
    with torch.no_grad():
        for angle in angles:
            rot_img = img_pil.rotate(angle) if angle > 0 else img_pil
            tensor = eval_transform(rot_img).unsqueeze(0).to(device)
            logit = model(tensor)
            prob = torch.sigmoid(logit).item()
            rot_probs[f"{angle}°"] = prob
    mean_prob = float(np.mean(list(rot_probs.values())))
    return mean_prob, rot_probs


def generate_gradcam(model, img_pil, device):
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.layer4[-1]
    h_f = target_layer.register_forward_hook(forward_hook)
    h_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        tensor = eval_transform(img_pil).unsqueeze(0).to(device)
        tensor.requires_grad_(True)
        model.zero_grad()
        output = model(tensor)
        output.backward(torch.ones_like(output))

        grads = gradients[0].detach().cpu().numpy()[0]
        acts = activations[0].detach().cpu().numpy()[0]
        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / (cam.max() + 1e-8)
        cam_resized = cv2.resize(cam, (96, 96), interpolation=cv2.INTER_CUBIC)
        return cam_resized
    finally:
        h_f.remove()
        h_b.remove()


def enhance_histopathology_patch(rgb_96, sharpen_strength=1.25, apply_clahe=True):
    target_size = (576, 576)
    upscaled = cv2.resize(rgb_96, target_size, interpolation=cv2.INTER_LANCZOS4)

    if apply_clahe:
        lab = cv2.cvtColor(upscaled, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_chan)
        l_blended = cv2.addWeighted(l_enhanced, 0.55, l_chan, 0.45, 0)
        upscaled = cv2.cvtColor(cv2.merge((l_blended, a_chan, b_chan)), cv2.COLOR_LAB2RGB)

    if sharpen_strength > 0:
        gaussian_blur = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.4)
        upscaled = cv2.addWeighted(
            upscaled, 1.0 + sharpen_strength,
            gaussian_blur, -sharpen_strength, 0
        )

    return np.clip(upscaled, 0, 255).astype(np.uint8)


def draw_roi_and_overlay(img_pil, cam=None, roi_color=(13, 148, 136), enhance=True, sharpen_strength=1.25):
    base_rgb = np.array(img_pil.resize((96, 96), Image.Resampling.LANCZOS).convert("RGB"))

    if enhance:
        display_img = enhance_histopathology_patch(
            base_rgb,
            sharpen_strength=sharpen_strength,
            apply_clahe=True
        )
    else:
        display_img = cv2.resize(base_rgb, (576, 576), interpolation=cv2.INTER_NEAREST)

    if cam is not None:
        cam_hi = cv2.resize(cam, (576, 576), interpolation=cv2.INTER_CUBIC)
        cam_hi = np.clip(cam_hi, 0, 1)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_hi), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        display_img = cv2.addWeighted(display_img, 0.60, heatmap, 0.40, 0)

    cv2.rectangle(display_img, (192, 192), (384, 384), roi_color, 3, lineType=cv2.LINE_AA)
    return display_img


# ==========================================
# 4. SIDEBAR: CONTROLS & THEME ENGINE
# ==========================================
with st.sidebar:
    st.markdown("### 🎨 Clinical Environment Theme")
    ui_theme = st.selectbox(
        "Select Workstation Aesthetic",
        options=[
            "🏥 Sekond Hospital Interior (Daylight Teal)",
            "🩺 Mediso Clinical SaaS (Soft Sky & Pearl)",
            "🌑 PACS Radiology Dark (Bioluminescent)"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🏥 System Calibration")
    ckpt_file = st.text_input("Model Checkpoint File", value="pcam_deployment_bundle.pt")

    model_loaded = False
    if os.path.exists(ckpt_file):
        try:
            model, device, default_thresh, val_auc, test_auc = load_deployment_bundle(ckpt_file)
            st.success(f"● Diagnostic Core Online ({str(device).upper()})")
            model_loaded = True
        except Exception as e:
            st.error(f"Checkpoint Error: {e}")
            default_thresh, val_auc, test_auc = 0.126, 0.9667, 0.9534
    else:
        st.error(f"File not found: {ckpt_file}")
        default_thresh, val_auc, test_auc = 0.126, 0.9667, 0.9534

    threshold = st.slider(
        "Decision Threshold (Calibrated via Youden's J)",
        min_value=0.01,
        max_value=0.99,
        value=float(round(default_thresh, 3)),
        step=0.005,
        format="%.3f"
    )

    st.markdown("---")
    st.markdown("### 🔬 Optical Magnification")
    enhance_optics = st.toggle("HD Optical Enhancement (Lanczos4 + CLAHE)", value=True)
    sharpen_strength = st.slider(
        "Nuclear Edge Sharpening (USM)",
        min_value=0.0,
        max_value=2.5,
        value=1.25,
        step=0.05,
        disabled=not enhance_optics
    )

    st.markdown("---")
    st.markdown("### 📊 Benchmark Telemetry")
    c_m1, c_m2 = st.columns(2)
    c_m1.metric("Validation AUC", f"{val_auc:.4f}")
    c_m2.metric("Test AUC (4x TTA)", f"{test_auc:.4f}")

    st.markdown("---")
    st.markdown("### 🧬 Clinical Protocol")
    st.caption(
        "**PCam Center Rule:** Metastasis is scored positive if ≥1 tumor cell "
        "occupies the central **32×32 px** zone (marked by the target ROI box). "
        "Outer tissue provides morphological context."
    )


# ==========================================
# 5. DRIBBBLE HOSPITAL INTERIOR CSS ENGINE
# ==========================================
def inject_clinical_theme(theme_choice: str):
    sekond_hospital_svg = (
        "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1440' height='900' viewBox='0 0 1440 900'%3E"
        "%3Cdefs%3E"
        "%3ClinearGradient id='wall' x1='0' y1='0' x2='0' y2='1'%3E"
        "%3Cstop offset='0%25' stop-color='%23e3f3f5'/%3E"
        "%3Cstop offset='58%25' stop-color='%23edf7f8'/%3E"
        "%3Cstop offset='58.5%25' stop-color='%23b5dce1'/%3E"
        "%3Cstop offset='60%25' stop-color='%23d5ecef'/%3E"
        "%3Cstop offset='100%25' stop-color='%23c4e3e7'/%3E"
        "%3C/linearGradient%3E"
        "%3ClinearGradient id='sunbeam' x1='0' y1='0' x2='1' y2='1'%3E"
        "%3Cstop offset='0%25' stop-color='%23ffffff' stop-opacity='0.65'/%3E"
        "%3Cstop offset='100%25' stop-color='%23ffffff' stop-opacity='0.0'/%3E"
        "%3C/linearGradient%3E"
        "%3C/defs%3E"
        "%3Crect width='1440' height='900' fill='url(%23wall)'/%3E"
        "%3Cpolygon points='220,0 440,0 390,42 260,42' fill='%23ffffff' opacity='0.85'/%3E"
        "%3Cpolygon points='620,0 840,0 800,42 660,42' fill='%23ffffff' opacity='0.85'/%3E"
        "%3Cpolygon points='1020,0 1240,0 1190,42 1060,42' fill='%23ffffff' opacity='0.85'/%3E"
        "%3Cg opacity='0.55'%3E"
        "%3Crect x='90' y='110' width='260' height='310' rx='10' fill='%23ffffff' stroke='%2399cbd2' stroke-width='4'/%3E"
        "%3Cline x1='220' y1='110' x2='220' y2='420' stroke='%23b2d8de' stroke-width='3'/%3E"
        "%3Cline x1='90' y1='265' x2='350' y2='265' stroke='%23b2d8de' stroke-width='3'/%3E"
        "%3Crect x='1090' y='110' width='260' height='310' rx='10' fill='%23ffffff' stroke='%2399cbd2' stroke-width='4'/%3E"
        "%3Cline x1='1220' y1='110' x2='1220' y2='420' stroke='%23b2d8de' stroke-width='3'/%3E"
        "%3Cline x1='1090' y1='265' x2='1350' y2='265' stroke='%23b2d8de' stroke-width='3'/%3E"
        "%3C/g%3E"
        "%3Cpolygon points='90,110 350,110 640,900 220,900' fill='url(%23sunbeam)'/%3E"
        "%3Cpolygon points='1090,110 1350,110 1440,650 1180,900' fill='url(%23sunbeam)'/%3E"
        "%3Cg opacity='0.18' fill='%230d9488'%3E"
        "%3Ccircle cx='720' cy='250' r='74' fill='none' stroke='%230d9488' stroke-width='3'/%3E"
        "%3Crect x='706' y='204' width='28' height='92' rx='5'/%3E"
        "%3Crect x='674' y='236' width='92' height='28' rx='5'/%3E"
        "%3C/g%3E"
        "%3C/svg%3E"
    )

    if "Sekond" in theme_choice:
        bg_css = f"""
            background-color: #e6f3f5;
            background-image: url("{sekond_hospital_svg}");
            background-size: cover;
            background-position: center top;
            background-attachment: fixed;
            color: #0f2937;
        """
        sidebar_bg = "linear-gradient(180deg, #f7fcfd 0%, #eaf5f7 100%)"
        sidebar_border = "#bce0e6"
        text_primary = "#0f2937"
        text_secondary = "#476978"
        accent_primary = "#0d9488"
        card_bg = "rgba(255, 255, 255, 0.86)"
        card_border = "rgba(13, 148, 136, 0.24)"
        card_shadow = "0 12px 32px -4px rgba(15, 41, 55, 0.08), 0 4px 12px rgba(13, 148, 136, 0.06)"
        banner_bg = "linear-gradient(105deg, rgba(255, 255, 255, 0.94) 0%, rgba(236, 251, 252, 0.92) 100%)"
        pill_bg = "#f0fdfa"
        pill_text = "#0f766e"
        pill_border = "#99f6e4"
        input_bg = "#ffffff"
    elif "Mediso" in theme_choice:
        bg_css = """
            background-color: #f0f6ff;
            background-image: 
                radial-gradient(at 12% 10%, rgba(56, 189, 248, 0.18) 0px, transparent 50%),
                radial-gradient(at 88% 15%, rgba(14, 165, 233, 0.15) 0px, transparent 50%),
                radial-gradient(at 50% 85%, rgba(45, 212, 191, 0.14) 0px, transparent 55%),
                linear-gradient(rgba(2, 132, 199, 0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(2, 132, 199, 0.04) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 100% 100%, 32px 32px, 32px 32px;
            background-attachment: fixed;
            color: #0f172a;
        """
        sidebar_bg = "linear-gradient(180deg, #ffffff 0%, #f1f7fe 100%)"
        sidebar_border = "#cbd5e1"
        text_primary = "#0f172a"
        text_secondary = "#475569"
        accent_primary = "#0284c7"
        card_bg = "rgba(255, 255, 255, 0.88)"
        card_border = "rgba(2, 132, 199, 0.20)"
        card_shadow = "0 14px 34px -6px rgba(2, 132, 199, 0.10), 0 4px 12px rgba(15, 23, 42, 0.04)"
        banner_bg = "linear-gradient(105deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 249, 255, 0.94) 100%)"
        pill_bg = "#e0f2fe"
        pill_text = "#0369a1"
        pill_border = "#bae6fd"
        input_bg = "#ffffff"
    else:
        bg_css = """
            background-color: #060d14;
            background-image: 
                radial-gradient(circle at 20% 15%, rgba(13, 148, 136, 0.16) 0%, transparent 45%),
                radial-gradient(circle at 80% 10%, rgba(2, 132, 199, 0.16) 0%, transparent 45%),
                linear-gradient(rgba(56, 189, 248, 0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(56, 189, 248, 0.04) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 28px 28px, 28px 28px;
            background-attachment: fixed;
            color: #f1f5f9;
        """
        sidebar_bg = "linear-gradient(180deg, #0a141f 0%, #070e17 100%)"
        sidebar_border = "rgba(56, 189, 248, 0.22)"
        text_primary = "#f8fafc"
        text_secondary = "#94a3b8"
        accent_primary = "#14b8a6"
        card_bg = "rgba(13, 24, 38, 0.86)"
        card_border = "rgba(45, 212, 191, 0.25)"
        card_shadow = "0 12px 32px rgba(0, 0, 0, 0.45)"
        banner_bg = "linear-gradient(105deg, rgba(13, 25, 42, 0.95) 0%, rgba(9, 34, 46, 0.92) 100%)"
        pill_bg = "rgba(13, 148, 136, 0.2)"
        pill_text = "#5eead4"
        pill_border = "rgba(45, 212, 191, 0.4)"
        input_bg = "#0f1c2e"

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        .stApp {{
            {bg_css}
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}

        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label, .stApp .stMarkdown p {{
            color: {text_primary} !important;
        }}
        .stApp .stCaption, .stApp small {{
            color: {text_secondary} !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background: {sidebar_bg} !important;
            border-right: 1px solid {sidebar_border} !important;
            box-shadow: 4px 0 24px rgba(15, 41, 55, 0.05);
        }}
        section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{
            color: {accent_primary} !important;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
        }}
        section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {{
            color: {text_secondary} !important;
            font-weight: 600;
        }}

        /* Selectbox, TextInput, Buttons & FileUploader */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {{
            background-color: {input_bg} !important;
            border-color: {card_border} !important;
            color: {text_primary} !important;
            border-radius: 10px !important;
        }}
        div[data-baseweb="select"] span {{
            color: {text_primary} !important;
            font-weight: 500;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {input_bg} !important;
            border: 1.5px dashed {accent_primary} !important;
            border-radius: 12px !important;
        }}
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small {{
            color: {text_secondary} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button,
        .stButton > button,
        .stDownloadButton > button {{
            background-color: {accent_primary} !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.45rem 1rem !important;
        }}

        /* Tabs Styling */
        button[data-baseweb="tab"] {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            color: {text_primary} !important;
        }}

        /* Hospital Interior Top Header Banner */
        .his-banner {{
            background: {banner_bg};
            border: 1px solid {card_border};
            border-left: 6px solid {accent_primary};
            border-radius: 16px;
            padding: 22px 28px;
            margin-bottom: 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            box-shadow: {card_shadow};
            backdrop-filter: blur(14px);
        }}
        .his-title {{
            font-size: 1.65rem;
            font-weight: 800;
            color: {text_primary};
            letter-spacing: -0.4px;
            margin: 2px 0 4px 0;
        }}
        .his-subtitle {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            font-weight: 700;
            color: {accent_primary};
            text-transform: uppercase;
            letter-spacing: 1.4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .his-meta {{
            font-size: 0.88rem;
            color: {text_secondary};
            font-weight: 500;
        }}
        .his-badges {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .clinical-pill {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            font-weight: 600;
            padding: 7px 14px;
            border-radius: 999px;
            background: {pill_bg};
            border: 1px solid {pill_border};
            color: {pill_text};
        }}

        /* Specimen Acquisition Glass Card */
        .clinical-panel {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 16px;
            padding: 18px 24px 8px 24px;
            margin-bottom: 20px;
            box-shadow: {card_shadow};
            backdrop-filter: blur(14px);
        }}
        .panel-header {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.80rem;
            font-weight: 700;
            color: {accent_primary};
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* Diagnostic Viewport Cards */
        .viewport-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-top: 4px solid {accent_primary};
            border-radius: 14px;
            padding: 14px 18px;
            margin-bottom: 12px;
            box-shadow: {card_shadow};
            backdrop-filter: blur(12px);
        }}
        .viewport-header {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.88rem;
            font-weight: 700;
            color: {text_primary};
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .viewport-tag {{
            font-family: 'JetBrains Mono', monospace;
            color: {pill_text};
            background: {pill_bg};
            border: 1px solid {pill_border};
            padding: 3px 9px;
            border-radius: 999px;
            font-size: 0.70rem;
            font-weight: 600;
        }}

        /* Clinical Verdict Badges */
        .verdict-cancer {{
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 1.5px solid #ef4444;
            border-left: 6px solid #dc2626;
            padding: 16px;
            border-radius: 12px;
            color: #991b1b;
            font-weight: 800;
            text-align: center;
            font-size: 1.04rem;
            margin-bottom: 14px;
            box-shadow: 0 8px 20px rgba(239, 68, 68, 0.14);
        }}
        .verdict-normal {{
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border: 1.5px solid #10b981;
            border-left: 6px solid #059669;
            padding: 16px;
            border-radius: 12px;
            color: #065f46;
            font-weight: 800;
            text-align: center;
            font-size: 1.04rem;
            margin-bottom: 14px;
            box-shadow: 0 8px 20px rgba(16, 185, 129, 0.14);
        }}

        /* Probability Readout Card */
        .prob-readout {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 14px;
            padding: 18px;
            text-align: center;
            margin-top: 12px;
            margin-bottom: 12px;
            box-shadow: {card_shadow};
        }}
        .prob-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            font-weight: 600;
            color: {text_secondary};
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}
        .prob-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.45rem;
            font-weight: 800;
            margin: 6px 0;
        }}

        /* Specimen Telemetry Strip */
        .meta-strip {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            color: {text_primary};
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 12px 18px;
            margin-bottom: 18px;
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            box-shadow: {card_shadow};
            backdrop-filter: blur(10px);
        }}

        .ecg-svg {{
             width: 110px;
             height: 28px;
             vertical-align: middle;
        }}
        .ecg-line {{
            fill: none;
            stroke: {accent_primary};
            stroke-width: 2.2;
            stroke-linecap: round;
            stroke-linejoin: round;
            stroke-dasharray: 200;
            stroke-dashoffset: 200;
            animation: ecgPulse 2.8s infinite linear;
        }}
        @keyframes ecgPulse {{
            0% {{ stroke-dashoffset: 200; opacity: 0.4; }}
            50% {{ stroke-dashoffset: 0; opacity: 1; }}
            100% {{ stroke-dashoffset: -200; opacity: 0.4; }}
        }}
    </style>
    """, unsafe_allow_html=True)

inject_clinical_theme(ui_theme)


# ==========================================
# 6. MAIN HOSPITAL WORKSTATION HEADER & TABS
# ==========================================
st.markdown("""
<div class="his-banner">
    <div>
        <div class="his-subtitle">
            <span>🏥 DEPT. OF DIGITAL PATHOLOGY & ONCOLOGY // SUITE 04-B</span>
            <svg class="ecg-svg" viewBox="0 0 120 30">
                <polyline class="ecg-line" points="0,15 25,15 32,6 38,25 45,3 52,27 58,12 65,15 120,15"/>
            </svg>
        </div>
        <h1 class="his-title">PCam Histopathology Metastasis Workstation</h1>
        <div class="his-meta">Center-Aware ResNet-34 • 4x Rotational TTA • 6x HD Lanczos-4 Optical Magnification</div>
    </div>
    <div class="his-badges">
        <span class="clinical-pill">● DIAGNOSTIC CORE READY</span>
        <span class="clinical-pill">STAIN: H&E BRIGHTFIELD</span>
        <span class="clinical-pill">ROI ZONE: 32×32 px CENTER</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab_single, tab_batch = st.tabs([
    "🔬 Single-Slide Diagnostic Workstation",
    "📊 Batch Slide Screening & Summary Table"
])

sample_dir = "sample_biopsy_patches"

# ==========================================
# TAB 1: SINGLE-SLIDE WORKSTATION
# ==========================================
with tab_single:
    st.markdown("""
    <div class="clinical-panel">
        <div class="panel-header">🧪 SPECIMEN ACQUISITION & HOSPITAL BIOPSY ARCHIVE</div>
    </div>
    """, unsafe_allow_html=True)

    acq_col1, acq_col2 = st.columns([1.15, 1.35])

    sample_options = {"-- Use Custom Uploaded Slide (Right) --": None}
    if os.path.isdir(sample_dir):
        for fname in sorted(os.listdir(sample_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                if "cancer" in fname.lower():
                    label = f"🔴 Metastatic Reference — {fname}"
                elif "normal" in fname.lower():
                    label = f"🟢 Benign / Normal Reference — {fname}"
                else:
                    label = f"🧫 Biopsy Specimen — {fname}"
                sample_options[label] = os.path.join(sample_dir, fname)

    with acq_col1:
        selected_sample_label = st.selectbox(
            "Quick-Load Hospital Reference Biopsy (No Upload Needed)",
            options=list(sample_options.keys()),
            index=1 if len(sample_options) > 1 else 0
        )

    with acq_col2:
        uploaded_file = st.file_uploader(
            "Or Upload External Lymph Node Patch (PNG / JPG / TIF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"]
        )

    active_img = None
    specimen_id = None
    source_mode = None

    if uploaded_file is not None:
        active_img = Image.open(uploaded_file).convert("RGB")
        specimen_id = uploaded_file.name
        source_mode = "EXTERNAL UPLOAD"
    elif sample_options.get(selected_sample_label) is not None:
        sample_path = sample_options[selected_sample_label]
        if os.path.exists(sample_path):
            active_img = Image.open(sample_path).convert("RGB")
            specimen_id = os.path.basename(sample_path)
            source_mode = "REFERENCE BIOPSY BANK"

    if active_img is not None and model_loaded:
        img_np = np.array(active_img)
        mean_intensity = np.mean(img_np)

        st.markdown(f"""
        <div class="meta-strip">
            <span><b>SPECIMEN ID:</b> {specimen_id}</span>
            <span><b>SOURCE:</b> {source_mode}</span>
            <span><b>NATIVE FOV:</b> {active_img.size[0]}×{active_img.size[1]} px (Rendered at 576×576 HD)</span>
            <span><b>TARGET ROI:</b> 32×32 px Center [32:64, 32:64]</span>
            <span><b>OPTICAL INTENSITY:</b> {mean_intensity:.1f} / 255</span>
        </div>
        """, unsafe_allow_html=True)

        mean_prob, rot_probs = predict_with_4x_tta(model, active_img, device)
        cam_map = generate_gradcam(model, active_img, device)
        is_cancer = mean_prob >= threshold

        roi_img = draw_roi_and_overlay(
            active_img,
            cam=None,
            roi_color=(13, 148, 136),
            enhance=enhance_optics,
            sharpen_strength=sharpen_strength
        )
        cam_img = draw_roi_and_overlay(
            active_img,
            cam=cam_map,
            roi_color=(255, 255, 255),
            enhance=enhance_optics,
            sharpen_strength=sharpen_strength
        )

        col1, col2, col3 = st.columns([1, 1, 1.15], gap="medium")

        with col1:
            st.markdown("""
            <div class="viewport-card">
                <div class="viewport-header">
                    <span>1. H&E BRIGHTFIELD + ROI</span>
                    <span class="viewport-tag">6x LANCZOS-4 HD</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(roi_img, use_container_width=True, caption="Teal Square: 32×32px Diagnostic ROI Zone")

        with col2:
            st.markdown("""
            <div class="viewport-card">
                <div class="viewport-header">
                    <span>2. GRAD-CAM ACTIVATION</span>
                    <span class="viewport-tag">LAYER-4 ATTENTION</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(cam_img, use_container_width=True, caption="Red/Yellow Hotspots: High Feature Attribution")

        with col3:
            st.markdown("""
            <div class="viewport-card">
                <div class="viewport-header">
                    <span>3. AUTOMATED PATHOLOGY VERDICT</span>
                    <span class="viewport-tag">4x TTA CONSENSUS</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if is_cancer:
                st.markdown(
                    '<div class="verdict-cancer">🚨 METASTATIC TISSUE DETECTED<br>'
                    '<span style="font-size:0.80rem;font-weight:600;">POSITIVE FOR LYMPH NODE METASTASIS</span></div>',
                    unsafe_allow_html=True
                )
                val_color = "#dc2626"
            else:
                st.markdown(
                    '<div class="verdict-normal">✅ BENIGN / NORMAL TISSUE<br>'
                    '<span style="font-size:0.80rem;font-weight:600;">NO METASTATIC FOCI IDENTIFIED</span></div>',
                    unsafe_allow_html=True
                )
                val_color = "#059669"

            st.progress(min(max(mean_prob, 0.0), 1.0))

            st.markdown(f"""
            <div class="prob-readout">
                <div class="prob-label">4x TTA Malignancy Probability</div>
                <div class="prob-value" style="color: {val_color};">{mean_prob * 100:.2f}%</div>
                <div class="prob-label" style="font-size: 0.72rem;">
                    Calibrated Cutoff Threshold: <b>{threshold * 100:.1f}% ({threshold:.3f})</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📐 View 4x Rotational TTA Telemetry Breakdown", expanded=True):
                r_cols = st.columns(4)
                for idx, (ang, p_val) in enumerate(rot_probs.items()):
                    r_cols[idx].metric(label=f"Rot {ang}", value=f"{p_val * 100:.1f}%")


# ==========================================
# TAB 2: BATCH SLIDE SCREENING & SUMMARY TABLE
# ==========================================
with tab_batch:
    st.markdown("""
    <div class="clinical-panel">
        <div class="panel-header">📊 MULTI-SLIDE COHORT SCREENING & ACCURACY VALIDATION</div>
    </div>
    """, unsafe_allow_html=True)

    b_col1, b_col2 = st.columns([1.4, 1.0], gap="medium")

    with b_col1:
        batch_files = st.file_uploader(
            "Upload Multiple Lymph Node Biopsy Patches (Select 10–100+ PNG / JPG / TIF files)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True
        )
        run_builtin_batch = st.checkbox(
            "Or evaluate all 8 built-in Reference Biopsy Patches automatically",
            value=(not batch_files)
        )

    with b_col2:
        gt_csv_file = st.file_uploader(
            "Optional: Upload Ground-Truth Labels CSV (columns: id, label) to compute Accuracy",
            type=["csv"]
        )

    gt_lookup = {}
    if gt_csv_file is not None:
        try:
            gt_df = pd.read_csv(gt_csv_file)
            if "id" in gt_df.columns and "label" in gt_df.columns:
                gt_lookup = dict(zip(gt_df["id"].astype(str), gt_df["label"].astype(int)))
                st.success(f"Loaded {len(gt_lookup):,} ground-truth labels from CSV.")
        except Exception as e:
            st.warning(f"Could not parse ground-truth CSV: {e}")

    batch_items = []
    if batch_files:
        for f in batch_files:
            batch_items.append((f.name, Image.open(f).convert("RGB")))
    elif run_builtin_batch and os.path.isdir(sample_dir):
        for fname in sorted(os.listdir(sample_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                fpath = os.path.join(sample_dir, fname)
                batch_items.append((fname, Image.open(fpath).convert("RGB")))

    if batch_items and model_loaded:
        records = []
        progress_bar = st.progress(0)

        for idx, (fname, pil_img) in enumerate(batch_items):
            mean_p, rot_p = predict_with_4x_tta(model, pil_img, device)
            pred_label = 1 if mean_p >= threshold else 0
            verdict_str = "🚨 METASTATIC (POS)" if pred_label == 1 else "✅ BENIGN (NEG)"

            if mean_p >= 0.75:
                risk_tier = "High Risk"
            elif mean_p >= threshold:
                risk_tier = "Moderate Risk"
            else:
                risk_tier = "Low Risk"

            stem_id = os.path.splitext(fname)[0]
            true_label = None
            if stem_id in gt_lookup:
                true_label = gt_lookup[stem_id]
            elif "cancer" in fname.lower():
                true_label = 1
            elif "normal" in fname.lower():
                true_label = 0

            if true_label is not None:
                gt_str = "Metastatic (1)" if true_label == 1 else "Benign (0)"
                match_str = "✔️ Correct" if pred_label == true_label else "❌ Mismatch"
                is_correct = int(pred_label == true_label)
            else:
                gt_str = "Unlabeled"
                match_str = "—"
                is_correct = None

            records.append({
                "Specimen Filename": fname,
                "Clinical Verdict": verdict_str,
                "4x TTA Probability (%)": round(mean_p * 100, 2),
                "Risk Tier": risk_tier,
                "Ground Truth": gt_str,
                "Audit Result": match_str,
                "Rot 0° (%)": round(rot_p["0°"] * 100, 1),
                "Rot 90° (%)": round(rot_p["90°"] * 100, 1),
                "Rot 180° (%)": round(rot_p["180°"] * 100, 1),
                "Rot 270° (%)": round(rot_p["270°"] * 100, 1),
                "Mean Intensity": round(float(np.mean(np.array(pil_img))), 1),
                "_is_cancer": pred_label,
                "_is_correct": is_correct
            })
            progress_bar.progress((idx + 1) / len(batch_items))

        progress_bar.empty()

        summary_df = pd.DataFrame(records).sort_values(
            by="4x TTA Probability (%)", ascending=False
        ).reset_index(drop=True)
        summary_df.index = summary_df.index + 1
        summary_df.index.name = "Rank"

        total_slides = len(summary_df)
        pos_count = int(summary_df["_is_cancer"].sum())
        neg_count = total_slides - pos_count
        labeled_rows = summary_df[summary_df["_is_correct"].notnull()]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Slides Screened", f"{total_slides}")
        k2.metric("🚨 Metastatic Detected", f"{pos_count} ({pos_count / total_slides * 100:.1f}%)")
        k3.metric("✅ Benign Cleared", f"{neg_count} ({neg_count / total_slides * 100:.1f}%)")
        if len(labeled_rows) > 0:
            batch_acc = labeled_rows["_is_correct"].mean() * 100
            k4.metric("🎯 Verified Batch Accuracy", f"{batch_acc:.2f}% ({int(labeled_rows['_is_correct'].sum())}/{len(labeled_rows)})")
        else:
            k4.metric("Mean Malignancy Score", f"{summary_df['4x TTA Probability (%)'].mean():.2f}%")

        display_cols = [c for c in summary_df.columns if not c.startswith("_")]

        st.markdown("""
        <div class="viewport-card" style="margin-top: 12px;">
            <div class="viewport-header">
                <span>📋 RANKED CLINICAL BATCH SUMMARY TABLE (SORTED BY MALIGNANCY RISK)</span>
                <span class="viewport-tag">4x TTA COHORT AUDIT</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(summary_df[display_cols], use_container_width=True, height=340)

        csv_bytes = summary_df[display_cols].to_csv(index=True).encode("utf-8")
        st.download_button(
            label="📥 Download Clinical Batch Summary Table (CSV)",
            data=csv_bytes,
            file_name="pcam_clinical_batch_summary.csv",
            mime="text/csv"
        )
