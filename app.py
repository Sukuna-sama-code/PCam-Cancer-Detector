import os
import cv2
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
    page_title="PATHO-AI | Clinical Metastasis Workstation",
    page_icon="🔬",
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
# 4. SIDEBAR: CONTROLS & WEB-GRADIENTS PALETTE
# ==========================================
with st.sidebar:
    st.markdown("### 🎨 WebGradients Aesthetics")
    gradient_theme = st.selectbox(
        "Select Color Palette",
        options=[
            "✨ Crystal Clear (WebGradients A1C4FD → C2E9FB)",
            "🌊 Deep Ocean Breeze (WebGradients 2E3192 → 1BFFFF)",
            "🍑 Warm Sunset Clinical (WebGradients FAD961 → F76B1C)",
            "🌑 Pure PACS Darkroom (High Contrast Neon)"
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
            st.success(f"● Core Online ({str(device).upper()})")
            model_loaded = True
        except Exception as e:
            st.error(f"Checkpoint Error: {e}")
            default_thresh, val_auc, test_auc = 0.126, 0.9667, 0.9534
    else:
        st.error(f"File not found: {ckpt_file}")
        default_thresh, val_auc, test_auc = 0.126, 0.9667, 0.9534

    threshold = st.slider(
        "Decision Threshold (Youden's J)",
        min_value=0.01,
        max_value=0.99,
        value=float(round(default_thresh, 3)),
        step=0.005,
        format="%.3f"
    )

    st.markdown("---")
    st.markdown("### 🔬 Optical Magnification")
    enhance_optics = st.toggle("HD Optical Enhancement", value=True)
    sharpen_strength = st.slider(
        "Nuclear Sharpening (USM)",
        min_value=0.0,
        max_value=2.5,
        value=1.25,
        step=0.05,
        disabled=not enhance_optics
    )

    st.markdown("---")
    st.markdown("### 📊 Benchmark Telemetry")
    c_m1, c_m2 = st.columns(2)
    c_m1.metric("Val AUC", f"{val_auc:.4f}")
    c_m2.metric("Test AUC", f"{test_auc:.4f}")


# ==========================================
# 5. HIGH-CONTRAST WEB-GRADIENTS CSS ENGINE
# ==========================================
def inject_webgradient_theme(theme_choice: str):
    if "Crystal Clear" in theme_choice:
        bg_css = "background-image: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); background-attachment: fixed;"
        text_primary = "#0f172a"
        text_secondary = "#334155"
        card_bg = "rgba(255, 255, 255, 0.88)"
        card_border = "rgba(255, 255, 255, 0.9)"
        card_shadow = "0 16px 36px rgba(31, 38, 135, 0.12)"
        accent_color = "#0284c7"
        sidebar_bg = "linear-gradient(180deg, #f8fafc 0%, #edf2f7 100%)"
        pill_bg = "#e0f2fe"
        pill_text = "#0369a1"
    elif "Deep Ocean Breeze" in theme_choice:
        bg_css = "background-image: linear-gradient(135deg, #2e3192 0%, #1bffff 100%); background-attachment: fixed;"
        text_primary = "#f8fafc"
        text_secondary = "#cbd5e1"
        card_bg = "rgba(15, 23, 42, 0.82)"
        card_border = "rgba(56, 189, 248, 0.25)"
        card_shadow = "0 16px 36px rgba(0, 0, 0, 0.35)"
        accent_color = "#38bdf8"
        sidebar_bg = "linear-gradient(180deg, #0b1329 0%, #070b19 100%)"
        pill_bg = "rgba(56, 189, 248, 0.2)"
        pill_text = "#7dd3fc"
    elif "Warm Sunset Clinical" in theme_choice:
        bg_css = "background-image: linear-gradient(135deg, #fad961 0%, #f76b1c 100%); background-attachment: fixed;"
        text_primary = "#1c1917"
        text_secondary = "#44403c"
        card_bg = "rgba(255, 255, 255, 0.89)"
        card_border = "rgba(255, 255, 255, 0.9)"
        card_shadow = "0 16px 36px rgba(120, 53, 15, 0.15)"
        accent_color = "#c2410c"
        sidebar_bg = "linear-gradient(180deg, #fffbeb 0%, #fef3c7 100%)"
        pill_bg = "#ffedd5"
        pill_text = "#c2410c"
    else:
        # PACS Darkroom
        bg_css = "background-color: #050b14; background-image: radial-gradient(circle at 50% 0%, rgba(14, 165, 233, 0.12) 0%, transparent 60%); background-attachment: fixed;"
        text_primary = "#f8fafc"
        text_secondary = "#94a3b8"
        card_bg = "rgba(11, 19, 32, 0.9)"
        card_border = "rgba(56, 189, 248, 0.2)"
        card_shadow = "0 16px 36px rgba(0, 0, 0, 0.5)"
        accent_color = "#14b8a6"
        sidebar_bg = "linear-gradient(180deg, #09101d 0%, #060a12 100%)"
        pill_bg = "rgba(20, 184, 166, 0.2)"
        pill_text = "#5eead4"

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        .stApp {{
            {bg_css}
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}

        /* Strict High-Contrast Text Visibility */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label, .stApp span, .stApp div {{
            color: {text_primary};
        }}
        .stApp p, .stApp .stCaption, .stApp small {{
            color: {text_secondary};
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background: {sidebar_bg} !important;
            border-right: 1px solid {card_border} !important;
        }}
        section[data-testid="stSidebar"] label, 
        section[data-testid="stSidebar"] .stMarkdown p {{
            color: {text_primary} !important;
            font-weight: 600;
        }}

        /* Crystal Clear Buttons & Inputs */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {{
            background-color: {card_bg} !important;
            border: 1px solid {accent_color} !important;
            border-radius: 10px !important;
            color: {text_primary} !important;
        }}
        div[data-baseweb="select"] span {{
            color: {text_primary} !important;
            font-weight: 600;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {card_bg} !important;
            border: 2px dashed {accent_color} !important;
            border-radius: 12px !important;
        }}
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small {{
            color: {text_primary} !important;
            font-weight: 600;
        }}

        /* Main Glassmorphic Cards */
        .clinical-banner {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-left: 6px solid {accent_color};
            border-radius: 16px;
            padding: 24px 30px;
            margin-bottom: 24px;
            box-shadow: {card_shadow};
            backdrop-filter: blur(16px);
        }}
        .banner-title {{
            font-size: 1.7rem;
            font-weight: 800;
            color: {text_primary} !important;
            letter-spacing: -0.5px;
            margin: 4px 0;
        }}
        .banner-subtitle {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 700;
            color: {accent_color};
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }}

        .glass-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 18px;
            box-shadow: {card_shadow};
            backdrop-filter: blur(14px);
        }}
        .card-header {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            font-weight: 700;
            color: {accent_color};
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card-tag {{
            background: {pill_bg};
            color: {pill_text};
            border: 1px solid {accent_color};
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
        }}

        /* Verdict Banners */
        .verdict-cancer {{
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border: 2px solid #ef4444;
            border-left: 8px solid #dc2626;
            padding: 18px;
            border-radius: 12px;
            color: #7f1d1d !important;
            font-weight: 800;
            text-align: center;
            font-size: 1.08rem;
            margin-bottom: 16px;
            box-shadow: 0 10px 25px rgba(239, 68, 68, 0.2);
        }}
        .verdict-normal {{
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border: 2px solid #10b981;
            border-left: 8px solid #059669;
            padding: 18px;
            border-radius: 12px;
            color: #064e3b !important;
            font-weight: 800;
            text-align: center;
            font-size: 1.08rem;
            margin-bottom: 16px;
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.2);
        }}

        .metric-box {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            margin-top: 10px;
            box-shadow: {card_shadow};
        }}
        .metric-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 800;
            margin: 4px 0;
        }}
    </style>
    """, unsafe_allow_html=True)

inject_webgradient_theme(gradient_theme)


# ==========================================
# 6. MAIN WORKSTATION UI LAYOUT
# ==========================================
st.markdown("""
<div class="clinical-banner">
    <div class="banner-subtitle">🏥 DEPT. OF DIGITAL PATHOLOGY & ONCOLOGY // WORKSTATION SUITE</div>
    <div class="banner-title">PCam Histopathology Metastasis Workstation</div>
    <div style="font-size: 0.9rem; margin-top: 4px; opacity: 0.85;">
        Center-Aware ResNet-34 • 4x Rotational TTA • 6x HD Lanczos-4 Optical Magnification
    </div>
</div>
""", unsafe_allow_html=True)

# Specimen Selection Panel
st.markdown("""
<div class="glass-card">
    <div class="card-header">
        <span>🧪 Specimen Acquisition & Reference Bank</span>
        <span class="card-tag">INPUT TILE: 96×96 PX</span>
    </div>
""", unsafe_allow_html=True)

acq_col1, acq_col2 = st.columns(2, gap="medium")

sample_dir = "sample_biopsy_patches"
sample_options = {"-- Select Custom Uploaded Slide (Right) --": None}

if os.path.isdir(sample_dir):
    for fname in sorted(os.listdir(sample_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            if "cancer" in fname.lower():
                label = f"🔴 Metastatic Reference — {fname}"
            elif "normal" in fname.lower():
                label = f"🟢 Benign Reference — {fname}"
            else:
                label = f"🧫 Biopsy Specimen — {fname}"
            sample_options[label] = os.path.join(sample_dir, fname)

with acq_col1:
    selected_sample_label = st.selectbox(
        "Quick-Load Hospital Reference Biopsy",
        options=list(sample_options.keys()),
        index=1 if len(sample_options) > 1 else 0
    )

with acq_col2:
    uploaded_file = st.file_uploader(
        "Or Upload External Lymph Node Patch (PNG/JPG/TIF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

st.markdown('</div>', unsafe_allow_html=True)

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


# ==========================================
# 7. VIEWPORTS & DIAGNOSTIC REPORT
# ==========================================
if active_img is not None and model_loaded:
    img_np = np.array(active_img)
    mean_intensity = np.mean(img_np)

    st.markdown(f"""
    <div class="glass-card" style="padding: 12px 20px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <span><b>SPECIMEN ID:</b> {specimen_id}</span>
        <span><b>SOURCE:</b> {source_mode}</span>
        <span><b>RENDER:</b> 576×576 HD Lanczos-4</span>
        <span><b>ROI ZONE:</b> 32×32 px Center</span>
        <span><b>INTENSITY:</b> {mean_intensity:.1f}/255</span>
    </div>
    """, unsafe_allow_html=True)

    mean_prob, rot_probs = predict_with_4x_tta(model, active_img, device)
    cam_map = generate_gradcam(model, active_img, device)
    is_cancer = mean_prob >= threshold

    roi_img = draw_roi_and_overlay(active_img, cam=None, roi_color=(13, 148, 136), enhance=enhance_optics, sharpen_strength=sharpen_strength)
    cam_img = draw_roi_and_overlay(active_img, cam=cam_map, roi_color=(255, 255, 255), enhance=enhance_optics, sharpen_strength=sharpen_strength)

    v_col1, v_col2, v_col3 = st.columns([1, 1, 1.15], gap="medium")

    with v_col1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <span>1. H&E Brightfield</span>
                <span class="card-tag">HD ZOOM</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.image(roi_img, use_container_width=True, caption="Target ROI: 32x32px Center Box")

    with v_col2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <span>2. Grad-CAM Map</span>
                <span class="card-tag">ATTENTION</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.image(cam_img, use_container_width=True, caption="Layer-4 Attention Hotspots")

    with v_col3:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <span>3. Pathology Verdict</span>
                <span class="card-tag">4x TTA</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if is_cancer:
            st.markdown(
                '<div class="verdict-cancer">🚨 METASTATIC TISSUE DETECTED<br>'
                '<span style="font-size:0.78rem; font-weight:700;">POSITIVE FOR LYMPH NODE METASTASIS</span></div>',
                unsafe_allow_html=True
            )
            val_color = "#dc2626"
        else:
            st.markdown(
                '<div class="verdict-normal">✅ BENIGN / NORMAL TISSUE<br>'
                '<span style="font-size:0.78rem; font-weight:700;">NO METASTATIC FOCI IDENTIFIED</span></div>',
                unsafe_allow_html=True
            )
            val_color = "#059669"

        st.progress(min(max(mean_prob, 0.0), 1.0))

        st.markdown(f"""
        <div class="metric-box">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">4x TTA Malignancy Probability</div>
            <div class="metric-value" style="color: {val_color};">{mean_prob * 100:.2f}%</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; opacity: 0.85;">
                Calibrated Threshold: <b>{threshold * 100:.1f}% ({threshold:.3f})</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📐 View 4x Rotational TTA Breakdown", expanded=True):
            r_cols = st.columns(4)
            for idx, (ang, p_val) in enumerate(rot_probs.items()):
                r_cols[idx].metric(label=f"Rot {ang}", value=f"{p_val * 100:.1f}%")
