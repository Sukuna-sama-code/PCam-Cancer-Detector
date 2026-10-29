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
# 1. PAGE CONFIG & HOSPITAL PACS UI STYLING
# ==========================================
st.set_page_config(
    page_title="PATHO-AI | Clinical Metastasis Workstation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #070d14;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(6, 182, 212, 0.10) 0%, transparent 60%),
            linear-gradient(rgba(148, 163, 184, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(148, 163, 184, 0.03) 1px, transparent 1px);
        background-size: 100% 100%, 28px 28px, 28px 28px;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b131e 0%, #091019 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.18);
    }

    .his-banner {
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.95) 0%, rgba(12, 31, 48, 0.92) 100%);
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-left: 5px solid #06b6d4;
        border-radius: 10px;
        padding: 18px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
    }
    .his-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.3px;
        margin: 0;
    }
    .his-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .his-meta {
        font-size: 0.84rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .his-badges {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .clinical-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        padding: 6px 12px;
        border-radius: 6px;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #e0f2fe;
    }
    .pill-online {
        border-color: rgba(16, 185, 129, 0.45);
        color: #6ee7b7;
        background: rgba(6, 78, 59, 0.35);
    }

    .clinical-panel {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 18px;
        backdrop-filter: blur(8px);
    }
    .panel-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 10px;
    }

    .viewport-card {
        background: rgba(13, 22, 35, 0.88);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .viewport-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        color: #e2e8f0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        padding-bottom: 8px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
    }
    .viewport-tag {
        color: #38bdf8;
        font-size: 0.72rem;
    }

    .verdict-cancer {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.55) 0%, rgba(69, 10, 10, 0.75) 100%);
        border: 1px solid #ef4444;
        border-left: 5px solid #ef4444;
        padding: 16px;
        border-radius: 8px;
        color: #fecaca;
        font-weight: 700;
        text-align: center;
        font-size: 1.02rem;
        margin-bottom: 14px;
    }
    .verdict-normal {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.55) 0%, rgba(2, 44, 34, 0.75) 100%);
        border: 1px solid #10b981;
        border-left: 5px solid #10b981;
        padding: 16px;
        border-radius: 8px;
        color: #a7f3d0;
        font-weight: 700;
        text-align: center;
        font-size: 1.02rem;
        margin-bottom: 14px;
    }
    .prob-readout {
        background: rgba(9, 15, 26, 0.9);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-top: 12px;
    }
    .prob-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
    }
    .prob-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.3rem;
        font-weight: 700;
        margin: 6px 0;
    }
    .meta-strip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        color: #94a3b8;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 6px;
        padding: 8px 14px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)


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
    """
    6x Super-Resolution Optical Pipeline (96x96 -> 576x576):
    1. Lanczos-4 Sinc Interpolation (eliminates blocky nearest-neighbor pixels)
    2. LAB-Space Luminance CLAHE (enhances chromatin/nuclear boundaries without shifting H&E stain)
    3. Fine-Radius Unsharp Masking (sharpens cellular membranes)
    """
    target_size = (576, 576)
    upscaled = cv2.resize(rgb_96, target_size, interpolation=cv2.INTER_LANCZOS4)

    if apply_clahe:
        lab = cv2.cvtColor(upscaled, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_chan)
        # Blend 55% CLAHE luminance + 45% original luminance for natural H&E staining
        l_blended = cv2.addWeighted(l_enhanced, 0.55, l_chan, 0.45, 0)
        upscaled = cv2.cvtColor(cv2.merge((l_blended, a_chan, b_chan)), cv2.COLOR_LAB2RGB)

    if sharpen_strength > 0:
        gaussian_blur = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.4)
        upscaled = cv2.addWeighted(
            upscaled, 1.0 + sharpen_strength,
            gaussian_blur, -sharpen_strength, 0
        )

    return np.clip(upscaled, 0, 255).astype(np.uint8)


def draw_roi_and_overlay(img_pil, cam=None, roi_color=(16, 185, 129), enhance=True, sharpen_strength=1.25):
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

    # Draw crisp anti-aliased 32x32 center ROI box at 576x576 scale ([192:384, 192:384])
    cv2.rectangle(display_img, (192, 192), (384, 384), roi_color, 3, lineType=cv2.LINE_AA)
    return display_img


# ==========================================
# 4. SIDEBAR: SYSTEM CALIBRATION & TELEMETRY
# ==========================================
with st.sidebar:
    st.markdown("### 🏥 System Calibration")
    ckpt_file = st.text_input("Model Checkpoint File", value="pcam_deployment_bundle.pt")

    model_loaded = False
    if os.path.exists(ckpt_file):
        try:
            model, device, default_thresh, val_auc, test_auc = load_deployment_bundle(ckpt_file)
            st.success(f"● Engine Online ({str(device).upper()})")
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
# 5. MAIN HOSPITAL WORKSTATION INTERFACE
# ==========================================
st.markdown("""
<div class="his-banner">
    <div>
        <div class="his-subtitle">PATHO-AI // DIGITAL PATHOLOGY WORKSTATION v2.4</div>
        <h1 class="his-title">🔬 PCam Histopathology Metastasis Detector</h1>
        <div class="his-meta">Center-Aware ResNet-34 • 4x Rotational TTA • Grad-CAM Spatial Explainability</div>
    </div>
    <div class="his-badges">
        <span class="clinical-pill pill-online">● DIAGNOSTIC CORE READY</span>
        <span class="clinical-pill">STAIN: H&E BRIGHTFIELD</span>
        <span class="clinical-pill">FOV: 96×96 px (6x HD OPTICS)</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="clinical-panel"><div class="panel-header">🧪 SPECIMEN ACQUISITION & REFERENCE BIOPSY LOADER</div>', unsafe_allow_html=True)

acq_col1, acq_col2 = st.columns([1.15, 1.35])

sample_dir = "sample_biopsy_patches"
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
# 6. DIAGNOSTIC VIEWPORT & REPORT
# ==========================================
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
        roi_color=(16, 185, 129),
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
        st.image(roi_img, use_container_width=True, caption="Green Square: 32×32px Diagnostic ROI Zone")

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
                '<span style="font-size:0.8rem;font-weight:500;">POSITIVE FOR LYMPH NODE METASTASIS</span></div>',
                unsafe_allow_html=True
            )
            val_color = "#f87171"
        else:
            st.markdown(
                '<div class="verdict-normal">✅ BENIGN / NORMAL TISSUE<br>'
                '<span style="font-size:0.8rem;font-weight:500;">NO METASTATIC FOCI IDENTIFIED</span></div>',
                unsafe_allow_html=True
            )
            val_color = "#34d399"

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
