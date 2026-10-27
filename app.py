import os
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

# 1. Page Configuration & Custom Dark Clinical Theme
st.set_page_config(
    page_title="PCam Histopathology Cancer Detector",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .status-cancer {
        background: rgba(220, 38, 38, 0.2);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 14px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
        font-size: 1.2rem;
    }
    .status-normal {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid #10b981;
        color: #6ee7b7;
        padding: 14px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# 2. Model Definition (Self-Contained Architecture matching Colab)
class PCamResNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet34(weights=None)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            backbone.bn1,
            backbone.relu,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.head = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        global_feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
        center_feat = F.adaptive_avg_pool2d(x[:, :, 4:8, 4:8], 1).flatten(1)
        return self.head(torch.cat([global_feat, center_feat], dim=1)).squeeze(1)


# 3. Cached Model Loader
@st.cache_resource
def load_diagnostic_bundle(bundle_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PCamResNet()

    if not os.path.exists(bundle_path):
        return None, None, None, device

    checkpoint = torch.load(bundle_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        opt_thresh = checkpoint.get("optimal_threshold", 0.1261)
        metrics = checkpoint.get("metrics", {})
    else:
        model.load_state_dict(checkpoint)
        opt_thresh = 0.1261
        metrics = {"test_roc_auc": 0.9534, "best_val_auc": 0.9667}

    model.to(device).eval()
    return model, opt_thresh, metrics, device


def preprocess_patch(pil_img: Image.Image, device: str) -> tuple[torch.Tensor, np.ndarray]:
    img_resized = pil_img.convert("RGB").resize((96, 96), Image.Resampling.BICUBIC)
    rgb_np = np.array(img_resized, dtype=np.uint8)

    tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    return tensor, rgb_np


def run_tta_and_gradcam(model: nn.Module, x: torch.Tensor) -> tuple[float, np.ndarray]:
    # 1. 4x Rotational Test-Time Augmentation Probability
    with torch.no_grad():
        p1 = torch.sigmoid(model(x))
        p2 = torch.sigmoid(model(torch.rot90(x, 1, dims=[-2, -1])))
        p3 = torch.sigmoid(model(torch.rot90(x, 2, dims=[-2, -1])))
        p4 = torch.sigmoid(model(torch.rot90(x, 3, dims=[-2, -1])))
        tta_prob = float(((p1 + p2 + p3 + p4) / 4.0).item())

    # 2. Grad-CAM on Layer 4
    gradients, activations = [], []
    handle_f = model.layer4[-1].register_forward_hook(lambda m, i, o: activations.append(o))
    handle_b = model.layer4[-1].register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    x_grad = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    logits = model(x_grad)
    logits.backward()

    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]
    handle_f.remove()
    handle_b.remove()

    weights = np.mean(grads, axis=(1, 2))
    cam = np.maximum(np.tensordot(weights, acts, axes=(0, 0)), 0)
    cam = cv2.resize(cam, (96, 96), interpolation=cv2.INTER_CUBIC)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return tta_prob, cam


# 4. Dashboard Layout
st.title("🔬 PCam Histopathology Metastasis Detector")
st.caption("Center-Aware ResNet-34 with 4x Rotational TTA & Grad-CAM Explainability | Research Use Only")

bundle_file = st.sidebar.text_input("Model Checkpoint File", value="pcam_deployment_bundle.pt")
model, default_thresh, metrics, device = load_diagnostic_bundle(bundle_file)

if model is None:
    st.error(f"Could not find `{bundle_file}` in your D drive folder. Make sure `pcam_deployment_bundle.pt` is in `D:\\PCam_Cancer_Detector`.")
    st.stop()

st.sidebar.success(f"Loaded on **{device.upper()}**")
threshold = st.sidebar.slider("Decision Threshold (Calibrated)", 0.01, 0.95, float(default_thresh), 0.005)

if metrics:
    st.sidebar.markdown("### Benchmark Performance")
    st.sidebar.metric("Validation ROC-AUC", f"{metrics.get('best_val_auc', 0.9667):.4f}")
    st.sidebar.metric("Test ROC-AUC (4x TTA)", f"{metrics.get('test_roc_auc', 0.9534):.4f}")

uploaded_file = st.file_uploader("Upload Lymph Node Histopathology Patch (PNG / JPG / TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    input_tensor, rgb_patch = preprocess_patch(pil_image, device)
    prob, cam_mask = run_tta_and_gradcam(model, input_tensor)

    # Draw 32x32 Center ROI Box
    roi_img = rgb_patch.copy()
    cv2.rectangle(roi_img, (32, 32), (64, 64), (0, 255, 100), 1)

    # Create Grad-CAM Overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay_img = cv2.addWeighted(rgb_patch, 0.55, heatmap, 0.45, 0)
    cv2.rectangle(overlay_img, (32, 32), (64, 64), (255, 255, 255), 1)

    col1, col2, col3 = st.columns([1, 1, 1.2])
    with col1:
        st.subheader("1. H&E Patch + Target ROI")
        st.image(roi_img, caption="Green Box: Center 32x32px Diagnostic Zone", use_container_width=True)

    with col2:
        st.subheader("2. Grad-CAM Activation")
        st.image(overlay_img, caption="Red/Yellow: High Cellular Malignancy Focus", use_container_width=True)

    with col3:
        st.subheader("3. Diagnostic Prediction")
        is_cancer = prob >= threshold
        if is_cancer:
            st.markdown('<div class="status-cancer">⚠️ METASTATIC TISSUE DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-normal">✅ NORMAL TISSUE (NO METASTASIS)</div>', unsafe_allow_html=True)

        st.write("")
        st.progress(min(max(prob, 0.0), 1.0))
        st.markdown(f"""
        <div class="metric-card">
            <h4>4x TTA Malignancy Probability</h4>
            <h1 style="color: {'#f87171' if is_cancer else '#4ade80'};">{prob * 100:.2f}%</h1>
            <p>Calibrated Cutoff Threshold: <b>{threshold:.3f}</b></p>
        </div>
        """, unsafe_allow_html=True)