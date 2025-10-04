# streamlit_app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Load model weights
MODEL_PATH = "outputs/checkpoints/best_model_dice.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"""
    ⚠️ **Model file not found!**
    
    Please download the pre-trained model weights:
    1. Run `python download_models.py` to download models automatically
    2. Or manually download `best_model_dice.pth` and place in `outputs/checkpoints/`
    
    Expected path: `{MODEL_PATH}`
    """)
    st.stop()

# Define your model architecture
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x): 
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

class ImprovedUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=4):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(0.1)
            )
        self.enc1 = conv_block(in_ch, 64); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128);   self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256);  self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(256, 512);  self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(512, 1024)
        # Attention blocks: first argument = upsampled features, second = skip connection
        self.att4 = AttentionBlock(512, 512, 256)   # up4(b): 512, e4: 512
        self.att3 = AttentionBlock(256, 256, 128)   # up3(d4): 256, e3: 256
        self.att2 = AttentionBlock(128, 128, 64)    # up2(d3): 128, e2: 128
        self.att1 = AttentionBlock(64, 64, 32)      # up1(d2): 64,  e1: 64
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2); self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2);  self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2);  self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2);   self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        up4_b = self.up4(b)
        d4 = self.dec4(torch.cat([up4_b, self.att4(up4_b, e4)], 1))

        up3_d4 = self.up3(d4)
        d3 = self.dec3(torch.cat([up3_d4, self.att3(up3_d4, e3)], 1))

        up2_d3 = self.up2(d3)
        d2 = self.dec2(torch.cat([up2_d3, self.att2(up2_d3, e2)], 1))

        up1_d2 = self.up1(d2)
        d1 = self.dec1(torch.cat([up1_d2, self.att1(up1_d2, e1)], 1))

        return self.final(d1)

@st.cache_resource
def load_model():
    model = ImprovedUNet(in_ch=3, out_ch=4).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('metrics', {})

def preprocess_image(uploaded_file):
    image_pil = Image.open(uploaded_file).convert("RGB").resize((256,256))
    image = np.array(image_pil) / 255.0
    image_torch = torch.tensor(image.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return image_pil, image_torch

def visualize_prediction(image_pil, pred_mask):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(image_pil)
    axes[0].set_title('Original')
    for i in range(4):
        axes[i+1].imshow((pred_mask[i] > 0.5), cmap='jet', alpha=0.5)
        axes[i+1].set_title(f'Defect {i+1} Mask')
    st.pyplot(fig)

def display_model_accuracy(metrics):
    """Display model training accuracy metrics"""
    st.subheader("📊 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Pixel Accuracy", f"{metrics.get('pixel_acc', 0)*100:.2f}%")
        st.metric("Dice Score", f"{metrics.get('dice', 0)*100:.2f}%")
    
    with col2:
        st.metric("IoU Score", f"{metrics.get('iou', 0)*100:.2f}%")
        st.metric("F1 Score", f"{metrics.get('f1_score', 0)*100:.2f}%")
    
    with col3:
        st.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
        st.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
    
    # Class-wise IoU
    st.subheader("🎯 Per-Class Performance (IoU)")
    class_cols = st.columns(4)
    defect_names = ["Defect 1", "Defect 2", "Defect 3", "Defect 4"]
    
    for i, col in enumerate(class_cols):
        iou_key = f'iou_class_{i}'
        iou_value = metrics.get(iou_key, 0) * 100
        col.metric(defect_names[i], f"{iou_value:.2f}%")

def calculate_prediction_confidence(pred_mask):
    """Calculate confidence scores for current prediction"""
    confidences = []
    for i in range(4):
        # Calculate mean confidence for detected regions
        mask_binary = pred_mask[i] > 0.5
        if np.sum(mask_binary) > 0:
            # Average confidence of detected pixels
            confidence = np.mean(pred_mask[i][mask_binary]) * 100
        else:
            # If no detection, use max probability as confidence
            confidence = np.max(pred_mask[i]) * 100
        confidences.append(confidence)
    return confidences

# Streamlit interface
st.title("🔍 Steel Defect Segmentation Prototype")
st.write("Upload a steel image to segment defects with your trained model.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Load model and display accuracy metrics
model, metrics = load_model()
if metrics:
    with st.expander("📈 View Model Training Accuracy", expanded=False):
        display_model_accuracy(metrics)

if uploaded_file is not None:
    image_pil, image_torch = preprocess_image(uploaded_file)
    st.image(image_pil, caption='Uploaded Image', use_container_width=True)
    
    with torch.no_grad():
        prediction = model(image_torch)
        pred_mask = torch.sigmoid(prediction)[0].cpu().numpy()  # (4, 256, 256)
    
    st.subheader('🎯 Defect Detection Results')
    visualize_prediction(image_pil, pred_mask)
    
    # Calculate prediction confidence
    confidences = calculate_prediction_confidence(pred_mask)
    
    # Display results with confidence
    st.subheader('📊 Detection Summary')
    result_cols = st.columns(4)
    
    for i, col in enumerate(result_cols):
        defect_area = np.sum(pred_mask[i] > 0.5)
        with col:
            if defect_area > 0:
                st.success(f"**Defect {i+1}**")
                st.write(f"Pixels: {defect_area}")
                st.write(f"Confidence: {confidences[i]:.1f}%")
            else:
                st.info(f"**Defect {i+1}**")
                st.write("No defects detected")
                st.write(f"Max confidence: {confidences[i]:.1f}%")
    
    # Overall summary
    total_defects = sum(1 for i in range(4) if np.sum(pred_mask[i] > 0.5) > 0)
    total_defect_pixels = sum(np.sum(pred_mask[i] > 0.5) for i in range(4))
    
    st.subheader('📋 Overall Summary')
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Defect Types Found", total_defects)
    with summary_col2:
        st.metric("Total Defect Pixels", total_defect_pixels)
    with summary_col3:
        defect_percentage = (total_defect_pixels / (256 * 256)) * 100
        st.metric("Defect Coverage", f"{defect_percentage:.2f}%")

