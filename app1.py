import streamlit as st
import tensorflow as tf
import keras
from keras import layers, models
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Brain Tumor Classifier Dashboard", layout="wide")

# Custom CSS for background and styling
st.markdown("""
    <style>
    /* Main background with medical theme */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                    url('https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Card-like containers */
    .stMarkdown, .stMetric, .element-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    /* Title styling */
    h1 {
        color: #1e3a8a !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* Headers */
    h2, h3 {
        color: #1e40af !important;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown strong {
        color: white !important;
    }
    
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    /* Force black text in custom divs */
    [data-testid="stSidebar"] div[style*="rgba(255, 255, 255"] p,
    [data-testid="stSidebar"] div[style*="rgba(255, 255, 255"] li,
    [data-testid="stSidebar"] div[style*="rgba(255, 255, 255"] strong {
        color: #000000 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        border: 2px dashed #3b82f6;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
    }
    
    [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 28px !important;
        font-weight: bold !important;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
    }
    </style>
""", unsafe_allow_html=True)

# Medical Disclaimer with enhanced styling
st.sidebar.markdown("""
<div style='background: rgba(255, 255, 255, 0.95); padding: 15px; border-radius: 10px; border: 2px solid #fbbf24;'>
    <h3 style='color: #1e40af !important; margin: 0;'>⚠️ MEDICAL DISCLAIMER</h3>
    <p style='color: #000000 !important; margin: 10px 0 0 0; font-weight: 600; font-size: 14px;'>
        This tool is for <strong style='color: #000000 !important;'>educational purposes only</strong>.<br><br>
        <strong style='color: #000000 !important;'>NOT for medical diagnosis.</strong><br><br>
        Consult healthcare professionals.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# 🔍 Grad-CAM Function
# -----------------------------
def generate_gradcam(model, img_array, class_index):
    try:
        base_model = model.layers[0]
        last_conv_layer_name = 'out_relu'
        
        grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[base_model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (224, 224))
        
        return heatmap
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM: {str(e)}")
        return None

# -----------------------------
# 🚀 Main App
# -----------------------------
st.title("🧠 Brain Tumor Image Classifier Dashboard")
st.markdown("<p style='font-size: 18px; color: #3b82f6; font-weight: 500;'>AI-powered MRI classification using Transfer Learning with MobileNetV2</p>", unsafe_allow_html=True)

# Prediction Mode Only
st.header("🩻 Brain Tumor Prediction Section")

# Check for model
model_path = None
if os.path.exists("brain_tumor_model.keras"):
    model_path = "brain_tumor_model.keras"
elif os.path.exists("brain_tumor_model.h5"):
    model_path = "brain_tumor_model.h5"

if model_path is None:
    st.error("⚠️ No trained model found. Please ensure 'brain_tumor_model.keras' is in the repository!")
    st.stop()

uploaded_file = st.file_uploader("📤 Upload a brain MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with st.spinner("Loading model..."):
        model = keras.models.load_model(model_path)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Original Image")
        st.image(uploaded_file, use_container_width=True)

    # Preprocess image
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_display = img_array.copy()
    
    # Normalize
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analyzing image..."):
        preds = model.predict(img_array, verbose=0)
    
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    with col2:
        st.subheader("🎯 Prediction Results")
        
        if pred_class == 'notumor':
            st.success(f"**Prediction:** NO TUMOR")
        else:
            st.warning(f"**Prediction:** {pred_class.upper()}")
        
        st.metric("Confidence", f"{confidence:.2f}%")
        
        st.write("**Class Probabilities:**")
        for i, class_name in enumerate(class_names):
            prob = preds[0][i] * 100
            st.progress(int(prob), text=f"{class_name}: {prob:.2f}%")

    # Grad-CAM Heatmap
    st.subheader("🔥 Activation Heatmap (Grad-CAM)")
    st.caption("Shows which regions influenced the model's decision")
    
    heatmap = generate_gradcam(model, img_array, np.argmax(preds))
    
    if heatmap is not None:
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        original = cv2.cvtColor(np.uint8(img_display), cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(heatmap_colored, caption="Heatmap", channels="BGR", use_container_width=True)
        with col2:
            st.image(superimposed, caption="Overlay", channels="BGR", use_container_width=True)
    
    st.warning("⚠️ **Remember:** This is an AI prediction for educational purposes only. Always consult qualified medical professionals for diagnosis.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: rgba(255, 255, 255, 0.95); padding: 15px; border-radius: 10px;'>
    <h3 style='color: #3b82f6 !important;'>📖 About</h3>
    <p style='color: #000000 !important; font-weight: 600; font-size: 14px;'>
        This application uses deep learning to classify brain MRI images into four categories:
    </p>
    <ul style='color: #000000 !important; font-weight: 600; font-size: 14px; list-style-type: none; padding-left: 0;'>
        <li style='color: #000000 !important; margin: 8px 0;'>🔴 Glioma</li>
        <li style='color: #000000 !important; margin: 8px 0;'>🟡 Meningioma</li>
        <li style='color: #000000 !important; margin: 8px 0;'>🟢 Pituitary tumor</li>
        <li style='color: #000000 !important; margin: 8px 0;'>🔵 No tumor</li>
    </ul>
    <p style='color: #000000 !important; margin-top: 15px; font-weight: 600; font-size: 14px; line-height: 1.8;'>
        <strong style='color: #000000 !important;'>Model:</strong> MobileNetV2<br>
        <strong style='color: #000000 !important;'>Framework:</strong> TensorFlow & Keras<br>
        <strong style='color: #000000 !important;'>Technique:</strong> Transfer Learning
    </p>
</div>
""", unsafe_allow_html=True)