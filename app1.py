import streamlit as st
import tensorflow as tf
import keras
from keras import layers, models
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import zipfile
import shutil

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
# 📦 Download Dataset from Google Drive
# -----------------------------
@st.cache_resource
def download_dataset_from_drive():
    """Download and extract dataset from Google Drive"""
    if os.path.exists("./dataset") and os.path.exists("./dataset/train") and os.path.exists("./dataset/test"):
        return True
    
    try:
        st.info("📥 Downloading dataset from cloud storage... (This is a one-time setup, ~2-3 minutes)")
        
        # Google Drive direct download link for the dataset
        # You need to upload your dataset.zip to Google Drive and get shareable link
        # Replace YOUR_FILE_ID with actual file ID
        file_id = "dataset"  # ⚠️ REPLACE THIS
        url = f"https://drive.google.com/drive/folders/1cp3G14iRPAxQ09MFg2z6XUIZ44c-_n-B?usp=drive_link"
        output = "dataset.zip"
        
        # Download
        gdown.download(url, output, quiet=False)
        
        # Extract
        st.info("📂 Extracting dataset...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Clean up
        os.remove(output)
        
        st.success("✅ Dataset ready!")
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to download dataset: {str(e)}")
        st.info("""
        **Setup Instructions:**
        1. Upload your dataset.zip to Google Drive
        2. Make it publicly accessible (Anyone with link can view)
        3. Get the file ID from the shareable link
        4. Replace YOUR_FILE_ID_HERE in the code
        
        Or train the model locally and upload the .keras file!
        """)
        return False

# -----------------------------
# ⚙️ Load dataset
# -----------------------------
@st.cache_resource
def load_dataset(data_dir="./dataset"):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        st.error("❌ Dataset folders not found!")
        return None, None, None

    img_size = (224, 224)
    batch_size = 32

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    train_ds = train_ds.cache().shuffle(500).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, test_ds, class_names

# -----------------------------
# 🧠 Build / Load Model
# -----------------------------
@st.cache_resource
def build_model(num_classes):
    base_model = keras.applications.MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

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

st.sidebar.markdown("<h3 style='color: #3b82f6 !important;'>🎯 Select Mode:</h3>", unsafe_allow_html=True)
option = st.sidebar.radio("", ["Train Model", "Predict from Image"], label_visibility="collapsed")

if option == "Train Model":
    st.header("📚 Model Training Section")
    
    # Download dataset if needed
    dataset_ready = download_dataset_from_drive()
    
    if not dataset_ready:
        st.stop()

    train_ds, test_ds, class_names = load_dataset("./dataset")
    
    if train_ds is None:
        st.stop()
    
    # Display class names in a better format
    st.markdown("**📋 Class Names:**")
    cols = st.columns(len(class_names))
    for i, name in enumerate(class_names):
        with cols[i]:
            st.markdown(f"**{i+1}.** `{name}`")
    
    st.info(f"📊 Training with {len(class_names)} classes")

    # Check if model exists
    if os.path.exists("brain_tumor_model.keras"):
        st.success("✅ Found existing trained model!")
        col1, col2 = st.columns(2)
        with col1:
            retrain = st.button("🔄 Retrain Model", use_container_width=True)
        with col2:
            use_existing = st.button("📦 Use Existing Model", use_container_width=True)
        
        if use_existing:
            st.info("Using existing model. Switch to 'Predict from Image' mode to test it!")
            st.stop()
    else:
        retrain = True
        st.warning("No trained model found. Training required.")

    if retrain or st.button("▶️ Start Training"):
        model = build_model(len(class_names))
        
        with st.spinner("Training in progress... ⏳ (This may take a few minutes)"):
            progress_bar = st.progress(0)
            
            epochs = 3
            history = model.fit(
                train_ds, 
                validation_data=test_ds, 
                epochs=epochs,
                verbose=0
            )
            progress_bar.progress(100)
        
        st.success("✅ Training Complete!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Training Accuracy", f"{history.history['accuracy'][-1]:.2%}")
        with col2:
            st.metric("Final Validation Accuracy", f"{history.history['val_accuracy'][-1]:.2%}")

        # Training and Validation Graphs
        st.subheader("📈 Training Performance Graphs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Accuracy Over Epochs**")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(history.history['accuracy'], marker='o', linewidth=2, label='Training Accuracy', color='#3b82f6')
            ax.plot(history.history['val_accuracy'], marker='s', linewidth=2, label='Validation Accuracy', color='#06b6d4')
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            ax.set_title('Model Accuracy', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8fafc')
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Loss Over Epochs**")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(history.history['loss'], marker='o', linewidth=2, label='Training Loss', color='#ef4444')
            ax.plot(history.history['val_loss'], marker='s', linewidth=2, label='Validation Loss', color='#f59e0b')
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax.set_title('Model Loss', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8fafc')
            st.pyplot(fig)

        # Save in new Keras 3 format
        model.save("brain_tumor_model.keras")
        st.success("💾 Model saved as 'brain_tumor_model.keras'")
        st.balloons()

elif option == "Predict from Image":
    st.header("🩻 Brain Tumor Prediction Section")
    
    # Check for both old and new model formats
    model_path = None
    if os.path.exists("brain_tumor_model.keras"):
        model_path = "brain_tumor_model.keras"
    elif os.path.exists("brain_tumor_model.h5"):
        model_path = "brain_tumor_model.h5"
    
    if model_path is None:
        st.error("⚠️ No trained model found. Please train the model first!")
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
        
        class_names = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
        pred_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

        with col2:
            st.subheader("🎯 Prediction Results")
            
            if pred_class == 'no_tumor':
                st.success(f"**Prediction:** {pred_class.upper()}")
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
        <strong style='color: #000000 !important;'>Framework:</strong> TensorFlow & Keras 3.0<br>
        <strong style='color: #000000 !important;'>Technique:</strong> Transfer Learning
    </p>
</div>
""", unsafe_allow_html=True)