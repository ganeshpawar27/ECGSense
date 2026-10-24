import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Page Configuration
st.set_page_config(
    page_title="ECG Arrhythmia Detection",
    page_icon="❤️",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_ecg_model():
    model = tf.keras.models.load_model("ecg_image_model.keras")
    return model

try:
    model = load_ecg_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model load nahi ho paya. Make sure 'ecg_image_model.keras' same folder me hai. Error: {e}")

st.title("ECG Signal Classification System")
st.write("CNN-based deep learning system to detect Normal vs. Abnormal ECG signals.")

# Tabs for Image Upload vs Raw Signal
tab1, tab2 = st.tabs(["Upload ECG Image", "Input Raw Signal (CSV values)"])

# ----------------- TAB 1: Image Upload -----------------
with tab1:
    uploaded_file = st.file_uploader("Upload ECG Trace Image (.png, .jpg)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None and model_loaded:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded ECG Image", width=250)

        # Preprocessing matching notebook (128x128, normalized 1./255)
        img_resized = image.resize((128, 128))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        if st.button("Classify Image"):
            with st.spinner("Analyzing ECG pattern..."):
                prediction = model.predict(img_batch, verbose=0)[0][0]

                # Notebook class indices: {'abnormal': 0, 'normal': 1}
                # Sigmoid output close to 1 -> Normal, close to 0 -> Abnormal
                if prediction >= 0.5:
                    confidence = prediction * 100
                    st.success(f"Result: **Normal ECG** (Confidence: {confidence:.2f}%)")
                else:
                    confidence = (1 - prediction) * 100
                    st.error(f"Result: **Abnormal ECG (Arrhythmia Detected)** (Confidence: {confidence:.2f}%)")

# ----------------- TAB 2: Signal to Image -----------------
with tab2:
    st.write("Comma-separated ECG signal values paste karein:")
    signal_text = st.text_area("Signal Values (e.g. 0.12, -0.4, 0.9, ...)", height=100)

    if st.button("Generate Image & Predict") and model_loaded:
        if signal_text.strip():
            try:
                signal_values = [float(x.strip()) for x in signal_text.split(",") if x.strip()]
                
                # Plot signal exact notebook style (2.5x2.5, dpi=80, black line, no axes)
                fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=80)
                ax.plot(signal_values, color="black", linewidth=1.5)
                ax.axis("off")

                # Save plot to in-memory buffer
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                buf.seek(0)

                # Process buffer as input image
                gen_image = Image.open(buf).convert("RGB")
                st.image(gen_image, caption="Generated Signal Trace", width=200)

                img_resized = gen_image.resize((128, 128))
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_batch, verbose=0)[0][0]

                if prediction >= 0.5:
                    confidence = prediction * 100
                    st.success(f"Result: **Normal ECG** (Confidence: {confidence:.2f}%)")
                else:
                    confidence = (1 - prediction) * 100
                    st.error(f"Result: **Abnormal ECG (Arrhythmia Detected)** (Confidence: {confidence:.2f}%)")
            except Exception as err:
                st.error(f"Signal parse karne me error: {err}")
        else:
            st.warning("Kripya comma-separated numbers enter karein.")