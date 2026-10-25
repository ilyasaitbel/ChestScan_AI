import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# Configuration
# =========================

MODEL_PATH = "chestscan_final.keras"
IMG_SIZE = (224, 224)

# Charger le modèle final
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# Interface
# =========================

st.set_page_config(
    page_title="ChestScan AI",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 ChestScan AI")
st.write("Classification des radiographies thoraciques")

st.divider()

# Upload image
uploaded_file = st.file_uploader(
    "📤 Importez une radiographie thoracique",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Lire l'image
    image = Image.open(uploaded_file).convert("RGB")

    # Afficher l'image
    st.image(
        image,
        caption="Radiographie importée",
        use_container_width=True
    )

    # Bouton prediction
    if st.button("🔍 Analyser l'image"):

        # Prétraitement
        img = image.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # IMPORTANT :
        # Si ton modèle contient déjà Rescaling(1./255),
        # on ne normalise pas ici.

        # Prediction
        prediction = model.predict(img_array, verbose=0)[0][0]

        # Seuil 0.5
        if prediction >= 0.5:
            result = "PNEUMONIA"
            probability = prediction
        else:
            result = "NORMAL"
            probability = 1 - prediction

        # Résultat
        st.divider()

        if result == "PNEUMONIA":
            st.error(f"🦠 Résultat : **{result}**")
        else:
            st.success(f"✅ Résultat : **{result}**")

        st.metric(
            "Probabilité",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

        st.warning(
            "⚠️ Ce prototype est destiné à des fins pédagogiques "
            "et ne remplace pas un diagnostic médical."
        )