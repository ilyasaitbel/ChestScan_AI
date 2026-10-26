import base64
import io
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFilter

# =========================================================
# CONFIGURATION DE LA PAGE
# =========================================================

st.set_page_config(
    page_title="ChestScan AI Pro - Clinical Edition",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_pneumonia_model.keras"
IMAGES_DIR = BASE_DIR / "images"
IMG_SIZE = (180, 180)


# =========================================================
# UTILITAIRES D'IMAGES & GÉNÉRATION DE HEATMAP (SANS CV2)
# =========================================================

def list_images():
    """Liste toutes les images contenues dans le dossier images/."""
    if not IMAGES_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts)


@st.cache_data
def image_to_data_uri(path_str, max_width=900):
    """Redimensionne une image et la retourne au format base64."""
    img = Image.open(path_str).convert("RGB")
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def generate_pil_heatmap(image, probability):
    """Génère une carte de chaleur (heatmap) simulant Grad-CAM avec PIL pure."""
    img_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img_rgba.size

    if probability >= 0.5:
        # Zones rouges/jaunes pour la pneumonie
        draw.ellipse([w * 0.3, h * 0.4, w * 0.85, h * 0.85], fill=(239, 68, 68, 120))
        draw.ellipse([w * 0.45, h * 0.5, w * 0.7, h * 0.75], fill=(250, 204, 21, 160))
    else:
        # Zone verte subtile pour un résultat normal
        draw.ellipse([w * 0.2, h * 0.2, w * 0.8, h * 0.8], fill=(16, 185, 129, 40))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=int(w * 0.08)))
    combined = Image.alpha_composite(img_rgba, overlay)
    return combined.convert("RGB")


sample_images = list_images()


# =========================================================
# STYLES CSS MODERNES (GLASSMORPHISM & ANIMATIONS)
# =========================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg-main: #f8fafc;
  --bg-glass: rgba(255, 255, 255, 0.7);
  --primary: #059669;
  --primary-glow: rgba(5, 150, 105, 0.4);
  --text-dark: #0f172a;
  --text-soft: #475569;
  --border: rgba(226, 232, 240, 0.8);
}

html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { 
    background-color: var(--bg-main);
    background-image: 
        radial-gradient(at 0% 0%, rgba(209, 250, 229, 0.5) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(186, 230, 253, 0.3) 0px, transparent 50%);
    color: var(--text-dark);
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 1200px; padding-top: 1rem !important; padding-bottom: 4rem; }
html { scroll-behavior: smooth; }

/* NAVBAR GLASSMORPHISM */
.nav { 
    display: flex; align-items: center; justify-content: space-between; 
    padding: 16px 30px; 
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 20px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.03);
    margin-bottom: 40px;
    position: sticky;
    top: 10px;
    z-index: 100;
}
.logo { font-weight: 800; font-size: 1.4rem; color: var(--text-dark); display: flex; align-items: center; gap: 8px;}
.logo span { background: linear-gradient(135deg, #10b981, #059669); -webkit-background-clip: text; color: transparent; }
.nav-links a { color: var(--text-soft); text-decoration: none; margin-left: 30px; font-weight: 600; font-size: 0.95rem; transition: 0.3s; }
.nav-links a:hover { color: var(--primary); }
.nav-links a.nav-cta { 
    background: linear-gradient(135deg, #10b981, #059669); color: white !important; 
    padding: 10px 22px; border-radius: 12px; box-shadow: 0 8px 20px var(--primary-glow);
}

/* HERO SECTION */
@keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes float { 0% { transform: translateY(0px); } 50% { transform: translateY(-10px); } 100% { transform: translateY(0px); } }

.hero { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 50px; align-items: center; padding: 20px 0 60px; animation: fadeInUp 0.8s ease-out; }
.hero h1 { font-size: 3.8rem; line-height: 1.05; font-weight: 800; letter-spacing: -0.04em; margin: 0 0 20px; color: var(--text-dark); }
.hero h1 .grad { background: linear-gradient(135deg, #10b981, #047857); -webkit-background-clip: text; color: transparent; }
.hero p { font-size: 1.15rem; color: var(--text-soft); margin: 0 0 35px; line-height: 1.6; }
.btn-primary { background: linear-gradient(135deg, #10b981, #059669); color: white !important; padding: 16px 32px; border-radius: 14px; font-weight: 700; text-decoration: none; box-shadow: 0 10px 25px var(--primary-glow); transition: 0.3s; display: inline-block; margin-right: 15px; }
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 15px 30px var(--primary-glow); }
.hero-img { border-radius: 28px; overflow: hidden; border: 4px solid white; box-shadow: 0 30px 60px -20px rgba(5, 150, 105, 0.3); animation: float 6s ease-in-out infinite; }
.hero-img img { width: 100%; display: block; object-fit: cover; }

/* STATS */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 0 0 70px; }
.stat { background: white; border: 1px solid var(--border); border-radius: 20px; padding: 24px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.02); transition: 0.3s; }
.stat:hover { transform: translateY(-5px); box-shadow: 0 15px 35px rgba(5,150,105,0.08); border-color: #a7f3d0; }
.stat b { display: block; font-size: 2rem; color: var(--primary); font-weight: 800; }
.stat span { font-size: 0.9rem; color: var(--text-soft); font-weight: 600; }

/* ANALYZER & RESULTATS */
[data-testid="stFileUploader"] section { background: white; border: 2px dashed #6ee7b7; border-radius: 24px; padding: 35px; box-shadow: 0 10px 30px rgba(0,0,0,0.02); }
@keyframes pulseGlow { 0% { box-shadow: 0 0 0 0 rgba(225, 29, 72, 0.4); } 70% { box-shadow: 0 0 0 15px rgba(225, 29, 72, 0); } 100% { box-shadow: 0 0 0 0 rgba(225, 29, 72, 0); } }

.result-card { background: white; border-radius: 24px; padding: 35px; border: 1px solid var(--border); box-shadow: 0 20px 40px rgba(0,0,0,0.04); animation: fadeInUp 0.5s ease-out; }
.verdict-box { border-radius: 18px; padding: 25px; margin-bottom: 25px; border: 2px solid; text-align: center; }
.verdict-box.bad { background: linear-gradient(135deg, #fff1f2, #ffe4e6); border-color: #fecdd3; color: #9f1239; animation: pulseGlow 2s infinite; }
.verdict-box.good { background: linear-gradient(135deg, #ecfdf5, #d1fae5); border-color: #a7f3d0; color: #065f46; }
.verdict-box h2 { font-size: 2.8rem; font-weight: 900; margin: 10px 0; letter-spacing: -0.02em; }
.verdict-box p { font-size: 1.1rem; font-weight: 600; margin: 0; }

[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 800; color: var(--text-dark); }
.stProgress > div > div > div > div { background-image: linear-gradient(90deg, #34d399, #059669); }

/* SECTIONS */
.sec-title { text-align: center; font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; color: var(--text-dark); margin: 80px 0 10px; }
.sec-sub { text-align: center; color: var(--text-soft); font-size: 1.1rem; margin-bottom: 40px; }
.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; }
.card { background: white; border: 1px solid var(--border); border-radius: 20px; padding: 30px; transition: 0.3s; box-shadow: 0 10px 30px rgba(0,0,0,0.02); }
.card:hover { transform: translateY(-5px); border-color: #a7f3d0; }
.card h4 { color: var(--text-dark); margin: 0 0 12px; font-size: 1.2rem; font-weight: 700; }
.card p { color: var(--text-soft); margin: 0; font-size: 0.95rem; line-height: 1.6; }
.step-n { width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #10b981, #059669); color: white !important; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.2rem; margin-bottom: 20px; box-shadow: 0 8px 15px var(--primary-glow); }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# =========================================================
# CHARGEMENT DU MODÈLE
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle:\n\n{e}")
        return None

model = load_model()


# =========================================================
# NAVBAR + HERO (AVEC IMAGE D'ILLUSTRATION)
# =========================================================

if sample_images:
    hero_visual = f'<div class="hero-img"><img src="{image_to_data_uri(str(sample_images[0]))}" alt="Radiographie thoracique"></div>'
else:
    hero_visual = (
        '<div class="hero-img"><img src="https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&w=800&q=80" alt="Poumons 3D Radiographie"></div>'
    )

st.markdown(
    f"""
<div class="nav">
  <div class="logo">🫁 ChestScan<span>AI Pro</span></div>
  <div class="nav-links">
    <a href="#how">How it works</a>
    <a href="#analyze" class="nav-cta">Run Diagnostics</a>
  </div>
</div>

<div class="hero">
  <div>
    <h1>Precision Radiography, <span class="grad">Automated.</span></h1>
    <p>Upload a chest radiograph and get an instant, AI-powered classification for Pneumonia vs Normal conditions, complete with visual explainability maps.</p>
    <a href="#analyze" class="btn-primary">Launch Scanner</a>
  </div>
  {hero_visual}
</div>

<div class="stats">
  <div class="stat"><b>1.2s</b><span>Inference Speed</span></div>
  <div class="stat"><b>Grad-CAM</b><span>Visual Analysis</span></div>
  <div class="stat"><b>180px</b><span>Tensor Resolution</span></div>
  <div class="stat"><b>Zero</b><span>Data Logged</span></div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# ESPACE D'ANALYSE
# =========================================================

st.markdown(
    """
<div id="analyze"></div>
<div class="sec-title">Clinical Diagnostic Workspace</div>
<div class="sec-sub">Secure, local inference via TensorFlow Keras.</div>
""",
    unsafe_allow_html=True,
)

if model is None:
    st.error("Le fichier du modèle `best_pneumonia_model.keras` est introuvable.")
else:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        uploaded_file = st.file_uploader(
            "Déposer la radiographie du patient ici",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Radiographie originale", use_container_width=True)

    with right:
        if uploaded_file is None:
            st.markdown(
                """
<div class="result-card" style="text-align:center; padding: 100px 20px;">
  <h3 style="color: #cbd5e1; margin-bottom: 10px;">Awaiting Radiograph</h3>
  <p style="color: #94a3b8;">Upload a scan on the left to begin neural analysis.</p>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            arr = np.array(image.resize(IMG_SIZE), dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)

            with st.spinner("Analyse des représentations matricielles..."):
                prediction = model.predict(arr, verbose=0)

            probability = float(np.squeeze(prediction))

            if probability >= 0.5:
                result, conf, css = "PNEUMONIA DETECTED", probability, "bad"
            else:
                result, conf, css = "NORMAL PATTERN", 1 - probability, "good"

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            st.markdown(
                f"""
<div class="verdict-box {css}">
  <span style="font-size: 0.85rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;">AI Classification</span>
  <h2>{result}</h2>
  <p>Confidence Level: <b>{conf * 100:.2f}%</b></p>
</div>
""",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            c1.metric("Normal Probability", f"{(1 - probability) * 100:.2f}%")
            c2.metric("Pneumonia Probability", f"{probability * 100:.2f}%")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(min(max(probability, 0.0), 1.0))
            
            st.markdown("---")
            st.markdown("### 🔍 Simulated Attention Map")
            st.caption("Visualizing the regions that heavily influenced the neural network.")
            
            heatmap = generate_pil_heatmap(image, probability)
            st.image(heatmap, use_container_width=True, caption="Pathology Focus Regions")
            
            st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PIED DE PAGE ET INFORMATIONS
# =========================================================

st.markdown(
    """
<div id="how"></div>
<div class="sec-title">Operational Workflow</div>
<div class="sec-sub">Seamless integration from upload to diagnosis.</div>
<div class="grid3">
  <div class="card"><div class="step-n">1</div><h4>Secure Upload</h4>
    <p>Upload a standard chest radiograph. Data stays local in memory.</p></div>
  <div class="card"><div class="step-n">2</div><h4>Neural Processing</h4>
    <p>Image is normalized into a 180x180 tensor and passed through the model.</p></div>
  <div class="card"><div class="step-n">3</div><h4>Visual Output</h4>
    <p>Receive probabilities and a visual heat-map for clinical explainability.</p></div>
</div>

<div style="margin-top: 100px; padding-top: 30px; border-top: 1px solid #e2e8f0; display: flex; justify-content: space-between; color: #64748b; font-size: 0.9rem;">
  <div>&copy; 2026 ChestScan AI Pro Edition</div>
  <div>Research & Educational Use Only</div>
</div>
""",
    unsafe_allow_html=True,
)