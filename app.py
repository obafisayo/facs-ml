"""
Face Access Control — Streamlit App
Stage 1: Covered / Uncovered  (MobileNetV2, my_model.keras)
Stage 2: Authorized / Unauthorized  (DeepFace VGG-Face, authorized_faces/)
"""

import os
import tempfile

import numpy as np
import streamlit as st
import tensorflow as tf
from deepface import DeepFace
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ── Constants ──────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
# Keras flow_from_directory sorts alphabetically: WithMask=0, WithoutMask=1
VISIBILITY_LABELS = {0: "Covered", 1: "Uncovered"}
AUTHORIZED_DIR = "authorized_faces"
os.makedirs(AUTHORIZED_DIR, exist_ok=True)


# ── Model ──────────────────────────────────────────────────────────
@st.cache_resource
def load_visibility_model():
    return tf.keras.models.load_model("my_model.keras")


# ── Core logic ─────────────────────────────────────────────────────
def check_visibility(model, img: Image.Image):
    arr = mobilenet_preprocess(np.array(img.convert("RGB").resize(IMG_SIZE), dtype=np.float32))
    pred = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    idx = int(np.argmax(pred))
    return VISIBILITY_LABELS[idx], float(pred[idx])


def check_authorization(img: Image.Image, authorized_dir: str = AUTHORIZED_DIR):
    refs = [
        f for f in os.listdir(authorized_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not refs:
        return "No authorized faces registered", 0.0

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.convert("RGB").save(tmp.name)
        probe_path = tmp.name

    best_distance = float("inf")
    best_name = None
    best_verified = False
    best_threshold = 0.4

    try:
        for fname in refs:
            ref_path = os.path.join(authorized_dir, fname)
            try:
                result = DeepFace.verify(
                    img1_path=probe_path,
                    img2_path=ref_path,
                    model_name="VGG-Face",
                    enforce_detection=False,
                    silent=True,
                )
                if result["distance"] < best_distance:
                    best_distance = result["distance"]
                    best_name = os.path.splitext(fname)[0]
                    best_verified = result["verified"]
                    best_threshold = result["threshold"]
            except Exception:
                continue
    finally:
        os.unlink(probe_path)

    if best_verified:
        confidence = max(0.0, 1.0 - best_distance / best_threshold)
        return f"Authorized — {best_name}", confidence

    return "Unauthorized", 0.0


# ── UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Face Access Control", layout="wide")
st.title("Face Access Control System")
st.caption("Stage 1: Covered/Uncovered  →  Stage 2: Authorized/Unauthorized")

# Sidebar — manage registered faces
with st.sidebar:
    st.header("Registered Faces")

    uploaded_refs = st.file_uploader(
        "Add authorized individuals (name the file after the person)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="refs",
    )
    if uploaded_refs:
        for f in uploaded_refs:
            dest = os.path.join(AUTHORIZED_DIR, f.name)
            with open(dest, "wb") as out:
                out.write(f.read())
        st.success(f"Saved {len(uploaded_refs)} reference image(s).")

    existing = [
        f for f in os.listdir(AUTHORIZED_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if existing:
        st.markdown("**Currently registered:**")
        for fname in existing:
            col1, col2 = st.columns([4, 1])
            col1.write(f"• {os.path.splitext(fname)[0]}")
            if col2.button("✕", key=f"del_{fname}"):
                os.remove(os.path.join(AUTHORIZED_DIR, fname))
                st.rerun()
    else:
        st.info("No authorized faces registered yet.\nUpload reference photos above.")

    st.divider()
    st.download_button(
        label="Download visibility model",
        data=open("my_model.keras", "rb").read(),
        file_name="my_model.keras",
        mime="application/octet-stream",
    )

# Main — test an image
model = load_visibility_model()

st.subheader("Test an Image")
test_file = st.file_uploader(
    "Upload a face image to run through the pipeline",
    type=["jpg", "jpeg", "png"],
    key="test",
)

if test_file:
    img = Image.open(test_file)
    col_img, col_res = st.columns(2)

    with col_img:
        st.image(img, caption="Input image", use_container_width=True)

    with col_res:
        st.markdown("### Results")

        with st.spinner("Stage 1 — checking visibility…"):
            vis_label, vis_conf = check_visibility(model, img)

        if vis_label == "Covered":
            st.error(f"**Stage 1: Covered** ({vis_conf:.0%} confidence)")
            st.warning("Face is covered — cannot proceed to authorization.")
        else:
            st.success(f"**Stage 1: Uncovered** ({vis_conf:.0%} confidence)")

            with st.spinner("Stage 2 — checking authorization…"):
                auth_label, auth_conf = check_authorization(img)

            if auth_label.startswith("Authorized"):
                st.success(f"**Stage 2: {auth_label}** ({auth_conf:.0%} match)")
            elif auth_label == "No authorized faces registered":
                st.warning(f"**Stage 2:** {auth_label}. Add reference photos in the sidebar.")
            else:
                st.error(f"**Stage 2: {auth_label}**")
