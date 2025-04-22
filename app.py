import streamlit as st
import numpy as np
import tensorflow as tf
from skimage import color, io, transform
from PIL import Image
import io as io_lib
import os

# Load pts_in_hull (313 ab cluster centers)
pts_in_hull = np.load("pts_in_hull.npy")  # shape (313, 2)

# Helper: Quantized prediction -> ab channels
def bin2ab(prob_map, T=0.48):
    prob_T = np.exp(np.log(prob_map + 1e-8) / T)
    prob_T /= np.sum(prob_T, axis=-1, keepdims=True)
    ab = np.dot(prob_T, pts_in_hull)  # shape: (H, W, 2)
    return ab

def ab2bin(ab_image):
    ab_flat = ab_image.reshape(-1, 2)
    dists = np.linalg.norm(ab_flat[:, None, :] - pts_in_hull[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    return labels.reshape(ab_image.shape[:2])

# Preprocess uploaded grayscale image
def preprocess_grayscale_image(uploaded_file):
    TARGET_SIZE = 224
    rgb_image = io.imread(uploaded_file)
    rgb_image = transform.resize(rgb_image, (TARGET_SIZE, TARGET_SIZE), anti_aliasing=True)
    lab = color.rgb2lab(rgb_image)

    l = lab[:, :, 0:1] / 100.0
    ab = lab[:, :, 1:3]
    ab_labels = ab2bin(ab)  # quantized label per pixel

    return l.astype(np.float32), ab_labels.astype(np.int32), rgb_image

# Post-process prediction
def lab_to_rgb(l_orig, ab_pred):
    lab_image = np.concatenate([l_orig[..., np.newaxis], ab_pred], axis=-1)
    rgb_image = color.lab2rgb(lab_image.astype(np.float32))
    return rgb_image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model2.keras", compile=False)


model = load_model()

# Streamlit UI
st.title("🎨 Deep Colorization App")
st.write("Upload a grayscale image and get it colorized using a deep learning model!")

uploaded_file = st.file_uploader("Upload a grayscale image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    l, _, orig_pil = preprocess_grayscale_image(uploaded_file)
    
    st.image(orig_pil, caption="Uploaded Grayscale Image", use_column_width=True)

    with st.spinner("Colorizing..."):
        l_input = np.expand_dims(l, 0)
        pred = model.predict(l_input)[0]  # shape (H, W, 313)
        ab = bin2ab(pred)
        lab_image = np.concatenate([l * 100.0, ab * 128.0], axis=-1)
        rgb_output = color.lab2rgb(lab_image.astype(np.float32))

    st.image(rgb_output , caption="Colorized Output", use_column_width=True)

