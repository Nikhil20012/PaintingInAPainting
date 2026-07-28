"""Streamlit frontend for hidden painting detection."""

import numpy as np
import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:5000"

st.set_page_config(page_title="Painting in a Painting", page_icon="🎨", layout="wide")

st.title("🎨 Painting in a Painting")
st.markdown("Detect hidden paintings beneath visible layers using deep learning")

uploaded = st.file_uploader("Upload a painting", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Painting")
        st.image(image, use_container_width=True)

    with st.spinner("Analyzing painting..."):
        files = {"image": uploaded.getvalue()}
        try:
            resp = requests.post(f"{API_URL}/predict", files={"image": uploaded})
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure the Flask server is running.")
            st.stop()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    with col2:
        st.subheader("Detection Results")

        # hidden layer status
        if result["hidden_detected"]:
            st.error(
                f"Hidden painting detected (confidence: {result['hidden_confidence']:.1%})"
            )
        else:
            st.success(
                f"No hidden painting detected (confidence: {1 - result['hidden_confidence']:.1%})"
            )

        # classification results
        st.markdown(f"**Style:** {result['style']}")
        st.markdown(f"**Artist:** {result['artist']}")
        st.markdown(f"**Genre:** {result['genre']}")

    # heatmap visualization
    if result["hidden_detected"] and "heatmap" in result:
        st.subheader("Hidden Layer Heatmap")
        st.markdown("Areas where hidden content bleeds through the visible surface")

        heatmap = np.array(result["heatmap"])

        col3, col4 = st.columns(2)
        with col3:
            st.image(image, caption="Original", use_container_width=True)
        with col4:
            fig_size = heatmap.shape[0]
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(np.array(image.resize((fig_size, fig_size))))
            ax.imshow(heatmap, cmap="jet", alpha=0.4)
            ax.axis("off")
            st.pyplot(fig)
            plt.close()

    # narrative
    if "narrative" in result and result["narrative"]:
        st.subheader("Art Historical Context")
        st.markdown(result["narrative"])