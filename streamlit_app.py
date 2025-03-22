import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("DCGAN Image Generator")

# Cache the loaded model so it doesn't reload on every interaction.
@st.cache_resource
def load_generator_model():
    # Update the path to where your saved generator model is located.
    model_path = "model/generator_model_final.h5"
    model = tf.keras.models.load_model(model_path)
    return model

# Load the model once.
generator = load_generator_model()

# Set noise dimension (should match training configuration)
NOISE_DIM = 100

# Function to generate an image from random noise.
def generate_image():
    noise = np.random.normal(0, 1, (1, NOISE_DIM))
    generated = generator.predict(noise)
    # Rescale from [-1, 1] to [0, 1]
    generated = (generated + 1) / 2.0
    # Remove the batch dimension and convert to uint8
    generated = np.squeeze(generated, axis=0)
    generated = (generated * 255).astype(np.uint8)
    return Image.fromarray(generated)

# Create a button to trigger image generation.
if st.button("Generate New Image"):
    img = generate_image()
    st.image(img, caption="Generated Image", use_container_width=True)

