import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, send_file
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load your saved generator model.
# Update the path below to point to your saved generator model.
GENERATOR_MODEL_PATH = 'model/generator_model_100.h5'
generator = tf.keras.models.load_model(GENERATOR_MODEL_PATH)

# Make sure the noise dimension matches what was used in training.
NOISE_DIM = 100

def generate_image():
    # Generate random noise and pass it through the generator.
    noise = np.random.normal(0, 1, (1, NOISE_DIM))
    generated = generator.predict(noise)
    # Rescale generated image values from [-1,1] to [0,1]
    generated = (generated + 1) / 2.0  
    # Remove the batch dimension.
    generated = np.squeeze(generated, axis=0)
    # Convert the numpy array to an image (assumes generated shape is (H, W, 3)).
    generated = (generated * 255).astype(np.uint8)
    img = Image.fromarray(generated)
    return img

@app.route('/')
def index():
    # Renders an HTML page with a button and an image element.
    return render_template('index.html')

@app.route('/generate', methods=['GET'])
def generate():
    # Generate an image and return it as a PNG.
    img = generate_image()
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    # Run the Flask app (set debug=False in production).
    app.run(debug=True, use_reloader=False, port=5001)


