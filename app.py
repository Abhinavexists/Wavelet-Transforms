from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
from transform import WaveletDenoiser
from skimage.util import random_noise
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image, params):
    """Process image with given parameters"""
    # Convert image to RGB mode if it isn't already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert image to numpy array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Add noise if specified
    if params.get('add_noise', False):
        noise_level = float(params.get('noise_level', 0.02))
        img_array = random_noise(img_array, mode='gaussian', var=noise_level)
    
    # Create denoiser
    denoiser = WaveletDenoiser(
        wavelet=params.get('wavelet', 'bior4.4'),
        level=int(params.get('level', 2)) if params.get('level') else None
    )
    
    # Denoise image
    if params.get('color_space') == 'YCbCr':
        denoised = denoiser.denoise_color_image(img_array, params.get('method', 'BayesShrink'))
    else:
        denoised = denoiser.denoise_image(img_array, params.get('method', 'BayesShrink'))
    
    # Ensure the output is in the correct range [0, 255] and uint8 format
    denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    denoised_img = Image.fromarray(denoised, mode='RGB')
    return denoised_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Read image
    image = Image.open(file.stream)
    
    # Get parameters
    params = {
        'wavelet': request.form.get('wavelet', 'bior4.4'),
        'method': request.form.get('method', 'BayesShrink'),
        'level': request.form.get('level'),
        'color_space': request.form.get('color_space', 'RGB'),
        'add_noise': request.form.get('add_noise') == 'true',
        'noise_level': request.form.get('noise_level', '0.02')
    }
    
    # Process image
    denoised_img = process_image(image, params)
    
    # Convert to base64 for sending to frontend
    buffered = io.BytesIO()
    denoised_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({
        'image': img_str
    })

if __name__ == '__main__':
    app.run(debug=True) 