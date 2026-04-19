#!/usr/bin/env python3
"""Flask backend for equation solver web interface."""
import os
import sys
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import tempfile
import re

app = Flask(__name__, static_folder='.', static_url_path='')

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'inkml'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load model once on startup
model = None
vocab = None
idx2token = None
device = None

def load_model_once():
    """Load model on first request."""
    global model, vocab, idx2token, device

    if model is not None:
        return True

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model_path = 'models/seq2seq_mathwriting.pt'
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}")
            return False

        from equation_solver.seq2seq_model import load_model
        model, vocab, idx2token = load_model(model_path, device=device)
        model.eval()
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_equation(image_path):
    """Predict equation from image using seq2seq model."""
    from equation_solver.mathwriting_loader import decode_sequence, parse_inkml_strokes, strokes_to_image

    try:
        if image_path.endswith('.inkml'):
            strokes, _ = parse_inkml_strokes(image_path)
            if not strokes:
                return None
            img = strokes_to_image(strokes)
            img_array = np.array(img, dtype=np.float32) / 255.0
        else:
            img = Image.open(image_path).convert('L')
            if img.size != (640, 480):
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0

        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            generated_indices = model.generate(img_tensor, vocab, max_len=300, device=device)

        tokens = decode_sequence(generated_indices[0], idx2token)
        return ''.join(tokens)

    except Exception as e:
        print(f"Error in predict_equation: {e}")
        return None

def parse_and_evaluate(latex_str):
    """Convert LaTeX to expression and evaluate."""
    if not latex_str:
        return None, None

    expr = latex_str.replace('\\times', '*')
    expr = expr.replace('\\cdot', '*')
    expr = expr.replace('\\div', '/')
    expr = expr.replace(' ', '')
    expr = re.sub(r'[^0-9+\-*/.=()]', '', expr)

    if not expr or '=' in expr:
        return expr, None

    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return expr, result
    except:
        return expr, None

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for equation prediction."""
    if not load_model_once():
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict equation
        latex = predict_equation(filepath)

        if not latex:
            return jsonify({'error': 'Failed to recognize equation'}), 400

        # Parse and evaluate
        expr, answer = parse_and_evaluate(latex)

        # Clean up temp file
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'equation': latex,
            'expression': expr,
            'answer': str(answer) if answer is not None else None
        })

    except Exception as e:
        print(f"Error in api_predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """Check if model is loaded and ready."""
    if load_model_once():
        return jsonify({
            'status': 'ready',
            'device': str(device),
            'model': 'seq2seq_mathwriting',
            'vocab_size': len(vocab) if vocab else 0
        })
    else:
        return jsonify({'status': 'error', 'message': 'Model not found'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Equation Solver Web Server")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
