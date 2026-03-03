from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import librosa
import numpy as np
import torch
from werkzeug.utils import secure_filename

from models.gan_model import Discriminator
from models.advanced_models import create_advanced_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'opus'}

Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

model = None
device = None
active_model_name = None


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def build_model(model_name: str):
    if model_name == 'cnn':
        return Discriminator(channels=1, dropout_rate=0.0)
    if model_name == 'resnet18':
        return create_advanced_model('resnet18', pretrained=False)
    raise ValueError(f'Unsupported model: {model_name}')


def extract_state_dict(checkpoint: dict, model_name: str):
    if model_name == 'cnn':
        if 'discriminator_state_dict' in checkpoint:
            return checkpoint['discriminator_state_dict']
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
    else:
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
    raise KeyError('No compatible state dict key found in checkpoint')


def load_model(model_name='resnet18', checkpoint_path=None):
    global model, device, active_model_name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if checkpoint_path is None:
        checkpoint_path = 'checkpoints/resnet18_best.pth' if model_name == 'resnet18' else 'checkpoints/cnn_best_model.pth'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(model_name).to(device)
    state_dict = extract_state_dict(checkpoint, model_name)
    model.load_state_dict(state_dict)
    model.eval()
    active_model_name = model_name

    print(f'Model loaded: {model_name} on {device}')


def audio_to_spectrogram(audio_path, sample_rate=16000, n_mels=128, max_len=128):
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        fmax=8000
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    mel_spec_db = mel_spec_db * 2.0 - 1.0

    if mel_spec_db.shape[1] < max_len:
        pad_width = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=-1)
    else:
        mel_spec_db = mel_spec_db[:, :max_len]

    return torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)


def model_predict(spec):
    with torch.no_grad():
        output = model(spec)

        if active_model_name == 'cnn':
            prob_real = float(output.item())
        else:
            prob_real = float(torch.sigmoid(output).item())

    is_real = prob_real >= 0.5
    confidence = prob_real if is_real else 1.0 - prob_real

    return {
        'prediction': 'REAL' if is_real else 'DEEPFAKE',
        'confidence': confidence,
        'raw_score': prob_real,
        'is_real': is_real,
        'model': active_model_name,
    }


def predict_audio(audio_path):
    if model is None:
        raise RuntimeError('Model not loaded. Call load_model() first.')

    spec = audio_to_spectrogram(audio_path).to(device)
    return model_predict(spec)


def save_uploaded_file(file_storage):
    original_name = secure_filename(file_storage.filename)
    ext = original_name.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file_storage.save(save_path)
    return unique_name, save_path, original_name


def save_file_from_url(audio_url: str):
    parsed = urlparse(audio_url)
    if parsed.scheme not in {'http', 'https'}:
        raise ValueError('Only http/https URLs are allowed')

    file_name = os.path.basename(parsed.path) or f'{uuid.uuid4().hex}.wav'
    file_name = secure_filename(file_name)

    if '.' in file_name and allowed_file(file_name):
        ext = file_name.rsplit('.', 1)[1].lower()
    else:
        ext = 'wav'

    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)

    with urlopen(audio_url, timeout=10) as response:
        content = response.read()

    if len(content) > app.config['MAX_CONTENT_LENGTH']:
        raise ValueError('File from URL exceeds max size limit (16MB)')

    with open(save_path, 'wb') as file_handle:
        file_handle.write(content)

    return unique_name, save_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict', methods=['POST'])
def predict_upload():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        allowed = ', '.join(sorted(app.config['ALLOWED_EXTENSIONS']))
        return jsonify({'error': f'Invalid file type. Allowed: {allowed}'}), 400

    try:
        stored_name, file_path, original_name = save_uploaded_file(file)
        result = predict_audio(file_path)
        result['file_url'] = f"/uploads/{stored_name}"
        result['file_name'] = original_name
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/predict-url', methods=['POST'])
def predict_from_url():
    payload = request.get_json(silent=True) or {}
    audio_url = (payload.get('audioUrl') or '').strip()

    if not audio_url:
        return jsonify({'error': 'audioUrl is required'}), 400

    try:
        stored_name, file_path = save_file_from_url(audio_url)
        result = predict_audio(file_path)
        result['file_url'] = f"/uploads/{stored_name}"
        result['file_name'] = os.path.basename(file_path)
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not initialized',
        'active_model': active_model_name,
    })


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Audio Deepfake Detection Web App')
    parser.add_argument('--model', choices=['cnn', 'resnet18'], default='resnet18', help='Model to use')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    try:
        load_model(model_name=args.model, checkpoint_path=args.checkpoint)
    except Exception as exc:
        print(f'Failed to load model: {exc}')

    print('\n' + '=' * 60)
    print('Starting Audio Deepfake Detection Web App')
    print(f'Model: {args.model}')
    print(f'Access at: http://localhost:{args.port}')
    print('=' * 60 + '\n')

    app.run(host=args.host, port=args.port, debug=args.debug)
