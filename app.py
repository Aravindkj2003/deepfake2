from flask import Flask, render_template, request, jsonify
import torch
import librosa
import numpy as np
from pathlib import Path
import os
from werkzeug.utils import secure_filename

from models.gan_model import Discriminator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}

# Create upload folder
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Global model variable
model = None
device = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model(checkpoint_path='checkpoints/best_model.pth'):
    """Load trained discriminator model."""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create discriminator
    model = Discriminator(channels=1, dropout_rate=0.0).to(device)  # No dropout for inference
    model.load_state_dict(checkpoint['discriminator_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Best validation accuracy: {checkpoint['history']['best_val_acc']:.4f}")


def audio_to_spectrogram(audio_path, sample_rate=16000, n_mels=128, max_len=128):
    """Convert audio file to Mel-spectrogram tensor."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Compute Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        fmax=8000
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [-1, 1]
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    mel_spec_db = mel_spec_db * 2.0 - 1.0
    
    # Pad or truncate to fixed width
    if mel_spec_db.shape[1] < max_len:
        pad_width = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=-1)
    else:
        mel_spec_db = mel_spec_db[:, :max_len]
    
    # Convert to tensor
    spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 128]
    
    return spec_tensor


def predict_audio(audio_path):
    """Predict if audio is real or deepfake."""
    global model, device
    
    if model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Convert audio to spectrogram
    spec = audio_to_spectrogram(audio_path).to(device)
    
    # Predict
    with torch.no_grad():
        pred = model(spec)
        prob = pred.item()
    
    # Interpret result
    is_real = prob > 0.5
    confidence = prob if is_real else (1 - prob)
    
    result = {
        'prediction': 'REAL' if is_real else 'DEEPFAKE',
        'confidence': float(confidence),
        'raw_score': float(prob),
        'is_real': bool(is_real)
    }
    
    return result


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle audio upload and prediction."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_audio(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not initialized'
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Deepfake Detection Web App')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load model
    if Path(args.checkpoint).exists():
        load_model(args.checkpoint)
    else:
        print(f"⚠ Warning: Checkpoint not found at {args.checkpoint}")
        print("Model will need to be loaded manually or train first.")
    
    # Run app
    print(f"\n{'='*60}")
    print(f"Starting Audio Deepfake Detection Web App")
    print(f"{'='*60}")
    print(f"Access at: http://localhost:{args.port}")
    print(f"{'='*60}\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
