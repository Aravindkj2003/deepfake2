# 🎵 Audio Deepfake Detection with GAN

A GAN-based audio deepfake detector with web interface. Built for fast training with anti-overfitting measures.

## 📁 Project Structure

```
deepfake_2/
├── models/
│   ├── gan_model.py          # Generator + Discriminator architecture
│   └── dataset.py             # Data loading and preprocessing
├── templates/
│   └── index.html             # Web interface
├── augment_dataset.py         # Audio augmentation pipeline
├── train.py                   # GAN training script
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" -m pip install -r requirements.txt
```

### 2. Augment Dataset (5 copies per file)

```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\augment_dataset.py --input-dir "C:\Users\Aravind KJ\Desktop\deepfake_2\for-2sec (1)\for-2seconds\training" --output-dir ".\data\for2sec\aug" --copies-per-file 5 --sample-rate 16000 --duration 2.0
```

**Expected time:** 10-15 minutes  
**Output:** 6x data (1 original + 5 augmented per file)

### 3. Train GAN Model

```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\train.py --data-dir ".\data\for2sec\aug" --checkpoint-dir ".\checkpoints" --epochs 50 --batch-size 32 --dropout-rate 0.3 --early-stopping-patience 10
```

**Expected time:** 10-16 hours (GPU recommended)  
**Anti-overfitting measures:**
- Dropout (0.3)
- Batch normalization
- Weight decay (L1/L2 regularization)
- Label smoothing
- Early stopping (patience=10)
- Data augmentation

### 4. Run Web Interface

```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\app.py --checkpoint ".\checkpoints\best_model.pth"
```

Access at: **http://localhost:5000**

## 📊 Dataset Details

**FoR-2sec Dataset:**
- Real vs Fake audio clips
- 2-second duration per clip
- 16kHz sample rate

**Augmentation techniques applied:**
1. 🔊 Pitch shifting (±2.5 semitones)
2. ⏱️ Time stretching (0.9-1.1x speed)
3. 🔇 Background noise (SNR 8-20 dB)
4. 📊 Volume changes (±6 dB)
5. ⏸️ Time shifting (±15% duration)
6. 🎵 Echo/Reverb effects

## 🧠 Model Architecture

### Generator
- Input: Random noise (100-dim latent vector)
- Output: Fake spectrogram (128x128)
- Architecture: 4 transposed conv layers with batch norm

### Discriminator (Used for Predictions)
- Input: Mel-spectrogram (128x128)
- Output: Real/Fake probability [0, 1]
- Architecture: 4 conv layers with dropout (0.3)
- Parameters: ~2.5M

## 📈 Training Strategy

**Key features to prevent overfitting:**
1. ✅ **Dropout layers** (0.3 rate)
2. ✅ **Batch normalization**
3. ✅ **Weight decay** (L2 regularization)
4. ✅ **Label smoothing** (0.9 instead of 1.0)
5. ✅ **Early stopping** (patience=10 epochs)
6. ✅ **Validation monitoring** (20% split)
7. ✅ **Data augmentation** (6x increase)

## 🎯 Usage Examples

### Training with custom parameters

```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\train.py --data-dir ".\data\for2sec\aug" --epochs 100 --batch-size 64 --lr-d 0.0001 --lr-g 0.0001 --dropout-rate 0.4
```

### Testing model checkpoint

```python
import torch
from models.gan_model import Discriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)

model = Discriminator(channels=1, dropout_rate=0.0).to(device)
model.load_state_dict(checkpoint['discriminator_state_dict'])
model.eval()

print(f"Best validation accuracy: {checkpoint['history']['best_val_acc']:.4f}")
```

## 📂 Output Files

After training:
- `checkpoints/best_model.pth` - Best model checkpoint
- `checkpoints/latest_checkpoint.pth` - Latest checkpoint
- `checkpoints/training_history.json` - Loss curves and metrics

## 🔧 Troubleshooting

**Issue:** Out of memory during training  
**Solution:** Reduce `--batch-size` to 16 or 8

**Issue:** Model overfitting  
**Solution:** Increase `--dropout-rate` to 0.4 or 0.5

**Issue:** Training too slow  
**Solution:** Reduce `--epochs` or use GPU

**Issue:** Web app can't find model  
**Solution:** Check checkpoint path: `.\checkpoints\best_model.pth`

## 📝 Presentation Tips

When showing to your teacher:

1. **Architecture diagram:** Draw Generator → Discriminator flow
2. **Anti-overfitting measures:** List all 7 techniques used
3. **Training curves:** Show from `training_history.json`
4. **Live demo:** Upload test audio via web interface
5. **Code walkthrough:** Explain key parts in `gan_model.py` and `train.py`

## ⏱️ Timeline (2 Days)

**Day 1:**
- ✅ Augmentation (15 min)
- ⏳ Training (10-16 hours) - **Run overnight**

**Day 2:**
- ✅ Test model predictions
- ✅ Setup web interface
- ✅ Create presentation
- ✅ Practice demo

## 🎓 Academic Integrity

This project uses a **GAN discriminator** trained through adversarial learning. The discriminator acts as a binary classifier for real vs fake audio - this is a legitimate and research-backed approach for deepfake detection.

**Explanation for teacher:**
> "I implemented a GAN where the discriminator learns to classify real vs fake audio spectrograms through adversarial training with a generator. The discriminator network is then used for predictions in the web application. This approach combines generative modeling principles with classification, providing robust feature learning through the adversarial process."

## 📚 References

- GAN Paper: Goodfellow et al. (2014)
- Audio Deepfake Detection: ASVspoof Challenge
- Mel-spectrograms: librosa documentation

---

**Built with:** PyTorch, Flask, librosa, NumPy  
**Author:** Your Name  
**Date:** March 2026
