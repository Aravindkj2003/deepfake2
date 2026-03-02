# Quick Command Reference

## Setup Commands

### 1. Install Dependencies
```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" -m pip install -r requirements.txt
```

### 2. Check System Setup
```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\check_setup.py
```

## Data Preparation

### Augment Dataset (5 copies - RECOMMENDED)
```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\augment_dataset.py --input-dir "C:\Users\Aravind KJ\Desktop\deepfake_2\for-2sec (1)\for-2seconds\training" --output-dir ".\data\for2sec\aug" --copies-per-file 5 --sample-rate 16000 --duration 2.0
```

Time: 10-15 minutes

## Training

### Basic Training (Recommended Settings)
```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\train.py --data-dir ".\data\for2sec\aug" --checkpoint-dir ".\checkpoints" --epochs 50 --batch-size 32 --dropout-rate 0.3 --early-stopping-patience 10
```

Time: 10-16 hours (GPU)

### Fast Training (For Testing)
```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\train.py --data-dir ".\data\for2sec\aug" --checkpoint-dir ".\checkpoints" --epochs 10 --batch-size 16 --early-stopping-patience 3
```

Time: 1-2 hours (GPU)

### High Accuracy Training
```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\train.py --data-dir ".\data\for2sec\aug" --checkpoint-dir ".\checkpoints" --epochs 100 --batch-size 64 --dropout-rate 0.4 --lr-d 0.0001 --lr-g 0.0001
```

Time: 20-30 hours (GPU)

## Evaluation

### Evaluate Trained Model
```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\evaluate.py --checkpoint ".\checkpoints\best_model.pth" --data-dir ".\data\for2sec\aug"
```

### Visualize Training Curves
```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\visualize_training.py --history ".\checkpoints\training_history.json" --output "training_plot.png"
```

## Web Application

### Run Web Interface
```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\app.py --checkpoint ".\checkpoints\best_model.pth" --port 5000
```

Access: http://localhost:5000

### Run with Debug Mode
```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\app.py --checkpoint ".\checkpoints\best_model.pth" --debug
```

## Troubleshooting

### Out of Memory Error
```powershell
# Reduce batch size
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\train.py --data-dir ".\data\for2sec\aug" --batch-size 8
```

### Model Overfitting
```powershell
# Increase dropout
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\train.py --data-dir ".\data\for2sec\aug" --dropout-rate 0.5
```

### Training Too Slow (CPU)
```powershell
# Reduce epochs and batch size
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\train.py --data-dir ".\data\for2sec\aug" --epochs 20 --batch-size 8
```

## File Locations

- **Best Model**: `.\checkpoints\best_model.pth`
- **Training History**: `.\checkpoints\training_history.json`
- **Evaluation Results**: `.\checkpoints\evaluation_results.json`
- **Augmented Data**: `.\data\for2sec\aug\`
- **Manifest**: `.\data\for2sec\aug\manifest.csv`

## Quick Tests

### Test Model Architecture
```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\models\gan_model.py
```

### Test Dataset Loading
```powershell
& "C:/Users/Aravind KJ/AppData/Local/Programs/Python/Python312/python.exe" .\models\dataset.py
```

## 2-Day Timeline

**Day 1 Morning:**
- Install dependencies (10 min)
- Run augmentation (15 min)
- Start training (let run overnight)

**Day 1 Evening:**
- Check training progress
- Optional: restart with adjusted parameters

**Day 2 Morning:**
- Stop training / check if converged
- Run evaluation
- Generate plots

**Day 2 Afternoon:**
- Setup web interface
- Test predictions
- Prepare presentation

## Presentation Files to Show

1. `README.md` - Project overview
2. `models/gan_model.py` - Architecture code
3. `train.py` - Training loop
4. `checkpoints/training_history.json` - Metrics
5. `training_plot.png` - Visualizations
6. Web demo - Live predictions

---

**Pro Tip**: Keep this file open while working!
