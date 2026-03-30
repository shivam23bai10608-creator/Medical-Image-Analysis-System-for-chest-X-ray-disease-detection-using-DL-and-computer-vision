# Medical Image Analyzer - Quick Start Guide ⚡

Get the X-ray analysis system running in **5 minutes**!

---

## Ultra-Quick Setup

### Prerequisites (Have These First)

- Python 3.7+
- pip
- ~2GB disk space (for models)
- 4GB+ RAM

### Setup Steps (Copy & Paste)

#### Linux/macOS:

```bash
# 1. Navigate to project
cd medical-imaging-cv

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download models (run once)
python -c "from medical_imaging_analyzer import DiseaseDetectionModel; m = DiseaseDetectionModel(); m.build_model()"

# 5. Run web server
python app.py

# 6. Open browser to http://localhost:5000 ✅
```

#### Windows:

```cmd
REM 1. Navigate to project
cd medical-imaging-cv

REM 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

REM 3. Install dependencies
pip install -r requirements.txt

REM 4. Download models
python -c "from medical_imaging_analyzer import DiseaseDetectionModel; m = DiseaseDetectionModel(); m.build_model()"

REM 5. Run web server
python app.py

REM 6. Open browser to http://localhost:5000 ✅
```

---

## What You Get

✅ Beautiful web interface  
✅ Drag-and-drop image upload  
✅ Real-time disease detection  
✅ Confidence scores  
✅ Risk level assessment  
✅ Batch processing  

---

## First Test

1. **Open**: http://localhost:5000
2. **Upload**: Any chest X-ray image (JPG, PNG, BMP)
3. **Wait**: 1-2 seconds for analysis
4. **See**: Disease prediction + confidence + all probabilities

---

## Example Code (CLI)

```python
from medical_imaging_analyzer import MedicalImageAnalyzer

# Initialize
analyzer = MedicalImageAnalyzer(model_type="mobilenetv2")

# Analyze image
result = analyzer.analyze_image("chest_xray.jpg")

# Print results
print(f"Disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']:.2%}")
print("\nAll Predictions:")
for disease, prob in result['all_predictions'].items():
    print(f"  {disease}: {prob:.2%}")
```

---

## Run Tests

```bash
python -m pytest test_medical_imaging.py -v
```

Should see: **11 tests PASSED** ✅

---

## Troubleshooting (30 seconds)

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow` |
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| `Port 5000 already in use` | `python app.py --port 5001` |
| `CUDA not available` | That's OK! Uses CPU (slower but works) |
| `Out of memory` | Reduce image size or batch size |

---

## Key Features

### Single Image Analysis
- Drag-drop X-ray image
- Get instant diagnosis
- See all 5 disease probabilities
- View confidence score
- Check risk level

### Batch Analysis
- Upload 5-50 images at once
- Process all automatically
- Get summary results
- Export findings

### Models Available
- **MobileNetV2** (Fast - 100ms)
- **ResNet50** (Accurate - 300ms)

---

## Important Notes

⚠️ **Medical Disclaimer**:
- For **educational purposes only**
- Not for clinical diagnosis
- Always consult medical professionals
- Results are not medical advice

---

## System Requirements

| Component | Requirement |
|-----------|------------|
| **RAM** | 4GB minimum, 8GB+ recommended |
| **Disk** | 2GB for models + OS |
| **GPU** | Optional (CPU works fine) |
| **Python** | 3.7+ (3.9+ recommended) |

---

## Performance

| Metric | Value |
|--------|-------|
| Single Image | 0.5-2 seconds |
| Batch (10 images) | 5-10 seconds |
| Memory | 200-400MB |
| Model Size | 50MB |

---

## Next Steps

1. ✅ Get it running (you just did!)
2. 📸 Test with real X-ray images
3. 📖 Read README.md for full docs
4. 📝 Read PROJECT_REPORT.md for methodology
5. 🧪 Run tests to verify installation
6. 🚀 Push to GitHub

---

## File Structure

```
medical-imaging-cv/
├── medical_imaging_analyzer.py  ← Main code (500+ lines)
├── app.py                       ← Web server
├── templates/index.html         ← Web UI
├── test_medical_imaging.py     ← Tests
├── README.md                   ← Full documentation
└── PROJECT_REPORT.md           ← Detailed analysis
```

---

## Support Resources

- **Setup Issues**: See README.md → Installation section
- **Usage Questions**: See README.md → Usage section
- **Technical Details**: See PROJECT_REPORT.md → Methodology
- **Diseases Detected**: See README.md → Disease Classes
- **Performance**: See README.md → Results and Performance

---

## One-Liners for Common Tasks

```bash
# Just run the web app
python app.py

# Run all tests
python -m pytest test_medical_imaging.py -v

# Analyze single image (CLI)
python -c "from medical_imaging_analyzer import *; MedicalImageAnalyzer().analyze_image('x.jpg')"

# Check if all packages installed
python -c "import tensorflow, cv2, flask; print('✓ All good!')"
```

---

**That's it! You're ready to analyze chest X-rays with AI! 🏥✨**
