# Construction Safety PPE Detection

A simple Streamlit demo app for detecting Personal Protective Equipment (PPE) on construction sites using a custom YOLOv8 model.

## Features

- **Image Upload** – Upload images (JPG, PNG, BMP) and get instant detection results
- **Video Upload** – Upload videos (MP4, AVI, MOV, MKV) with frame-by-frame detection
- **Webcam** – Real-time PPE detection using your laptop camera

## Detected Classes

| Class | Color | Meaning |
|-------|-------|---------|
| person | Cyan | Worker detected |
| helmet | Green | Safety compliant |
| vest | Green | Safety compliant |
| no-helmet | Red | Violation |
| no-vest | Red | Violation |

## Class Weights

Class weights were applied during training to handle class imbalance:

| Class | Weight | Count |
|-------|--------|-------|
| person | 0.91 | 7,275 |
| helmet | 0.86 | 7,714 |
| no-helmet | 0.85 | 7,855 |
| vest | 1.20 | 5,559 |
| no-vest | 1.39 | 4,785 |

Higher weights for `vest` and `no-vest` classes help the model better detect safety vest violations.

## Dataset

Dataset sourced from Roboflow:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="driNxP0m2Aa7pojif1wF")
project = rf.workspace("rohit-yadav-fswnv").project("safety-tucje-ba78y")
dataset = project.version(1).download("yolov8")
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your model

Place your `best.pt` model file in the project root folder:

```
.
├── app.py
├── best.pt              <-- your trained model
├── ppe_train.ipynb      <-- training notebook
├── requirements.txt
└── README.md
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Requirements

- Python 3.8+
- Webcam (for live detection feature)
- `best.pt` model file

## File Structure

```
.
├── app.py              # Main Streamlit application
├── best.pt             # Trained YOLOv8 model
├── ppe_train.ipynb     # Training notebook (Roboflow dataset + class weights)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Usage

1. Launch the app with `streamlit run app.py`
2. Select your input source from the sidebar:
   - **Image Upload** – Drag & drop or browse an image
   - **Video Upload** – Select a video file and click Start
   - **Webcam** – Click Start Webcam to begin live detection
3. Adjust the confidence threshold in the sidebar if needed
4. Click **Stop** to end webcam or video processing

## Training Details

- **Model:** YOLOv8
- **Dataset:** Roboflow construction safety dataset
- **Class weights applied:** Yes (to handle imbalance between vest/no-vest classes)
- **Training notebook:** `ppe_train.ipynb`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model file not found" | Make sure `best.pt` is in the same folder as `app.py` |
| Webcam not working | Check camera permissions; webcam only works locally |
| Slow video processing | Increase "Process every N frames" slider to skip frames |
| App won't start | Ensure all dependencies are installed: `pip install -r requirements.txt` |

## License

This is a demo project for educational purposes.
