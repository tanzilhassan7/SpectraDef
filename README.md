# SpectraDef — Image Input & Baseline YOLO Detection Pipeline

> **Owner:** Tanzil  
> **Module Scope:** Image Upload, Live Camera Input, OpenCV Preprocessing, YOLOv8 Inference, and Standardized Detection Extraction (`[class_name, confidence, bounding_boxes]`).  
> **Target GitHub Repository:** [https://github.com/tanzilhassan7/SpectraDef](https://github.com/tanzilhassan7/SpectraDef)

---

## 🌟 Overview & Architecture

**SpectraDef** is an end-to-end, standardized object detection pipeline built for real-world computer vision applications. It ingests images from uploaded files or live camera video streams, applies modular OpenCV preprocessing routines, runs ultralytics YOLOv8 baseline inference, and returns clean, structured detection results ready for downstream components in a multi-person project.

```
+------------------+     +-----------------------+     +-------------------+     +-------------------------+
|   Image Input    | --> | OpenCV Preprocessor   | --> | YOLOv8 Inference  | --> | Standardized JSON Result|
| (Upload / Webcam)|     | (Blur, CLAHE, Color)  |     | (Bounding Boxes)  |     | (Class, Conf, Pixels)   |
+------------------+     +-----------------------+     +-------------------+     +-------------------------+
```

---

## ✨ Features

- 📷 **Multiple Input Sources:** Supports single image files (JPG, PNG, WEBP), raw numpy matrices, base64 strings, and live browser camera streaming.
- ⚙️ **OpenCV Preprocessing Engine:** Customizable pipeline including Gaussian Blur, CLAHE Adaptive Histogram Equalization, Brightness/Contrast adjustment, Sharpening, and Color Space Conversions (RGB, BGR, HSV, Grayscale).
- 🎯 **YOLOv8 Detection:** Ultralytics YOLOv8 inference wrapper (`yolov8n.pt` lightweight default model) with configurable confidence threshold and NMS IoU settings.
- 📐 **Standardized Output Schema:** Provides class IDs, class names, confidence scores, pixel bounding box coordinates `[x1, y1, x2, y2]`, normalized coordinates `[x_center, y_center, w, h]`, execution timing breakdown (latency in ms), and base64 annotated visualizations.
- 🌐 **Modern Web Dashboard UI:** Sleek glassmorphism dark-mode UI with drag-and-drop file upload, live webcam stream with real-time FPS/latency gauges, interactive preprocessing studio sliders, and a 1-click JSON inspector.
- 🚀 **FastAPI Backend:** Fully asynchronous REST API endpoints ready for integration.

---

## 📂 Project Structure

```
yolo-pipeline/
├── app/
│   ├── main.py                # FastAPI web server & API endpoints
│   └── static/
│       ├── index.html         # Interactive Glassmorphism Dashboard UI
│       ├── css/style.css      # Dark mode styling & animations
│       └── js/app.js          # Web camera streaming & API client
├── src/
│   ├── __init__.py            # Package initialization
│   ├── schemas.py             # Pydantic standardized output definitions
│   ├── preprocessor.py        # OpenCV image preprocessing engine
│   ├── yolo_detector.py       # Ultralytics YOLOv8 inference engine
│   └── pipeline.py            # Master YOLOPipeline wrapper
├── test_cli.py                # CLI test script for terminal testing
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore configuration
└── README.md                  # Complete project documentation
```

---

## 🚀 Quick Start Guide (Step-by-Step for Beginners)

### 1. Prerequisite: Python Installation
Ensure you have Python 3.9 or higher installed on your system.

### 2. Set Up Virtual Environment & Install Dependencies
Open your terminal (PowerShell or Command Prompt in Antigravity) inside the project folder:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / Mac:
source venv/bin/activate

# Upgrade pip and install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 How to Run & Test

### Option A: Interactive Web Dashboard & API Server
Run the FastAPI development server:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open your web browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

You will see:
1. **Image Upload Tab:** Drag & drop image files to view annotated YOLO outputs & metrics.
2. **Live Camera Tab:** Activate your webcam for real-time bounding box detection & FPS counter.
3. **OpenCV Preprocessor:** Tune Gaussian blur, CLAHE contrast, and sharpening sliders live.
4. **Standard Output Schema:** View and copy the clean JSON output payload.

---

### Option B: Terminal CLI Testing
Test the pipeline directly from your command line without starting the web server:

```bash
# Run with a synthetic test image
python test_cli.py

# Run on a custom image file with custom confidence threshold
python test_cli.py --image path/to/your/image.jpg --conf 0.35 --clahe
```

---

## 🐍 How to Use Tanzil's Pipeline in Code

Other team members can easily import and use Tanzil's pipeline in their scripts:

```python
from src.pipeline import YOLOPipeline
from src.schemas import PreprocessConfig

# 1. Instantiate the Pipeline
pipeline = YOLOPipeline(model_name="yolov8n.pt", conf_threshold=0.25)

# 2. Set OpenCV Preprocessing Config (Optional)
config = PreprocessConfig(
    clahe_contrast=True,
    gaussian_blur=3,
    sharpen=True
)

# 3. Process Image (file path, raw numpy array, or bytes)
result = pipeline.process(image_input="test.jpg", config=config)

# 4. Access Clean Standardized Results
print(f"Total objects detected: {result.total_detections}")
for detection in result.detections:
    print(f" -> {detection.class_name}: {detection.confidence * 100:.1f}% confidence")
    print(f"    Bounding Box (Pixels): [{detection.bbox.x1}, {detection.bbox.y1}, {detection.bbox.x2}, {detection.bbox.y2}]")
    print(f"    Bounding Box (Normalized): [{detection.bbox.x_center_norm}, {detection.bbox.y_center_norm}, {detection.bbox.width_norm}, {detection.bbox.height_norm}]")

# 5. Extract Summary JSON payload for API response
summary_json = result.to_summary()
```

---

## 📄 Standardized System Output JSON Format

```json
{
  "success": true,
  "total_detections": 2,
  "detections": [
    {
      "id": 1,
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.8942,
      "bbox": {
        "x1": 120.5,
        "y1": 45.0,
        "x2": 310.2,
        "y2": 450.8,
        "width": 189.7,
        "height": 405.8,
        "x_center_norm": 0.3365,
        "y_center_norm": 0.5165,
        "width_norm": 0.2964,
        "height_norm": 0.8454
      }
    }
  ],
  "metadata": {
    "width": 640,
    "height": 480,
    "channels": 3
  },
  "preprocess_time_ms": 3.42,
  "inference_time_ms": 14.85,
  "total_time_ms": 18.27
}
```

---

## 🐙 Step-by-Step Guide to Push Code to GitHub

Follow these exact steps in your terminal to publish your repository to GitHub:

### Step 1: Initialize Git
```bash
git init
```

### Step 2: Set Main Branch
```bash
git branch -M main
```

### Step 3: Add Files to Git Stage
```bash
git add .
```

### Step 4: Create Initial Commit
```bash
git commit -m "Add Tanzil's Image Input and Baseline YOLO Detection Pipeline"
```

### Step 5: Link Local Repo to GitHub Remote
```bash
git remote add origin https://github.com/tanzilhassan7/SpectraDef.git
```

### Step 6: Push to GitHub
```bash
git push -u origin main
```

---

## 🛠️ Tech Stack

- **Computer Vision:** OpenCV (`opencv-python`), Ultralytics YOLOv8 (`ultralytics`), PyTorch
- **Backend API:** FastAPI, Uvicorn, Pydantic, NumPy
- **Frontend Dashboard:** HTML5, Modern CSS3 (Glassmorphism), Vanilla JavaScript, FontAwesome
