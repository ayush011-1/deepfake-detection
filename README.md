<p align="center">
  <img src="https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch Logo" width="180"/>
</p>

<h1 align="center">Deepfake Detection with Explainable AI (XAI)</h1>

<p align="center">
  <b>State-of-the-Art Deepfake Detection leveraging XceptionNet, Grad-CAM, and LIME interpretability with an interactive Web Dashboard and CLI.</b>
</p>

---

## 📌 Overview
This repository provides an end-to-end deepfake detection and model interpretability framework. Deep neural networks often act as black boxes; this project integrates **Explainable AI (XAI)** techniques (**Grad-CAM** and **LIME**) alongside **XceptionNet** architectures to pinpoint exactly which facial features and synthesis artifacts drive the model's classifications.

- **Deep Learning Model**: Pretrained and fine-tuned **XceptionNet** architectures (FaceForensics++ and Celeb-DF v2 checkpoints).
- **Explainable AI (XAI)**:
  - **Grad-CAM**: Generates class activation heatmaps on intermediate separable convolution feature maps to highlight forgery boundaries, warping seams, and unnatural textures.
  - **LIME**: Identifies superpixel segments with highest contribution to real vs. fake predictions.
- **Serving & UI**:
  - **Interactive Web App**: Modern Flask web dashboard with drag-and-drop image upload, real-time face detection, and dual-view Grad-CAM analysis.
  - **CLI Pipeline**: Command-line tool supporting single images, image batches, and video analysis.

> 📖 **Comprehensive Project Documentation**: For an exhaustive, in-depth technical analysis covering the mathematical formulation of Depthwise Separable Convolutions, Grad-CAM/LIME derivations, dataset comparison tables, confusion matrices, and real-world considerations, see [PROJECT_DOCUMENTATION.md](file:///Users/ayushsonawane/deepfake-detection/PROJECT_DOCUMENTATION.md).

---

## 🗂️ Datasets
1. **FaceForensics++ (FF++)**: Benchmark dataset for facial manipulation detection covering multiple manipulation techniques (Deepfakes, Face2Face, FaceSwap, NeuralTextures). [Repository](https://github.com/ondyari/FaceForensics)
2. **Celeb-DF (v2)**: Large-scale deepfake dataset of celebrity interview clips under diverse lighting conditions, angles, and facial expressions. [Repository](https://github.com/yuezunli/celeb-deepfakeforensics)

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/ayush011-1/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

### 2. Run the Interactive Web Dashboard
Launch the web interface locally:
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5001
```
From the dashboard:
- Switch between **Celeb-DF** and **FaceForensics++** trained models.
- Upload any facial photo or test with one-click benchmark samples.
- View face bounding box detection, fake probability, confidence score, and side-by-side **Grad-CAM heatmap**.

---

## 🖥️ Command-Line Interface (CLI)

Use `detect.py` to run predictions directly from your terminal:

```bash
# Analyze a single image with Celeb-DF model & generate Grad-CAM
python detect.py --input sample_data/fake_sample_1.jpg --model celebdf --gradcam

# Analyze with FaceForensics++ model
python detect.py --input sample_data/real_sample_1.jpg --model ffpp --gradcam

# Analyze an entire folder of images
python detect.py --input sample_data/ --model celebdf --output-dir runs/detect

# Analyze a video file
python detect.py --input path/to/video.mp4 --model ffpp
```

Detection outputs and Grad-CAM overlays are saved automatically to `runs/detect/`.

### 🔬 Classical ML Baselines (SVM, Logistic Regression, Random Forest)
To benchmark classical machine learning algorithms against deep representations:
```bash
python evaluate_classical_ml_baseline.py --features all
```

---

## 📁 Repository Structure
```
├── app.py                             # Interactive Flask web application & dashboard
├── detect.py                          # Command-line interface for image/video detection
├── infer.py                           # DeepfakeDetector inference engine (XceptionNet + Grad-CAM)
├── evaluate_classical_ml_baseline.py  # Classical ML benchmark (SVM, Logistic Regression, Random Forest)
├── requirements.txt                   # Project dependencies
├── PROJECT_DOCUMENTATION.md           # Comprehensive master documentation & derivations
├── models/
│   ├── CelebDF_model_20_epochs_99acc.pt # Trained XceptionNet weights (Celeb-DF)
│   └── FFPP_model_20_epochs_99acc.pt    # Trained XceptionNet weights (FaceForensics++)
├── sample_data/                       # Benchmark test images (Real & Fake)
├── FaceForensics_XceptionNet_Code.ipynb # Training notebook for FaceForensics++
├── CelebDF_XceptionNet_Code.ipynb     # Training notebook for Celeb-DF
├── grad_cam.ipynb                     # Grad-CAM implementation & experiments
├── lime_code.ipynb                    # LIME explainability experiments
├── plot_curves.ipynb                  # Training loss & accuracy curve plots
├── test_predictions.ipynb             # Batch test set evaluation notebook
├── face_extraction_code_from_videos.py # Face crop preprocessing utility
├── outputs/                           # Sample Grad-CAM & LIME visualizations
└── plots/                             # Training evaluation curves
```

---

## 🔬 Explainable AI (XAI) Methodology
1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**:
   - Computes gradients of the target class score with respect to feature maps in the final separable convolutional block of XceptionNet.
   - Weights feature maps by global average-pooled gradients to highlight discriminative regions (e.g. blend boundaries around the mouth or eyes).
2. **LIME (Local Interpretable Model-agnostic Explanations)**:
   - Perturbs superpixels with Quickshift segmentation to determine local feature contributions to the prediction.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
