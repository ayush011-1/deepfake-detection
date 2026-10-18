import os
import sys
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string, send_from_directory

from infer import DeepfakeDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# Cache model instances to avoid reloading weights on every request
detectors = {
    'celebdf': None,
    'ffpp': None
}

def get_detector(model_type):
    model_type = model_type.lower()
    if model_type not in detectors:
        model_type = 'celebdf'
    if detectors[model_type] is None:
        detectors[model_type] = DeepfakeDetector(model_type=model_type)
    return detectors[model_type]

def img_to_base64(img_bgr):
    if img_bgr is None:
        return ""
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepFake Detection with Explainable AI (XAI)</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0d14;
            --bg-secondary: #111726;
            --bg-card: rgba(18, 24, 38, 0.7);
            --border: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(255, 255, 255, 0.18);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --real-color: #10b981;
            --real-glow: rgba(16, 185, 129, 0.25);
            --fake-color: #f43f5e;
            --fake-glow: rgba(244, 63, 94, 0.25);
            --radius-lg: 16px;
            --radius-md: 10px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 15% 20%, rgba(6, 182, 212, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 75%, rgba(59, 130, 246, 0.08) 0%, transparent 40%);
        }

        header {
            padding: 24px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(12px);
            background: rgba(10, 13, 20, 0.8);
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 800;
            font-size: 18px;
            box-shadow: 0 4px 16px rgba(6, 182, 212, 0.35);
        }

        .brand-text h1 {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .brand-text span {
            font-size: 12px;
            color: var(--text-secondary);
        }

        .badge-model {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            padding: 6px 12px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            border-radius: 20px;
            color: var(--text-secondary);
        }

        main {
            flex: 1;
            max-width: 1280px;
            width: 100%;
            margin: 0 auto;
            padding: 32px 24px;
            display: grid;
            grid-template-columns: 1fr 1.25fr;
            gap: 28px;
        }

        @media (max-width: 960px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .glass-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            backdrop-filter: blur(16px);
            padding: 24px;
            transition: all 0.3s ease;
        }

        .glass-card:hover {
            border-color: var(--border-hover);
        }

        .card-header {
            margin-bottom: 20px;
        }

        .card-header h2 {
            font-size: 16px;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-header p {
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        .model-select-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        .model-btn {
            flex: 1;
            padding: 12px 14px;
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
        }

        .model-btn.active {
            background: rgba(6, 182, 212, 0.12);
            border-color: var(--accent-cyan);
            color: var(--text-primary);
        }

        .model-btn small {
            display: block;
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 3px;
            font-weight: 400;
        }

        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.14);
            border-radius: var(--radius-lg);
            padding: 36px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(0, 0, 0, 0.18);
            position: relative;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent-cyan);
            background: rgba(6, 182, 212, 0.05);
        }

        .dropzone svg {
            width: 48px;
            height: 48px;
            stroke: var(--text-secondary);
            margin-bottom: 12px;
            transition: stroke 0.2s;
        }

        .dropzone:hover svg {
            stroke: var(--accent-cyan);
        }

        .dropzone p {
            font-size: 14px;
            color: var(--text-primary);
            font-weight: 600;
        }

        .dropzone span {
            font-size: 12px;
            color: var(--text-muted);
            display: block;
            margin-top: 6px;
        }

        input[type="file"] {
            display: none;
        }

        .sample-section {
            margin-top: 24px;
        }

        .sample-label {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            margin-bottom: 10px;
            display: block;
        }

        .sample-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
        }

        .sample-item {
            position: relative;
            aspect-ratio: 1;
            border-radius: var(--radius-md);
            overflow: hidden;
            border: 1px solid var(--border);
            cursor: pointer;
            transition: transform 0.2s, border-color 0.2s;
            background: #000;
        }

        .sample-item:hover {
            transform: translateY(-2px);
            border-color: var(--accent-cyan);
        }

        .sample-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .sample-tag {
            position: absolute;
            bottom: 4px;
            left: 4px;
            right: 4px;
            font-size: 10px;
            font-weight: 700;
            text-align: center;
            padding: 2px 4px;
            border-radius: 4px;
            backdrop-filter: blur(6px);
        }

        .sample-tag.real {
            background: rgba(16, 185, 129, 0.85);
            color: #fff;
        }

        .sample-tag.fake {
            background: rgba(244, 63, 94, 0.85);
            color: #fff;
        }

        .btn-analyze {
            width: 100%;
            margin-top: 20px;
            padding: 14px 20px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            color: white;
            font-size: 15px;
            font-weight: 700;
            border: none;
            border-radius: var(--radius-md);
            cursor: pointer;
            box-shadow: 0 4px 18px rgba(6, 182, 212, 0.35);
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn-analyze:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .btn-analyze:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* Results Panel */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            min-height: 420px;
            text-align: center;
            color: var(--text-muted);
        }

        .empty-state svg {
            width: 64px;
            height: 64px;
            stroke: rgba(255, 255, 255, 0.12);
            margin-bottom: 16px;
        }

        .result-container {
            display: none;
            animation: fadeIn 0.4s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .verdict-banner {
            padding: 20px;
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        }

        .verdict-banner.real {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
            border: 1px solid rgba(16, 185, 129, 0.35);
            box-shadow: 0 4px 20px var(--real-glow);
        }

        .verdict-banner.fake {
            background: linear-gradient(135deg, rgba(244, 63, 94, 0.15), rgba(244, 63, 94, 0.05));
            border: 1px solid rgba(244, 63, 94, 0.35);
            box-shadow: 0 4px 20px var(--fake-glow);
        }

        .verdict-title {
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .verdict-banner.real .verdict-title { color: var(--real-color); }
        .verdict-banner.fake .verdict-title { color: var(--fake-color); }

        .verdict-sub {
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 2px;
        }

        .confidence-circle {
            text-align: right;
        }

        .confidence-val {
            font-size: 24px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
        }

        .confidence-txt {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }

        .image-box {
            background: rgba(0, 0, 0, 0.3);
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
            overflow: hidden;
            text-align: center;
        }

        .image-box-header {
            padding: 8px 12px;
            font-size: 12px;
            font-weight: 700;
            color: var(--text-secondary);
            background: rgba(255, 255, 255, 0.03);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .image-box img {
            width: 100%;
            height: 220px;
            object-fit: cover;
            display: block;
        }

        .explanation-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 16px;
        }

        .explanation-card h3 {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .explanation-card p {
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        .metrics-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 16px;
        }

        .metric-card {
            padding: 12px;
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            text-align: center;
        }

        .metric-card span {
            font-size: 11px;
            color: var(--text-muted);
            display: block;
            margin-bottom: 4px;
        }

        .metric-card strong {
            font-size: 15px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
        }

        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="brand-icon">X</div>
            <div class="brand-text">
                <h1>DeepFake Detection</h1>
                <span>Explainable AI Engine (XceptionNet + Grad-CAM)</span>
            </div>
        </div>
        <div class="badge-model" id="activeModelBadge">Model: Celeb-DF v2</div>
    </header>

    <main>
        <!-- Left Panel: Input & Settings -->
        <section class="glass-card">
            <div class="card-header">
                <h2>Input & Configuration</h2>
                <p>Upload a facial photo or choose a benchmark sample</p>
            </div>

            <label class="sample-label">Select Pretrained Weights</label>
            <div class="model-select-group">
                <button type="button" class="model-btn active" id="btnModelCeleb" onclick="selectModel('celebdf')">
                    Celeb-DF v2
                    <small>High precision on celebrity synthesis</small>
                </button>
                <button type="button" class="model-btn" id="btnModelFFPP" onclick="selectModel('ffpp')">
                    FaceForensics++
                    <small>FaceSwap, Face2Face, NeuralTextures</small>
                </button>
            </div>

            <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p id="dropzoneText">Drag & drop facial image here or click to browse</p>
                <span>Supports JPG, PNG, WEBP</span>
            </div>
            <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">

            <div class="sample-section">
                <span class="sample-label">Instant Test Samples</span>
                <div class="sample-grid">
                    <div class="sample-item" onclick="loadSample('real_sample_1.jpg', 'celebdf')">
                        <img src="/samples/real_sample_1.jpg" alt="Real 1">
                        <span class="sample-tag real">Real #1</span>
                    </div>
                    <div class="sample-item" onclick="loadSample('real_sample_2.jpg', 'celebdf')">
                        <img src="/samples/real_sample_2.jpg" alt="Real 2">
                        <span class="sample-tag real">Real #2</span>
                    </div>
                    <div class="sample-item" onclick="loadSample('fake_sample_1.jpg', 'ffpp')">
                        <img src="/samples/fake_sample_1.jpg" alt="Fake 1">
                        <span class="sample-tag fake">Fake #1</span>
                    </div>
                    <div class="sample-item" onclick="loadSample('fake_sample_2.jpg', 'ffpp')">
                        <img src="/samples/fake_sample_2.jpg" alt="Fake 2">
                        <span class="sample-tag fake">Fake #2</span>
                    </div>
                </div>
            </div>

            <button type="button" class="btn-analyze" id="btnAnalyze" onclick="runAnalysis()">
                <span id="btnText">Run Deepfake Analysis</span>
            </button>
        </section>

        <!-- Right Panel: Results & XAI Visualizer -->
        <section class="glass-card">
            <div class="card-header">
                <h2>Analysis & Interpretability</h2>
                <p>Evaluation metrics and Grad-CAM neural attention heatmaps</p>
            </div>

            <div class="empty-state" id="emptyState">
                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <p>Upload an image or click a sample to see AI prediction and Grad-CAM interpretability.</p>
            </div>

            <div class="result-container" id="resultContainer">
                <!-- Verdict Banner -->
                <div class="verdict-banner" id="verdictBanner">
                    <div>
                        <div class="verdict-title" id="verdictText">GENUINE REAL</div>
                        <div class="verdict-sub" id="verdictSub">No synthetic facial manipulation detected</div>
                    </div>
                    <div class="confidence-circle">
                        <div class="confidence-val" id="confidenceVal">99.8%</div>
                        <div class="confidence-txt">Confidence</div>
                    </div>
                </div>

                <!-- Side-by-side Images: Detected Face vs Grad-CAM -->
                <div class="comparison-grid">
                    <div class="image-box">
                        <div class="image-box-header">
                            <span>Detected Face</span>
                            <small id="faceCountBadge">1 face</small>
                        </div>
                        <img id="faceImg" src="" alt="Face crop">
                    </div>
                    <div class="image-box">
                        <div class="image-box-header">
                            <span>Grad-CAM Attention Map</span>
                            <small style="color: var(--accent-cyan)">Layer: Block 7</small>
                        </div>
                        <img id="gradcamImg" src="" alt="GradCAM">
                    </div>
                </div>

                <!-- Explanation & XAI Insight -->
                <div class="explanation-card">
                    <h3>
                        <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Explainable AI (XAI) Insight
                    </h3>
                    <p id="explanationText"></p>

                    <div class="metrics-row">
                        <div class="metric-card">
                            <span>Raw Logit</span>
                            <strong id="metricLogit">-12.84</strong>
                        </div>
                        <div class="metric-card">
                            <span>P(Manipulation)</span>
                            <strong id="metricProb">0.0001</strong>
                        </div>
                        <div class="metric-card">
                            <span>Architecture</span>
                            <strong>XceptionNet</strong>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <script>
        let currentModel = 'celebdf';
        let currentFile = null;
        let selectedSampleName = null;

        function selectModel(modelName) {
            currentModel = modelName;
            document.getElementById('btnModelCeleb').classList.toggle('active', modelName === 'celebdf');
            document.getElementById('btnModelFFPP').classList.toggle('active', modelName === 'ffpp');
            document.getElementById('activeModelBadge').textContent = 
                modelName === 'celebdf' ? 'Model: Celeb-DF v2' : 'Model: FaceForensics++';
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                currentFile = file;
                selectedSampleName = null;
                document.getElementById('dropzoneText').textContent = file.name;
            }
        }

        // Setup drag & drop
        const dropzone = document.getElementById('dropzone');
        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        dropzone.addEventListener('dragleave', () => { dropzone.classList.remove('dragover'); });
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                currentFile = e.dataTransfer.files[0];
                selectedSampleName = null;
                document.getElementById('dropzoneText').textContent = currentFile.name;
            }
        });

        function loadSample(sampleName, recommendedModel) {
            selectedSampleName = sampleName;
            currentFile = null;
            selectModel(recommendedModel);
            document.getElementById('dropzoneText').textContent = 'Selected: ' + sampleName;
            runAnalysis();
        }

        async function runAnalysis() {
            if (!currentFile && !selectedSampleName) {
                alert('Please upload an image or click a sample first!');
                return;
            }

            const btn = document.getElementById('btnAnalyze');
            const btnText = document.getElementById('btnText');
            btn.disabled = true;
            btnText.innerHTML = '<span class="spinner"></span> Analyzing Image with XceptionNet...';

            const formData = new FormData();
            formData.append('model', currentModel);
            if (currentFile) {
                formData.append('file', currentFile);
            } else if (selectedSampleName) {
                formData.append('sample', selectedSampleName);
            }

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    displayResult(data);
                }
            } catch (err) {
                alert('Inference error: ' + err.message);
            } finally {
                btn.disabled = false;
                btnText.textContent = 'Run Deepfake Analysis';
            }
        }

        function displayResult(data) {
            document.getElementById('emptyState').style.display = 'none';
            const resultBox = document.getElementById('resultContainer');
            resultBox.style.display = 'block';

            const banner = document.getElementById('verdictBanner');
            const isFake = data.label === 'FAKE';

            banner.className = 'verdict-banner ' + (isFake ? 'fake' : 'real');
            document.getElementById('verdictText').textContent = isFake ? 'MANIPULATED / FAKE' : 'AUTHENTIC / REAL';
            document.getElementById('verdictSub').textContent = isFake 
                ? 'High-confidence facial manipulation or synthesis artifacts detected' 
                : 'Natural facial features and boundary consistency verified';

            document.getElementById('confidenceVal').textContent = (data.confidence * 100).toFixed(1) + '%';
            document.getElementById('metricLogit').textContent = data.raw_logit.toFixed(2);
            document.getElementById('metricProb').textContent = data.prob_fake.toFixed(4);

            document.getElementById('faceImg').src = data.face_image_base64;
            document.getElementById('gradcamImg').src = data.gradcam_image_base64;
            document.getElementById('faceCountBadge').textContent = data.faces_detected + ' face(s) detected';

            document.getElementById('explanationText').textContent = data.explanation;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/samples/<path:filename>')
def serve_sample(filename):
    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
    return send_from_directory(sample_dir, filename)

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    model_name = request.form.get('model', 'celebdf').lower()
    sample_name = request.form.get('sample', '')
    
    detector = get_detector(model_name)
    
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Failed to decode image file'}), 400
    elif sample_name:
        sample_path = os.path.join(os.path.dirname(__file__), 'sample_data', sample_name)
        if not os.path.exists(sample_path):
            return jsonify({'error': f'Sample {sample_name} not found'}), 404
        img = cv2.imread(sample_path)
    else:
        return jsonify({'error': 'No file or sample provided'}), 400

    try:
        res = detector.predict_image(img, return_gradcam=True)
        primary = res['primary']

        label = primary['label']
        conf = primary['confidence']
        logit = primary['raw_logit']
        prob = primary['prob_fake']

        if label == 'FAKE':
            explanation = (
                f"The XceptionNet model identified strong generative manipulation cues "
                f"(P(Fake)={prob:.4f}, confidence={conf*100:.1f}%). "
                f"The Grad-CAM heatmap highlights concentrated attention (red/warm zones) "
                f"around boundary seams, blending regions, or facial features (mouth/eyes) "
                f"characteristic of deepfake synthesis artifacts."
            )
        else:
            explanation = (
                f"The model detected authentic biometric features with no significant manipulation artifacts "
                f"(P(Real)={(1.0-prob):.4f}, confidence={conf*100:.1f}%). "
                f"The Grad-CAM activation map displays uniform, natural gradient response across the facial structure."
            )

        return jsonify({
            'label': label,
            'confidence': conf,
            'prob_fake': prob,
            'raw_logit': logit,
            'model': model_name,
            'faces_detected': res['faces_count'],
            'face_image_base64': img_to_base64(primary['face_crop']),
            'gradcam_image_base64': img_to_base64(primary['gradcam_overlay']),
            'explanation': explanation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting Deepfake Detection Dashboard on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)
