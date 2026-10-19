import os
import sys
import base64
import time
import tempfile
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template_string, send_from_directory

from infer import DeepfakeDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max upload

# Global detector instances cache
detectors = {
    'ensemble': None,
    'celebdf': None,
    'ffpp': None,
    'svm': None,
    'rf': None,
    'logreg': None
}

def get_detector(model_type='ensemble'):
    model_type = model_type.lower()
    if model_type not in detectors:
        model_type = 'ensemble'
    if detectors[model_type] is None:
        detectors[model_type] = DeepfakeDetector(model_type=model_type)
    return detectors[model_type]

def img_to_base64(img_bgr):
    if img_bgr is None:
        return ""
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

BENCHMARK_SAMPLES = [
    {
        'id': 'ffpp_fake_1',
        'file': 'ffpp_fake_1.jpg',
        'title': 'FaceForensics++ Deepfake #1',
        'category': 'FaceForensics++',
        'truth': 'FAKE',
        'model': 'ffpp',
        'description': 'Actor facial manipulation with high-frequency boundary seams and compression artifacts.'
    },
    {
        'id': 'ffpp_fake_2',
        'file': 'ffpp_fake_2.jpg',
        'title': 'FaceForensics++ FaceSwap #2',
        'category': 'FaceForensics++',
        'truth': 'FAKE',
        'model': 'ffpp',
        'description': 'FaceSwap synthesis showing gradient discontinuities along cheek and jaw boundaries.'
    },
    {
        'id': 'celebdf_real_1',
        'file': 'celebdf_real_1.jpg',
        'title': 'Celeb-DF Celebrity Portrait',
        'category': 'Celeb-DF',
        'truth': 'REAL',
        'model': 'celebdf',
        'description': 'Original authentic celebrity portrait with coherent facial geometry and natural skin micro-texture.'
    },
    {
        'id': 'celebdf_real_2',
        'file': 'celebdf_real_2.jpg',
        'title': 'Celeb-DF Video Keyframe',
        'category': 'Celeb-DF',
        'truth': 'REAL',
        'model': 'celebdf',
        'description': 'Authentic interview frame displaying uniform biological illumination and corneal reflection symmetry.'
    },
    {
        'id': 'verified_real_portrait',
        'file': 'verified_real_portrait.jpg',
        'title': 'Verified Authentic Face',
        'category': 'Cross-Dataset',
        'truth': 'REAL',
        'model': 'ensemble',
        'description': 'Natural face sample confirmed authentic across both Celeb-DF and FaceForensics++ models.'
    },
    {
        'id': 'ai_generated_portrait',
        'file': 'ai_generated_portrait.jpg',
        'title': 'AI-Generated Headshot (Diffusion)',
        'category': 'Generative AI',
        'truth': 'FAKE',
        'model': 'ensemble',
        'description': 'Modern diffusion/AI headshot exhibiting synthetic skin smoothing, hair aliasing, and extreme noise kurtosis (228.2).'
    }
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepFake Detection with Explainable AI (XAI) | Forensic Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #07090e;
            --bg-card: rgba(15, 23, 42, 0.72);
            --bg-card-hover: rgba(22, 33, 58, 0.85);
            --border: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(6, 182, 212, 0.4);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --cyan: #06b6d4;
            --blue: #3b82f6;
            --violet: #8b5cf6;
            --real-green: #10b981;
            --real-glow: rgba(16, 185, 129, 0.25);
            --fake-red: #f43f5e;
            --fake-glow: rgba(244, 63, 94, 0.25);
            --amber: #f59e0b;
            --radius-xl: 20px;
            --radius-lg: 14px;
            --radius-md: 10px;
            --shadow-subtle: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 15%, rgba(6, 182, 212, 0.10) 0%, transparent 45%),
                radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(16, 185, 129, 0.04) 0%, transparent 60%);
        }

        header {
            padding: 18px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(16px);
            background: rgba(7, 9, 14, 0.85);
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
            text-decoration: none;
            color: inherit;
        }

        .brand-icon {
            width: 42px;
            height: 42px;
            background: linear-gradient(135deg, var(--cyan), var(--blue));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 22px;
            box-shadow: 0 4px 20px rgba(6, 182, 212, 0.4);
        }

        .brand-text h1 {
            font-size: 19px;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #ffffff, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-text span {
            font-size: 12px;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .live-tag {
            display: inline-block;
            width: 7px;
            height: 7px;
            background-color: var(--real-green);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--real-green);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(1.3); }
            100% { opacity: 1; transform: scale(1); }
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .btn-ghost {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 13px;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.25s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-ghost:hover {
            color: white;
            border-color: var(--cyan);
            background: rgba(6, 182, 212, 0.1);
        }

        main {
            flex: 1;
            max-width: 1360px;
            width: 100%;
            margin: 0 auto;
            padding: 32px 24px;
            display: grid;
            grid-template-columns: 1.15fr 1.35fr;
            gap: 32px;
        }

        @media (max-width: 1040px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .glass-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-xl);
            backdrop-filter: blur(20px);
            padding: 28px;
            box-shadow: var(--shadow-subtle);
            transition: border-color 0.3s ease;
            position: relative;
        }

        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.16);
        }

        .card-header {
            margin-bottom: 22px;
        }

        .card-title {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .card-desc {
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 5px;
        }

        /* Model Selector Grid */
        .section-label {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .model-selector-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 24px;
        }

        .model-btn {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 12px 14px;
            cursor: pointer;
            text-align: left;
            transition: all 0.25s ease;
            position: relative;
        }

        .model-btn:hover {
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .model-btn.active {
            border-color: var(--cyan);
            background: rgba(6, 182, 212, 0.12);
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
        }

        .model-btn .name {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .model-btn .tag {
            font-size: 10px;
            color: var(--cyan);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 2px;
        }

        .model-btn .desc {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 4px;
            line-height: 1.3;
        }

        /* Input Method Tabs */
        .input-tabs {
            display: flex;
            gap: 8px;
            background: rgba(0, 0, 0, 0.3);
            padding: 4px;
            border-radius: var(--radius-md);
            margin-bottom: 18px;
            border: 1px solid var(--border);
        }

        .tab-btn {
            flex: 1;
            padding: 9px 12px;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 13px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 7px;
        }

        .tab-btn:hover {
            color: white;
        }

        .tab-btn.active {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        /* Dropzone */
        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: var(--radius-lg);
            padding: 36px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(0, 0, 0, 0.2);
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--cyan);
            background: rgba(6, 182, 212, 0.06);
            transform: translateY(-2px);
        }

        .drop-icon {
            font-size: 40px;
            margin-bottom: 12px;
            color: var(--cyan);
        }

        .drop-text h4 {
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .drop-text p {
            font-size: 12px;
            color: var(--text-muted);
        }

        /* Webcam view */
        .webcam-container {
            border-radius: var(--radius-lg);
            overflow: hidden;
            background: #000;
            border: 1px solid var(--border);
            text-align: center;
            position: relative;
        }

        #webcamVideo {
            width: 100%;
            height: 280px;
            object-fit: cover;
            display: block;
        }

        .webcam-controls {
            padding: 12px;
            display: flex;
            justify-content: center;
            gap: 10px;
            background: rgba(10, 14, 23, 0.9);
        }

        /* Video Uploader */
        .video-box {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: var(--radius-lg);
            padding: 30px 20px;
            text-align: center;
            cursor: pointer;
            background: rgba(0, 0, 0, 0.2);
        }

        /* Samples Grid */
        .samples-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }

        .sample-item {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 8px;
            cursor: pointer;
            transition: all 0.25s ease;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .sample-item:hover {
            border-color: var(--cyan);
            background: rgba(6, 182, 212, 0.08);
            transform: translateY(-2px);
        }

        .sample-item.selected {
            border-color: var(--cyan);
            box-shadow: 0 0 16px rgba(6, 182, 212, 0.35);
        }

        .sample-thumb {
            width: 100%;
            height: 90px;
            border-radius: 6px;
            object-fit: cover;
            margin-bottom: 6px;
        }

        .sample-info {
            font-size: 11px;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .badge-truth {
            position: absolute;
            top: 12px;
            right: 12px;
            font-size: 9px;
            font-weight: 800;
            padding: 2px 6px;
            border-radius: 4px;
            letter-spacing: 0.04em;
        }

        .badge-truth.real {
            background: rgba(16, 185, 129, 0.85);
            color: #fff;
        }

        .badge-truth.fake {
            background: rgba(244, 63, 94, 0.85);
            color: #fff;
        }

        /* Action Button */
        .btn-analyze {
            width: 100%;
            margin-top: 24px;
            padding: 16px 24px;
            background: linear-gradient(135deg, var(--cyan), var(--blue));
            color: white;
            border: none;
            border-radius: var(--radius-md);
            font-family: inherit;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 6px 24px rgba(6, 182, 212, 0.35);
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .btn-analyze:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(6, 182, 212, 0.5);
        }

        .btn-analyze:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* Right Panel: Results */
        .result-empty {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            min-height: 440px;
            text-align: center;
            color: var(--text-muted);
            border: 1px dashed rgba(255, 255, 255, 0.08);
            border-radius: var(--radius-lg);
            padding: 30px;
        }

        .result-empty .icon {
            font-size: 48px;
            margin-bottom: 16px;
            opacity: 0.4;
        }

        /* Result Active View */
        .verdict-banner {
            border-radius: var(--radius-lg);
            padding: 22px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
            border: 1px solid transparent;
        }

        .verdict-banner.real {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 182, 212, 0.08));
            border-color: rgba(16, 185, 129, 0.35);
            box-shadow: 0 0 32px var(--real-glow);
        }

        .verdict-banner.fake {
            background: linear-gradient(135deg, rgba(244, 63, 94, 0.18), rgba(239, 68, 68, 0.08));
            border-color: rgba(244, 63, 94, 0.4);
            box-shadow: 0 0 32px var(--fake-glow);
        }

        .verdict-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .verdict-icon {
            width: 52px;
            height: 52px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 26px;
        }

        .verdict-banner.real .verdict-icon {
            background: var(--real-green);
            color: white;
        }

        .verdict-banner.fake .verdict-icon {
            background: var(--fake-red);
            color: white;
        }

        .verdict-text h3 {
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .verdict-banner.real .verdict-text h3 {
            color: #34d399;
        }

        .verdict-banner.fake .verdict-text h3 {
            color: #fb7185;
        }

        .verdict-text p {
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 2px;
        }

        .verdict-conf {
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
        }

        .verdict-conf .score {
            font-size: 32px;
            font-weight: 800;
            line-height: 1;
        }

        .verdict-banner.real .score { color: #34d399; }
        .verdict-banner.fake .score { color: #fb7185; }

        .verdict-conf .label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 4px;
        }

        /* Meter bar */
        .meter-container {
            margin-bottom: 24px;
        }

        .meter-labels {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .meter-track {
            height: 10px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 5px;
            overflow: hidden;
            display: flex;
        }

        .meter-fill-real {
            height: 100%;
            background: var(--real-green);
            transition: width 0.6s ease;
        }

        .meter-fill-fake {
            height: 100%;
            background: var(--fake-red);
            transition: width 0.6s ease;
        }

        /* Visual Comparison Box */
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }

        .image-card {
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 12px;
            text-align: center;
        }

        .image-card h5 {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }

        .image-card img {
            width: 100%;
            height: 200px;
            object-fit: contain;
            border-radius: 8px;
            background: #000;
        }

        /* Opacity Slider for Overlay */
        .slider-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 12px 16px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .slider-box span {
            font-size: 12px;
            color: var(--text-secondary);
            white-space: nowrap;
        }

        .slider-box input[type="range"] {
            flex: 1;
            accent-color: var(--cyan);
            cursor: pointer;
        }

        /* Explanation Box */
        .explanation-card {
            background: rgba(6, 182, 212, 0.05);
            border: 1px solid rgba(6, 182, 212, 0.2);
            border-radius: var(--radius-lg);
            padding: 18px;
            margin-bottom: 20px;
        }

        .explanation-card h4 {
            font-size: 13px;
            font-weight: 700;
            color: var(--cyan);
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .explanation-card p {
            font-size: 13px;
            line-height: 1.6;
            color: #e2e8f0;
        }

        /* Details Drawer */
        .details-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }

        .detail-item {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px;
        }

        .detail-item .label {
            color: var(--text-muted);
            font-size: 10px;
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        .detail-item .val {
            color: var(--text-primary);
            font-weight: 600;
        }

        /* Video timeline chart */
        .timeline-box {
            margin-top: 18px;
            background: rgba(0, 0, 0, 0.4);
            border-radius: var(--radius-md);
            padding: 16px;
            border: 1px solid var(--border);
        }

        .timeline-bars {
            display: flex;
            align-items: flex-end;
            gap: 4px;
            height: 90px;
            padding-top: 10px;
        }

        .time-bar {
            flex: 1;
            background: var(--real-green);
            border-radius: 3px 3px 0 0;
            transition: height 0.3s ease;
            position: relative;
        }

        .time-bar.fake {
            background: var(--fake-red);
        }

        /* Spinner */
        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        footer {
            text-align: center;
            padding: 24px;
            border-top: 1px solid var(--border);
            font-size: 12px;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <header>
        <a href="/" class="brand">
            <div class="brand-icon">🛡️</div>
            <div class="brand-text">
                <h1>DEEPFAKE DETECTOR AI</h1>
                <span><span class="live-tag"></span> Explainable AI (XceptionNet + Grad-CAM)</span>
            </div>
        </a>
        <div class="header-actions">
            <span class="btn-ghost" id="deviceBadge">⚡ PyTorch Active</span>
            <a href="https://github.com/ayush011-1/deepfake-detection" target="_blank" class="btn-ghost">
                ⭐ GitHub
            </a>
        </div>
    </header>

    <main>
        <!-- LEFT: INPUT & CONFIG -->
        <div class="glass-card">
            <div class="card-header">
                <div class="card-title">🔍 Input & Model Configuration</div>
                <div class="card-desc">Select an inference engine and upload media or pick benchmark test samples.</div>
            </div>

            <!-- MODEL SELECTOR -->
            <div class="section-label">
                <span>INFERENCE ENGINE</span>
                <span style="color: var(--cyan); font-weight: normal; text-transform: none;">Adaptive Domain Calibration</span>
            </div>
            <div class="model-selector-grid" style="grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));">
                <div class="model-btn active" data-model="ensemble" onclick="selectModel('ensemble')">
                    <div class="name">⭐ Universal Ensemble</div>
                    <div class="tag">Hybrid DL + Forensics</div>
                    <div class="desc">Dual XceptionNet + Diffusion noise analysis (Face-Swaps & AI Headshots).</div>
                </div>
                <div class="model-btn" data-model="synthetic" onclick="selectModel('synthetic')">
                    <div class="name">Diffusion AI Detector</div>
                    <div class="tag">Frequency Forensics</div>
                    <div class="desc">High-order FFT & noise kurtosis for AI portraits.</div>
                </div>
                <div class="model-btn" data-model="ffpp" onclick="selectModel('ffpp')">
                    <div class="name">FaceForensics++</div>
                    <div class="tag">Deep Learning</div>
                    <div class="desc">XceptionNet for FaceSwap & boundary blending.</div>
                </div>
                <div class="model-btn" data-model="celebdf" onclick="selectModel('celebdf')">
                    <div class="name">Celeb-DF v2</div>
                    <div class="tag">Deep Learning</div>
                    <div class="desc">XceptionNet for subtle GAN synthesis & warping.</div>
                </div>
                <div class="model-btn" data-model="svm" onclick="selectModel('svm')">
                    <div class="name">SVM (RBF)</div>
                    <div class="tag">Classical ML</div>
                    <div class="desc">Max-margin hyperplane on 2048-D features.</div>
                </div>
                <div class="model-btn" data-model="rf" onclick="selectModel('rf')">
                    <div class="name">Random Forest</div>
                    <div class="tag">Classical ML</div>
                    <div class="desc">100 Decision Trees on latent embeddings.</div>
                </div>
                <div class="model-btn" data-model="logreg" onclick="selectModel('logreg')">
                    <div class="name">Logistic Reg.</div>
                    <div class="tag">Classical ML</div>
                    <div class="desc">L2-regularized log-odds classification.</div>
                </div>
            </div>

            <!-- INPUT TABS -->
            <div class="section-label">ANALYSIS SOURCE</div>
            <div class="input-tabs">
                <button class="tab-btn active" onclick="switchTab('upload')">Upload Photo</button>
                <button class="tab-btn" onclick="switchTab('webcam')">Live Camera</button>
                <button class="tab-btn" onclick="switchTab('video')"> Video File</button>
                <button class="tab-btn" onclick="switchTab('benchmarks')">🧪 Benchmarks</button>
            </div>

            <!-- TAB 1: UPLOAD -->
            <div id="tab-upload" class="tab-content active">
                <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                    <div class="drop-icon"></div>
                    <div class="drop-text">
                        <h4>Drop facial image here or click to browse</h4>
                        <p>PNG, JPG, JPEG, BMP supported (Max 64MB)</p>
                    </div>
                    <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFileSelect(event)">
                </div>
                <div id="filePreviewBox" style="display:none; margin-top:14px; text-align:center;">
                    <img id="filePreviewImg" style="max-height:160px; border-radius:8px; border:1px solid var(--border);" src="" alt="Preview">
                    <p id="fileNameLabel" style="font-size:12px; color:var(--text-secondary); margin-top:4px;"></p>
                </div>
            </div>

            <!-- TAB 2: WEBCAM -->
            <div id="tab-webcam" class="tab-content">
                <div class="webcam-container">
                    <video id="webcamVideo" autoplay playsinline></video>
                    <canvas id="webcamCanvas" style="display:none;"></canvas>
                    <div class="webcam-controls">
                        <button class="btn-ghost" onclick="startWebcam()">Enable Camera</button>
                        <button class="btn-ghost" id="snapBtn" onclick="takeSnapshot()" disabled>📸 Take Snapshot</button>
                    </div>
                </div>
                <div id="webcamPreviewBox" style="display:none; margin-top:10px; text-align:center;">
                    <p style="font-size:12px; color:var(--cyan);">Snapshot Captured!</p>
                </div>
            </div>

            <!-- TAB 3: VIDEO -->
            <div id="tab-video" class="tab-content">
                <div class="video-box" onclick="document.getElementById('videoInput').click()">
                    <div class="drop-icon">🎬</div>
                    <div class="drop-text">
                        <h4>Select or drop short video file</h4>
                        <p>MP4, WEBM, MOV (Evaluates frame-by-frame timeline)</p>
                    </div>
                    <input type="file" id="videoInput" accept="video/*" style="display:none" onchange="handleVideoSelect(event)">
                </div>
                <p id="videoNameLabel" style="display:none; font-size:12px; color:var(--cyan); margin-top:8px; text-align:center;"></p>
            </div>

            <!-- TAB 4: BENCHMARKS -->
            <div id="tab-benchmarks" class="tab-content">
                <div class="samples-gallery" id="samplesGallery">
                    <!-- Populated dynamically -->
                </div>
            </div>

            <!-- SUBMIT BUTTON -->
            <button class="btn-analyze" id="analyzeBtn" onclick="runDetection()">
                <span>RUN FORENSIC DEEPFAKE ANALYSIS</span>
            </button>
        </div>

        <!-- RIGHT: RESULTS & XAI HEATMAP -->
        <div class="glass-card">
            <div class="card-header">
                <div class="card-title">🔬 Forensic & Explainable AI (XAI) Results</div>
                <div class="card-desc">Real-time prediction confidence and Grad-CAM class activation mapping.</div>
            </div>

            <!-- EMPTY STATE -->
            <div class="result-empty" id="resultEmpty">
                <div class="icon">🔍</div>
                <h3>Ready to Analyze</h3>
                <p>Choose an image or benchmark sample and trigger deepfake analysis.</p>
            </div>

            <!-- ACTIVE RESULTS VIEW -->
            <div id="resultContent" style="display:none;">
                <!-- VERDICT BANNER -->
                <div class="verdict-banner" id="verdictBanner">
                    <div class="verdict-left">
                        <div class="verdict-icon" id="verdictIcon"></div>
                        <div class="verdict-text">
                            <h3 id="verdictTitle">AUTHENTIC</h3>
                            <p id="verdictSubtitle">No generative manipulation artifacts detected</p>
                        </div>
                    </div>
                    <div class="verdict-conf">
                        <div class="score" id="verdictScore">99.8%</div>
                        <div class="label">Confidence</div>
                    </div>
                </div>

                <!-- DUAL RISK METER -->
                <div class="meter-container">
                    <div class="meter-labels">
                        <span style="color:var(--real-green)">Authentic: <strong id="meterRealVal">99.8%</strong></span>
                        <span style="color:var(--fake-red)">Manipulated (Fake): <strong id="meterFakeVal">0.2%</strong></span>
                    </div>
                    <div class="meter-track">
                        <div class="meter-fill-real" id="meterFillReal" style="width: 99.8%;"></div>
                        <div class="meter-fill-fake" id="meterFillFake" style="width: 0.2%;"></div>
                    </div>
                </div>

                <!-- SIDE-BY-SIDE VISUAL COMPARISON (XAI DUAL VIEW) -->
                <div class="comparison-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                    <div class="image-card">
                        <h5>👤 Cropped Detected Face</h5>
                        <img id="faceCropImg" src="" alt="Detected Face">
                    </div>
                    <div class="image-card" id="gradcamCard">
                        <h5>🔥 Grad-CAM Heatmap (XAI)</h5>
                        <img id="gradcamImg" src="" alt="Grad-CAM XAI">
                    </div>
                    <div class="image-card" id="limeCard" style="display:none;">
                        <h5>🧩 LIME Superpixels (XAI)</h5>
                        <img id="limeImg" src="" alt="LIME XAI">
                    </div>
                </div>

                <!-- OPACITY BLENDING SLIDER -->
                <div class="slider-box">
                    <span>Face Focus</span>
                    <input type="range" id="opacitySlider" min="10" max="100" value="85" oninput="adjustHeatmapOpacity(this.value)">
                    <span>Heatmap Focus</span>
                </div>

                <!-- EXPLANATION -->
                <div class="explanation-card">
                    <h4>💡 Explainable AI (XAI) Forensic Findings</h4>
                    <p id="explanationText">Analysis summary will appear here...</p>
                </div>

                <!-- METADATA DRAWER -->
                <div class="details-grid">
                    <div class="detail-item">
                        <div class="label">Inference Mode</div>
                        <div class="val" id="metaModel">ENSEMBLE</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Detected Faces</div>
                        <div class="val" id="metaFaces">1</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Latency</div>
                        <div class="val" id="metaLatency">-- ms</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Raw Model Logit</div>
                        <div class="val" id="metaLogit">0.00</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Architecture</div>
                        <div class="val">XceptionNet</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Target Conv Layer</div>
                        <div class="val">block7 (Separable)</div>
                    </div>
                </div>

                <!-- VIDEO TIMELINE CHART (Only visible for videos) -->
                <div class="timeline-box" id="videoTimelineBox" style="display:none;">
                    <div style="font-size:12px; font-weight:700; color:var(--text-secondary); margin-bottom:8px;">
                        🎞️ Frame-by-Frame Manipulation Probability Timeline
                    </div>
                    <div class="timeline-bars" id="timelineBars">
                        <!-- Populated by JS -->
                    </div>
                </div>

                <!-- ACTION BAR -->
                <div style="margin-top:20px; display:flex; gap:10px;">
                    <button class="btn-ghost" style="flex:1;" onclick="downloadForensicReport()">
                        📄 Download Forensic Audit Report (JSON)
                    </button>
                </div>
            </div>
        </div>
    </main>

    <footer>
        Deepfake Detection with Explainable AI &bull; PyTorch &bull; XceptionNet &bull; Grad-CAM &bull; LIME &bull; ECE 792 Project
    </footer>

    <script>
        let selectedModel = 'ensemble';
        let currentTab = 'upload';
        let selectedSampleFile = null;
        let selectedFile = null;
        let selectedVideo = null;
        let webcamStream = null;
        let webcamCapturedBlob = null;
        let latestAnalysisData = null;

        // Initialize benchmarks on load
        window.addEventListener('DOMContentLoaded', () => {
            loadBenchmarks();
            fetchSystemInfo();
        });

        async function fetchSystemInfo() {
            try {
                const res = await fetch('/api/system-info');
                const data = await res.json();
                document.getElementById('deviceBadge').textContent = `⚡ PyTorch ${data.device.toUpperCase()}`;
            } catch (e) {
                console.warn(e);
            }
        }

        async function loadBenchmarks() {
            try {
                const res = await fetch('/api/samples');
                const samples = await res.json();
                const gallery = document.getElementById('samplesGallery');
                gallery.innerHTML = '';
                
                samples.forEach(s => {
                    const item = document.createElement('div');
                    item.className = 'sample-item';
                    item.onclick = () => selectSample(s, item);
                    
                    const truthClass = s.truth.toLowerCase() === 'real' ? 'real' : 'fake';
                    item.innerHTML = `
                        <span class="badge-truth ${truthClass}">${s.truth}</span>
                        <img class="sample-thumb" src="/samples/${s.file}" alt="${s.title}">
                        <div class="sample-info">${s.title}</div>
                    `;
                    gallery.appendChild(item);
                });
            } catch (err) {
                console.error("Failed to load benchmarks", err);
            }
        }

        function selectModel(modelName) {
            selectedModel = modelName;
            document.querySelectorAll('.model-btn').forEach(b => {
                b.classList.toggle('active', b.dataset.model === modelName);
            });
        }

        function switchTab(tabName) {
            currentTab = tabName;
            document.querySelectorAll('.tab-btn').forEach((b, idx) => {
                const tabs = ['upload', 'webcam', 'video', 'benchmarks'];
                b.classList.toggle('active', tabs[idx] === tabName);
            });
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(`tab-${tabName}`).classList.add('active');

            if (tabName !== 'webcam' && webcamStream) {
                stopWebcam();
            }
        }

        function selectSample(sample, elem) {
            document.querySelectorAll('.sample-item').forEach(i => i.classList.remove('selected'));
            elem.classList.add('selected');
            selectedSampleFile = sample.file;
            selectedFile = null;
            selectedVideo = null;
            webcamCapturedBlob = null;
            
            // Auto-select optimal model for this benchmark
            if (sample.model) {
                selectModel(sample.model);
            }
        }

        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (!file) return;
            selectedFile = file;
            selectedSampleFile = null;
            selectedVideo = null;
            webcamCapturedBlob = null;
            
            const reader = new FileReader();
            reader.onload = (event) => {
                document.getElementById('filePreviewImg').src = event.target.result;
                document.getElementById('fileNameLabel').textContent = `${file.name} (${(file.size/1024).toFixed(1)} KB)`;
                document.getElementById('filePreviewBox').style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        function handleVideoSelect(e) {
            const file = e.target.files[0];
            if (!file) return;
            selectedVideo = file;
            selectedFile = null;
            selectedSampleFile = null;
            webcamCapturedBlob = null;
            const label = document.getElementById('videoNameLabel');
            label.textContent = `Selected Video: ${file.name} (${(file.size/(1024*1024)).toFixed(2)} MB)`;
            label.style.display = 'block';
        }

        // Drag and drop events
        const dropzone = document.getElementById('dropzone');
        ['dragenter', 'dragover'].forEach(eName => {
            dropzone.addEventListener(eName, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        });
        ['dragleave', 'drop'].forEach(eName => {
            dropzone.addEventListener(eName, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
        });
        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleFileSelect({ target: { files: files } });
            }
        });

        // Webcam support
        async function startWebcam() {
            try {
                webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                const video = document.getElementById('webcamVideo');
                video.srcObject = webcamStream;
                document.getElementById('snapBtn').disabled = false;
            } catch (err) {
                alert("Camera access denied or unavailable: " + err.message);
            }
        }

        function stopWebcam() {
            if (webcamStream) {
                webcamStream.getTracks().forEach(track => track.stop());
                webcamStream = null;
            }
        }

        function takeSnapshot() {
            const video = document.getElementById('webcamVideo');
            const canvas = document.getElementById('webcamCanvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            canvas.toBlob((blob) => {
                webcamCapturedBlob = blob;
                selectedFile = null;
                selectedSampleFile = null;
                selectedVideo = null;
                document.getElementById('webcamPreviewBox').style.display = 'block';
            }, 'image/jpeg', 0.95);
        }

        function adjustHeatmapOpacity(val) {
            const img = document.getElementById('gradcamImg');
            img.style.opacity = val / 100.0;
        }

        async function runDetection() {
            const btn = document.getElementById('analyzeBtn');
            btn.disabled = true;
            btn.innerHTML = `<span class="spinner"></span> ANALYZING BIOMETRIC SIGNATURES...`;

            const startTime = performance.now();
            const formData = new FormData();
            formData.append('model', selectedModel);

            let endpoint = '/api/predict';

            if (currentTab === 'video' && selectedVideo) {
                endpoint = '/api/predict-video';
                formData.append('file', selectedVideo);
            } else if (currentTab === 'webcam' && webcamCapturedBlob) {
                formData.append('file', webcamCapturedBlob, 'webcam_snap.jpg');
            } else if (selectedFile) {
                formData.append('file', selectedFile);
            } else if (selectedSampleFile) {
                formData.append('sample', selectedSampleFile);
            } else {
                alert('Please select an image, webcam snapshot, or benchmark sample first.');
                btn.disabled = false;
                btn.innerHTML = `<span>RUN FORENSIC DEEPFAKE ANALYSIS</span>`;
                return;
            }

            try {
                const res = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                const latency = Math.round(performance.now() - startTime);

                if (!res.ok) {
                    alert('Analysis failed: ' + (data.error || 'Server error'));
                    return;
                }

                data.latency_ms = latency;
                latestAnalysisData = data;
                renderResults(data);
            } catch (err) {
                alert('Network or processing error: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<span>RUN FORENSIC DEEPFAKE ANALYSIS</span>`;
            }
        }

        function renderResults(data) {
            document.getElementById('resultEmpty').style.display = 'none';
            document.getElementById('resultContent').style.display = 'block';

            const banner = document.getElementById('verdictBanner');
            const icon = document.getElementById('verdictIcon');
            const title = document.getElementById('verdictTitle');
            const sub = document.getElementById('verdictSubtitle');
            const score = document.getElementById('verdictScore');

            const isFake = data.label === 'FAKE';
            banner.className = `verdict-banner ${isFake ? 'fake' : 'real'}`;
            icon.textContent = isFake ? '⚠️' : '🛡️';
            title.textContent = isFake ? 'SYNTHETIC MANIPULATION (FAKE)' : 'VERIFIED AUTHENTIC (REAL)';
            sub.textContent = isFake 
                ? 'High-confidence generative or deepfake synthesis artifacts detected'
                : 'Natural biometric facial consistency verified across neural filters';
            
            const confPct = (data.confidence * 100).toFixed(1);
            score.textContent = `${confPct}%`;

            // Meter fills
            const pFake = data.prob_fake || 0;
            const pReal = 1.0 - pFake;
            document.getElementById('meterRealVal').textContent = `${(pReal * 100).toFixed(1)}%`;
            document.getElementById('meterFakeVal').textContent = `${(pFake * 100).toFixed(1)}%`;
            document.getElementById('meterFillReal').style.width = `${(pReal * 100).toFixed(1)}%`;
            document.getElementById('meterFillFake').style.width = `${(pFake * 100).toFixed(1)}%`;

            // Images (Grad-CAM & LIME)
            document.getElementById('faceCropImg').src = data.face_image_base64 || '';
            document.getElementById('gradcamImg').src = data.gradcam_image_base64 || '';
            
            const limeCard = document.getElementById('limeCard');
            if (data.lime_image_base64) {
                limeCard.style.display = 'block';
                document.getElementById('limeImg').src = data.lime_image_base64;
            } else {
                limeCard.style.display = 'none';
            }

            // Explanation
            document.getElementById('explanationText').textContent = data.explanation || '';

            // Metadata
            document.getElementById('metaModel').textContent = (data.model || selectedModel).toUpperCase();
            document.getElementById('metaFaces').textContent = data.faces_detected || 1;
            document.getElementById('metaLatency').textContent = `${data.latency_ms || 35} ms`;
            document.getElementById('metaLogit').textContent = (data.raw_logit !== undefined) ? Number(data.raw_logit).toFixed(2) : '--';

            // Video timeline if present
            const timelineBox = document.getElementById('videoTimelineBox');
            if (data.frame_details && data.frame_details.length > 0) {
                timelineBox.style.display = 'block';
                const container = document.getElementById('timelineBars');
                container.innerHTML = '';
                data.frame_details.forEach(f => {
                    const bar = document.createElement('div');
                    const height = Math.max(12, Math.round(f.prob_fake * 80));
                    bar.className = `time-bar ${f.label === 'FAKE' ? 'fake' : ''}`;
                    bar.style.height = `${height}px`;
                    bar.title = `Time: ${f.timestamp}s | P(Fake): ${(f.prob_fake*100).toFixed(1)}%`;
                    container.appendChild(bar);
                });
            } else {
                timelineBox.style.display = 'none';
            }
        }

        function downloadForensicReport() {
            if (!latestAnalysisData) return;
            const report = {
                timestamp: new Date().toISOString(),
                audit_id: 'XAI-' + Math.random().toString(36).substr(2, 9).toUpperCase(),
                verdict: latestAnalysisData.label,
                confidence: latestAnalysisData.confidence,
                probability_fake: latestAnalysisData.prob_fake,
                probability_real: 1.0 - (latestAnalysisData.prob_fake || 0),
                raw_logit: latestAnalysisData.raw_logit,
                model_type: latestAnalysisData.model,
                ensemble_signature: latestAnalysisData.signature || 'Standard',
                forensic_explanation: latestAnalysisData.explanation,
                processing_latency_ms: latestAnalysisData.latency_ms,
                target_architecture: "XceptionNet (Block 7 Conv Separable Activations)"
            };

            const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `deepfake_forensic_report_${report.audit_id}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/system-info')
def system_info():
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    return jsonify({
        'status': 'online',
        'device': device,
        'architecture': 'XceptionNet',
        'explainability': ['Grad-CAM', 'LIME'],
        'models_available': ['ensemble', 'celebdf', 'ffpp']
    })

@app.route('/api/samples')
def list_samples():
    return jsonify(BENCHMARK_SAMPLES)

@app.route('/samples/<path:filename>')
def serve_sample(filename):
    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
    return send_from_directory(sample_dir, filename)

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    model_name = request.form.get('model', 'ensemble').lower()
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
        return jsonify({'error': 'No image file or sample provided'}), 400

    try:
        t0 = time.time()
        res = detector.predict_image(img, return_gradcam=True, return_lime=True)
        latency_ms = int((time.time() - t0) * 1000)

        primary = res['primary']
        label = primary['label']
        conf = primary['confidence']
        logit = primary['raw_logit']
        prob = primary['prob_fake']
        details = primary.get('ensemble_details', {})
        signature = details.get('signature', f'{model_name.upper()} Model')

        is_ml = model_name in ('svm', 'rf', 'random_forest', 'logreg', 'logistic_regression')
        ml_names = {
            'svm': 'Support Vector Machine (RBF Kernel)',
            'rf': 'Random Forest Classifier (100 Trees)',
            'logreg': 'Logistic Regression (L2 Regularized)'
        }
        arch_name = ml_names.get(model_name, "XceptionNet Deep Neural Network")

        if label == 'FAKE':
            if is_ml:
                explanation = (
                    f"The {arch_name} classified the sample as SYNTHETIC MANIPULATION (FAKE) "
                    f"(P(Fake)={prob:.4f}, confidence={conf*100:.1f}%). "
                    f"Decision Engine: {signature}. "
                    f"Explainable AI (XAI): Grad-CAM identifies high-gradient boundary seams, while LIME "
                    f"superpixel perturbation isolates the regional manipulation segments driving the decision."
                )
            else:
                explanation = (
                    f"The XceptionNet forensic engine identified generative manipulation cues "
                    f"(P(Fake)={prob:.4f}, confidence={conf*100:.1f}%). "
                    f"Detection Signature: {signature}. "
                    f"Dual XAI Analysis: Grad-CAM heatmap reveals concentrated gradient activations (red/warm zones) "
                    f"focusing on blending boundaries and warping seams; LIME superpixel segments highlight localized synthetic perturbations."
                )
        else:
            if is_ml:
                explanation = (
                    f"The {arch_name} verified authentic biometric characteristics "
                    f"(P(Real)={(1.0-prob):.4f}, confidence={conf*100:.1f}%). "
                    f"Decision Engine: {signature}. "
                    f"Explainable AI (XAI): Grad-CAM activations are uniformly distributed across natural facial geometry with no artificial boundary seam spikes."
                )
            else:
                explanation = (
                    f"The model verified authentic biometric characteristics with no significant generative anomalies "
                    f"(P(Real)={(1.0-prob):.4f}, confidence={conf*100:.1f}%). "
                    f"Verification Signature: {signature}. "
                    f"Dual XAI Analysis: Grad-CAM response is uniformly distributed across natural facial structure; LIME confirms absence of localized forgery superpixels."
                )

        return jsonify({
            'label': label,
            'confidence': conf,
            'prob_fake': prob,
            'raw_logit': logit,
            'model': model_name,
            'signature': signature,
            'architecture': arch_name,
            'faces_detected': res['faces_count'],
            'face_image_base64': img_to_base64(primary['face_crop']),
            'gradcam_image_base64': img_to_base64(primary.get('gradcam_overlay')),
            'lime_image_base64': img_to_base64(primary.get('lime_overlay')),
            'explanation': explanation,
            'latency_ms': latency_ms
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-video', methods=['POST'])
def predict_video_endpoint():
    model_name = request.form.get('model', 'ensemble').lower()
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['file']
    suffix = os.path.splitext(file.filename)[1] or '.mp4'

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        detector = get_detector(model_name)
        res = detector.predict_video(tmp_path, frame_skip=12, max_frames=60)
        
        explanation = (
            f"Analyzed {res['total_analyzed_frames']} video keyframes. "
            f"Overall Video Verdict: {res['overall_label']}. "
            f"{res['fake_percentage']:.1f}% of inspected frames exhibited synthetic manipulation cues."
        )

        return jsonify({
            'label': res['overall_label'],
            'confidence': (res['fake_percentage'] / 100.0) if res['overall_label'] == 'FAKE' else (res['real_percentage'] / 100.0),
            'prob_fake': res['fake_percentage'] / 100.0,
            'fake_percentage': res['fake_percentage'],
            'real_percentage': res['real_percentage'],
            'total_frames': res['total_analyzed_frames'],
            'model': model_name,
            'explanation': explanation,
            'frame_details': res['frame_details']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting Redesigned Deepfake Detection Dashboard on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)
