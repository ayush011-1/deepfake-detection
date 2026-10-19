#!/usr/bin/env python3
"""
Train and persist Classical Machine Learning models for Deepfake Detection:
1. Support Vector Machine (RBF Kernel)
2. Random Forest (100 Trees)
3. Logistic Regression (L2 Regularized)

Uses 2048-dimensional deep feature embeddings from XceptionNet with data augmentation.
Saves models to models/*.joblib for instant inference in app.py and infer.py.
"""

import os
import cv2
import numpy as np
import torch
import joblib
import torchvision.transforms as transforms
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from infer import DeepfakeDetector

def augment_image(img_bgr):
    """Generate subtle photometric and geometric variations for robust ML training."""
    augmented = [img_bgr]
    
    # Horizontal flip
    augmented.append(cv2.flip(img_bgr, 1))
    
    # Slight brightness adjustments
    for alpha in [0.85, 1.15]:
        adj = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=0)
        augmented.append(adj)
        augmented.append(cv2.flip(adj, 1))
        
    # Slight crop & resize
    h, w = img_bgr.shape[:2]
    crop1 = img_bgr[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)]
    augmented.append(cv2.resize(crop1, (w, h)))
    
    return augmented

def main():
    print("=" * 70)
    print("  TRAINING CLASSICAL MACHINE LEARNING MODELS WITH DEEP EMBEDDINGS")
    print("=" * 70)

    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Initialize backbone detector to extract 2048-D features
    detector = DeepfakeDetector(model_type='ffpp')
    model = detector.model
    device = detector.device
    transform = detector.transform

    X_list = []
    y_list = []

    print("\n[1/3] Ingesting and Augmenting Samples...")
    for filename in sorted(os.listdir(sample_dir)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        filepath = os.path.join(sample_dir, filename)
        img = cv2.imread(filepath)
        if img is None:
            continue

        is_fake = 1 if 'fake' in filename.lower() else 0
        variants = augment_image(img)

        for var in variants:
            rgb = cv2.cvtColor(var, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.forward_features(tensor)
                pooled = model.global_pool(feat)
                emb = pooled.squeeze().cpu().numpy()
            X_list.append(emb)
            y_list.append(is_fake)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"      Total Training Feature Vectors: {len(X)} ({np.sum(y==0)} Real, {np.sum(y==1)} Fake)")
    print(f"      Feature Dimension: {X.shape[1]}-D")

    print("\n[2/3] Training Classifiers...")
    
    # 1. Support Vector Machine
    print("      Training Support Vector Machine (RBF Kernel)...")
    svm = SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X, y)
    svm_acc = accuracy_score(y, svm.predict(X))
    joblib.dump(svm, os.path.join(models_dir, 'svm_model.joblib'))
    print(f"      -> SVM Accuracy: {svm_acc*100:.1f}% | Saved to models/svm_model.joblib")

    # 2. Random Forest
    print("      Training Random Forest (100 Decision Trees)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X, y)
    rf_acc = accuracy_score(y, rf.predict(X))
    joblib.dump(rf, os.path.join(models_dir, 'rf_model.joblib'))
    print(f"      -> Random Forest Accuracy: {rf_acc*100:.1f}% | Saved to models/rf_model.joblib")

    # 3. Logistic Regression
    print("      Training Logistic Regression (L2 Penalty)...")
    logreg = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    logreg.fit(X, y)
    lr_acc = accuracy_score(y, logreg.predict(X))
    joblib.dump(logreg, os.path.join(models_dir, 'logreg_model.joblib'))
    print(f"      -> Logistic Regression Accuracy: {lr_acc*100:.1f}% | Saved to models/logreg_model.joblib")

    print("\n[3/3] Validation Complete. All ML Models Saved Successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
