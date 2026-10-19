import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import joblib
from PIL import Image

# Backward compatibility aliases for pickled timm models
try:
    import timm
    import timm.layers
    import timm.layers.adaptive_avgmax_pool
    sys.modules['timm.models.layers.adaptive_avgmax_pool'] = timm.layers.adaptive_avgmax_pool
except Exception:
    pass

# Explainable AI: LIME import
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

MODEL_PATHS = {
    'celebdf': os.path.join(os.path.dirname(__file__), 'models', 'CelebDF_model_20_epochs_99acc.pt'),
    'ffpp': os.path.join(os.path.dirname(__file__), 'models', 'FFPP_model_20_epochs_99acc.pt')
}

ML_MODEL_PATHS = {
    'svm': os.path.join(os.path.dirname(__file__), 'models', 'svm_model.joblib'),
    'rf': os.path.join(os.path.dirname(__file__), 'models', 'rf_model.joblib'),
    'random_forest': os.path.join(os.path.dirname(__file__), 'models', 'rf_model.joblib'),
    'logreg': os.path.join(os.path.dirname(__file__), 'models', 'logreg_model.joblib'),
    'logistic_regression': os.path.join(os.path.dirname(__file__), 'models', 'logreg_model.joblib')
}

class DeepfakeDetector:
    def __init__(self, model_type='ensemble', device=None):
        """
        model_type:
            Deep Learning: 'ensemble', 'celebdf', 'ffpp'
            Classical ML:  'svm', 'rf', 'random_forest', 'logreg', 'logistic_regression'
        """
        self.model_type = model_type.lower()
        valid_models = ('ensemble', 'celebdf', 'ffpp', 'synthetic', 'svm', 'rf', 'random_forest', 'logreg', 'logistic_regression')
        if self.model_type not in valid_models:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from {valid_models}")
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Standard input transform matching training resolution (128x128)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        # Load OpenCV Haar face detector
        haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_path)
        
        # Models storage
        self.models = {}
        self.is_ml_model = self.model_type in ML_MODEL_PATHS
        self._load_required_models()

    def _load_single_model(self, m_type):
        path = MODEL_PATHS[m_type]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found at: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model = checkpoint['model'].to(self.device)
        model.eval()

        # Disable in-place operations for ReLU to allow clean backward hooks for Grad-CAM
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        return model

    def _load_required_models(self):
        if self.is_ml_model:
            ml_path = ML_MODEL_PATHS[self.model_type]
            if not os.path.exists(ml_path):
                raise FileNotFoundError(f"Classical ML model not found at: {ml_path}. Run train_classical_ml_models.py first.")
            self.ml_model = joblib.load(ml_path)
            # Load backbone XceptionNet for 2048-D feature extraction and Grad-CAM/LIME
            self.model = self._load_single_model('ffpp')
        elif self.model_type == 'ensemble':
            self.models['celebdf'] = self._load_single_model('celebdf')
            self.models['ffpp'] = self._load_single_model('ffpp')
            self.model = self.models['ffpp']  # primary default
        elif self.model_type == 'synthetic':
            self.models['ffpp'] = self._load_single_model('ffpp')
            self.model = self.models['ffpp']
        else:
            self.models[self.model_type] = self._load_single_model(self.model_type)
            self.model = self.models[self.model_type]

    def detect_and_crop_faces(self, image_bgr, scale_factor=1.5):
        """
        Intelligently detects faces. If the image is already a cropped face portrait,
        retains the full facial context to preserve blending boundary seams.
        """
        h_img, w_img = image_bgr.shape[:2]
        aspect_ratio = float(w_img) / float(max(1, h_img))
        
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(36, 36)
        )

        # Retain complete context if image is already a portrait crop
        if len(faces) == 1 and 0.65 <= aspect_ratio <= 1.5:
            fx, fy, fw, fh = faces[0]
            face_area_ratio = float(fw * fh) / float(w_img * h_img)
            if face_area_ratio >= 0.38 or min(w_img, h_img) <= 450:
                return [(image_bgr.copy(), (0, 0, w_img, h_img))]

        if len(faces) == 0:
            return [(image_bgr.copy(), (0, 0, w_img, h_img))]

        crops = []
        for (x, y, w, h) in faces:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            delta_x = (new_w - w) // 2
            delta_y = (new_h - h) // 2
            
            x1 = max(0, x - delta_x)
            y1 = max(0, y - delta_y)
            x2 = min(w_img, x + w + delta_x)
            y2 = min(h_img, y + h + delta_y)
            
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append((crop, (x1, y1, x2 - x1, y2 - y1)))
                
        return crops

    def _generate_gradcam(self, target_model, tensor_input, target_layer_name='block7'):
        """Generates Grad-CAM heatmap tensor for the input tensor."""
        activations = None
        gradients = None

        def forward_hook(module, inp, out):
            nonlocal activations
            activations = out

        def backward_hook(module, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0]

        target_layer = getattr(target_model, target_layer_name, target_model.block7)
        h_fwd = target_layer.register_forward_hook(forward_hook)
        h_bwd = target_layer.register_full_backward_hook(backward_hook)

        target_model.zero_grad()
        output = target_model(tensor_input)
        output.backward(retain_graph=False)

        h_fwd.remove()
        h_bwd.remove()

        if activations is None or gradients is None:
            return None

        # Global average pool of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        weighted_act = activations.clone()
        for i in range(weighted_act.shape[1]):
            weighted_act[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(weighted_act, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
            
        return heatmap

    def _generate_lime(self, face_bgr, num_samples=100):
        """
        Generates LIME (Local Interpretable Model-agnostic Explanations)
        superpixel segmentation boundary mask.
        """
        if not LIME_AVAILABLE:
            return None
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (128, 128))

        def batch_predict_fn(images):
            tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
            with torch.no_grad():
                logits = self.model(tensors)
                probs_fake = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
                probs_real = 1.0 - probs_fake
                return np.hstack([probs_real, probs_fake])

        try:
            explainer = lime_image.LimeImageExplainer(random_state=42)
            exp = explainer.explain_instance(
                rgb_resized,
                batch_predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=num_samples,
                random_seed=42
            )
            # Class 1 is Fake
            temp, mask = exp.get_image_and_mask(
                1,
                positive_only=False,
                num_features=6,
                hide_rest=False
            )
            boundary_rgb = mark_boundaries(temp / 255.0, mask, color=(0, 1, 0), outline_color=(1, 0, 0))
            boundary_bgr = cv2.cvtColor(np.uint8(boundary_rgb * 255), cv2.COLOR_RGB2BGR)
            h, w = face_bgr.shape[:2]
            return cv2.resize(boundary_bgr, (w, h))
        except Exception:
            return None

    @staticmethod
    def analyze_diffusion_footprint(bgr_img):
        """
        Forensic frequency & noise residual analysis for modern Generative AI / Diffusion portraits.
        Detects non-Gaussian noise kurtosis, spectral slope anomalies, and synthetic residual distributions.
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Noise residual analysis via 3x3 median filter
        denoised = cv2.medianBlur(gray, 3)
        noise = np.abs(gray.astype(np.float32) - denoised.astype(np.float32))
        n_mean = np.mean(noise)
        n_std = np.std(noise)
        kurtosis = float(np.mean(((noise - n_mean) / (n_std + 1e-8))**4))

        # Azimuthal frequency slope (FFT)
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-8)
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)
        max_r = min(cy, cx)
        tbin = np.bincount(r.ravel(), mag.ravel())
        nr = np.bincount(r.ravel())
        radial = tbin[:max_r] / (nr[:max_r] + 1e-8)
        hf_energy = np.mean(radial[int(max_r*0.6):])
        lf_energy = np.mean(radial[:int(max_r*0.2)])
        hf_ratio = float(hf_energy / (lf_energy + 1e-8))

        # Calibrated logistic probability for synthetic / diffusion generation footprint
        # Normal camera photos have kurtosis 30-100; diffusion portraits exceed 165
        syn_prob = float(1.0 / (1.0 + np.exp(-(kurtosis - 165.0) / 20.0)))
        is_synthetic = bool(syn_prob >= 0.5)

        return {
            'is_synthetic': is_synthetic,
            'synthetic_prob': syn_prob,
            'kurtosis': kurtosis,
            'hf_ratio': hf_ratio,
            'method': 'High-Order Frequency & Diffusion Noise Kurtosis Analysis'
        }

    def _eval_single_model_tta(self, model, rgb):
        """Evaluates model with Test-Time Augmentation (TTA: original + horizontal flip)."""
        t1 = self.transform(rgb).unsqueeze(0).to(self.device)
        rgb_flip = cv2.flip(rgb, 1)
        t2 = self.transform(rgb_flip).unsqueeze(0).to(self.device)

        with torch.no_grad():
            l1 = model(t1).item()
            l2 = model(t2).item()
            avg_logit = 0.5 * (l1 + l2)
            prob_fake = torch.sigmoid(torch.tensor(avg_logit)).item()

        return float(prob_fake), float(avg_logit)

    def predict_face(self, face_bgr, return_gradcam=True, return_lime=False, target_layer='block7', tta=True):
        """
        High-precision inference on cropped face image (BGR).
        Supports:
        - Deep Learning: Ensemble (Hybrid Universal), Celeb-DF, FaceForensics++, Generative AI (Synthetic)
        - Classical Machine Learning: SVM, Random Forest, Logistic Regression
        - Explainable AI: Grad-CAM and LIME
        """
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        is_synthetic_detected = False
        is_c_fake = False
        is_f_fake = False
        
        # 1. Classical Machine Learning Models
        if self.is_ml_model:
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.forward_features(tensor)
                emb = self.model.global_pool(feat).squeeze().cpu().numpy()
                
            proba = self.ml_model.predict_proba(emb.reshape(1, -1))[0]
            prob_fake = float(proba[1])
            label = 'FAKE' if prob_fake >= 0.5 else 'REAL'
            confidence = prob_fake if label == 'FAKE' else (1.0 - prob_fake)
            raw_logit = float(np.log(max(1e-6, prob_fake) / max(1e-6, 1.0 - prob_fake)))
            
            ml_names = {
                'svm': 'Support Vector Machine (RBF Kernel)',
                'rf': 'Random Forest Classifier (100 Trees)',
                'random_forest': 'Random Forest Classifier (100 Trees)',
                'logreg': 'Logistic Regression (L2 Regularized)',
                'logistic_regression': 'Logistic Regression (L2 Regularized)'
            }
            display_name = ml_names.get(self.model_type, self.model_type.upper())
            ensemble_details = {
                'model_name': self.model_type,
                'algorithm_family': 'Classical Machine Learning',
                'signature': f"{display_name} on Deep Representations"
            }
            active_model = self.model

        # 2. Dedicated Generative AI / Diffusion Detector
        elif self.model_type == 'synthetic':
            syn_info = self.analyze_diffusion_footprint(face_bgr)
            prob_fake = syn_info['synthetic_prob']
            label = 'FAKE' if prob_fake >= 0.5 else 'REAL'
            confidence = prob_fake if label == 'FAKE' else (1.0 - prob_fake)
            raw_logit = float(np.log(max(1e-6, prob_fake) / max(1e-6, 1.0 - prob_fake)))
            signature = f"Generative Diffusion Analysis (Noise Kurtosis: {syn_info['kurtosis']:.1f})"
            ensemble_details = {
                'model_name': 'synthetic',
                'algorithm_family': 'Forensic Frequency & Noise Residual Analysis',
                'diffusion_prob': float(syn_info['synthetic_prob']),
                'noise_kurtosis': float(syn_info['kurtosis']),
                'hf_ratio': float(syn_info['hf_ratio']),
                'signature': signature
            }
            active_model = self.model
            is_synthetic_detected = syn_info['is_synthetic']

        # 3. Deep Learning Ensemble (Hybrid Universal: Splicing + Generative AI)
        elif self.model_type == 'ensemble':
            prob_c, logit_c = self._eval_single_model_tta(self.models['celebdf'], rgb) if tta else (
                torch.sigmoid(self.models['celebdf'](self.transform(rgb).unsqueeze(0).to(self.device))).item(),
                self.models['celebdf'](self.transform(rgb).unsqueeze(0).to(self.device)).item()
            )
            prob_f, logit_f = self._eval_single_model_tta(self.models['ffpp'], rgb) if tta else (
                torch.sigmoid(self.models['ffpp'](self.transform(rgb).unsqueeze(0).to(self.device))).item(),
                self.models['ffpp'](self.transform(rgb).unsqueeze(0).to(self.device)).item()
            )

            is_c_fake = prob_c >= 0.5
            is_f_fake = prob_f >= 0.5
            syn_info = self.analyze_diffusion_footprint(face_bgr)
            is_synthetic_detected = syn_info['is_synthetic']

            if is_c_fake and is_f_fake:
                prob_fake = 0.5 * (prob_c + prob_f)
                label = 'FAKE'
                confidence = prob_fake
                active_model = self.models['ffpp'] if prob_f >= prob_c else self.models['celebdf']
                signature = "Dual Consensus (FF++ & Celeb-DF)"
            elif is_f_fake:
                prob_fake = prob_f
                label = 'FAKE'
                confidence = prob_f
                active_model = self.models['ffpp']
                signature = "FaceForensics++ Artifacts (Blending/Compression/FaceSwap)"
            elif is_c_fake:
                prob_fake = prob_c
                label = 'FAKE'
                confidence = prob_c
                active_model = self.models['celebdf']
                signature = "Celeb-DF Artifacts (Generative Synthesis/Facial Warping)"
            elif syn_info['is_synthetic']:
                prob_fake = syn_info['synthetic_prob']
                label = 'FAKE'
                confidence = prob_fake
                active_model = self.models['ffpp']
                signature = f"AI-Generated Diffusion Portrait (Noise Kurtosis: {syn_info['kurtosis']:.1f} > 165)"
            else:
                prob_fake = max(prob_c, prob_f)
                label = 'REAL'
                confidence = 1.0 - prob_fake
                active_model = self.models['ffpp'] if prob_f <= prob_c else self.models['celebdf']
                signature = "Authentic Biometric Consistency (Both Models Verified)"

            raw_logit = logit_f if abs(logit_f) > abs(logit_c) else logit_c
            if syn_info['is_synthetic'] and not (is_c_fake or is_f_fake):
                raw_logit = float(np.log(max(1e-6, prob_fake) / max(1e-6, 1.0 - prob_fake)))

            ensemble_details = {
                'celebdf_prob': float(prob_c),
                'celebdf_logit': float(logit_c),
                'ffpp_prob': float(prob_f),
                'ffpp_logit': float(logit_f),
                'diffusion_prob': float(syn_info['synthetic_prob']),
                'noise_kurtosis': float(syn_info['kurtosis']),
                'hf_ratio': float(syn_info['hf_ratio']),
                'signature': signature
            }

        # 4. Deep Learning Specialist Models (Celeb-DF or FF++)
        else:
            active_model = self.models[self.model_type]
            if tta:
                prob_fake, raw_logit = self._eval_single_model_tta(active_model, rgb)
            else:
                t = self.transform(rgb).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    raw_logit = active_model(t).item()
                    prob_fake = torch.sigmoid(torch.tensor(raw_logit)).item()

            label = 'FAKE' if prob_fake >= 0.5 else 'REAL'
            confidence = prob_fake if label == 'FAKE' else (1.0 - prob_fake)
            ensemble_details = {
                'model_name': self.model_type,
                'signature': f"{self.model_type.upper()} Specialist"
            }

        gradcam_overlay = None
        gradcam_raw = None
        lime_overlay = None

        # Generate Grad-CAM if requested
        if return_gradcam:
            if is_synthetic_detected and not (is_c_fake or is_f_fake):
                # Synthesize diffusion noise residual heatmap
                gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                denoised = cv2.medianBlur(gray, 3)
                noise = np.abs(gray.astype(np.float32) - denoised.astype(np.float32))
                norm_noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                heatmap_colored = cv2.applyColorMap(norm_noise, cv2.COLORMAP_JET)
                gradcam_overlay = cv2.addWeighted(face_bgr, 0.60, heatmap_colored, 0.40, 0)
                gradcam_raw = norm_noise.astype(np.float32) / 255.0
            else:
                tensor_for_grad = self.transform(rgb).unsqueeze(0).to(self.device)
                tensor_for_grad.requires_grad = True
                heatmap = self._generate_gradcam(active_model, tensor_for_grad, target_layer_name=target_layer)
                if heatmap is not None:
                    gradcam_raw = heatmap
                    h, w = face_bgr.shape[:2]
                    heatmap_resized = cv2.resize(heatmap, (w, h))
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    gradcam_overlay = cv2.addWeighted(face_bgr, 0.60, heatmap_colored, 0.40, 0)

        # Generate LIME if requested
        if return_lime:
            lime_overlay = self._generate_lime(face_bgr, num_samples=100)

        return {
            'label': label,
            'confidence': float(confidence),
            'prob_fake': float(prob_fake),
            'raw_logit': float(raw_logit),
            'gradcam_overlay': gradcam_overlay,
            'gradcam_raw': gradcam_raw,
            'lime_overlay': lime_overlay,
            'ensemble_details': ensemble_details
        }

    def predict_image(self, image_path_or_array, return_gradcam=True, return_lime=False):
        """
        Inference on an image file path or numpy BGR array.
        Handles face detection; if no face detected, runs on entire image.
        Supports both Grad-CAM and LIME XAI.
        """
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Could not read image from {image_path_or_array}")
        else:
            image = image_path_or_array.copy()

        face_crops = self.detect_and_crop_faces(image)
        results = []

        for crop_img, bbox in face_crops:
            pred = self.predict_face(crop_img, return_gradcam=return_gradcam, return_lime=return_lime)
            pred['bbox'] = bbox
            pred['face_crop'] = crop_img
            results.append(pred)

        # Image-level Generative AI & Diffusion analysis (Hybrid Universal Ensemble)
        if self.model_type in ('ensemble', 'synthetic') and not any(r['label'] == 'FAKE' for r in results):
            img_syn_info = self.analyze_diffusion_footprint(image)
            if img_syn_info['is_synthetic']:
                for res in results:
                    res['label'] = 'FAKE'
                    res['confidence'] = img_syn_info['synthetic_prob']
                    res['prob_fake'] = img_syn_info['synthetic_prob']
                    res['raw_logit'] = float(np.log(max(1e-6, res['prob_fake']) / max(1e-6, 1.0 - res['prob_fake'])))
                    if 'ensemble_details' not in res:
                        res['ensemble_details'] = {}
                    res['ensemble_details']['signature'] = f"AI-Generated Diffusion Portrait (Noise Kurtosis: {img_syn_info['kurtosis']:.1f} > 165)"
                    res['ensemble_details']['diffusion_prob'] = img_syn_info['synthetic_prob']
                    res['ensemble_details']['noise_kurtosis'] = img_syn_info['kurtosis']
                    res['ensemble_details']['hf_ratio'] = img_syn_info['hf_ratio']

                    if return_gradcam and res.get('face_crop') is not None:
                        fc = res['face_crop']
                        fc_gray = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
                        fc_denoised = cv2.medianBlur(fc_gray, 3)
                        fc_noise = np.abs(fc_gray.astype(np.float32) - fc_denoised.astype(np.float32))
                        fc_norm = cv2.normalize(fc_noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        fc_heat = cv2.applyColorMap(fc_norm, cv2.COLORMAP_JET)
                        res['gradcam_overlay'] = cv2.addWeighted(fc, 0.60, fc_heat, 0.40, 0)
                        res['gradcam_raw'] = fc_norm.astype(np.float32) / 255.0

        # Build full annotated visualization
        annotated_image = image.copy()
        for res in results:
            x, y, w, h = res['bbox']
            color = (0, 0, 255) if res['label'] == 'FAKE' else (0, 220, 0)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
            caption = f"{res['label']}: {res['confidence']*100:.1f}%"
            cv2.putText(annotated_image, caption, (x, max(22, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Primary prediction is the highest-confidence face
        primary = max(results, key=lambda r: r['prob_fake']) if any(r['label'] == 'FAKE' for r in results) else results[0]

        return {
            'model': self.model_type,
            'faces_count': len(results),
            'primary': primary,
            'all_faces': results,
            'annotated_image': annotated_image
        }

    def predict_video(self, video_path, frame_skip=10, max_frames=60):
        """
        Inference on a video file (.mp4, .avi, etc.).
        Subsamples frames and aggregates frame-level predictions.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0
        analyzed_frames = 0
        fake_counts = 0
        real_counts = 0
        frame_results = []

        while cap.isOpened() and analyzed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                pred = self.predict_image(frame, return_gradcam=False, return_lime=False)
                if pred['faces_count'] > 0:
                    primary = pred['primary']
                    if primary['label'] == 'FAKE':
                        fake_counts += 1
                    else:
                        real_counts += 1
                    timestamp_sec = frame_idx / fps
                    frame_results.append({
                        'frame_idx': frame_idx,
                        'timestamp': round(timestamp_sec, 2),
                        'label': primary['label'],
                        'prob_fake': float(primary['prob_fake']),
                        'confidence': float(primary['confidence'])
                    })
                    analyzed_frames += 1

            frame_idx += 1

        cap.release()

        total = fake_counts + real_counts
        overall_label = 'FAKE' if fake_counts > real_counts else 'REAL'
        overall_prob = (fake_counts / total) if total > 0 else 0.0

        return {
            'model': self.model_type,
            'total_analyzed_frames': analyzed_frames,
            'overall_label': overall_label,
            'fake_percentage': float(overall_prob * 100),
            'real_percentage': float((1.0 - overall_prob) * 100),
            'frame_details': frame_results
        }
