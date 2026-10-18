import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Setup backward compatibility aliases for older pickled timm models
try:
    import timm
    import timm.layers
    import timm.layers.adaptive_avgmax_pool
    sys.modules['timm.models.layers.adaptive_avgmax_pool'] = timm.layers.adaptive_avgmax_pool
except Exception:
    pass

MODEL_PATHS = {
    'celebdf': os.path.join(os.path.dirname(__file__), 'models', 'CelebDF_model_20_epochs_99acc.pt'),
    'ffpp': os.path.join(os.path.dirname(__file__), 'models', 'FFPP_model_20_epochs_99acc.pt')
}

class DeepfakeDetector:
    def __init__(self, model_type='ensemble', device=None):
        """
        model_type: 'ensemble', 'celebdf', or 'ffpp'
        """
        self.model_type = model_type.lower()
        if self.model_type not in ('ensemble', 'celebdf', 'ffpp'):
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'ensemble', 'celebdf', or 'ffpp'.")
        
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
        if self.model_type == 'ensemble':
            self.models['celebdf'] = self._load_single_model('celebdf')
            self.models['ffpp'] = self._load_single_model('ffpp')
            self.model = self.models['ffpp']  # primary default
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

        # Check if this image is already an isolated face portrait (e.g. square crop)
        # where Haar detects a face covering more than 40% of the image area
        if len(faces) == 1 and 0.65 <= aspect_ratio <= 1.5:
            fx, fy, fw, fh = faces[0]
            face_area_ratio = float(fw * fh) / float(w_img * h_img)
            if face_area_ratio >= 0.38 or min(w_img, h_img) <= 450:
                # Retain the complete face with boundaries untouched
                return [(image_bgr.copy(), (0, 0, w_img, h_img))]

        if len(faces) == 0:
            # Fallback: Treat the whole image as face
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
        """
        Generates Grad-CAM heatmap tensor for the input tensor.
        """
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

    def _eval_single_model_tta(self, model, rgb):
        """
        Evaluates model with Test-Time Augmentation (TTA: original + horizontal flip).
        """
        t1 = self.transform(rgb).unsqueeze(0).to(self.device)
        rgb_flip = cv2.flip(rgb, 1)
        t2 = self.transform(rgb_flip).unsqueeze(0).to(self.device)

        with torch.no_grad():
            l1 = model(t1).item()
            l2 = model(t2).item()
            avg_logit = 0.5 * (l1 + l2)
            prob_fake = torch.sigmoid(torch.tensor(avg_logit)).item()

        return float(prob_fake), float(avg_logit)

    def predict_face(self, face_bgr, return_gradcam=True, target_layer='block7', tta=True):
        """
        High-precision inference on cropped face image (BGR).
        Supports Ensemble mode and individual specialist models.
        """
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        
        if self.model_type == 'ensemble':
            prob_c, logit_c = self._eval_single_model_tta(self.models['celebdf'], rgb) if tta else (
                torch.sigmoid(self.models['celebdf'](self.transform(rgb).unsqueeze(0).to(self.device))).item(),
                self.models['celebdf'](self.transform(rgb).unsqueeze(0).to(self.device)).item()
            )
            prob_f, logit_f = self._eval_single_model_tta(self.models['ffpp'], rgb) if tta else (
                torch.sigmoid(self.models['ffpp'](self.transform(rgb).unsqueeze(0).to(self.device))).item(),
                self.models['ffpp'](self.transform(rgb).unsqueeze(0).to(self.device)).item()
            )

            # Ensemble Decision Logic:
            # - FaceForensics++ is sensitive to compression/seam synthesis artifacts.
            # - Celeb-DF is sensitive to high-fidelity face swap blending.
            # If either specialist detects synthetic manipulation with high confidence, flag as FAKE.
            is_c_fake = prob_c >= 0.5
            is_f_fake = prob_f >= 0.5

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
            else:
                # Both models agree REAL
                prob_fake = max(prob_c, prob_f)  # Conservative upper bound on fake probability
                label = 'REAL'
                confidence = 1.0 - prob_fake
                active_model = self.models['ffpp'] if prob_f <= prob_c else self.models['celebdf']
                signature = "Authentic Biometric Consistency (Both Models Verified)"

            raw_logit = logit_f if abs(logit_f) > abs(logit_c) else logit_c
            ensemble_details = {
                'celebdf_prob': float(prob_c),
                'celebdf_logit': float(logit_c),
                'ffpp_prob': float(prob_f),
                'ffpp_logit': float(logit_f),
                'signature': signature
            }
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

        if return_gradcam:
            tensor_for_grad = self.transform(rgb).unsqueeze(0).to(self.device)
            tensor_for_grad.requires_grad = True
            heatmap = self._generate_gradcam(active_model, tensor_for_grad, target_layer_name=target_layer)
            if heatmap is not None:
                gradcam_raw = heatmap
                h, w = face_bgr.shape[:2]
                heatmap_resized = cv2.resize(heatmap, (w, h))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                gradcam_overlay = cv2.addWeighted(face_bgr, 0.60, heatmap_colored, 0.40, 0)

        return {
            'label': label,
            'confidence': float(confidence),
            'prob_fake': float(prob_fake),
            'raw_logit': float(raw_logit),
            'gradcam_overlay': gradcam_overlay,
            'gradcam_raw': gradcam_raw,
            'ensemble_details': ensemble_details
        }

    def predict_image(self, image_path_or_array, return_gradcam=True):
        """
        Inference on an image file path or numpy BGR array.
        Handles face detection; if no face detected, runs on entire image.
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
            pred = self.predict_face(crop_img, return_gradcam=return_gradcam)
            pred['bbox'] = bbox
            pred['face_crop'] = crop_img
            results.append(pred)

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
                pred = self.predict_image(frame, return_gradcam=False)
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
